# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This source code was adapted from https://github.com/facebookresearch/speech-resynthesis by Xiaoxiao Miao (NII, Japan).

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

from multiprocessing import Manager, Pool
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write
from tqdm import tqdm
import soundfile as sf

from dataset_test import latentDataset, mel_spectrogram, \
    MAX_WAV_VALUE
from utils import AttrDict
from models_test import latentGenerator,SoftPredictor
import joblib
import fairseq

h = None
xvector_cache = {}  # Global cache for xvectors: {utt_id: numpy_array}
scp_dict = {}  # Global dict for scp lookup: {utt_id: xvector_array}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))


def collate_fn_variable_length(batch):
    """Custom collate function to handle variable-length audio tensors.
    
    PyTorch's default collate_fn uses torch.stack which requires all tensors
    to have identical shapes. This function handles variable-length sequences
    by padding them to the maximum length in the batch.
    
    Args:
        batch: List of samples from dataset, each is a tuple:
               (x_dict, gt_audio, _, filename)
    
    Returns:
        (x_batch, gt_audio_batch, _, filenames)
        - x_batch: dict with keys like 'mel_loss' containing padded tensors
        - gt_audio_batch: padded ground truth audio tensor or list of tensors
        - filenames: list of utterance filenames
    """
    if len(batch) == 0:
        return None
    
    # Separate components
    x_list = [item[0] for item in batch]  # List of dicts
    gt_audio_list = [item[1] for item in batch]  # List of audio tensors
    empty_list = [item[2] for item in batch]  # Empty list (usually None)
    filenames = [item[3] for item in batch]  # List of filenames
    
    # Handle x_dict (contains mel_loss, pitch, etc. with variable lengths)
    x_batch = {}
    if isinstance(x_list[0], dict):
        keys = x_list[0].keys()
        
        for key in keys:
            tensors = [x[key] for x in x_list]
            
            # Find max length in this batch for this key
            max_len = max(t.shape[-1] if t.ndim > 0 else 1 for t in tensors)
            
            # Pad all tensors to max length
            padded = []
            for t in tensors:
                if t.ndim == 0:  # Scalar
                    padded.append(t)
                elif t.ndim == 1:
                    # 1D tensor: pad to max_len
                    if t.shape[0] < max_len:
                        pad_size = max_len - t.shape[0]
                        t = torch.nn.functional.pad(t, (0, pad_size), mode='constant', value=0)
                    padded.append(t)
                else:  # 2D or higher: pad last dimension
                    if t.shape[-1] < max_len:
                        pad_size = max_len - t.shape[-1]
                        # Create padding tuple: (0, pad_size) for last dim, (0, 0) for others
                        pad_tuple = (0, pad_size) + (0, 0) * (t.ndim - 1)
                        t = torch.nn.functional.pad(t, pad_tuple, mode='constant', value=0)
                    padded.append(t)
            
            # Stack padded tensors
            x_batch[key] = torch.stack(padded, dim=0)
    else:
        # If x is not a dict, use default stacking (assume same size)
        x_batch = torch.stack(x_list, dim=0)
    
    # Handle gt_audio (variable length)
    if gt_audio_list[0] is not None:
        max_audio_len = max(a.shape[-1] if a.ndim > 0 else 1 for a in gt_audio_list)
        padded_audio = []
        
        for audio in gt_audio_list:
            if audio.ndim == 0:  # Scalar
                padded_audio.append(audio)
            elif audio.ndim == 1:
                # 1D audio: pad to max length
                if audio.shape[0] < max_audio_len:
                    pad_size = max_audio_len - audio.shape[0]
                    audio = torch.nn.functional.pad(audio, (0, pad_size), mode='constant', value=0)
                padded_audio.append(audio)
            else:
                # Multi-dim: pad last dimension
                if audio.shape[-1] < max_audio_len:
                    pad_size = max_audio_len - audio.shape[-1]
                    pad_tuple = (0, pad_size) + (0, 0) * (audio.ndim - 1)
                    audio = torch.nn.functional.pad(audio, pad_tuple, mode='constant', value=0)
                padded_audio.append(audio)
        
        gt_audio_batch = torch.stack(padded_audio, dim=0)
    else:
        gt_audio_batch = None
    
    return x_batch, gt_audio_batch, empty_list, filenames


def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, x, xv_path):
    start = time.time()
    y_g_hat = generator.gen_vpc(xv_path,**x).to(device)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


def init_worker(arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global dataset
    global device
    global a
    global h
    global scp_dict

    a = arguments

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = latentGenerator(h).to(device)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])

    file_list = []
    for line in open(a.input_test_file):
        temp = line.strip().split(" ")[-1]
        file_list.append(temp)
    
    dataset = latentDataset(file_list, -1, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss,  device=device)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ============================================
    # OPTIMIZATION 1: Pre-load .scp file into dictionary (O(1) lookup)
    # ============================================
    print("Pre-loading xvector .scp file...")
    scp_file = str(a.xv_dir) + '/pseudo_xvector.scp'
    ark_file = str(a.xv_dir) + '/pseudo_xvector.ark'
 
    if os.path.exists(scp_file) and os.path.exists(ark_file):
        from kaldiio import ReadHelper
        
        # Read all xvectors from archive into memory
        scp_dict = {}
        with ReadHelper(f'scp:{scp_file}') as reader:
            for key, mat in reader:
                scp_dict[key] = mat.astype(np.float32)
        
        print(f"✓ Loaded {len(scp_dict)} xvectors into memory")
    else:
        print(f"Warning: {scp_file} or {ark_file} not found")



@torch.no_grad()
def inference(item_index):
    x, gt_audio, _, filename = dataset[item_index]
    x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
    gt_audio = torch.autograd.Variable(gt_audio.to(device, non_blocking=False))
    fname_out_name = Path(filename).stem
    if a.step == None:
        xv_path = str(a.xv_dir) + '/' + fname_out_name +  '.xvector'
    else:
        xv_path = str(a.xv_dir) + '/' + fname_out_name +  '_{}.xvector'.format(a.step)
    audio, rtf = generate(h, generator, x, xv_path)
    output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
    audio = librosa.util.normalize(audio.astype(np.float32))
    write(output_file, h.sampling_rate, audio)

    if gt_audio is not None:
        output_file = os.path.join(a.output_original_dir, fname_out_name + '_gt.wav')
        gt_audio = librosa.util.normalize(gt_audio.squeeze().cpu().numpy().astype(np.float32))
        write(output_file, h.sampling_rate, gt_audio)

# @torch.no_grad()
# def inference_scp(item_index):
#     """Inference using pre-loaded xvectors from memory (OPTIMIZED).
    
#     Optimizations:
#     1. O(1) dictionary lookup instead of scanning .scp file
#     2. Xvectors kept in memory (scp_dict) instead of temp files
#     3. No disk I/O per inference call
#     """
#     x, gt_audio, _, filename = dataset[item_index]
#     x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
#     gt_audio = torch.autograd.Variable(gt_audio.to(device, non_blocking=False))
#     fname_out_name = Path(filename).stem
    
#     # OPTIMIZATION 1: O(1) lookup from pre-loaded dictionary
#     if fname_out_name not in scp_dict:
#         print(f"Warning: xvector not found for {fname_out_name}")
#         return
    
#     xvector = scp_dict[fname_out_name]
    
#     # OPTIMIZATION 2: Keep xvector in memory, pass directly via temp file
#     # Note: generate() expects file path, so we write only when needed
#     import tempfile
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.xvector') as tmp:
#         xvector.tofile(tmp.name)
#         xv_path = tmp.name
    
#     try:
#         audio, rtf = generate(h, generator, x, xv_path)
#         output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
#         audio = librosa.util.normalize(audio.astype(np.float32))
#         write(output_file, h.sampling_rate, audio)

#         if gt_audio is not None and a.write_original:
#             if not os.path.exists(a.output_original_dir):
#                 os.makedirs(a.output_original_dir, exist_ok=True)
#             output_file = os.path.join(a.output_original_dir, fname_out_name + '.wav')
#             gt_audio = librosa.util.normalize(gt_audio.squeeze().cpu().numpy().astype(np.float32))
#             write(output_file, h.sampling_rate, gt_audio)
#     finally:
#         # Clean up temporary file
#         if os.path.exists(xv_path):
#             os.remove(xv_path)

@torch.no_grad()
def inference_scp_batch(x_batch, gt_audio_batch, filenames, pbar=None):
    """Process entire batch with GPU optimization and original audio length tracking.
    
    Args:
        x_batch: Dict with batched tensors {key: (batch_size, ...)}
        gt_audio_batch: Batched ground truth audio (batch_size, ...)
        filenames: List of output filenames
        pbar: tqdm progress bar object
    """
    # Move batch to GPU once
    x_batch = {k: v.to(device) for k, v in x_batch.items()}
    gt_audio_batch = gt_audio_batch.to(device)
    
    batch_size = len(filenames)
    
    # Pre-load all xvectors for this batch (reduces I/O overhead)
    import tempfile
    xvector_paths = {}
    original_lengths = {}
    temp_files = []
    
    for fname in filenames:
        fname_out_name = Path(fname).stem
        if fname_out_name not in scp_dict:
            print(f"Warning: xvector not found for {fname_out_name}")
            continue
        
        # Get original audio length from file (before padding)
        try:
            info = sf.info(fname)
            original_lengths[fname_out_name] = info.frames
        except:
            original_lengths[fname_out_name] = None
        
        # Pre-create temp xvector file
        xvector = scp_dict[fname_out_name]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xvector')
        xvector.tofile(tmp.name)
        xvector_paths[fname_out_name] = tmp.name
        temp_files.append(tmp.name)
        tmp.close()
    
    # Process batch through generator with GPU-kept tensors
    for i, fname in enumerate(filenames):
        fname_out_name = Path(fname).stem
        if fname_out_name not in xvector_paths:
            if pbar:
                pbar.update(1)
            continue
        
        # Extract single item features from batch (stays on GPU)
        x_single = {k: v[i:i+1] for k, v in x_batch.items()}
        
        # Squeeze extra dimensions for f0 if present
        if 'f0' in x_single and x_single['f0'].dim() == 4:
            x_single['f0'] = x_single['f0'].squeeze(1)
        
        gt_audio_single = gt_audio_batch[i:i+1] if gt_audio_batch is not None else None
        xv_path = xvector_paths[fname_out_name]
        
        try:
            audio, rtf = generate(h, generator, x_single, xv_path)
            
            # Trim to original audio length (fix for padded audios)
            if fname_out_name in original_lengths and original_lengths[fname_out_name]:
                max_samples = original_lengths[fname_out_name]
                audio = audio[:max_samples]
            
            output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
            audio = librosa.util.normalize(audio.astype(np.float32))
            write(output_file, h.sampling_rate, audio)
            
            if gt_audio_single is not None and a.write_original:
                if not os.path.exists(a.output_original_dir):
                    os.makedirs(a.output_original_dir, exist_ok=True)
                output_file = os.path.join(a.output_original_dir, fname_out_name + '.wav')
                gt_audio_single_cpu = librosa.util.normalize(
                    gt_audio_single.squeeze().cpu().numpy().astype(np.float32)
                )
                write(output_file, h.sampling_rate, gt_audio_single_cpu)
        finally:
            if pbar:
                pbar.update(1)
    
    # Clean up all temp files
    for tmp_path in temp_files:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@torch.no_grad()
def inference_with_prefetch(data_loader, total_items):
    """Process batches with GPU stream prefetching and progress bar.
    
    Overlaps:
    - Stream 0: Current batch GPU processing
    - Stream 1: Next batch CPU→GPU transfer + preprocessing
    """
    stream_compute = torch.cuda.Stream()
    stream_prefetch = torch.cuda.Stream()
    
    iterator = iter(data_loader)
    
    # Prefetch first batch
    try:
        batch = next(iterator)
    except StopIteration:
        return
    
    # Create progress bar
    with tqdm(total=total_items, desc="Generating audio", unit="utterance") as pbar:
        while True:
            # Prefetch next batch on background stream
            try:
                next_batch = next(iterator)
                prefetch_available = True
            except StopIteration:
                prefetch_available = False
            
            # Process current batch on main stream
            with torch.cuda.stream(stream_compute):
                x_batch, gt_audio_batch, _, filenames = batch
                inference_scp_batch(x_batch, gt_audio_batch, filenames, pbar=pbar)
            
            # Synchronize only when necessary
            torch.cuda.current_stream().wait_stream(stream_compute)
            
            if not prefetch_available:
                break
            
            batch = next_batch
    
    torch.cuda.synchronize()

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_test_file', default=None)
    parser.add_argument('--test_wav_dir', default=None)
    parser.add_argument('--feat_model', type=Path)
    parser.add_argument('--kmeans_model', type=Path,nargs="?",default=None)
    parser.add_argument('--soft_model', type=Path,nargs="?",default=None)
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--output_original_dir', default='generated_files_original')
    parser.add_argument('--write_original', action='store_true')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--f0_dir', type=Path)
    parser.add_argument('--xv_dir', type=Path)
    parser.add_argument('--step', default=None)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader (OPTIMIZATION 3)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes (default 0 for GPU inference)')

    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    file_list = []
    for line in open(a.input_test_file):
        file_list.append(line.strip())

    init_worker(a)

    # ============================================
    # OPTIMIZATION 3: Use PyTorch DataLoader for batching with custom collate
    # ============================================
    print(f"Creating DataLoader with batch_size={a.batch_size}, num_workers={a.num_workers}")
    
    from torch.utils.data import DataLoader
    
    # Use custom collate function to handle variable-length audio
    data_loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_variable_length  # Custom collate for variable-length tensors
    )

    # Process batches instead of individual items
    total_items = len(dataset)
    processed = 0
    
    print(f"Starting inference on {total_items} items with batch processing...")
    
    # for batch_idx, batch in enumerate(data_loader):
    #     if batch is None:
    #         continue
            
    #     x_batch, gt_audio_batch, _, filenames = batch
    #     batch_size = len(filenames)
        
    #     for item_idx in range(batch_size):
    #         # Extract single item from batch
    #         inference_scp(processed)
    #         bar = progbar(processed, total_items)
    #         message = f'{bar} {processed}/{total_items}'
    #         stream(message)
    #         processed += 1

    inference_with_prefetch(data_loader, total_items)

if __name__ == '__main__':
    main()

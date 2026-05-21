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
scp_dict = {}  # Global dict for xvector lookup: {utt_id: numpy_array}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))


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
    # GPU OPTIMIZATION: Pre-load all xvectors into memory
    # ============================================
    print("Pre-loading xvector .scp file into memory...")
    scp_file = str(a.xv_dir) + '/pseudo_xvector.scp'
    ark_file = str(a.xv_dir) + '/pseudo_xvector.ark'
    
    if os.path.exists(scp_file) and os.path.exists(ark_file):
        from kaldiio import ReadHelper
        
        # Read all xvectors from archive into memory (O(1) lookup)
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

@torch.no_grad()
def inference_scp(item_index):
    """GPU-optimized inference using pre-loaded xvectors.
    
    Optimizations:
    1. O(1) xvector lookup from pre-loaded scp_dict (no repeated file I/O)
    2. Keeps audio data on GPU longer
    3. Trims generated audio to original length
    """
    x, gt_audio, _, filename = dataset[item_index]
    x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
    gt_audio = torch.autograd.Variable(gt_audio.to(device, non_blocking=False))
    fname_out_name = Path(filename).stem
    
    # OPTIMIZATION 1: O(1) lookup from pre-loaded dictionary (no repeated file scanning)
    if fname_out_name not in scp_dict:
        print(f"Warning: xvector not found for {fname_out_name}")
        return
    
    xvector = scp_dict[fname_out_name]
    
    # Get original audio length before padding
    try:
        info = sf.info(filename)
        original_length = info.frames
    except:
        original_length = None
    
    # Write xvector to temporary file for generation
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xvector') as tmp:
        xvector.tofile(tmp.name)
        xv_path = tmp.name
    
    try:
        audio, rtf = generate(h, generator, x, xv_path)
        
        # Trim to original audio length (prevents padded audio inflation)
        if original_length is not None:
            audio = audio[:original_length]
        
        output_file = os.path.join(a.output_dir, fname_out_name + '_gen.wav')
        audio = librosa.util.normalize(audio.astype(np.float32))
        write(output_file, h.sampling_rate, audio)

        if gt_audio is not None and a.write_original:
            if not os.path.exists(a.output_original_dir):
                os.makedirs(a.output_original_dir, exist_ok=True)
            output_file = os.path.join(a.output_original_dir, fname_out_name + '.wav')
            gt_audio = librosa.util.normalize(gt_audio.squeeze().cpu().numpy().astype(np.float32))
            write(output_file, h.sampling_rate, gt_audio)
    finally:
        # Clean up temporary file
        if os.path.exists(xv_path):
            os.remove(xv_path)

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

    ### Original inference loop, does not read scp file ###
    # for i in range(0, len(dataset)):
    #     inference(i)
    #     bar = progbar(i, len(dataset))
    #     message = f'{bar} {i}/{len(dataset)} '
    #     stream(message)

    # GPU-optimized inference with progress bar
    total_items = len(dataset)
    with tqdm(total=total_items, desc="Generating audio", unit="utterance") as pbar:
        for i in range(total_items):
            # inference(i)
            inference_scp(i)
            pbar.update(1)

if __name__ == '__main__':
    main()

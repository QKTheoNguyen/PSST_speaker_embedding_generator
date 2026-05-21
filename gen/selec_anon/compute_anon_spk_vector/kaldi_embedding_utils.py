"""Utilities for reading/writing embeddings in Kaldi format."""
import torch
import numpy as np
from pathlib import Path
from kaldiio import ReadHelper, WriteHelper

class KaldiEmbeddings:
    """Wrapper for Kaldi xvector format compatible with GANAnonymizer."""
    
    def __init__(self, xvector_scp, spk2gender_file=None, utt2spk_file=None):
        self.xvector_scp = xvector_scp
        self.embeddings = {}
        self.utts = []
        self.genders = []
        self.speakers = []
        
        # Load embeddings from Kaldi
        with ReadHelper(f'scp:{xvector_scp}') as reader:
            for key, mat in reader:
                self.embeddings[key] = torch.from_numpy(mat).float().clone()
                self.utts.append(key)
                self.speakers.append(key)
        
        # Load metadata
        self.spk2gender = {}
        if spk2gender_file and Path(spk2gender_file).exists():
            with open(spk2gender_file) as f:
                for line in f:
                    spk, gender = line.strip().split()
                    self.spk2gender[spk] = gender
                    
            # Map utterances to genders
            self.genders = [self.spk2gender.get(utt, 'M') for utt in self.utts]
        else:
            self.genders = ['M'] * len(self.utts)
        
        self.utt2spk = {}
        if utt2spk_file and Path(utt2spk_file).exists():
            with open(utt2spk_file) as f:
                for line in f:
                    utt, spk = line.strip().split()
                    self.utt2spk[utt] = spk
    
    def __len__(self):
        return len(self.utts)
    
    def __getitem__(self, idx):
        utt = self.utts[idx]
        return utt, self.embeddings[utt]
    
    def write_kaldi(self, output_dir, prefix='pseudo_xvector'):
        """Write anonymized embeddings back to Kaldi format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scp_file = output_dir / f'{prefix}.scp'
        ark_file = output_dir / f'{prefix}.ark'
        
        with WriteHelper(f'ark,scp:{ark_file},{scp_file}') as writer:
            for utt in self.utts:
                # writer(utt, self.embeddings[utt].numpy())
                writer(utt, self.embeddings[utt].cpu().numpy())
        
        print(f"Wrote {len(self.utts)} embeddings to {output_dir}")
    
    def update_embedding(self, utt, new_embedding):
        """Update embedding for utterance."""
        self.embeddings[utt] = new_embedding


def load_speaker_embeddings(data_dir, vector_dir, spk_level=True):
    """Load speaker-level embeddings (aggregate utterances by speaker)."""
    # Read metadata
    utt2spk_file = data_dir / 'utt2spk'
    spk2gender_file = data_dir / 'spk2gender'
    xvector_scp = vector_dir / 'xvector.scp'
    
    # Load utterance-level embeddings
    utt_embeddings = KaldiEmbeddings(str(xvector_scp), 
                                      str(spk2gender_file),
                                      str(utt2spk_file))
    
    if not spk_level:
        return utt_embeddings
    
    # Aggregate to speaker-level by averaging
    spk_embeddings = KaldiEmbeddings(str(xvector_scp), 
                                      str(spk2gender_file),
                                      str(utt2spk_file))
    
    # Average utterances by speaker
    spk_dict = {}
    for i, utt in enumerate(utt_embeddings.utts):
        spk = utt_embeddings.utt2spk.get(utt, utt)
        if spk not in spk_dict:
            spk_dict[spk] = []
        spk_dict[spk].append(utt_embeddings.embeddings[utt])
    
    # Create speaker-level embeddings
    spk_embeddings.embeddings = {}
    spk_embeddings.utts = []
    spk_embeddings.speakers = []
    spk_embeddings.genders = []
    
    for spk, embs in spk_dict.items():
        avg_emb = torch.mean(torch.stack(embs), dim=0)
        spk_embeddings.embeddings[spk] = avg_emb
        spk_embeddings.utts.append(spk)
        spk_embeddings.speakers.append(spk)
        spk_embeddings.genders.append(spk_embeddings.spk2gender.get(spk, 'M'))
    
    return spk_embeddings
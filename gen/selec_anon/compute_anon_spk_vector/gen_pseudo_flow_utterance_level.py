#!/usr/bin/env python3
"""Generate utterance-level pseudo xvectors using flow matching-based anonymization.

This script:
1. Reads original utterance-level embeddings
2. For each utterance, generates a dissimilar flow-matched embedding
3. Outputs utterance-level pseudo_xvector.scp/.ark for inference
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import cosine
from kaldiio import ReadHelper, WriteHelper

sys.path.insert(0, str(Path(__file__).parent))
from flow_matching.embeddings_generator import EmbeddingsGenerator
from kaldi_embedding_utils import KaldiEmbeddings


class FlowAnonymizer:
    """Anonymizer using flow matching to generate pseudo-speaker vectors."""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.generator = None
    
    def load_flow_model(self, flow_model_dir, flow_model_name='flow.pt'):
        """Load flow matching model.
        
        Args:
            flow_model_dir: Directory with flow.pt checkpoint
            flow_model_name: Name of flow matching checkpoint
            steps: ODE solver steps for generation
        """
        print(f"Loading flow matching model from {flow_model_dir}...")
        flow_path = Path(flow_model_dir) / flow_model_name
        
        if not flow_path.exists():
            raise FileNotFoundError(f"Flow model not found at {flow_path}")
        
        self.generator = EmbeddingsGenerator(flow_path=str(flow_path), device=self.device)

    def generate_from_speaker(self, spk_vec, w=0.7, steps=20):
        """Generate anonymized speaker vector.
        
        Args:
            spk_vec: Original speaker vector (192,)
            w: Speaker weight controlling the strength of original speaker identity
            
        Returns:
            torch.Tensor: Pseudo-speaker vector
        """

        z_orig = self.generator.generate_pseudo_speaker(spk_vec, w, steps)
        return z_orig


def generate_utterance_level_pseudo_vectors(
    data_dir: Path,
    vector_dir: Path,
    flow_model_dir: Path,
    output_dir: Path,
    speaker_weight: float = 0.7,
    flow_steps: int = 20,
    device: str = 'cuda:0'
):
    """Generate utterance-level pseudo-speaker vectors using flow matching.
    
    Args:
        data_dir: Data directory (with utt2spk, spk2gender)
        vector_dir: Directory with original xvector.scp
        flow_model_dir: Directory with flow matching model (flow.pt)
        output_dir: Output directory for pseudo_xvector.{scp,ark}
        num_flow_samples: Number of flow samples to generate
        similarity_threshold: Max similarity between original and flow vector
        flow_steps: ODE solver steps for flow matching generation
        device: GPU device
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # Load original utterance-level embeddings
    # ============================================
    print("Loading original utterance-level embeddings...")
    original_embeddings = KaldiEmbeddings(
        xvector_scp=str(vector_dir / 'xvector.scp'),
        spk2gender_file=str(data_dir / 'spk2gender'),
        utt2spk_file=str(data_dir / 'utt2spk')
    )
    
    print(f"Loaded {len(original_embeddings)} utterances")
    
    # ============================================
    # Initialize flow matching anonymizer
    # ============================================
    print(f"Loading flow matching model from {flow_model_dir}...")
    anonymizer = FlowAnonymizer(device=device)
    anonymizer.load_flow_model(flow_model_dir, flow_model_name='flow.pt')
    
    # ============================================
    # Generate pseudo vectors (utterance level)
    # ============================================
    print(f"Generating pseudo vectors for {len(original_embeddings)} utterances...")
    
    pseudo_xvecs = {}
    for i in tqdm(range(len(original_embeddings))):
        utt_id, orig_vec = original_embeddings[i]
        
        # Select dissimilar flow-matched vector
        pseudo_vec = anonymizer.generate_from_speaker(spk_vec=orig_vec, w=speaker_weight, steps=flow_steps)
        pseudo_xvecs[utt_id] = pseudo_vec.cpu().numpy()
    
    # ============================================
    # Write output as individual .xvector files
    # ============================================
    print(f"Writing pseudo vectors to {output_dir}...")
    
    # Also create ark,scp format for reference
    scp_file = output_dir / 'pseudo_xvector.scp'
    ark_file = output_dir / 'pseudo_xvector.ark'
    
    with WriteHelper(f'ark,scp:{ark_file},{scp_file}') as writer:
        for utt_id in sorted(pseudo_xvecs.keys()):
            writer(utt_id, pseudo_xvecs[utt_id])
    
    # Also write individual .xvector files (one per utterance)
    print(f"Writing individual .xvector files...")
    for utt_id in tqdm(sorted(pseudo_xvecs.keys())):
        xvector_file = output_dir / f'{utt_id}.xvector'
        pseudo_xvecs[utt_id].astype(np.float32).tofile(xvector_file)
    
    print(f"✓ Wrote {len(pseudo_xvecs)} pseudo vectors")
    print(f"  Individual files: {output_dir}/*.xvector")
    print(f"  Kaldi format: {scp_file} + {ark_file}")
    
    return pseudo_xvecs


def main():
    parser = argparse.ArgumentParser(
        description='Generate utterance-level pseudo xvectors with flow matching'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory (with utt2spk, spk2gender)')
    parser.add_argument('--vector_dir', type=str, required=True,
                        help='Directory with original xvector.scp')
    parser.add_argument('--flow_model_dir', type=str, required=True,
                        help='Directory with flow matching model (flow.pt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudo_xvector.{scp,ark}')
    parser.add_argument('--speaker_weight', type=float, default=0.7,
                        help='Speaker weight')
    parser.add_argument('--flow_steps', type=int, default=50,
                        help='ODE solver steps for flow generation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device')
    
    args = parser.parse_args()
    
    generate_utterance_level_pseudo_vectors(
        data_dir=Path(args.data_dir),
        vector_dir=Path(args.vector_dir),
        flow_model_dir=Path(args.flow_model_dir),
        output_dir=Path(args.output_dir),
        speaker_weight=args.speaker_weight,
        flow_steps=args.flow_steps,
        device=args.device
    )
    
    print("✓ Utterance-level pseudo-speaker generation complete!")


if __name__ == '__main__':
    main()

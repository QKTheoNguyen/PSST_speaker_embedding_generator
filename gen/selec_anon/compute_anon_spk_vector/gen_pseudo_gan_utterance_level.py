#!/usr/bin/env python3
"""Generate utterance-level pseudo xvectors using GAN-based anonymization.

This script:
1. Reads original utterance-level embeddings
2. For each utterance, generates a dissimilar GAN embedding
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
from kaldi_embedding_utils import KaldiEmbeddings, load_speaker_embeddings
from WGAN.embeddings_generator import EmbeddingsGenerator
sys.path.insert(0, str(Path(__file__).parent))


class GANAnonymizer:
    """Speaker anonymization using GAN-generated embeddings."""

    def __init__(self, device=None, sim_threshold=0.7, **kwargs):
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sim_threshold = sim_threshold
        
        self.gan_vectors = None
        self.unused_indices = None
        self.gan_model_name = None
        self.n = 1000

    def load_gan_model(self, model_dir: Path, gan_model_name='gan.pt', n_samples=1000):
        """Load or generate GAN-based embeddings."""
        self.gan_model_name = gan_model_name
        self.n = n_samples
        
        vectors_file = model_dir / 'gan_vectors.pt'
        
        if vectors_file.exists():
            print(f"Loading pre-generated GAN vectors from {vectors_file}")
            self.gan_vectors = torch.load(vectors_file, map_location=self.device)
            self.unused_indices = torch.arange(len(self.gan_vectors))
        else:
            self._generate_gan_vectors(model_dir, gan_model_name, n_samples)
    
    def _generate_gan_vectors(self, model_dir: Path, gan_model_name, n):
        """Generate artificial embeddings using WGAN."""
        print(f'Generating {n} artificial embeddings with WGAN...')
        generator = EmbeddingsGenerator(gan_path=model_dir / gan_model_name, device=self.device)
        self.gan_vectors = generator.generate_embeddings(n=n)
        self.unused_indices = torch.arange(len(self.gan_vectors))
        
        # Cache for reuse
        torch.save(self.gan_vectors, model_dir / 'gan_vectors.pt')
        print(f"Cached GAN vectors to {model_dir / 'gan_vectors.pt'}")

    def anonymize_embeddings(self, original_embeddings: KaldiEmbeddings) -> KaldiEmbeddings:
        """Replace original embeddings with dissimilar GAN-generated ones.
        
        Args:
            original_embeddings: KaldiEmbeddings object with original speaker vectors
            
        Returns:
            KaldiEmbeddings object with anonymized vectors
        """
        if self.gan_vectors is None:
            raise ValueError("GAN model not loaded. Call load_gan_model() first.")
        
        anon_embeddings = KaldiEmbeddings.__new__(KaldiEmbeddings)
        anon_embeddings.embeddings = {}
        anon_embeddings.utts = original_embeddings.utts
        anon_embeddings.speakers = original_embeddings.speakers
        anon_embeddings.genders = original_embeddings.genders
        anon_embeddings.utt2spk = original_embeddings.utt2spk
        anon_embeddings.spk2gender = original_embeddings.spk2gender
        anon_embeddings.xvector_scp = original_embeddings.xvector_scp
        
        print(f"Anonymizing {len(original_embeddings)} embeddings with GAN...")
        for i in tqdm(range(len(original_embeddings))):
            utt, orig_vec = original_embeddings[i]
            anon_vec = self._select_gan_vector(spk_vec=orig_vec)
            anon_embeddings.embeddings[utt] = anon_vec
        
        return anon_embeddings

    def _select_gan_vector(self, spk_vec):
        """Select dissimilar GAN vector to original speaker."""
        limit = 20
        for attempt in range(limit):
            idx = int(np.random.choice(self.unused_indices.cpu().numpy()))
            anon_vec = self.gan_vectors[idx]
            
            # Compute cosine similarity
            sim = 1 - cosine(spk_vec.cpu().numpy(), anon_vec.cpu().numpy())
            
            if sim < self.sim_threshold:
                # Remove from pool to avoid reuse (or track usage)
                self.unused_indices = self.unused_indices[self.unused_indices != idx]
                return anon_vec
        
        # Fallback: if similarity threshold not met, still return vector
        idx = int(np.random.choice(self.unused_indices.cpu().numpy()))
        return self.gan_vectors[idx]

def generate_utterance_level_pseudo_vectors(
    data_dir: Path,
    vector_dir: Path,
    gan_model_dir: Path,
    output_dir: Path,
    num_gan_samples: int = 10000,
    similarity_threshold: float = 0.7,
    device: str = 'cuda:0'
):
    """Generate utterance-level pseudo-speaker vectors.
    
    Args:
        data_dir: Data directory (with utt2spk, spk2gender)
        vector_dir: Directory with original xvector.scp
        gan_model_dir: Directory with GAN model (gan.pt)
        output_dir: Output directory for pseudo_xvector.{scp,ark}
        num_gan_samples: Number of GAN samples to generate
        similarity_threshold: Max similarity between original and GAN vector
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
    # Initialize GAN anonymizer
    # ============================================
    print(f"Loading GAN model from {gan_model_dir}...")
    anonymizer = GANAnonymizer(device=device, sim_threshold=similarity_threshold)
    anonymizer.load_gan_model(gan_model_dir, gan_model_name='gan.pt', n_samples=num_gan_samples)
    
    # ============================================
    # Generate pseudo vectors (utterance level)
    # ============================================
    print(f"Generating pseudo vectors for {len(original_embeddings)} utterances...")
    
    pseudo_xvecs = {}
    for i in tqdm(range(len(original_embeddings))):
        utt_id, orig_vec = original_embeddings[i]
        
        # Select dissimilar GAN vector
        pseudo_vec = anonymizer._select_gan_vector(spk_vec=orig_vec)
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
        description='Generate utterance-level pseudo xvectors with GAN'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory (with utt2spk, spk2gender)')
    parser.add_argument('--vector_dir', type=str, required=True,
                        help='Directory with original xvector.scp')
    parser.add_argument('--gan_model_dir', type=str, required=True,
                        help='Directory with GAN model (gan.pt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudo_xvector.{scp,ark}')
    parser.add_argument('--num_gan_samples', type=int, default=10000,
                        help='Number of GAN samples to generate')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Max similarity between original and GAN vector')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device')
    
    args = parser.parse_args()
    
    generate_utterance_level_pseudo_vectors(
        data_dir=Path(args.data_dir),
        vector_dir=Path(args.vector_dir),
        gan_model_dir=Path(args.gan_model_dir),
        output_dir=Path(args.output_dir),
        num_gan_samples=args.num_gan_samples,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )
    
    print("✓ Utterance-level pseudo-speaker generation complete!")


if __name__ == '__main__':
    main()

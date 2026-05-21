import torch
import torch.nn as nn
import math

from flow_matcher import VelocityNet, FlowMatcher


class EmbeddingsGenerator:
    """Generator for speaker embeddings using flow matching."""

    def __init__(self, flow_path, device):
        self.device = device
        self.flow_path = flow_path

        self.mean = None
        self.std = None
        self.flow_matcher = None

        self._load_model(self.flow_path)

    def generate_embeddings(self, n=1000, steps=20):
        """Generate embeddings using flow matching.
        
        Args:
            n: Number of embeddings to generate
            steps: Number of ODE solver steps
            
        Returns:
            Generated embeddings (n, 192)
        """
        samples = self.flow_matcher.sample(n_samples=n, steps=steps)
        return self._inverse_normalize(samples, self.mean, self.std)
    
    def generate_pseudo_speaker(self, spk_vec, w, steps=20):

        # Speaker Encoding (ODE-1)
        spk_vec = self._normalize(spk_vec, self.mean, self.std)
        z_orig = self.flow_matcher.invert_embedding(spk_vec, steps=20)

        # Speaker Obscuration
        z_rand = torch.rand_like(z_orig).to(self.device)
        z_anon = ((1.0 - w) * z_rand + w * z_orig) / math.sqrt((1.0 - w)**2 + w**2)

        # Speaker Generation (ODE-2)
        pseudo_spk_vec = self.flow_matcher.embedding(z_anon, steps=steps)
        pseudo_spk_vec = self._inverse_normalize(pseudo_spk_vec, self.mean, self.std)

        return pseudo_spk_vec

    def _load_model(self, path):
        """Load trained flow matching model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Initialize velocity network
        model_params = checkpoint.get('model_parameters', {})
        velocity_net = VelocityNet(
            input_dim=model_params.get('input_dim', 192),
            cond_dim=model_params.get('cond_dim', 256)
        )
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Load state dict with DataParallel handling
            state_dict = checkpoint['model_state_dict']
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
        else:
            state_dict = checkpoint

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        velocity_net.load_state_dict(state_dict)
        velocity_net = velocity_net.to(self.device)
        velocity_net.eval()
        
        # Create flow matcher
        # self.velocity_net = velocity_net
        self.flow_matcher = FlowMatcher(velocity_net, device=self.device)


    def _normalize(self, tensor, mean, std):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def _inverse_normalize(self, tensor, mean, std):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    


if __name__ == '__main__':
    print('Test')

    import sys
    from pathlib import Path
    path_root = Path(__file__).parents[1]
    sys.path.append(str(path_root))
    from kaldi_embedding_utils import KaldiEmbeddings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_path = 'trained/flow_models/ecapa_flow/flow.pt'
    generator = EmbeddingsGenerator(flow_path=str(flow_path), device=device)
    batch_size = 4
    original_embeddings = KaldiEmbeddings(
        xvector_scp=str("../compute_ori_spk_vector/output_ori_spk_vector/IEMOCAP_test/xvector.scp"),
        spk2gender_file=str("../data/IEMOCAP_test/spk2gender"),
        utt2spk_file=str("../data/IEMOCAP_test/utt2spk")
    )
    spk_idx = 67
    utt_id, orig_vec = original_embeddings[spk_idx]
    x = orig_vec
    print(f"Input x shape: {x.shape}")

    try:

        # Pseudo speaker generation verification
        print(f"####### Pseudo speaker generation verification ########")
        pseudo_spk_vec = generator.generate_pseudo_speaker(x, w=0.5, steps=20)
        print(f"Pseudo speaker generation successful!")
        print(f"Pseudo speaker shape: {pseudo_spk_vec.shape}")
        
        # Verification checks
        assert pseudo_spk_vec.shape == x.shape, f"Shape mismatch: Expected {x.shape}, got {pseudo_spk_vec.shape}"
        assert not torch.isnan(pseudo_spk_vec).any(), "Model produced NaN values"

        print(f"####### Values consistency verification ########")

        for steps in [10, 20, 50, 100]:
            x_norm = generator._normalize(x, generator.mean, generator.std)
            x1_norm = generator.flow_matcher.embedding(x_norm, steps=steps)
            x0_norm = generator.flow_matcher.invert_embedding(x1_norm, steps=steps)
            x0 = generator._inverse_normalize(x0_norm, generator.mean, generator.std)

            diff = (x - x0).abs().max()
            print(f"steps: {steps}, max_diff: {diff.item()}")
        # If it scales down with more steps, the model has been trained correctly


    except Exception as e:
        print(f"\nTest Result: FAILED")
        print(f"Error encountered: {e}")

        # import traceback
        # traceback.print_exc()
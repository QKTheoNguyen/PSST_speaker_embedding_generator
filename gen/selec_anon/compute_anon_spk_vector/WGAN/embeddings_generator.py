import torch
import torch.nn as nn

from init_wgan import create_wgan


class EmbeddingsGenerator:

    def __init__(self, gan_path, device):
        self.device = device
        self.gan_path = gan_path

        self.mean = None
        self.std = None
        self.wgan = None

        self._load_model(self.gan_path)

    def generate_embeddings(self, n=1000):
        samples = self.wgan.sample_generator(num_samples=n, nograd=True)
        return self._inverse_normalize(samples, self.mean, self.std)

    def _load_model(self, path):
        gan_checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.wgan = create_wgan(parameters=gan_checkpoint['model_parameters'], device=self.device)
        
        # Handle DataParallel mismatch:
        # If state dict keys don't have "module." prefix, add them
        gen_state = gan_checkpoint['generator_state_dict']
        crit_state = gan_checkpoint['critic_state_dict']
        
        gen_state = self._fix_dataparallel_keys(gen_state)
        crit_state = self._fix_dataparallel_keys(crit_state)
        
        self.wgan.G.load_state_dict(gen_state)
        self.wgan.D.load_state_dict(crit_state)

        self.mean = gan_checkpoint["mean"]
        self.std = gan_checkpoint["std"]
    
    def _fix_dataparallel_keys(self, state_dict):
        """Add 'module.' prefix to keys if loading into DataParallel model."""
        # Check if keys need "module." prefix
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        
        if not has_module_prefix:
            # Add "module." prefix to all keys
            new_state = {}
            for k, v in state_dict.items():
                new_state[f'module.{k}'] = v
            return new_state
        
        return state_dict

    def _inverse_normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
### code adapted from https://github.com/facebookresearch/DiT

import torch
import torch.nn.functional as F
import math
from torch import nn

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    MLP replaces MHSA for non-sequential embeddings.
    """
    def __init__(self, hidden_size, cond_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )

        self.linear_final = nn.Linear(hidden_size, hidden_size)
        self.linear_inter = nn.Linear(hidden_size, hidden_size)



    def forward(self, x, c):
        shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = self.adaLN_modulation(c).chunk(6, dim=1)

        x_1 = x + gate_1 * self.mlp1(modulate(self.norm1(x), shift_1, scale_1))
        x_2 = self.linear_final(x) + self.linear_inter(x_1) +  gate_2 * self.mlp2(modulate(self.norm2(x_1), shift_2, scale_2))

        return x_2


class VelocityNet(nn.Module):
    """
    U-Net architecture for speaker embeddings: 
    192 -> 96 -> 48 -> 24 (Bottleneck) -> 48 -> 96 -> 192
    """
    def __init__(self, input_dim=192, cond_dim=256):
        super().__init__()
        self.time_gen = TimestepEmbedder(cond_dim)
        
        # --- Encoder (Downsampling) ---
        self.enc1 = nn.Linear(input_dim, 96)
        self.block1 = DiTBlock(96, cond_dim)
        
        self.enc2 = nn.Linear(96, 48)
        self.block2 = DiTBlock(48, cond_dim)
        
        self.enc3 = nn.Linear(48, 24)
        self.bottleneck = DiTBlock(24, cond_dim)
        
        # --- Decoder (Upsampling) ---
        self.dec1 = nn.Linear(24, 48)
        self.block_up1 = DiTBlock(48, cond_dim)
        
        self.dec2 = nn.Linear(48, 96)
        self.block_up2 = DiTBlock(96, cond_dim)
        
        self.out_proj = nn.Linear(96, input_dim)

    def forward(self, x, t):
        t_emb = self.time_gen(t)
        
        # Encoder
        h1 = self.block1(F.silu(self.enc1(x)), t_emb)    # 96
        h2 = self.block2(F.silu(self.enc2(h1)), t_emb)   # 48
        h3 = self.bottleneck(F.silu(self.enc3(h2)), t_emb) # 24 (Bottleneck)
        
        # Decoder with Skip Connections (Residual Addition)
        # Level 48
        x_up = self.block_up1(F.silu(self.dec1(h3)), t_emb)
        x_up = x_up + h2 # Skip connection
        
        # Level 96
        x_up = self.block_up2(F.silu(self.dec2(x_up)), t_emb)
        x_up = x_up + h1 # Skip connection
        
        # Final Output
        return self.out_proj(x_up)
    
class FlowMatcher:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def get_loss(self, x1):
        """
        Computes the Conditional Flow Matching MSE loss.
        x1: Batch of real speaker embeddings (Batch, 192)
        """
        batch_size = x1.shape[0]
        
        # 1. Sample Gaussian Noise (Source distribution)
        x0 = torch.randn_like(x1).to(self.device)
        
        # 2. Sample random time t from Uniform(0, 1)
        t = torch.rand(batch_size, 1).to(self.device)

        # 3. Construct the probability path (Optimal Transport/Straight Line)
        xt = (1.0 - t) * x0 + t * x1
        
        # 4. Define the conditional vector field (the target velocity)
        # For straight paths, the velocity is constant: v = x1 - x0
        ut = x1 - x0
    
        if t.dim() > 1:
            t = t.squeeze(-1)

        # 5. Predict the velocity using the U-Net/DiT VelocityNet
        vt = self.model(xt, t)
        
        # 6. Minimize the MSE between predicted and actual velocity
        return F.mse_loss(vt, ut)

    @torch.no_grad()
    def sample(self, n_samples, steps=20):
        """
        Generates pseudo-speaker embeddings using the Euler method.
        n_samples: Number of speakers to generate
        steps: Number of integration steps
        """
        self.model.eval()
        
        # 1. Start from pure Gaussian noise
        xt = torch.randn(n_samples, 192).to(self.device)
        dt = 1.0 / steps
        
        # 2. Iteratively solve the ODE: dx/dt = v(x, t)
        for i in range(steps):
            # Current time scalar (normalized to 0-1)
            t_val = i / steps
            t = torch.full((n_samples, 1), t_val).to(self.device)
            
            if t.dim() > 1:
                t = t.squeeze(-1)

            # Predict the velocity at the current position and time
            vt = self.model(xt, t)
            
            # Euler step: Move slightly in the direction of the velocity
            xt = xt + vt * dt
            
        # 3. Project back to unit sphere (optional, but standard for ECAPA-TDNN)
        # ECAPA-TDNN embeddings are usually L2-normalized.
        xt = F.normalize(xt, p=2, dim=-1)
        
        return xt
    
    @torch.no_grad()
    def embedding(self, x0, steps=20):
        """
        Finds the given speaker_embedding x1 from the noise vector x0.
        spk_vec: Speaker embedding vector, make sure it is normalized
        steps: Number of integration steps
        """

        self.model.eval()
        xt = x0.to(self.device)

        dt = 1.0 / steps

        for i in range(steps):

            t_val = i / steps
            t = torch.full((xt.shape[0],), t_val).to(self.device)

            if t.dim() > 1:
                t = t.squeeze(-1)
            
            # Predict the velocity at the current position and time
            vt = self.model(xt, t)
            
            # Euler step: Move slightly in the direction of the velocity
            xt = xt + vt * dt

        return xt
    
    @torch.no_grad()
    def invert_embedding(self, x1, steps=20):
        """
        Finds the noise vector x0 that produces the given speaker_embedding x1.
        x: Speaker embedding vector, make sure it is normalized
        steps: Number of integration steps
        """

        self.model.eval()
        xt = x1.to(self.device)

        dt = 1.0 / steps

        for i in range(steps):

            t_val = 1.0 - (i * dt)
            t = torch.full((xt.shape[0],), t_val).to(self.device)

            if t.dim() > 1:
                t = t.squeeze(-1)

            # Predict the velocity at the current position and time
            v = self.model(xt, t)

            # Update x by moving backwards
            # xt-dt = xt - (velocity * dt)
            xt = xt - v * dt

        return xt


if __name__ == '__main__':
    print('Test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VelocityNet(input_dim=192, cond_dim=256).to(device)
    flow_matcher = FlowMatcher(model, device=device)
    batch_size = 4
    x = torch.randn(batch_size, 192).to(device)
    t = torch.rand(batch_size).to(device)

    print(f"Input x shape: {x.shape}")
    print(f"Input t shape: {t.shape}")

    try:


        # Forward pass verification
        print(f"####### Forward pass verification ########")
        with torch.no_grad():
            output = model(x, t)
        print(f"Forward pass successful!")
        print(f"Output (Velocity) shape: {output.shape}")
        
        # Verification checks
        assert output.shape == x.shape, f"Shape mismatch: Expected {x.shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Model produced NaN values"

        # Embedding function verification
        print(f"####### Flow matcher embedding function verification ########")
        x1 = flow_matcher.embedding(x, steps=20)
        print(f"Flow matcher embedding function successful!")
        print(f"x1 shape: {x1.shape}")
        
        # Verification checks
        assert x1.shape == x.shape, f"Shape mismatch: Expected {x.shape}, got {x1.shape}"
        assert not torch.isnan(x1).any(), "Model produced NaN values"

        # Invert embedding function verification
        print(f"####### Flow matcher invert embedding function verification ########")
        x0 = flow_matcher.invert_embedding(x1, steps=20)
        print(f"Flow matcher invert embedding function successful!")
        print(f"x0 shape: {x0.shape}")
        
        # Verification checks
        assert x0.shape == x1.shape, f"Shape mismatch: Expected {x1.shape}, got {x0.shape}"
        assert not torch.isnan(x0).any(), "Model produced NaN values"

        print("\nTest Result: PASSED")

    except Exception as e:
        print(f"\nTest Result: FAILED")
        print(f"Error encountered: {e}")

        # import traceback
        # traceback.print_exc()
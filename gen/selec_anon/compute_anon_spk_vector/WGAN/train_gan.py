#!/usr/bin/env python3
"""Train WGAN to generate speaker embeddings (192-dim ECAPA-TDNN compatible).

Usage:
    python train_gan.py --embeddings_dir path/to/xvector/dir --output_dir ./gan_checkpoint --epochs 300
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from init_wgan import create_wgan
from embeddings_generator import EmbeddingsGenerator
from torch.utils.tensorboard import SummaryWriter


class EmbeddingDataset(torch.utils.data.Dataset):
    """Load ECAPA-TDNN embeddings from Kaldi format."""
    
    def __init__(self, embeddings_list):
        """
        Args:
            embeddings_list: List of numpy arrays, each shape (192,)
        """
        self.embeddings = torch.from_numpy(np.array(embeddings_list)).float()
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


def load_embeddings_from_kaldi(xvector_scp_paths):
    """Load embeddings from Kaldi xvector.scp files.
    
    Args:
        xvector_scp_paths: List of paths to xvector.scp files
        
    Returns:
        List of embedding numpy arrays (each flattened to 1D)
    """
    from kaldiio import ReadHelper
    
    all_embeddings = []
    
    for scp_path in xvector_scp_paths:
        print(f"Loading embeddings from {scp_path}...")
        with ReadHelper(f'scp:{scp_path}') as reader:
            for key, mat in tqdm(reader):
                # Flatten to 1D (remove any extra dimensions)
                mat_flat = mat.flatten()
                all_embeddings.append(mat_flat)
    
    print(f"Loaded {len(all_embeddings)} embeddings total")
    return all_embeddings


def normalize_embeddings(embeddings):
    """Normalize embeddings to mean=0, std=1.
    
    Args:
        embeddings: List of numpy arrays or torch tensors
        
    Returns:
        (normalized_embeddings, mean, std) where normalized shape is (N, 192)
    """
    embeddings_array = np.array(embeddings)
    
    # Flatten to 2D: (N, 192) if embeddings had extra dimensions
    if embeddings_array.ndim > 2:
        embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)
    
    print(f"Flattened embeddings shape: {embeddings_array.shape}")
    
    mean = embeddings_array.mean(axis=0)
    std = embeddings_array.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized = (embeddings_array - mean) / std
    return normalized, mean, std


def train_gan(generator, discriminator, train_loader, config, device, output_dir, writer=None):
    """Train WGAN for embedding generation.
    
    Args:
        generator: Generator model
        discriminator: Discriminator (Critic) model
        train_loader: DataLoader
        config: Configuration dict
        device: GPU device
        output_dir: Directory to save checkpoints
        writer: TensorBoard SummaryWriter (optional)
    """
    from wgan_qc import WassersteinGanQuadraticCost
    
    # Initialize optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(), 
        lr=config['learning_rate'],
        betas=config.get('betas', (0.0, 0.9))
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config['learning_rate'],
        betas=config.get('betas', (0.0, 0.9))
    )
    
    criterion = nn.MSELoss()
    
    # Create WGAN trainer
    gan = WassersteinGanQuadraticCost(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=optimizer_g,
        dis_optimizer=optimizer_d,
        criterion=criterion,
        epochs=config['epochs'],
        n_max_iterations=config['n_max_iterations'],
        data_dimensions=config['data_dim'],
        batch_size=config['batch_size'],
        device=device,
        gamma=config.get('gamma', 0.1),
        K=config.get('K', -1),
        milestones=config.get('milestones', [150000, 250000]),
        lr_anneal=config.get('lr_anneal', 1.0)
    )
    
    print(f"Starting GAN training for {config['epochs']} epochs...")
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        epoch_loss_wd = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, real_embeddings in enumerate(pbar):
            real_embeddings = real_embeddings.to(device)
            batch_size = real_embeddings.size(0)
            
            # ============================================
            # Train Discriminator (Critic)
            # ============================================
            gan.D_opt.zero_grad()
            
            # Real embeddings
            real_output = gan.D(real_embeddings)
            
            # Generated embeddings
            z = torch.randn(batch_size, config['z_dim'], device=device)
            fake_embeddings = gan.G(z)
            fake_output = gan.D(fake_embeddings.detach())
            
            # Wasserstein distance (quadratic cost)
            wd = gan._quadratic_wasserstein_distance_(real_embeddings, fake_embeddings)
            wd_loss = wd.mean()
            
            # Critic loss
            d_loss = fake_output.mean() - real_output.mean() + gan.LAMBDA * wd_loss
            d_loss.backward()
            gan.D_opt.step()
            
            # ============================================
            # Train Generator
            # ============================================
            gan.G_opt.zero_grad()
            
            z = torch.randn(batch_size, config['z_dim'], device=device)
            fake_embeddings = gan.G(z)
            fake_output = gan.D(fake_embeddings)
            
            g_loss = -fake_output.mean()
            g_loss.backward()
            gan.G_opt.step()
            
            # Logging
            epoch_loss_d += d_loss.item()
            epoch_loss_g += g_loss.item()
            epoch_loss_wd += wd_loss.item()
            
            pbar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'WD': wd_loss.item()
            })
            
            # Log to TensorBoard
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss/generator', g_loss.item(), global_step)
                writer.add_scalar('Loss/wasserstein_distance', wd_loss.item(), global_step)
                writer.add_scalar('LR/discriminator', gan.D_opt.param_groups[0]['lr'], global_step)
                writer.add_scalar('LR/generator', gan.G_opt.param_groups[0]['lr'], global_step)
            
            gan.num_steps += 1
            
            # Checkpoint every N iterations
            if (gan.num_steps + 1) % config.get('checkpoint_interval', 10000) == 0:
                save_checkpoint(
                    gan, config, 
                    output_dir, 
                    epoch, 
                    gan.num_steps,
                    mean=None,  # Will set during final save
                    std=None
                )
        
        # Learning rate scheduling
        gan.schedulerD.step()
        gan.schedulerG.step()
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"D_loss: {epoch_loss_d/len(train_loader):.4f}, "
              f"G_loss: {epoch_loss_g/len(train_loader):.4f}, "
              f"WD: {epoch_loss_wd/len(train_loader):.4f}")
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss_Epoch/discriminator', epoch_loss_d/len(train_loader), epoch + 1)
            writer.add_scalar('Loss_Epoch/generator', epoch_loss_g/len(train_loader), epoch + 1)
            writer.add_scalar('Loss_Epoch/wasserstein_distance', epoch_loss_wd/len(train_loader), epoch + 1)
        
        # # Save checkpoint every N epochs
        # if (epoch + 1) % config.get('save_epoch_interval', 50) == 0:
        #     save_checkpoint(
        #         gan, config,
        #         output_dir,
        #         epoch,
        #         gan.num_steps,
        #         mean=None,
        #         std=None
        #     )
    
    return gan


def save_checkpoint(gan, config, output_dir, epoch, num_steps, mean, std):
    """Save GAN checkpoint."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'num_steps': num_steps,
        'generator_state_dict': gan.G.module.state_dict() if isinstance(gan.G, nn.DataParallel) else gan.G.state_dict(),
        'critic_state_dict': gan.D.module.state_dict() if isinstance(gan.D, nn.DataParallel) else gan.D.state_dict(),
        'model_parameters': config,
        'mean': mean,
        'std': std
    }
    
    checkpoint_path = Path(output_dir) / f'gan_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Also save as 'gan.pt' (latest)
    latest_path = Path(output_dir) / 'gan.pt'
    torch.save(checkpoint, latest_path)


def main():
    parser = argparse.ArgumentParser(description='Train WGAN for speaker embeddings')
    parser.add_argument('--embeddings_dir', type=str, nargs='+', required=True,
                        help='Path(s) to directories with xvector.scp files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to JSON config file (optional)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================
    # Load config
    # ============================================
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path) as f:
        config = json.load(f)

    # ============================================
    # Create output directory (for now the name only shows the epochs)
    # ============================================
    output_dir = f"./trained/gan_models/gan_{config.get('epochs')}_epochs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ============================================
    # Load and preprocess embeddings
    # ============================================
    print("Loading embeddings...")
    
    # Collect xvector.scp paths
    scp_paths = []
    for emb_dir in args.embeddings_dir:
        scp_path = Path(emb_dir) / 'xvector.scp'
        if scp_path.exists():
            scp_paths.append(str(scp_path))
    
    if not scp_paths:
        raise ValueError(f"No xvector.scp found in {args.embeddings_dir}")
    
    # Load embeddings from Kaldi format
    embeddings = load_embeddings_from_kaldi(scp_paths)
    
    # Normalize
    embeddings_normalized, mean, std = normalize_embeddings(embeddings)
    print(f"Embeddings shape: {embeddings_normalized.shape}")
    print(f"Mean: {mean[:5]}..., Std: {std[:5]}...")
    
    # Create dataset and dataloader
    dataset = EmbeddingDataset(embeddings_normalized)
    train_loader = DataLoader(dataset, batch_size=config.get('batch_size'), shuffle=True)
    
    # ============================================
    # Initialize GAN
    # ============================================
    print("Creating GAN models...")
    gan = create_wgan(config, device=device, optimizer='adam')
    
    # ============================================
    # Setup TensorBoard logging
    # ============================================
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # ============================================
    # Train GAN
    # ============================================
    trained_gan = train_gan(gan.G.module, gan.D.module, train_loader, config, device, output_dir, writer)
    
    # ============================================
    # Save final checkpoint
    # ============================================
    print("Saving final checkpoint...")
    final_checkpoint = {
        'epoch': config.get('epochs'),
        'num_steps': gan.num_steps,
        'generator_state_dict': gan.G.module.state_dict() if isinstance(gan.G, nn.DataParallel) else gan.G.state_dict(),
        'critic_state_dict': gan.D.module.state_dict() if isinstance(gan.D, nn.DataParallel) else gan.D.state_dict(),
        'model_parameters': config,
        'mean': mean,
        'std': std
    }
    
    final_path = Path(output_dir) / 'gan.pt'
    torch.save(final_checkpoint, final_path)
    print(f"✓ Final checkpoint saved: {final_path}")
    
    # Save config for reference
    config_path = Path(output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        config_json = config.copy()
        config_json['milestones'] = [int(m) for m in config_json['milestones']]
        json.dump(config_json, f, indent=4)
    print(f"✓ Config saved: {config_path}")
    
    print("✓ Training complete!")
    
    # ============================================
    # Test generation
    # ============================================
    print("\nTesting embedding generation...")
    generator = EmbeddingsGenerator(gan_path=str(final_path), device=device)
    test_samples = generator.generate_embeddings(n=5)
    print(f"Generated sample shape: {test_samples.shape}")
    print(f"Generated sample [0]: {test_samples[0][:5]}...")
    
    # Close TensorBoard writer
    writer.flush()
    writer.close()
    print(f"\n✓ TensorBoard logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir={log_dir} --port=6006")


if __name__ == '__main__':
    main()
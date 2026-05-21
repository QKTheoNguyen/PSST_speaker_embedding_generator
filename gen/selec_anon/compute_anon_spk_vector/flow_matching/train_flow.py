#!/usr/bin/env python3
"""Train flow-matching to generate speaker embeddings (192-dim ECAPA-TDNN compatible).

Usage:
    python train_flow.py --embeddings_dir path/to/xvector/dir --output_dir ./gan_checkpoint --epochs 300
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

from flow_matcher import VelocityNet, FlowMatcher
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

def train_flow(model, train_loader, config, device, output_dir, mean, std, writer=None):
    """Train flow matching model for embedding generation.
    
    Args:
        model: FlowMatcher instance
        train_loader: DataLoader with training embeddings
        config: Configuration dict
        device: GPU device
        output_dir: Directory to save checkpoints
        mean: Mean of training embeddings (for checkpoint)
        std: Std of training embeddings (for checkpoint)
        writer: TensorBoard SummaryWriter (optional)
    """
    
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=config['learning_rate'],
        betas=config.get('betas', (0.9, 0.999)),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'], # This becomes the peak LR
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader), # Needs the number of batches per epoch
        pct_start=0.1, # Spends 10% of time warming up
        anneal_strategy='cos',
        div_factor=25.0, # Initial_lr = max_lr / 25
        final_div_factor=1000.0 # Final_lr = initial_lr / 1000
    )
    
    print(f"Starting flow matching training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, real_embeddings in enumerate(pbar):
            real_embeddings = real_embeddings.to(device)
            
            # Compute CFM loss
            loss = model.get_loss(real_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/batch', loss.item(), global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
        
            
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.6f}")
        
        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/epoch', avg_loss, epoch + 1)
        
        # # Save checkpoint periodically
        # if (epoch + 1) % config.get('save_epoch_interval', 50) == 0:
        #     save_checkpoint(model, config, output_dir, epoch, mean, std)
    
    return model


def save_checkpoint(model, config, output_dir, epoch, mean, std):
    """Save flow matching checkpoint."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract velocity net from DataParallel wrapper if needed
    velocity_net = model.model

    if hasattr(velocity_net, 'module'):
        state_dict = velocity_net.module.state_dict()
    else:
        state_dict = velocity_net.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'model_parameters': config,
        'mean': mean,
        'std': std
    }
    
    checkpoint_path = Path(output_dir) / f'flow_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Also save as 'flow.pt' (latest)
    latest_path = Path(output_dir) / 'flow.pt'
    torch.save(checkpoint, latest_path)


def main():
    parser = argparse.ArgumentParser(description='Train flow matching for speaker embeddings')
    parser.add_argument('--embeddings_dir', type=str, nargs='+', required=True,
                        help='Path(s) to directories with training xvector.scp files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU device')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to JSON config file')
    parser.add_argument('--epochs', type=float,
                        help='Number of training epochs, outwrites config if set')


    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # ============================================
    # Load config
    # ============================================
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path) as f:
        config = json.load(f)
    # Override config with command-line args if provided
    if args.epochs is not None:
        config['epochs'] = int(args.epochs)
    print(f"Config: {json.dumps(config, indent=2)}")

    # ============================================
    # Create output directory (for now the name only shows the epochs)
    # ============================================
    output_dir = f"./trained/flow_models/flow_{config.get('epochs')}_epochs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ============================================
    # Load and preprocess embeddings
    # ============================================
    print("Loading embeddings...")
    
    # Collect xvector.scp paths
    scp_paths = []
    for emb_dir in args.embeddings_dir:
        scp_file = Path(emb_dir) / 'xvector.scp'
        if scp_file.exists():
            scp_paths.append(str(scp_file))
    
    if not scp_paths:
        raise RuntimeError(f"No xvector.scp found in {args.embeddings_dir}")
    
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
    # Initialize flow matching
    # ============================================
    from init_flow import create_flow_matching
    
    print("Creating flow matching model...")
    flow_matcher = create_flow_matching(config, device=device)
    
    # ============================================
    # Setup TensorBoard logging
    # ============================================
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # ============================================
    # Train flow matching
    # ============================================
    trained_flow = train_flow(flow_matcher, train_loader, config, device, output_dir, mean, std, writer)
    
    # ============================================
    # Save final checkpoint
    # ============================================
    print("Saving final checkpoint...")
    velocity_net = trained_flow.model
    if isinstance(velocity_net, nn.DataParallel):
        state_dict = velocity_net.module.state_dict()
    else:
        state_dict = velocity_net.state_dict()
    
    final_checkpoint = {
        'epoch': config.get('epochs'),
        'model_state_dict': state_dict,
        'model_parameters': config,
        'mean': mean,
        'std': std
    }
    
    final_path = Path(output_dir) / 'flow.pt'
    torch.save(final_checkpoint, final_path)
    print(f"✓ Final checkpoint saved: {final_path}")
    
    # Save config for reference
    config_path = Path(output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"✓ Config saved: {config_path}")
    
    print("✓ Training complete!")
    
    # ============================================
    # Test generation
    # ============================================
    print("\nTesting embedding generation...")
    from embeddings_generator import EmbeddingsGenerator
    generator = EmbeddingsGenerator(flow_path=str(final_path), device=device)
    test_samples = generator.generate_embeddings(n=5, steps=20)
    print(f"Generated sample shape: {test_samples.shape}")
    print(f"Generated sample [0]: {test_samples[0][:5]}...")
    
    writer.flush()
    writer.close()
    print(f"\n✓ TensorBoard logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir={log_dir} --port=6006")

if __name__ == '__main__':
    main()

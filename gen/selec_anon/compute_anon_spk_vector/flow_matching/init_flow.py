import torch
import torch.nn as nn
from flow_matcher import VelocityNet, FlowMatcher


def create_flow_matching(config, device='cuda'):
    """Create and initialize flow matching model.
    
    Args:
        config: Configuration dict with keys:
            - input_dim: Embedding dimension (default: 192)
            - cond_dim: cond_dim (default: 256)

        device: GPU device
        
    Returns:
        flow_matcher: FlowMatcher instance
    """
    
    input_dim = config.get('input_dim', 192)
    cond_dim = config.get('cond_dim', 256)
    
    # Create velocity network
    velocity_net = VelocityNet(
        input_dim=input_dim,
        cond_dim=cond_dim
    )
    
    # Wrap with DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        velocity_net = nn.DataParallel(velocity_net)
    
    velocity_net = velocity_net.to(device)
    
    # Create flow matcher
    flow_matcher = FlowMatcher(velocity_net)
    
    return flow_matcher

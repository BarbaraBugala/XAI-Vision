# src/XAI_methods/baselines.py
import torch
import torchvision.transforms.functional as TF

def get_baseline(input_batch: torch.Tensor, baseline_type: str = 'zeros') -> torch.Tensor:
    """
    Generates a baseline tensor matching the shape of the input batch.
    
    Args:
        input_batch (torch.Tensor): The input image tensor, shape (B, C, H, W) or (C, H, W).
        baseline_type (str): Type of baseline ('zeros', 'ones', 'average', 'blurred').
        
    Returns:
        torch.Tensor: The generated baseline tensor of the same shape.
    """
    baseline_type = baseline_type.lower()
    
    if baseline_type == 'zeros':
        return torch.zeros_like(input_batch)
        
    elif baseline_type == 'ones':
        return torch.ones_like(input_batch)
        
    elif baseline_type == 'average':
        # Handles both (B, C, H, W) and (C, H, W) dynamically
        spatial_dims = tuple(range(input_batch.ndim - 2, input_batch.ndim)) # Gets last two dims (H, W)
        
        # Calculate mean across H and W per channel
        mean_vals = input_batch.mean(dim=spatial_dims, keepdim=True)
        return mean_vals.expand_as(input_batch)
        
    elif baseline_type == 'blurred':
        return TF.gaussian_blur(input_batch, kernel_size=(51, 51), sigma=(15.0, 15.0))
        
    else:
        raise ValueError(f"Unknown baseline type: '{baseline_type}'. "
                         f"Allowed types: ['zeros', 'ones', 'average', 'blurred', 'random']")
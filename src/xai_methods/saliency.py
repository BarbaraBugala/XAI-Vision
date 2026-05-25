# src/XAI_methods/saliency.py
import torch
from captum.attr import Saliency

def saliency(model, input_tensor, target):
    """
    Computes standard Saliency maps (vanilla gradients) for a given input tensor.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): Shape (B, C, H, W) or (C, H, W).
        target (int): Target class index.
        
    Returns:
        torch.Tensor: Saliency attributions mapping back to the input shape.
    """
    sal = Saliency(model)
    
    # Standardize dimensions to 4D for Captum
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    input_tensor.requires_grad_()
    
    # Compute attributions
    attributions = sal.attribute(
        inputs=input_tensor,
        target=target
    )
    
    return attributions
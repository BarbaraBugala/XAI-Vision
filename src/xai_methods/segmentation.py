# src/XAI_methods/segmentation.py
import torch
import numpy as np
from skimage.segmentation import slic

def get_superpixels(image, n_segments=50, compactness=10, sigma=1):
    """
    Segments an image into superpixels and returns a PyTorch tensor mask.
    
    Returns:
        torch.Tensor: A tensor of shape (1, 1, H, W) containing integer labels.
    """
    # 1. Convert input safely to a NumPy array (H, W, C) for skimage
    if isinstance(image, torch.Tensor):
        # If it has a batch dim (1, C, H, W), strip it to (C, H, W)
        if image.ndim == 4:
            image = image.squeeze(0)
        # Move channels to the end: (H, W, C)
        img_np = image.cpu().detach().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif not isinstance(image, np.ndarray):
        # Assume PIL Image
        img_np = np.array(image)

    # 2. Run the SLIC algorithm
    # skimage expects an image where pixel values make sense (e.g., standard float or 0-255 uint8)
    segments_np = slic(
        img_np, 
        n_segments=n_segments, 
        compactness=compactness, 
        sigma=sigma, 
        start_label=0
    )
    
    # 3. Convert the resulting NumPy array back to a PyTorch Long Tensor
    segments_tensor = torch.from_numpy(segments_np).long()
    
    # 4. Format shape to match the required (1, 1, H, W) structure
    segments_tensor = segments_tensor.unsqueeze(0).unsqueeze(0)
    
    return segments_tensor
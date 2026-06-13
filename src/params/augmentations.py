import torch
import numpy as np


def add_gaussian_noise(input_tensor: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Adds random Gaussian noise. std controls the strength."""
    noise = torch.randn_like(input_tensor) * std
    return input_tensor + noise

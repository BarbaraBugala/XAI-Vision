# src/params/segmentation_registry.py
import torch
from params.segmentation import get_superpixels, get_grid

SEGMENTATION_REGISTRY = {
    "superpixels": get_superpixels,
    "grid": get_grid,
}

SEGMENTATION_PARAMS = {
    "superpixels": ("n_segments", "compactness", "sigma"),
    "grid": ("n_segments",),
}

def get_segmentation(method: str, image, config: dict) -> torch.Tensor:
    if method not in SEGMENTATION_REGISTRY:
        raise ValueError(f"Segmentation '{method}' not found. "
                         f"Choose from: {list(SEGMENTATION_REGISTRY.keys())}")
    allowed = SEGMENTATION_PARAMS[method]
    kwargs = {k: config[k] for k in allowed if k in config}
    return SEGMENTATION_REGISTRY[method](image, **kwargs)
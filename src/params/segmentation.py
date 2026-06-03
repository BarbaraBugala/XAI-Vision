import torch
import numpy as np
from skimage.segmentation import slic

def _to_hwc_numpy(image) -> np.ndarray:
    """Shared helper: converts tensor/PIL/numpy to (H, W, C) numpy array."""
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image.squeeze(0)        # (C, H, W)
        return image.cpu().detach().numpy().transpose(1, 2, 0)
    elif not isinstance(image, np.ndarray):
        return np.array(image)              # PIL
    return image


def _to_tensor(segments_np) -> torch.Tensor:
    """Shared helper: converts (H, W) int array to (1, 1, H, W) long tensor."""
    return torch.from_numpy(segments_np).long().unsqueeze(0).unsqueeze(0)


def get_superpixels(image, n_segments=50, compactness=10, sigma=1) -> torch.Tensor:
    img_np = _to_hwc_numpy(image)
    segments_np = slic(img_np, n_segments=n_segments, compactness=compactness,
                       sigma=sigma, start_label=0)
    return _to_tensor(segments_np)


def get_grid(image, n_segments=100) -> torch.Tensor:
    """Divides the image into a uniform rectangular grid of patches."""
    img_np = _to_hwc_numpy(image)
    h, w = img_np.shape[:2]
    n_cols = int(np.sqrt(n_segments))
    n_rows = int(np.ceil(n_segments / n_cols))

    segments_np = np.zeros((h, w), dtype=np.int64)
    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = int(row * h / n_rows), int((row + 1) * h / n_rows)
            x0, x1 = int(col * w / n_cols), int((col + 1) * w / n_cols)
            segments_np[y0:y1, x0:x1] = row * n_cols + col

    return _to_tensor(segments_np)
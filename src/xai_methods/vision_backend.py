"""Adapter from XAI-Vision experiment config to shapiq.vision."""

from __future__ import annotations

import math

import numpy as np
import torch

from shapiq.approximator import ProxySHAP
from shapiq.explainer.configuration import setup_approximator
from shapiq.game_theory.indices import is_empty_value_the_baseline
from shapiq.vision import (
    BlurMasking,
    CNNArchitecture,
    HuggingFacePixelArchitecture,
    MeanColorMasking,
    ZeroMasking,
)
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.players import GridStrategy, SuperpixelStrategy

MASKING = {
    "zeros": lambda: ZeroMasking(0.0),
    "ones": lambda: ZeroMasking(1.0),
    "average": MeanColorMasking,
    "blurred": BlurMasking,
}

SHAP_CONFIG = {
    "kernel_shap": {"index": "SV", "max_order": 1, "approximator": "auto", "budget_key": "num_samples"},
    "proxy_shap": {"index": "SII", "max_order": 2, "approximator": "proxy", "budget_key": "num_samples"},
}


def _player_strategy(segmentation: str, n_segments: int):
    if segmentation == "superpixels":
        return SuperpixelStrategy(n_segments=n_segments)
    if segmentation == "grid":
        n_cols = int(math.sqrt(n_segments))
        n_rows = int(math.ceil(n_segments / n_cols))
        return GridStrategy(grid_shape=(n_rows, n_cols))
    raise ValueError(f"Unknown segmentation: {segmentation}")


def _architecture(model, *, architecture, class_idx, baseline, segmentation, n_segments, processor):
    baseline = baseline.lower()
    if baseline not in MASKING:
        raise ValueError(f"Unknown baseline: {baseline}")
    masking = MASKING[baseline]()
    players = _player_strategy(segmentation, n_segments)

    if architecture == "cnn":
        return CNNArchitecture(model=model, masking_strategy=masking, player_strategy=players)
    if architecture == "hf_pixel":
        if processor is None:
            raise ValueError("processor is required for hf_pixel models")
        return HuggingFacePixelArchitecture(
            model=model,
            processor=processor,
            class_id=class_idx,
            masking_strategy=masking,
            player_strategy=players,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def to_attribution_tensor(interaction_values, player_masks) -> torch.Tensor:
    """Map first-order values to a (1, 3, H, W) tensor for plotting."""
    if isinstance(player_masks, torch.Tensor):
        player_masks = player_masks.cpu().numpy()

    first_order = interaction_values.get_n_order_values(1)
    heatmap = np.zeros(player_masks.shape[1:], dtype=np.float32)
    for i, mask in enumerate(player_masks):
        heatmap[mask] = float(first_order[i])

    return torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).contiguous()


def run_shap(
    model,
    img_np: np.ndarray,
    class_idx: int,
    *,
    method: str,
    architecture: str = "cnn",
    processor=None,
    config: dict,
) -> tuple[torch.Tensor, np.ndarray]:
    """Run kernel_shap or proxy_shap. Returns (attribution_tensor, player_masks)."""
    if method not in SHAP_CONFIG:
        raise ValueError(f"Unknown shap method: {method}")

    spec = SHAP_CONFIG[method]
    baseline = config.get("baseline_type", "zeros")
    n_segments = config.get("n_segments", 50)
    budget = config.get(spec["budget_key"], 200 if method == "kernel_shap" else 512)
    max_order = config.get("max_order", spec["max_order"])
    index = config.get("index", spec["index"])

    arch = _architecture(
        model,
        architecture=architecture,
        class_idx=class_idx,
        baseline=baseline,
        segmentation=config.get("segmentation", "superpixels"),
        n_segments=n_segments,
        processor=processor,
    )
    imputer = ImageImputer(model_architecture=arch, image=img_np, batch_size=32)

    if spec["approximator"] == "proxy":
        approx = ProxySHAP(
            n=imputer.n_features,
            max_order=max_order,
            index=index,
            adjustment=config.get("adjustment", "msr"),
            random_state=0,
        )
    else:
        approx = setup_approximator(
            approximator="auto",
            index=index,
            max_order=max_order,
            n_players=imputer.n_features,
            random_state=0,
        )

    interaction_values = approx.approximate(budget=budget, game=imputer)
    interaction_values.baseline_value = imputer.empty_prediction
    if is_empty_value_the_baseline(interaction_values.index):
        interaction_values[()] = interaction_values.baseline_value

    player_masks = imputer.player_masks
    return to_attribution_tensor(interaction_values, player_masks), player_masks

# src/XAI_methods/proxy_shap.py
import torch
import numpy as np
import shapiq
from shapiq.approximator import ProxySHAP
from params.baselines import get_baseline


def proxy_shap(model, input_tensor, target, baseline=None, segments=None,
               max_order=2, budget=512, index="SII", adjustment="msr"):

    # 1. Standardize dimensions (match your KernelSHAP convention)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    if segments is not None and segments.ndim == 3:
        segments = segments.unsqueeze(0)

    if baseline is None:
        baseline = get_baseline(input_tensor, baseline_type='zeros')
    if isinstance(baseline, str):
        baseline = get_baseline(input_tensor, baseline_type=baseline)

    # 2. Build the value function (shapiq's equivalent of Captum's model wrapper)
    #    shapiq expects a callable: coalition mask (np array) -> scalar output
    input_np = input_tensor.cpu().numpy()       # (1, C, H, W)
    baseline_np = baseline.cpu().numpy()

    if segments is not None:
        # Segment-based: each "player" is a superpixel segment (like your KernelSHAP)
        seg_np = segments.cpu().numpy().squeeze()           # (H, W)
        n_players = int(seg_np.max()) + 1

        def value_function(coalitions: np.ndarray) -> np.ndarray:
            # coalitions: (n_coalitions, n_players) boolean mask
            outputs = []
            for coalition in coalitions:
                masked = baseline_np.copy()
                for seg_id, active in enumerate(coalition):
                    if active:
                        masked[0, :, seg_np == seg_id] = input_np[0, :, seg_np == seg_id]
                inp = torch.tensor(masked, dtype=input_tensor.dtype).to(input_tensor.device)
                with torch.no_grad():
                    out = model(inp)
                # Handle classification: extract target class logit
                if out.ndim > 1:
                    out = out[:, target]
                outputs.append(out.item())
            return np.array(outputs)
    else:
        # Feature-based: each pixel/channel is a player
        flat_input = input_np.flatten()
        flat_baseline = baseline_np.flatten()
        n_players = len(flat_input)

        def value_function(coalitions: np.ndarray) -> np.ndarray:
            outputs = []
            for coalition in coalitions:
                masked = np.where(coalition, flat_input, flat_baseline)
                inp = torch.tensor(
                    masked.reshape(input_np.shape),
                    dtype=input_tensor.dtype
                ).to(input_tensor.device)
                with torch.no_grad():
                    out = model(inp)
                if out.ndim > 1:
                    out = out[:, target]
                outputs.append(out.item())
            return np.array(outputs)

    # 3. Run ProxySHAP
    approximator = ProxySHAP(
        n=n_players,
        max_order=max_order,
        index=index,
        adjustment=adjustment,
    )
    interaction_values = approximator.approximate(budget=budget, game=value_function)

    return interaction_values

def interaction_values_to_tensor(iv, segments, image_shape):
    """Convert per-segment SII attributions to a spatial tensor for visualization."""
    seg_np = segments.cpu().numpy().squeeze()   # (H, W)
    heatmap = np.zeros(seg_np.shape, dtype=np.float32)

    # iv.interactions is a dict: {(player_tuple,): value, ...}
    for players, value in iv.interactions.items():
        if len(players) == 1:                   # single-feature attributions only
            seg_id = players[0]
            heatmap[seg_np == seg_id] = float(value)

    # Expand to (1, 3, H, W) so squeeze(0).permute(1,2,0) → (H,W,3) for Captum
    return torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).contiguous()
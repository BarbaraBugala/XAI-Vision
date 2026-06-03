# src/XAI_methods/proxy_shap.py
import torch
import numpy as np
from shapiq.approximator import ProxySHAP
from params.baselines import get_baseline


def proxy_shap(model, input_tensor, target, baseline=None, segments=None,
               max_order=2, budget=512, index="SII", adjustment="msr"):

    # 1. Standardize dimensions
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    if baseline is None:
        baseline = get_baseline(input_tensor, baseline_type='zeros')
    if isinstance(baseline, str):
        baseline = get_baseline(input_tensor, baseline_type=baseline)

    input_np = input_tensor.cpu().numpy()       # (1, C, H, W)
    baseline_np = baseline.cpu().numpy()
    seg_np = segments.cpu().numpy().squeeze()   # (H, W)
    n_players = int(seg_np.max()) + 1

    def value_function(coalitions: np.ndarray) -> np.ndarray:
        outputs = []
        for coalition in coalitions:
            masked = baseline_np.copy()
            for seg_id, active in enumerate(coalition):
                if active:
                    masked[0, :, seg_np == seg_id] = input_np[0, :, seg_np == seg_id]
            inp = torch.tensor(masked, dtype=input_tensor.dtype).to(input_tensor.device)
            with torch.no_grad():
                out = model(inp)
            if out.ndim > 1:
                out = out[:, target]
            outputs.append(out.item())
        return np.array(outputs)

    # 2. Run ProxySHAP
    approximator = ProxySHAP(
        n=n_players,
        max_order=max_order,
        index=index,
        adjustment=adjustment,
    )
    return approximator.approximate(budget=budget, game=value_function)


def interaction_values_to_tensor(iv, segments, image_shape):
    seg_np = segments.cpu().numpy().squeeze()   # (H, W)
    heatmap = np.zeros(seg_np.shape, dtype=np.float32)

    for players, value in iv.interactions.items():
        if len(players) == 1:
            heatmap[seg_np == players[0]] = float(value)

    return torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).contiguous()
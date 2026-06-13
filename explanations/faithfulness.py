import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from params.baselines import get_baseline


def _player_masks(segments):
    """Accept (n, H, W) bool masks or legacy (1, 1, H, W) label map."""
    if isinstance(segments, torch.Tensor):
        segments = segments.cpu().numpy()
    if segments.ndim == 4:
        labels = segments.squeeze()
        return np.stack([labels == i for i in range(int(labels.max()) + 1)])
    return segments


def aggregate_scores(attributions, segments, method="sum"):
    if attributions.ndim == 4:
        attributions = attributions.squeeze(0)
    attr_2d = attributions.mean(dim=0).cpu().numpy()
    masks = _player_masks(segments)

    scores = {}
    for i, mask in enumerate(masks):
        vals = attr_2d[mask]
        scores[i] = vals.sum() if method == "sum" else vals.mean()
    return scores


def ranking_from_scores(scores, descending=True):
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=descending)


def id_curve(model, input_tensor, class_idx, segments, ranking,
             baseline_type="blurred", mode="insertion"):
    device = input_tensor.device
    baseline = get_baseline(input_tensor, baseline_type).to(device)
    masks = _player_masks(segments)

    model.eval()
    with torch.no_grad():
        f_full = F.softmax(model(input_tensor)[0], dim=0)[class_idx].item()
        f_empty = F.softmax(model(baseline)[0], dim=0)[class_idx].item()

    confidences = []
    with torch.no_grad():
        for k in range(len(ranking) + 1):
            if mode == "insertion":
                current = baseline.clone()
                for player_id in ranking[:k]:
                    mask = masks[player_id]
                    current[0, :, mask] = input_tensor[0, :, mask]
            else:
                current = input_tensor.clone()
                for player_id in ranking[:k]:
                    mask = masks[player_id]
                    current[0, :, mask] = baseline[0, :, mask]
            prob = F.softmax(model(current)[0], dim=0)[class_idx].item()
            normalized = (prob - f_empty) / (f_full - f_empty + 1e-8)
            confidences.append(float(np.clip(normalized, 0, 1)))
    return np.array(confidences)


def aid_for_ranking(insertion_curve, deletion_curve):
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    x = np.linspace(0, 1, len(insertion_curve))
    return float(trapz(insertion_curve, x) - trapz(deletion_curve, x))


def run_faithfulness(model, input_tensor, class_idx, attributions, segments,
                     baseline_type="blurred", aggregation="sum"):
    scores = aggregate_scores(attributions, segments, method=aggregation)
    ranking = ranking_from_scores(scores)
    ins = id_curve(model, input_tensor, class_idx, segments, ranking,
                   baseline_type=baseline_type, mode="insertion")
    del_ = id_curve(model, input_tensor, class_idx, segments, ranking,
                    baseline_type=baseline_type, mode="deletion")
    return {
        "insertion_curve": ins,
        "deletion_curve": del_,
        "aid": aid_for_ranking(ins, del_),
        "ranking": ranking,
    }

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from params.baselines import get_baseline


def aggregate_scores(attributions, segments, method="sum"):
    if attributions.ndim == 4:
        attributions = attributions.squeeze(0)
    attr_2d = attributions.mean(dim=0)
    seg_2d  = segments.squeeze(0).squeeze(0)
    scores = {}
    for seg_id in seg_2d.unique():
        seg_id = seg_id.item()
        mask   = (seg_2d == seg_id)
        vals   = attr_2d[mask]
        scores[seg_id] = vals.sum().item() if method == "sum" else vals.mean().item()
    return scores


def ranking_from_scores(scores, descending=True):
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=descending)


def id_curve(model, input_tensor, class_idx, segments, ranking,
             baseline_type="blurred", mode="insertion"):
    device   = input_tensor.device
    baseline = get_baseline(input_tensor, baseline_type).to(device)
    seg_2d   = segments.squeeze(0).squeeze(0)
    n        = len(ranking)

    model.eval()
    with torch.no_grad():
        f_full  = F.softmax(model(input_tensor)[0], dim=0)[class_idx].item()
        f_empty = F.softmax(model(baseline)[0],     dim=0)[class_idx].item()

    confidences = []
    with torch.no_grad():
        for k in range(n + 1):
            if mode == "insertion":
                current = baseline.clone()
                for seg_id in ranking[:k]:
                    mask = (seg_2d == seg_id)
                    current[0, :, mask] = input_tensor[0, :, mask]
            else:
                current = input_tensor.clone()
                for seg_id in ranking[:k]:
                    mask = (seg_2d == seg_id)
                    current[0, :, mask] = baseline[0, :, mask]
            prob       = F.softmax(model(current)[0], dim=0)[class_idx].item()
            normalized = (prob - f_empty) / (f_full - f_empty + 1e-8)
            confidences.append(float(np.clip(normalized, 0, 1)))
    return np.array(confidences)


def aid_for_ranking(insertion_curve, deletion_curve):
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    x     = np.linspace(0, 1, len(insertion_curve))
    return float(trapz(insertion_curve, x) - trapz(deletion_curve, x))


def run_faithfulness(model, input_tensor, class_idx, attributions, segments,
                     baseline_type="blurred", aggregation="sum"):
    scores  = aggregate_scores(attributions, segments, method=aggregation)
    ranking = ranking_from_scores(scores)
    ins  = id_curve(model, input_tensor, class_idx, segments, ranking,
                    baseline_type=baseline_type, mode="insertion")
    del_ = id_curve(model, input_tensor, class_idx, segments, ranking,
                    baseline_type=baseline_type, mode="deletion")
    aid  = aid_for_ranking(ins, del_)
    return {
        "insertion_curve": ins,
        "deletion_curve":  del_,
        "aid":             aid,
        "ranking":         ranking
    }

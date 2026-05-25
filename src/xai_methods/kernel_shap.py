# src/XAI_methods/kernel_shap.py
import torch
from captum.attr import KernelShap
from xai_methods.baselines import get_baseline

def kernel_shap(model, input_tensor, target, baseline=None, segments=None, n_samples=500):

    ks = KernelShap(model)
    
    # 1. Standardize dimensions (Ensure 4D Batch dimension for Captum)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
        
    if segments is not None and segments.ndim == 3:
        segments = segments.unsqueeze(0)

    if baseline is None:
        baseline = get_baseline(input_tensor, baseline_type='zeros')
    if isinstance(baseline, str):
        baseline = get_baseline(input_tensor, baseline_type=baseline)

    # 3. Compute and return attributions
    attributions = ks.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target,
        feature_mask=segments,
        n_samples=n_samples
    )
    
    return attributions
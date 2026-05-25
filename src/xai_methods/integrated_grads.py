# src/XAI_methods/integrated_gradients.py
import torch
from captum.attr import IntegratedGradients
from xai_methods.baselines import get_baseline

def integrated_gradients(model, input_tensor, target, baseline=None, n_steps=50):
    """
    Computes Integrated Gradients for a given input tensor.
    """
    ig = IntegratedGradients(model)
    
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
        
    if baseline is None:
        baseline = get_baseline(input_tensor, baseline_type='zeros')
    elif isinstance(baseline, str):
        baseline = get_baseline(input_tensor, baseline_type=baseline)


    attributions = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target,
        n_steps=n_steps
    )
    return attributions
# src/XAI_methods/guided_grad_cam.py
import torch
from captum.attr import GuidedGradCam

def guided_grad_cam(model, input_tensor, target, target_layer_name="layer4"):
    """
    Computes Guided Grad-CAM by combining Guided Backpropagation and Grad-CAM.
    """
    try:
        target_layer = getattr(model, target_layer_name)
    except AttributeError:
        raise AttributeError(f"Layer '{target_layer_name}' not found in the model structure.")
        
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # 1. Compute Coarse Grad-CAM
    ggc = GuidedGradCam(model, target_layer)
    gc_attr = ggc.attribute(inputs=input_tensor, target=target, interpolate_mode='bilinear')

    return gc_attr
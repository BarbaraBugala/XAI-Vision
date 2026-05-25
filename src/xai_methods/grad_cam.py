# src/XAI_methods/grad_cam.py
import torch
from captum.attr import LayerGradCam

def grad_cam(model, input_tensor, target, target_layer_name="layer4"):
    """
    Computes Grad-CAM attributions for a specified convolutional layer.
    
    Args:
        target_layer_name (str): The attribute name of the target layer in your model.
                                 For standard ResNets, 'layer4' is the final block.
    """
    # Dynamically fetch the layer object from the model using its string name
    try:
        target_layer = getattr(model, target_layer_name)
    except AttributeError:
        raise AttributeError(f"Layer '{target_layer_name}' not found in the model structure.")

    lgc = LayerGradCam(model, target_layer)
    
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
        
    attributions = lgc.attribute(
        inputs=input_tensor,
        target=target
    )
    
    # LayerGradCam outputs map to the layer's spatial size (e.g., 1x1x7x7).
    # To compare it with other pixel-level methods, we interpolate it back to the image size.
    spatial_shape = input_tensor.shape[2:] # (H, W)
    attributions = LayerGradCam.interpolate(attributions, spatial_shape)
    
    return attributions
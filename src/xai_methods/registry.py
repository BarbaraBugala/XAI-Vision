# src/XAI_methods/registry.py
import torch
from xai_methods.kernel_shap import kernel_shap
from xai_methods.saliency import saliency
from xai_methods.integrated_grads import integrated_gradients
from xai_methods.grad_cam import grad_cam
from xai_methods.guided_grad_cam import guided_grad_cam
from xai_methods.proxy_shap import proxy_shap

class XAIRegistry:
    def __init__(self, model, method_name: str):
        self.model = model
        
        # Maps strings directly to your raw functions
        self._methods = {
            "kernel_shap": kernel_shap,
            "saliency": saliency,
            "integrated_gradients": integrated_gradients,
            "grad_cam": grad_cam,
            "guided_grad_cam": guided_grad_cam,
            "proxy_shap": proxy_shap,
        }
        
        if method_name not in self._methods:
            raise ValueError(f"Method '{method_name}' not found. Choose from: {list(self._methods.keys())}")
            
        # Assign the function to self.method
        self.method = self._methods[method_name]

    def compute(self, image, class_idx, **kwargs):
        """
        Executes the chosen XAI method. All method-specific arguments 
        (like baseline, segmentation, n_samples) are passed via **kwargs.
        """
        return self.method(
            model=self.model, 
            input_tensor=image, 
            target=class_idx, 
            **kwargs
        )
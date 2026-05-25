import json
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from captum.attr import visualization as viz

# Internal project imports
from xai_methods.registry import XAIRegistry
from xai_methods.segmentation import get_superpixels

class XAIPipeline:
    def __init__(self, device=None):
        """
        Sets up the environment, model, and standard preprocessing weights.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load stable model configuration
        self.weights = models.ResNet18_Weights.DEFAULT
        self.labels = self.weights.meta["categories"]
        self.model = models.resnet18(weights=self.weights).to(self.device)
        self.model.eval()
        
        # Core ImageNet Transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Inverse Normalization for pristine Matplotlib colors
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def _preprocess_image(self, image_path):
        """Loads and formats a local image into PyTorch and back into clean NumPy."""
        orig_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(orig_image).unsqueeze(0).to(self.device)
        
        # Prepare background display array
        img_tensor_unnorm = self.inv_normalize(input_tensor.squeeze(0))
        img_np = img_tensor_unnorm.permute(1, 2, 0).detach().cpu().numpy()
        
        return input_tensor, img_np

    def _get_prediction(self, input_tensor):
        """Performs forward inference pass and tracks target class indices safely."""
        with torch.no_grad():
            output = self.model(input_tensor)
            
        probabilities = F.softmax(output[0], dim=0)
        prob, class_idx = torch.max(probabilities, dim=0)
        return class_idx.item(), prob.item()

    def run_experiment(self, image_path, method_name, config):
        """
        Orchestrates an XAI interpretation task and saves plots directly.
        """
        # 1. Output location generation
        output_dir = os.path.join("xai_results",os.path.splitext(os.path.basename(image_path))[0], method_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Extract Data
        input_tensor, img_np = self._preprocess_image(image_path)
        class_idx, confidence = self._get_prediction(input_tensor)
        
        print(f"File: {image_path} | Pred: {self.labels[class_idx]} ({confidence*100:.1f}%)")
        print(f"Calculating attributions via '{method_name}'...")

        # 3. Dynamic Kwargs Mapping based on targeted features
        xai_kwargs = {}
        if method_name == "kernel_shap":
            segments = get_superpixels(
                input_tensor, 
                n_segments=config.get("n_segments", 50), 
                compactness=config.get("compactness", 10), 
                sigma=config.get("sigma", 1)
            ).to(self.device)
            
            xai_kwargs = {
                "baseline": config.get("baseline_type", "zeros"),
                "segments": segments,
                "n_samples": config.get("num_samples", 200)
            }
        elif method_name in ["integrated_gradients"]:
            xai_kwargs = {"baseline": config.get("baseline_type", "zeros"), "n_steps": 50}
        elif method_name in ["grad_cam", "guided_grad_cam"]:
            xai_kwargs = {"target_layer_name": config.get("target_layer_name", "layer4")}

        # 4. Generate Core Attributions
        explainer = XAIRegistry(model=self.model, method_name=method_name)
        attributions = explainer.compute(input_tensor, class_idx=class_idx, **xai_kwargs)

        # 1. Prepare your dimensions exactly how you always do
        attr = attributions.squeeze(0)
        attr = attr.permute(1, 2, 0)
        attr = attr.detach().cpu().numpy()

        # 2. Create a single figure and axis (no layout array)
        fig, ax = plt.subplots(figsize=(6, 6))

        # Extract the configuration properties dynamically from your runtime options
        baseline_type = xai_kwargs.get("baseline", "none") if method_name in ["kernel_shap", "integrated_gradients"] else "N/A"
        sign_type = "all" if method_name in ["integrated_gradients", "saliency", "guided_grad_cam", "kernel_shap"] else "positive"
        out_method = "heat_map" if method_name in ["grad_cam", "kernel_shap"] else "heat_map"

        # 3. Call the visualizer with your single 'ax' element
        viz.visualize_image_attr(
            attr, 
            img_np, 
            method=out_method, 
            sign=sign_type, 
            show_colorbar=True, 
            title=f"{method_name} with {baseline_type} baseline", 
            plt_fig_axis=(fig, ax)
        )

        image_number = np.random.randint(0, 10000)
        # 4. Export the resulting visualization
        output_filename = os.path.join(output_dir, f"{image_number}_{method_name}.png")
        plt.tight_layout()
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"Saved visualization to: {output_filename}\n" + "-"*50)

        # 8. Save Experiment Configuration to JSON
        # Create a clean copy of the config dictionary to save as text
        saveable_config = {
            "image_path": image_path,
            "prediction": self.labels[class_idx],
            "confidence": f"{confidence * 100:.2f}%",
            **{k: v for k, v in config.items() if not isinstance(v, torch.Tensor)} 
        }
        output_dir_config = os.path.join(output_dir, "configs")
        os.makedirs(output_dir_config, exist_ok=True)
        config_filename = os.path.join(output_dir_config, f"{image_number}_config.json")
        with open(config_filename, "w") as f:
            json.dump(saveable_config, f, indent=4)

        print(f"Saved configuration to: {config_filename}\n" + "-"*50)

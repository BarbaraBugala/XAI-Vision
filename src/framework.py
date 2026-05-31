# src/framework.py
import os
import json
import torch
import numpy as np

from xai_methods.registry import XAIRegistry
from params.segmentation import get_superpixels
from utils.model_prediction import run_prediction
from utils.plotting import save_xai_visualization 

class XAIPipeline:
    def run_experiment(self, image_path, method_name, config):
        """
        Orchestrates an XAI interpretation task using precomputed predictions.
        """
        # 1. Extract unified prediction outputs directly
        pred_data = run_prediction(image_path)
        
        model = pred_data["model"]
        input_tensor = pred_data["input_tensor"]
        img_np = pred_data["img_np"]
        class_idx = pred_data["class_idx"]
        device = pred_data["device"]

        # 2. Output location generation
        baseline_str = config.get("baseline_type", "no_baseline")
        superpixels_str = f"{config.get('n_segments', 'no_segments')}segments"
        output_dir = os.path.join("xai_results", os.path.splitext(os.path.basename(image_path))[0], method_name, superpixels_str, baseline_str)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"File: {image_path} | Pred: {pred_data['pred_label']} ({pred_data['confidence']})")
        print(f"Calculating attributions via '{method_name}'...")

        # 3. Dynamic Kwargs Mapping based on targeted features
        xai_kwargs = {}
        if method_name == "kernel_shap":
            segments = get_superpixels(
                input_tensor, 
                n_segments=config.get("n_segments", 50), 
                compactness=config.get("compactness", 10), 
                sigma=config.get("sigma", 1)
            ).to(device)
            
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
        explainer = XAIRegistry(model=model, method_name=method_name)
        attributions = explainer.compute(input_tensor, class_idx=class_idx, **xai_kwargs)

        # 5. Format dimensions for Plotting
        attr_np = attributions.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Generate unique image tag
        image_number = np.random.randint(0, 10000)

        # 6 & 7. Call the externalized plotting function
        actual_baseline = xai_kwargs.get("baseline", "none") if method_name in ["kernel_shap", "integrated_gradients"] else "N/A"
        save_xai_visualization(
            attr=attr_np,
            img_np=img_np,
            method_name=method_name,
            baseline_type=actual_baseline,
            output_dir=output_dir,
            image_number=image_number
        )

        # 8. Save Experiment Configuration to JSON
        saveable_config = {
            "image_path": image_path,
            "prediction": pred_data["pred_label"],
            "confidence": pred_data["confidence"],
            **{k: v for k, v in config.items() if not isinstance(v, torch.Tensor)} 
        }
        output_dir_config = os.path.join(output_dir, "configs")
        os.makedirs(output_dir_config, exist_ok=True)
        config_filename = os.path.join(output_dir_config, f"{image_number}_config.json")
        with open(config_filename, "w") as f:
            json.dump(saveable_config, f, indent=4)

        print(f"Saved configuration to: {config_filename}\n" + "-"*50)
        
        return {
            "model": model,
            "input_tensor": input_tensor,
            "class_idx": class_idx,
            "attributions": attributions,
            "segments": xai_kwargs.get("segments")
        }
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

        # 4. Generate Core Attributions
        explainer = XAIRegistry(model=model, method_name=method_name)
        attributions = explainer.compute(input_tensor, class_idx=class_idx, **xai_kwargs)


        if config.get("save_file"):

            pred_data["image_path"] = image_path 
            
            _, image_number = save_xai_visualization(
                attributions=attributions, # Pass the raw tensor directly!
                img_np=img_np,
                method_name=method_name,
                config=config,
                pred_data=pred_data,
                xai_kwargs=xai_kwargs,
                output_dir=output_dir
            )

        if config.get("insertion-deletion_score"):
            pass
        
        return {
            "model": model,
            "input_tensor": input_tensor,
            "class_idx": class_idx,
            "attributions": attributions,
            "segments": xai_kwargs.get("segments")
        }
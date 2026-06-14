# src/framework.py
import os

from xai_methods.registry import XAIRegistry
from params.segmentation_registry import get_segmentation
from models.model_prediction import run_prediction
from plots.plotting import save_xai_visualization

SHAP_METHODS = {"kernel_shap", "proxy_shap"}


class XAIPipeline:
    def run_experiment(self, image_path, method_name, config, model_name="resnet18"):
        """Orchestrates an XAI interpretation task using precomputed predictions."""
        pred_data = run_prediction(image_path, model_name=model_name)
        model = pred_data["model"]
        input_tensor = pred_data["input_tensor"]
        img_np = pred_data["img_np"]
        class_idx = pred_data["class_idx"]
        device = pred_data["device"]

        baseline_str = config.get("baseline_type", "no_baseline")
        n_segments = config.get("n_segments", "no_segments")
        segmentation = config.get("segmentation", "no_segmentation")
        output_dir = os.path.join(
            "xai_results",
            os.path.splitext(os.path.basename(image_path))[0],
            model_name,
            "shapiq_vision",
            segmentation,
            f"{n_segments}segments",
            baseline_str,
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"File: {image_path} | Pred: {pred_data['pred_label']} ({pred_data['confidence']})")
        print(f"Calculating attributions via '{method_name}'...")

        if method_name in SHAP_METHODS:
            xai_kwargs = {
                "img_np": img_np,
                "architecture": pred_data["architecture"],
                "processor": pred_data.get("processor"),
                **config,
            }
            segments = None
        else:
            segments = get_segmentation(
                method=config.get("segmentation", "superpixels"),
                image=input_tensor,
                config=config,
            ).to(device)
            xai_kwargs = {}

        result = XAIRegistry(model=model, method_name=method_name).compute(
            input_tensor, class_idx=class_idx, **xai_kwargs
        )

        if method_name in SHAP_METHODS:
            attributions, segments = result
        else:
            attributions = result

        if config.get("save_file"):
            pred_data["image_path"] = image_path
            save_xai_visualization(
                attributions=attributions,
                img_np=img_np,
                method_name=method_name,
                config=config,
                pred_data=pred_data,
                xai_kwargs=xai_kwargs,
                output_dir=output_dir,
            )

        return {
            "model": model,
            "input_tensor": input_tensor,
            "class_idx": class_idx,
            "attributions": attributions,
            "segments": segments,
        }

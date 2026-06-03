import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz

def save_xai_visualization(attributions, img_np, method_name, config, pred_data, xai_kwargs, output_dir):
    """
    Handles Captum attribution transformation, PNG rendering, and JSON metadata export.
    """
    # 1. Generate unique image tag right here
    image_number = np.random.randint(0, 10000)
    actual_baseline = xai_kwargs.get("baseline", "none")

    # 2. Convert PyTorch tensor to NumPy (H, W, C) for Captum
    attr_np = attributions.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # 3. Build and save the matplotlib visualization
    fig, ax = plt.subplots(figsize=(6, 6))

    sign_type = "all"
    out_method = "heat_map"

    viz.visualize_image_attr(
        attr_np,  # Using the transformed numpy array
        img_np, 
        method=out_method, 
        sign=sign_type, 
        show_colorbar=True, 
        title=f"{method_name} with {actual_baseline} baseline", 
        plt_fig_axis=(fig, ax)
    )

    output_filename = os.path.join(output_dir, f"{image_number}_{method_name}.png")
    plt.tight_layout()
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved visualization to: {output_filename}")

    # 4. Save Experiment Configuration to JSON
    saveable_config = {
        "image_path": pred_data.get("image_path", "unknown"), # Fallback if not injected
        "prediction": pred_data["pred_label"],
        "confidence": pred_data["confidence"],
        **{k: v for k, v in config.items() if not isinstance(v, (torch.Tensor, torch.device))} 
    }
    
    output_dir_config = os.path.join(output_dir, "configs")
    os.makedirs(output_dir_config, exist_ok=True)
    config_filename = os.path.join(output_dir_config, f"{image_number}_config.json")
    
    with open(config_filename, "w") as f:
        json.dump(saveable_config, f, indent=4)

    print(f"Saved configuration to: {config_filename}\n" + "-"*50)
    
    return output_filename, image_number
# src/utils/plotting.py
import os
import matplotlib.pyplot as plt
from captum.attr import visualization as viz

def save_xai_visualization(attr, img_np, method_name, baseline_type, output_dir, image_number):
    """
    Handles Captum attribution visualization and exports the result as a PNG.
    """
    # 1. Build the Visualizations
    fig, ax = plt.subplots(figsize=(6, 6))

    # Determine sign configuration based on the XAI method
    if method_name in ["integrated_gradients", "saliency", "guided_grad_cam", "kernel_shap"]:
        sign_type = "all"
    else:
        sign_type = "positive"
        
    out_method = "heat_map"

    viz.visualize_image_attr(
        attr, 
        img_np, 
        method=out_method, 
        sign=sign_type, 
        show_colorbar=True, 
        title=f"{method_name} with {baseline_type} baseline", 
        plt_fig_axis=(fig, ax)
    )

    # 2. Export the resulting visualization
    output_filename = os.path.join(output_dir, f"{image_number}_{method_name}.png")
    plt.tight_layout()
    fig.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved visualization to: {output_filename}\n" + "-"*50)
    return output_filename
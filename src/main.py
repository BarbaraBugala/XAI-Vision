import os
from framework import XAIPipeline

# ------------------------------------- SPECIFY  ----------------------------------------

image = "cat1.png"
method = "proxy_shap"

experiment_config = {
        "n_segments": 20,
        "baseline_type": "blurred",          # options ["zeros", "blurred", "average", "ones"]
        "compactness": 10,                   # for superpixel segmentation
        "sigma": 1,                          # for superpixel segmentation
        "num_samples": 200,                  # for Kernel SHAP sampling
        "save_file": False,                  # if you want to save output image set to True
        "insertion-deletion_score": False,
        "max_order": 2,       # 1 = attributions only, 2 = pairwise interactions
        "budget": 512,        # analogous to num_samples in KernelSHAP
        "index": "SII",       # or "k-SII", "BII"
        "adjustment": "msr",  # or "svarm", "kernelshapiq"
        "save_file": True,
    }

# -----------------------------------------------------------------------------------------

# Initialize your pipeline asset once
pipeline = XAIPipeline()


outputs = pipeline.run_experiment(
            image_path=os.path.join("data", image),
            method_name=method,                    
            config=experiment_config
        )


'''
outputs contains:
{
    "model": model,
    "input_tensor": input_tensor,
    "class_idx": class_idx,
    "attributions": attributions,               # shape [1, 3, 224, 224]
    "segments": xai_kwargs.get("segments")      # shape [1, 1, 224, 224]
}

To get the attributions, you can access it via:
attrs = outputs["attributions"]
'''
import os
from framework import XAIPipeline

# ------------------------------------- SPECIFY  ----------------------------------------

method = "kernel_shap"
model_name = "resnet18"
image = "ood1.png"
slic_segmentations = [20, 40, 60]
grid_segmentations = [50, 75, 100]
baselines = ["zeros", "blurred", "average", "ones"]        

experiment_config = {
        "segmentation": "grid",          # options ["superpixels", "grid"]
        "n_segments": 50,
        "baseline_type": "blurred",          # options ["zeros", "blurred", "average", "ones"]
        "compactness": 10,                   # for superpixel segmentation
        "sigma": 1,                          # for superpixel segmentation
        "num_samples": 200,                  # for Kernel SHAP sampling
        "max_order": 2,                      # 1 = attributions only, 2 = pairwise interactions (for proxy shap)
        "save_file": True,                   # if you want to save output image set to True
    }

# -----------------------------------------------------------------------------------------
# Initialize your pipeline asset once
pipeline = XAIPipeline()


for segm in grid_segmentations:
    for baseline in baselines:
        experiment_config["n_segments"] = segm
        experiment_config["baseline_type"] = baseline
        outputs = pipeline.run_experiment(
                model_name=model_name,
                image_path=os.path.join("data", image),
                method_name=method,         
                config=experiment_config
            )
                

# image = "Bella.png"
# method = "proxy_shap"
# model_name = "resnet18"                      # options: ["resnet18", "vitb16"]

# experiment_config = {
#         "segmentation": "superpixels",          # options ["superpixels", "grid"]
#         "n_segments": 50,
#         "baseline_type": "blurred",          # options ["zeros", "blurred", "average", "ones"]
#         "compactness": 10,                   # for superpixel segmentation
#         "sigma": 1,                          # for superpixel segmentation
#         "num_samples": 200,                  # for Kernel SHAP sampling
#         "max_order": 2,                      # 1 = attributions only, 2 = pairwise interactions (for proxy shap)
#         "save_file": True,                   # if you want to save output image set to True
#     }

# outputs = pipeline.run_experiment(
#             model_name=model_name,
#             image_path=os.path.join("data", image),
#             method_name=method,                    
#             config=experiment_config
#         )


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
import os
from framework import XAIPipeline

# ------------------------------------- SPECIFY  ----------------------------------------

image = "cat1.png"
method = "kernel_shap"
segments_list = [10] #, 20, 30, 40, 50]
baseline_types = ["zeros"] #, "blurred", "average", "ones"]
# layer_list = ["layer4", "layer3", "layer2", "layer1"]   # for grad cam methods

experiment_config = {
        "compactness": 10,              # for superpixel segmentation
        "sigma": 1,                     # for superpixel segmentation
        "num_samples": 200,             # for Kernel SHAP sampling
    }

# -----------------------------------------------------------------------------------------

# Initialize your pipeline asset once
pipeline = XAIPipeline()


# Loop through each segment count and baseline and run the experiment
for segments in segments_list:
    for baseline in baseline_types:
        print(f"\n--- Running experiment with n_segments = {segments}, baseline = {baseline} ---")
        
        # Update the config dynamically
        experiment_config["n_segments"] = segments
        experiment_config["baseline_type"] = baseline
        
        # Run the pipeline for the current segment configuration
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
import os
from framework import XAIPipeline


segments_list = [10, 20, 30, 40, 50]
baseline_types = ["zeros", "blurred", "average", "ones"]

experiment_config = {
        "n_segments": 50,               # for superpixel segmentation
        "compactness": 10,              # for superpixel segmentation
        "sigma": 1,                     # for superpixel segmentation
        "num_samples": 200,             # for Kernel SHAP sampling
        "target_layer_name": "layer4",  # for Grad-CAM methods
        "baseline_type": "zeros"      # for baseline generation (options: 'zeros', 'blurred', 'average', 'ones')
    }

# Initialize your pipeline asset once
pipeline = XAIPipeline()
    
# Loop through each segment count and run the experiment
for segments in segments_list:
    for baseline in baseline_types:
        print(f"\n--- Running experiment with n_segments = {segments}, baseline = {baseline} ---")
        
        # Update the config dynamically
        experiment_config["n_segments"] = segments
        experiment_config["baseline_type"] = baseline
        
        # Run the pipeline for the current segment configuration
        outputs = pipeline.run_experiment(
            image_path=os.path.join("data", "cat1.png"),
            method_name="kernel_shap", 
            config=experiment_config
        )


'''
outputs contains:
{
    "model": model,
    "input_tensor": input_tensor,
    "class_idx": class_idx,
    "attributions": attributions,
    "segments": xai_kwargs.get("segments")
}

To get the attributions, you can access it via:
attrs = outputs["attributions"]
'''
import os
from framework import XAIPipeline


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
    
# Run it! It reads wonderfully now:
pipeline.run_experiment(
        image_path=os.path.join("data", "cat2.png"),
        method_name="integrated_gradients", 
        config=experiment_config
    )
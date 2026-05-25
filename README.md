How to work with the code for now:


Specify config, data and method with your own values and run 

```text
python3 src/main.py 
```
The plots will appear in xai_results.


Minimal working example on how to get attributions
```text
import os
import torch
from xai_methods.registry import XAIRegistry
from xai_methods.segmentation import get_superpixels
from run_prediction import run_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_file = os.path.join("data", "cat1.png")

pred_data = run_prediction(image_file, device=device)
model, input_tensor, class_idx = pred_data["model"], pred_data["input_tensor"], pred_data["class_idx"]

segments = get_superpixels(input_tensor, n_segments=50, compactness=10).to(device)

xai_kwargs = {
    "baseline": "blurred",
    "segments": segments,
    "n_samples": 500
}

explainer = XAIRegistry(model=model, method_name="kernel_shap")
attributions = explainer.compute(input_tensor, class_idx=class_idx, **xai_kwargs)

print("Raw Attributions Shape:", attributions.shape)
```
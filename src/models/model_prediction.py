# src/run_prediction.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.registry import get_model_config


def run_prediction(image_path, model_name="resnet18", device=None, top_prediction=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model + transforms from registry
    cfg = get_model_config(model_name)
    model = cfg["model_fn"](weights=cfg["weights"]).to(device)
    model.eval()

    # 2. Preprocess image
    orig_image = Image.open(image_path).convert("RGB")
    input_tensor = cfg["transform"](orig_image).unsqueeze(0).to(device)

    img_tensor_unnorm = cfg["inv_normalize"](input_tensor.squeeze(0))
    img_np = img_tensor_unnorm.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # 3. Predict
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output[0], dim=0)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=3)

    return {
        "model": model,
        "input_tensor": input_tensor,
        "img_np": img_np,
        "class_idx": top_indices[top_prediction].item(),
        "pred_label": cfg["labels"][top_indices[top_prediction].item()],
        "confidence": f"{top_probs[top_prediction].item() * 100:.2f}%",
        "device": device,
    }
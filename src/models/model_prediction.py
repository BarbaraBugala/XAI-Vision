# src/run_prediction.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models.registry import get_model_config


def _predict_torchvision(model, input_tensor, labels, top_prediction):
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = F.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, k=3)
    return {
        "class_idx": top_indices[top_prediction].item(),
        "pred_label": labels[top_indices[top_prediction].item()],
        "confidence": f"{top_probs[top_prediction].item() * 100:.2f}%",
    }


def _predict_hf(model, input_tensor, labels, top_prediction):
    with torch.no_grad():
        output = model(pixel_values=input_tensor)
    probabilities = F.softmax(output.logits[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, k=3)
    return {
        "class_idx": top_indices[top_prediction].item(),
        "pred_label": labels[top_indices[top_prediction].item()],
        "confidence": f"{top_probs[top_prediction].item() * 100:.2f}%",
    }


def run_prediction(image_path, model_name="resnet18", device=None, top_prediction=0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_model_config(model_name)
    orig_image = Image.open(image_path).convert("RGB")
    architecture = cfg["architecture"]

    if architecture == "cnn":
        model = cfg["model_fn"](weights=cfg["weights"]).to(device)
        model.eval()
        input_tensor = cfg["transform"](orig_image).unsqueeze(0).to(device)
        img_tensor_unnorm = cfg["inv_normalize"](input_tensor.squeeze(0))
        img_np = img_tensor_unnorm.permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        pred = _predict_torchvision(model, input_tensor, cfg["labels"], top_prediction)
        processor = None
    elif architecture in ("transformer", "hf_pixel"):
        model, processor, labels = cfg["loader_fn"]()
        model = model.to(device)
        model.eval()
        inputs = processor(images=orig_image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(device)
        img_np = np.asarray(orig_image, dtype=np.float32) / 255.0
        pred = _predict_hf(model, input_tensor, labels, top_prediction)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return {
        "model": model,
        "input_tensor": input_tensor,
        "img_np": img_np,
        "class_idx": pred["class_idx"],
        "pred_label": pred["pred_label"],
        "confidence": pred["confidence"],
        "device": device,
        "architecture": architecture,
        "processor": processor,
        "model_name": model_name,
    }

# src/run_prediction.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

def run_prediction(image_path, device=None):
    """
    Loads ResNet18, processes an image, runs prediction, 
    and returns everything required for an XAI task.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1. Setup Model
    weights = models.ResNet18_Weights.DEFAULT
    labels = weights.meta["categories"]
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    
    # 2. Transformers
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # 3. Preprocess Image
    orig_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(orig_image).unsqueeze(0).to(device)
    
    img_tensor_unnorm = inv_normalize(input_tensor.squeeze(0))
    img_np = img_tensor_unnorm.permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    # 4. Predict
    with torch.no_grad():
        output = model(input_tensor)
        
    probabilities = F.softmax(output[0], dim=0)
    prob, class_idx = torch.max(probabilities, dim=0)
    
    # Pack readable details
    pred_label = labels[class_idx.item()]
    confidence_pct = f"{prob.item() * 100:.2f}%"
    
    return {
        "model": model,
        "input_tensor": input_tensor,
        "img_np": img_np,
        "class_idx": class_idx.item(),
        "pred_label": pred_label,
        "confidence": confidence_pct,
        "device": device
    }
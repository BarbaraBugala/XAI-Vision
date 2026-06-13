# src/models/model_registry.py
import torchvision.models as models
import torchvision.transforms as transforms

def get_model_config(model_name: str) -> dict:
    """
    Returns model, weights, transform, and inv_transform for a given model name.
    """
    configs = {
        "resnet18": {
            "model_fn": models.resnet18,
            "weights": models.ResNet18_Weights.DEFAULT,
        },
        "vitb16": {
            "model_fn": models.vit_b_16,
            "weights": models.ViT_B_16_Weights.DEFAULT,
        },
    }

    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found. Choose from: {list(configs.keys())}")

    cfg = configs[model_name]
    weights = cfg["weights"]

    # Use the transform baked into the weights (works for all torchvision models)
    transform = weights.transforms()

    # Compute inverse normalize from the transform's mean/std
    mean = transform.mean if hasattr(transform, "mean") else [0.485, 0.456, 0.406]
    std  = transform.std  if hasattr(transform, "std")  else [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    return {
        "model_fn": cfg["model_fn"],
        "weights": weights,
        "labels": weights.meta["categories"],
        "transform": transform,
        "inv_normalize": inv_normalize,
    }
# src/models/model_registry.py
import torchvision.models as models
import torchvision.transforms as transforms

# architecture types:
#   "cnn"      — torchvision models; pixel masking via CNNArchitecture
#   "hf_pixel" — HuggingFace classifiers; pixel masking via HuggingFacePixelArchitecture
#   "transformer" — HuggingFace ViT; token masking via TransformerArchitecture


def _hf_vit_b16():
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model_id = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return model, processor, labels


def get_model_config(model_name: str) -> dict:
    """
    Returns model factory, weights/transform, and shapiq vision architecture metadata.
    """
    configs = {
        "resnet18": {
            "architecture": "cnn",
            "model_fn": models.resnet18,
            "weights": models.ResNet18_Weights.DEFAULT,
        },
        "vitb16": {
            # Torchvision ViT is a plain nn.Module — pixel masking is the correct path.
            "architecture": "cnn",
            "model_fn": models.vit_b_16,
            "weights": models.ViT_B_16_Weights.DEFAULT,
        },
        "vit_hf_b16": {
            # HuggingFace ViT with pixel masking — supports the same grid/superpixel sweeps.
            "architecture": "hf_pixel",
            "loader_fn": _hf_vit_b16,
        },
    }

    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found. Choose from: {list(configs.keys())}")

    cfg = dict(configs[model_name])

    if cfg["architecture"] == "cnn":
        weights = cfg["weights"]
        transform = weights.transforms()
        mean = transform.mean if hasattr(transform, "mean") else [0.485, 0.456, 0.406]
        std = transform.std if hasattr(transform, "std") else [0.229, 0.224, 0.225]
        inv_normalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std],
        )
        cfg.update(
            {
                "labels": weights.meta["categories"],
                "transform": transform,
                "inv_normalize": inv_normalize,
            }
        )

    return cfg

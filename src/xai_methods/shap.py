from xai_methods.vision_backend import run_shap


def kernel_shap(model, input_tensor, target, img_np=None, architecture="cnn", processor=None, **config):
    if img_np is None:
        raise ValueError("img_np is required for shapiq vision methods.")
    return run_shap(
        model, img_np, target,
        method="kernel_shap",
        architecture=architecture,
        processor=processor,
        config=config,
    )


def proxy_shap(model, input_tensor, target, img_np=None, architecture="cnn", processor=None, **config):
    if img_np is None:
        raise ValueError("img_np is required for shapiq vision methods.")
    return run_shap(
        model, img_np, target,
        method="proxy_shap",
        architecture=architecture,
        processor=processor,
        config=config,
    )

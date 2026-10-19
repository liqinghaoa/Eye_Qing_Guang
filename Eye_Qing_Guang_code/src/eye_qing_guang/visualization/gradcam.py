"""Grad-CAM on the last convolution, using saved inference preprocessing."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from eye_qing_guang.contracts.checkpoints import load_checkpoint
from eye_qing_guang.datasets.metadata import normalize_age
from eye_qing_guang.datasets.transforms import build_transforms, fov_crop, letterbox
from eye_qing_guang.utils.reproducibility import choose_device


def generate_gradcam(checkpoint: Path, image_path: Path, output: Path,
                     age=None, device_name="auto", class_index=1):
    if output.exists():
        raise FileExistsError(output)
    if class_index not in {0, 1} or output.suffix.lower() != ".png":
        raise ValueError("class_index must be 0 or 1; output must be .png.")
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    data = payload["config"]["data"]
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        inputs = build_transforms(data)(rgb).unsqueeze(0).to(device)
        if data.get("preprocessing", {}).get("version") == "fov_letterbox_v1":
            display, _ = fov_crop(rgb, data["preprocessing"])
            display = letterbox(display, data["image_size"])
        else:
            display = rgb.resize((data["image_size"], data["image_size"]))
        background = np.asarray(display, dtype=float) / 255
    age_tensor = torch.tensor([[normalize_age(age, payload["age_stats"])]],
                              dtype=torch.float32, device=device)
    layers = [layer for layer in model.encoder.modules() if isinstance(layer, nn.Conv2d)]
    if not layers:
        raise ValueError("Model has no convolutional layer for Grad-CAM.")
    activation = {}

    def capture(module, inputs, output_tensor):
        output_tensor.retain_grad()
        activation["tensor"] = output_tensor

    hook = layers[-1].register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(inputs, age_tensor).sum()
        (logit if class_index == 1 else -logit).backward()
        feature = activation["tensor"]
        weights = feature.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam /= max(float(cam.max()), 1e-12)
    finally:
        hook.remove()
    from matplotlib import colormaps

    heat = colormaps["jet"](cam)[..., :3]
    overlay = np.clip(0.6 * background + 0.4 * heat, 0, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        Image.fromarray((overlay * 255).astype(np.uint8)).save(handle, format="PNG")

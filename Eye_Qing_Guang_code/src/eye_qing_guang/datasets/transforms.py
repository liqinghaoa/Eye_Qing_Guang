"""Baseline RGB resize, augmentation and normalization."""

import math

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms


def scaled_size(width, height, size, upscale=True):
    ratio = min(size / width, size / height, 1.0 if not upscale else float("inf"))
    return (min(size, max(1, math.floor(width * ratio + 0.5))),
            min(size, max(1, math.floor(height * ratio + 0.5))))


def fov_crop(image, settings):
    """Conservative half-open crop; coverage is measured in preview coordinates."""
    width, height = image.size
    pw, ph = scaled_size(width, height, settings["preview_size"], upscale=False)
    preview = image.resize((pw, ph), Image.Resampling.BILINEAR)
    mask = np.asarray(preview).max(axis=2) > settings["threshold"]
    total = int(mask.sum())
    full = (0, 0, width, height)
    box, reason, coverage = full, "", 1.0
    if total == 0:
        reason = "empty_foreground"
    else:
        ys = np.flatnonzero(mask.mean(axis=1) >= settings["min_line_fraction"])
        xs = np.flatnonzero(mask.mean(axis=0) >= settings["min_line_fraction"])
        if not len(xs) or not len(ys):
            reason = "invalid_box"
        else:
            left, top = math.floor(xs[0] * width / pw), math.floor(ys[0] * height / ph)
            right = math.ceil((xs[-1] + 1) * width / pw)
            bottom = math.ceil((ys[-1] + 1) * height / ph)
            mx, my = math.ceil((right - left) * settings["margin"]), math.ceil((bottom - top) * settings["margin"])
            candidate = (max(0, left - mx), max(0, top - my),
                         min(width, right + mx), min(height, bottom + my))
            l, t, r, b = candidate
            area = (r - l) * (b - t) / (width * height)
            pl, pt = max(0, math.floor(l * pw / width)), max(0, math.floor(t * ph / height))
            pr, pb = min(pw, math.ceil(r * pw / width)), min(ph, math.ceil(b * ph / height))
            coverage = float(mask[pt:pb, pl:pr].sum() / total)
            if r <= l or b <= t:
                reason = "invalid_box"
            elif area < settings["min_area"]:
                reason = "small_area"
            elif coverage < settings["min_coverage"]:
                reason = "foreground_loss"
            else:
                box = candidate
    l, t, r, b = box
    info = {"crop_box": list(box), "area_ratio": (r-l)*(b-t)/(width*height),
            "fallback_reason": reason, "candidate_coverage": coverage}
    return image.crop(box), info


def letterbox(image, size):
    width, height = scaled_size(*image.size, size)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


OFFLINE_VERSION = "fov_presized_letterbox_v2"


def prepare_offline_image(image, data, split):
    """Deterministic uint8 RGB prefix; training intentionally resizes before augmentation."""
    image, info = fov_crop(image, data["preprocessing"])
    if split == "train":
        image = image.resize(scaled_size(*image.size, data["image_size"]), Image.Resampling.BILINEAR)
    elif split in ("valid", "test"):
        image = letterbox(image, data["image_size"])
    else:
        raise ValueError(f"Unknown cache split: {split}")
    return image, info


class FOVLetterbox:
    def __init__(self, data, training=False):
        self.data, self.training = data, training
        self.normalize = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"]),
        ])

    @staticmethod
    def uniform(low, high):
        return float(torch.empty(1).uniform_(low, high).item())

    def augment(self, image):
        if self.training:
            aug = self.data["augmentation"]
            if torch.rand(1).item() < aug["horizontal_flip"]:
                image = ImageOps.mirror(image)
            angle = self.uniform(-aug["rotation_degrees"], aug["rotation_degrees"])
            image = image.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=(0, 0, 0))
            image = ImageEnhance.Brightness(image).enhance(self.uniform(*aug["brightness"]))
            image = ImageEnhance.Contrast(image).enhance(self.uniform(*aug["contrast"]))
        return image

    def __call__(self, image):
        image, _ = fov_crop(image, self.data["preprocessing"])
        image = self.augment(image)
        return self.normalize(letterbox(image, self.data["image_size"]))


class OfflineFOVLetterbox(FOVLetterbox):
    """Versioned resize-before-augmentation recipe, with explicit raw fallback."""

    def from_cache(self, image, split):
        size = self.data["image_size"]
        if image.mode != "RGB" or (max(image.size) != size if split == "train" else image.size != (size, size)):
            raise ValueError(f"Invalid {split} cache image: {image.mode}/{image.size}")
        if self.training and split != "train":
            raise ValueError("Only official train images may receive training augmentation.")
        if split == "train":
            return self.normalize(letterbox(self.augment(image), size))
        if split not in ("valid", "test"):
            raise ValueError(f"Unknown cache split: {split}")
        return self.normalize(image)

    def __call__(self, image):
        split = "train" if self.training else "valid"
        prepared, _ = prepare_offline_image(image, self.data, split)
        return self.from_cache(prepared, split)


def build_transforms(data: dict, training: bool = False):
    version = data.get("preprocessing", {}).get("version", "legacy_resize_v1")
    if version == OFFLINE_VERSION:
        return OfflineFOVLetterbox(data, training)
    if version == "fov_letterbox_v1":
        return FOVLetterbox(data, training)
    if version != "legacy_resize_v1":
        raise ValueError(f"Unsupported preprocessing version: {version}")
    size = int(data["image_size"])
    operations = [transforms.Resize((size, size), antialias=True)]
    if training:
        augmentation = data.get("augmentation", {})
        operations.extend([
            transforms.RandomHorizontalFlip(float(augmentation.get("horizontal_flip", 0))),
            transforms.RandomRotation(float(augmentation.get("rotation_degrees", 0))),
        ])
    operations.extend([transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"])])
    return transforms.Compose(operations)

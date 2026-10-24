"""Exact two-stage timm encoder and a fully trainable single-logit head."""

import importlib.metadata
import hashlib
from pathlib import Path

from eye_qing_guang.contracts.convnext_config import SUPPORTED_MODEL_NAMES
from eye_qing_guang.models.image_classifier import ImageClassifier
from eye_qing_guang.utils.paths import resolve_path


def _timm(settings):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("The timm experiment needs the explicitly pinned timm package; no model fallback is allowed.") from exc
    version = str(importlib.metadata.version("timm"))
    if version != settings["timm_version"]:
        raise RuntimeError(f"timm version mismatch: installed={version}, frozen={settings['timm_version']}")
    return timm, version


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_pretrained_file(settings, root):
    """Resolve only existing local files; this function never requests a download."""
    if settings["pretrained_file"] is not None:
        path = resolve_path(settings["pretrained_file"], root)
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise RuntimeError("Local HF cache lookup needs huggingface_hub; alternatively specify the prepared exact pretrained_file. No download fallback is allowed.") from exc

        model_name = settings["name"]
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported ConvNeXt model identity: {model_name}")
        repo = f"timm/{model_name}"
        path = None
        for filename in ("model.safetensors", "pytorch_model.bin"):
            cached = try_to_load_from_cache(repo, filename, revision=settings["pretrained_revision"])
            if isinstance(cached, str) and Path(cached).is_file():
                path = Path(cached)
                break
        if path is None:
            raise FileNotFoundError("Two-stage pretrained weights are not prepared locally. Prepare the exact timm weights separately; automatic downloading is disabled.")
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty pretrained file: {path}")
    return path


def build_convnext_model(settings, root, *, initialize_pretrained=True):
    model_name = settings.get("name")
    if (settings["library"] != "timm" or model_name not in SUPPORTED_MODEL_NAMES
            or settings["use_age"] is not False or settings["dropout"] != 0.0
            or settings["drop_path_rate"] != 0.0):
        raise ValueError("Unexpected model identity.")
    timm, version = _timm(settings)
    cfg = timm.get_pretrained_cfg(model_name)
    cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    if (list(cfg.get("mean", ())) != [0.485, 0.456, 0.406] or list(cfg.get("std", ())) != [0.229, 0.224, 0.225]
            or cfg.get("num_classes") != 1000 or cfg.get("fixed_input_size") is not False
            or cfg.get("hf_hub_id") != f"timm/{model_name}"):
        raise ValueError("Installed timm pretrained metadata differs from the frozen two-stage recipe.")
    source = {"repo_id": f"timm/{model_name}", "requested_revision": settings["pretrained_revision"],
              "hf_hub_id": cfg.get("hf_hub_id"), "url": cfg.get("url"), "pretrained_cfg": cfg}
    options = {}
    if initialize_pretrained:
        path = prepared_pretrained_file(settings, root)
        expected_names = {f"{model_name}.safetensors", f"{model_name}.bin", f"{model_name}.pth"}
        if settings["pretrained_file"] is not None and path.name not in expected_names:
            raise ValueError(f"Prepared weight filename does not identify {model_name}: {path.name}")
        options["pretrained_cfg_overlay"] = {"file": str(path), "source": "file"}
        source.update(local_file=str(path), local_bytes=path.stat().st_size, sha256=_sha256(path))
        # A HF cache snapshot directory identifies the resolved immutable revision.
        source["cache_revision"] = next((p.name for p in path.parents if p.parent.name == "snapshots"), None)
    encoder = timm.create_model(model_name, pretrained=initialize_pretrained, num_classes=0,
                                global_pool="avg", drop_rate=0.0, drop_path_rate=0.0, **options)
    if encoder.num_features != 768:
        raise ValueError(f"Unexpected encoder.num_features: {encoder.num_features}")
    model = ImageClassifier(encoder, encoder.num_features, dropout=0.0)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    metadata = {"timm_version": version, "model_name": model_name, "num_features": 768,
                "num_classes": 0, "global_pool": "avg", "initialization": "pretrained" if initialize_pretrained else "checkpoint_reload",
                "source": source, "normalization": "layernorm", "preprocessing_difference": "512 RGB/FOV/BILINEAR/letterbox; no timm center crop"}
    return model, metadata

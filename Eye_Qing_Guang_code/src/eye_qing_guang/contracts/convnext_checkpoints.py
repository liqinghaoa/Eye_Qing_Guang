"""Format 3: only best/last inference weights, with safe strict reloading."""

import math
from pathlib import Path

import torch

from eye_qing_guang.contracts.convnext_config import SUPPORTED_MODEL_NAMES, validate_convnext_config
from eye_qing_guang.models.timm_convnext import build_convnext_model
from eye_qing_guang.utils.timm_artifacts import atomic_write


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Checkpoint metadata must be finite.")
        return value
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: plain(v) for k, v in value.items()}
    raise TypeError(f"Unsafe checkpoint metadata type: {type(value).__name__}")


class ConvNeXtCheckpointStore:
    """Only this instance may replace the two files it created in a fresh run."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Checkpoint directory must be empty: {self.directory}")
        self.created = set()

    def save(self, name, model, epoch, config, metrics, model_metadata):
        if name not in {"best.pt", "last.pt"}:
            raise ValueError("Only best.pt and last.pt are allowed.")
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Only completed positive epochs may be saved.")
        score = metrics.get("roc_auc")
        if score is None or not math.isfinite(score):
            raise ValueError("Invalid validation must never be saved as a completed checkpoint.")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        state = model.state_dict()
        if any(k.startswith("module.") for k in state):
            raise ValueError("Save the unwrapped model, never DDP-prefixed parameters.")
        model_name = config["model"]["name"]
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported ConvNeXt checkpoint identity: {model_name}")
        if model_metadata.get("model_name", model_name) != model_name:
            raise ValueError("Checkpoint model metadata does not match the frozen configuration.")
        metadata = dict(model_metadata)
        metadata["model_name"] = model_name
        metadata["num_features"] = 768
        source = dict(metadata.get("source") or {})
        source.setdefault("repo_id", f"timm/{model_name}")
        source.setdefault("requested_revision", config["model"]["pretrained_revision"])
        # Real model construction always supplies the SHA256.  The nullable
        # field keeps synthetic contract tests independent of local weights.
        source.setdefault("sha256", None)
        metadata["source"] = source
        payload = plain({"format_version": 3, "checkpoint_kind": "inference_weight", "epoch": epoch,
                         "config": config, "age_stats": None, "pos_weight": 5.0, "model_state": dict(state),
                         "metrics": metrics, "timm_version": str(metadata["timm_version"]),
                         "model_name": model_name, "num_features": 768, "model_metadata": metadata})
        destination = self.directory / name
        atomic_write(destination, lambda handle: torch.save(payload, handle), replace=name in self.created)
        self.created.add(name)


def load_convnext_checkpoint(path, device, root):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 3
            or payload.get("checkpoint_kind") != "inference_weight"
            or payload.get("model_name") not in SUPPORTED_MODEL_NAMES or payload.get("num_features") != 768
            or payload.get("pos_weight") != 5.0 or payload.get("age_stats") is not None
            or any(k in payload for k in ("optimizer_state", "scheduler", "scaler", "rng", "training_state"))):
        raise ValueError("Invalid format-3 timm inference checkpoint; resume is unsupported.")
    validate_convnext_config(payload["config"])
    if payload["config"]["model"]["name"] != payload["model_name"]:
        raise ValueError("Checkpoint model identity does not match its frozen configuration.")
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("model_name") != payload["model_name"] \
            or metadata.get("num_features") != 768 or not isinstance(metadata.get("source"), dict):
        raise ValueError("Checkpoint is missing strict ConvNeXt model metadata.")
    source = metadata["source"]
    if (source.get("repo_id") != f"timm/{payload['model_name']}"
            or source.get("requested_revision") != payload["config"]["model"]["pretrained_revision"]
            or "sha256" not in source):
        raise ValueError("Checkpoint pretrained source does not match its model identity.")
    if payload.get("timm_version") != payload["config"]["model"]["timm_version"]:
        raise ValueError("Checkpoint timm version does not match its frozen config.")
    if type(payload.get("epoch")) is not int or payload["epoch"] < 1:
        raise ValueError("Invalid checkpoint epoch.")
    model, _ = build_convnext_model(payload["config"]["model"], root, initialize_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload

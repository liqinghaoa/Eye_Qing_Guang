"""Format-3 final-fit checkpoints with no validation/best semantics."""

import math
from pathlib import Path
import torch

from eye_qing_guang.contracts.convnext_config import SUPPORTED_MODEL_NAMES
from eye_qing_guang.contracts.convnext_final_config import validate_convnext_final_config
from eye_qing_guang.contracts.convnext_checkpoints import plain
from eye_qing_guang.models.timm_convnext import build_convnext_model
from eye_qing_guang.utils.timm_artifacts import atomic_write


class FinalCheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Checkpoint directory must be empty: {self.directory}")
        self.created = set()

    def save(self, name, model, epoch, config, metrics, model_metadata):
        if name not in {"final.pt", "last.pt"}:
            raise ValueError("Final-fit checkpoints may only be final.pt and last.pt.")
        if type(epoch) is not int or epoch < 1:
            raise ValueError("Invalid final-fit checkpoint epoch.")
        model_name = config["model"]["name"]
        if model_name not in SUPPORTED_MODEL_NAMES or model_metadata.get("model_name") != model_name:
            raise ValueError("Final-fit checkpoint model identity mismatch.")
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        metadata = dict(model_metadata)
        source = dict(metadata.get("source") or {})
        source.setdefault("repo_id", f"timm/{model_name}")
        source.setdefault("requested_revision", config["model"]["pretrained_revision"])
        source.setdefault("sha256", None)
        metadata.update(model_name=model_name, num_features=768, source=source)
        payload = plain({"format_version": 3, "checkpoint_kind": "inference_weight",
                         "checkpoint_variant": "final_fit", "epoch": epoch,
                         "config": config, "age_stats": None, "pos_weight": 5.0,
                         "model_state": dict(model.state_dict()), "metrics": metrics,
                         "timm_version": str(metadata["timm_version"]),
                         "model_name": model_name, "num_features": 768,
                         "model_metadata": metadata})
        atomic_write(self.directory / name, lambda handle: torch.save(payload, handle),
                     replace=name in self.created)
        self.created.add(name)


def load_final_checkpoint(path, device, root):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != 3
            or payload.get("checkpoint_kind") != "inference_weight"
            or payload.get("checkpoint_variant") != "final_fit"
            or payload.get("model_name") not in SUPPORTED_MODEL_NAMES
            or payload.get("num_features") != 768 or payload.get("pos_weight") != 5.0
            or payload.get("age_stats") is not None
            or any(k in payload for k in ("optimizer_state", "scheduler", "scaler", "rng", "training_state"))):
        raise ValueError("Invalid final-fit checkpoint; resume and best-selection state are unsupported.")
    validate_convnext_final_config(payload["config"])
    if payload["config"]["model"]["name"] != payload["model_name"]:
        raise ValueError("Final-fit checkpoint model identity does not match its config.")
    metadata = payload.get("model_metadata")
    source = metadata.get("source") if isinstance(metadata, dict) else None
    if (not isinstance(metadata, dict) or metadata.get("model_name") != payload["model_name"]
            or metadata.get("num_features") != 768 or not isinstance(source, dict)
            or source.get("repo_id") != f"timm/{payload['model_name']}"
            or source.get("requested_revision") != payload["config"]["model"]["pretrained_revision"]
            or "sha256" not in source):
        raise ValueError("Final-fit checkpoint pretrained source metadata is invalid.")
    if payload.get("timm_version") != payload["config"]["model"]["timm_version"]:
        raise ValueError("Final-fit checkpoint timm version mismatch.")
    if payload.get("epoch") != payload["config"]["training"]["final_epochs"]:
        raise ValueError("Final-fit checkpoint is not the frozen final epoch.")
    model, _ = build_convnext_model(payload["config"]["model"], root, initialize_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload

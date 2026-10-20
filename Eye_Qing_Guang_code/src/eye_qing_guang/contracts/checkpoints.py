"""Versioned checkpoints with complete inference preprocessing metadata."""

from pathlib import Path

import torch

from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.models.factory import build_model


def save_checkpoint(path: Path, model, optimizer, epoch, config, age_stats, pos_weight, metrics,
                    training_state=None):
    payload = {
        "format_version": 2 if training_state is not None else 1,
        "epoch": int(epoch),
        "config": config,
        "age_stats": age_stats,
        "pos_weight": float(pos_weight),
        "model_state": (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "training_state": training_state,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path: Path, device):
    # Only load tensor/primitive payloads; do not unpickle arbitrary model objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported checkpoint format.")
    validate_config(payload["config"])
    if payload["config"]["model"]["use_age"] and payload.get("age_stats") is None:
        raise ValueError("Age-fusion checkpoint is missing training age statistics.")
    model = build_model(payload["config"]["model"], initialize_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device), payload

"""Format 4 inference-only student weights; no teacher file access on reload."""

import math
from pathlib import Path

import torch

from eye_qing_guang.contracts.convnext_checkpoints import plain
from eye_qing_guang.contracts.convnext_kd_config import MODEL_NAME, validate_kd_config, validate_teacher_provenance
from eye_qing_guang.models.timm_convnext import build_convnext_model
from eye_qing_guang.utils.timm_artifacts import atomic_write

PAYLOAD_FIELDS = {
    "format_version", "checkpoint_kind", "model_name", "num_features", "timm_version",
    "pos_weight", "age_stats", "epoch", "config", "distillation_variant", "model_state",
    "metrics", "model_metadata", "teacher_provenance",
}


def validate_student_payload(payload):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise ValueError("Invalid KD student checkpoint fields; only inference state is allowed.")
    config = validate_kd_config(payload["config"])
    if (payload["format_version"] != 4 or payload["checkpoint_kind"] != "kd_student_inference_weight"
            or payload["model_name"] != MODEL_NAME or payload["num_features"] != 768
            or payload["timm_version"] != config["model"]["timm_version"] or payload["pos_weight"] != 5.0
            or payload["age_stats"] is not None
            or payload["distillation_variant"] != config["distillation"]["variant"]
            or type(payload["epoch"]) is not int or not 1 <= payload["epoch"] <= config["training"]["epochs"]):
        raise ValueError("KD student identity, variant or completed epoch differs from its frozen config.")
    metrics = payload["metrics"]
    if (not isinstance(metrics, dict) or type(metrics.get("roc_auc")) not in (int, float)
            or not math.isfinite(metrics["roc_auc"]) or not 0 <= metrics["roc_auc"] <= 1):
        raise ValueError("A finite student validation AUROC is required.")
    metadata = payload["model_metadata"]
    if (not isinstance(metadata, dict) or metadata.get("model_name") != MODEL_NAME
            or metadata.get("num_features") != 768 or metadata.get("timm_version") != "1.0.22"
            or metadata.get("initialization") != "pretrained"
            or metadata.get("num_classes") != 0 or metadata.get("global_pool") != "avg"
            or metadata.get("normalization") != "layernorm"
            or metadata.get("preprocessing_difference") != "512 RGB/FOV/BILINEAR/letterbox; no timm center crop"):
        raise ValueError("Invalid student model/initialization metadata.")
    source = metadata.get("source", {})
    sha = source.get("sha256") if isinstance(source, dict) else None
    if (not isinstance(source, dict) or source.get("repo_id") != f"timm/{MODEL_NAME}"
            or source.get("requested_revision") != config["model"]["pretrained_revision"]
            or not isinstance(source.get("local_file"), str) or not source["local_file"]
            or not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("The student must record its prepared ImageNet pretrained source.")
    validate_teacher_provenance(payload["teacher_provenance"])
    state = payload["model_state"]
    if (not isinstance(state, dict) or not state
            or any(not isinstance(name, str) or name.startswith("module.") or not isinstance(value, torch.Tensor)
                   for name, value in state.items())):
        raise ValueError("Only the unwrapped student state_dict is allowed.")
    if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Non-finite student checkpoint state.")
    return payload


class KDCheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"KD checkpoint directory must be empty: {directory}")
        self.created = set()

    def save(self, name, student, epoch, config, metrics, metadata, teacher_provenance):
        if name not in {"best.pt", "last.pt"}:
            raise ValueError("KD stores only student best.pt and last.pt.")
        payload = plain({
            "format_version": 4, "checkpoint_kind": "kd_student_inference_weight",
            "model_name": MODEL_NAME, "num_features": 768, "timm_version": config["model"]["timm_version"],
            "pos_weight": 5.0, "age_stats": None, "epoch": epoch, "config": config,
            "distillation_variant": config["distillation"]["variant"], "model_state": dict(student.state_dict()),
            "metrics": metrics, "model_metadata": metadata, "teacher_provenance": teacher_provenance,
        })
        validate_student_payload(payload)
        atomic_write(self.directory / name, lambda handle: torch.save(payload, handle), replace=name in self.created)
        self.created.add(name)


def load_kd_student_checkpoint(path, device, root):
    payload = validate_student_payload(torch.load(path, map_location="cpu", weights_only=True))
    # This builds only the student architecture. Neither pretrained nor teacher files are loaded.
    student, _ = build_convnext_model(payload["config"]["model"], root, initialize_pretrained=False,
                                      image_size=payload["config"]["data"]["image_size"])
    student.load_state_dict(payload["model_state"], strict=True)
    return student.to(device).float().eval(), payload

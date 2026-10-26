"""Frozen v1 recipe for standard and rectified three-teacher distillation."""

import copy
from pathlib import Path

import yaml

from eye_qing_guang.contracts.convnext_config import PROTOCOL as MANIFEST_PROTOCOL, validate_convnext_config

PROTOCOL = "timm_convnext_kd_official_v1"
MODEL_NAME = "convnext_tiny.fb_in1k"
EXPERIMENT_NAMES = {
    "standard": "timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1",
    "rectified": "timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_rect_official_single3090_s42_v1",
}
IMAGENET_MEAN, IMAGENET_STD = [.485, .456, .406], [.229, .224, .225]
TEACHERS = [
    {"id": "T1", "model_name": MODEL_NAME, "loader": "convnext_format3",
     "checkpoint": "experiments/timm_convnext_tiny_fb_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt",
     "expected_epoch": 7, "weight": 1 / 3},
    {"id": "T2", "model_name": "convnext_tiny.fb_in22k_ft_in1k", "loader": "convnext_format3",
     "checkpoint": "experiments/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt",
     "expected_epoch": 9, "weight": 1 / 3},
    {"id": "T3", "model_name": "tf_efficientnetv2_s.in21k_ft_in1k", "loader": "timm_v2s_format3",
     "checkpoint": "experiments/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt",
     "expected_epoch": 7, "weight": 1 / 3},
]
FROZEN_TRAINING = {
    "optimizer": "adamw", "optimizer_parameter_groups": "all_trainable",
    "epochs": 100, "batch_size": 48, "eval_batch_size": 16, "accumulation_steps": 16,
    "learning_rate": .0003, "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8,
    "loss": "bce", "pos_weight": 5.0, "auto_pos_weight": False, "device": "cuda:0",
    "precision": "bf16", "amp": True, "grad_clip": 5.0, "max_consecutive_skips": None,
    "scheduler": "warmup_cosine_success_v1", "schedule_total_epochs": 100,
    "warmup_epochs": 5, "warmup_start_lr": .0001, "min_lr": 1e-6,
    "monitor": "roc_auc", "early_stopping_enabled": True,
    "min_epochs": 30, "min_delta": .0001, "patience": 15,
}
FROZEN_DISTRIBUTED = {
    "enabled": False, "world_size": 1, "gradient_transport": "none", "cpus_per_rank": 6,
    "backend": "none", "timeout_seconds": 900, "multiprocessing_context": "spawn",
    "log_every_n_groups": 10,
}
FROZEN_PREFLIGHT = {
    "max_sample_instances": 64, "validation_samples": 8, "gradient_relative_l2_max": .1,
    "gradient_max_abs": .1, "reload_atol": 1e-6, "reload_rtol": 1e-5,
}
DISTILLATION_FIELDS = {
    "variant", "fusion", "temperature", "alpha", "warmup_epochs", "loss", "correction",
    "epsilon", "teacher_precision", "teacher_forward_chunk", "shared_augmented_view", "teachers",
}


def teacher_identity(index):
    spec = TEACHERS[index]
    return {
        "id": spec["id"], "model_name": spec["model_name"], "checkpoint": spec["checkpoint"],
        "epoch": spec["expected_epoch"], "mean": IMAGENET_MEAN if index < 2 else [.5] * 3,
        "std": IMAGENET_STD if index < 2 else [.5] * 3, "num_features": 768 if index < 2 else 1280,
        "timm_version": "1.0.22", "image_size": 512, "pos_weight": 5.0,
        "protocol": MANIFEST_PROTOCOL if index < 2 else "timm_official_trainvalid_v1",
    }


def validate_teacher_provenance(provenance):
    """Validate metadata only; student inference must never access teacher files."""
    if not isinstance(provenance, list) or len(provenance) != 3:
        raise ValueError("Exactly three ordered teacher provenance records are required.")
    for index, item in enumerate(provenance):
        expected = teacher_identity(index)
        if (not isinstance(item, dict) or set(item) != set(expected) | {"size_bytes", "pretrained_source"}
                or any(item.get(key) != value for key, value in expected.items())
                or type(item["size_bytes"]) is not int or item["size_bytes"] <= 0
                or not isinstance(item["pretrained_source"], dict)
                or item["pretrained_source"].get("repo_id") != f"timm/{expected['model_name']}"):
            raise ValueError(f"Invalid teacher provenance for {TEACHERS[index]['id']}.")
    return provenance


def validate_kd_config(config):
    if not isinstance(config, dict) or "distillation" not in config:
        raise ValueError("A complete independent KD configuration is required.")
    kd = config["distillation"]
    if not isinstance(kd, dict) or set(kd) != DISTILLATION_FIELDS:
        raise ValueError("KD distillation fields must match the complete v1 recipe.")
    variant = kd["variant"]
    if variant not in EXPERIMENT_NAMES:
        raise ValueError("KD variant must be standard or rectified.")
    fixed = {
        "fusion": "probability_mean", "temperature": 1.0, "alpha": 1.0, "warmup_epochs": 0,
        "loss": "soft_bce", "correction": "none" if variant == "standard" else "arm_binary_v1",
        "epsilon": 1e-6, "teacher_precision": "fp32", "teacher_forward_chunk": 8,
        "shared_augmented_view": True,
    }
    if any(kd[key] != value or type(kd[key]) is not type(value) for key, value in fixed.items()):
        raise ValueError("KD v1 freezes fusion, loss, correction, precision, temperature, alpha and chunk size.")
    if kd["teachers"] != TEACHERS or not isinstance(kd["teachers"], list):
        raise ValueError("KD v1 requires the three exact ordered best checkpoints and equal weights.")
    if abs(sum(item["weight"] for item in kd["teachers"]) - 1) > 1e-12:
        raise ValueError("Teacher weights must sum to one.")
    base = copy.deepcopy(config)
    base.pop("distillation")
    if base["experiment"]["protocol"] != PROTOCOL or base["experiment"]["name"] != EXPERIMENT_NAMES[variant]:
        raise ValueError("KD protocol and fixed experiment name must match the variant.")
    base["experiment"]["protocol"] = MANIFEST_PROTOCOL
    validate_convnext_config(base)
    if (config["model"]["name"] != MODEL_NAME or config["model"]["timm_version"] != "1.0.22"
            or not config["model"]["pretrained_file"] or config["data"]["image_size"] != 512
            or config["data"]["num_workers"] != 6 or config["data"].get("cache", {}).get("enabled") is not True
            or config["data"]["preprocessing"]["version"] != "fov_presized_letterbox_v2"):
        raise ValueError("KD v1 requires the prepared ImageNet-1K student and the 512 offline-cache recipe.")
    if (config["training"] != FROZEN_TRAINING or config["distributed"] != FROZEN_DISTRIBUTED
            or config["preflight"] != FROZEN_PREFLIGHT):
        raise ValueError("KD v1 training and single-device resource settings are frozen by the plan.")
    if (config["recipe"]["implementation"] != "convnext_kd_single_device_v1"
            or config["recipe"]["implementation_path"] != "src/eye_qing_guang/trainers/convnext_kd.py"):
        raise ValueError("KD v1 must name its independent training implementation.")
    return config


def load_kd_config(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_kd_config(yaml.safe_load(handle))

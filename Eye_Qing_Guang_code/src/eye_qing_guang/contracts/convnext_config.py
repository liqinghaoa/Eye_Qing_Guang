"""Independent, fail-closed configuration for the official ConvNeXt experiment."""

import math
from pathlib import Path

import yaml

PROTOCOL = "timm_convnext_official_trainvalid_v1"
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
EXPECTED_COUNTS = {"train": 51396, "valid": 5849, "test": 5855}
PREPROCESSING = {"version": "fov_letterbox_v1", "preview_size": 512, "threshold": 8,
                 "min_line_fraction": 0.01, "margin": 0.02, "min_area": 0.5, "min_coverage": 0.995}
AUGMENTATION = {"horizontal_flip": 0.5, "rotation_degrees": 10,
                "brightness": [0.9, 1.1], "contrast": [0.9, 1.1]}
FIELDS = {
    "experiment": "name protocol seed output_root",
    "recipe": "implementation implementation_path parameter_notes",
    "data": "raw_root manifest_dir image_column label_column age_column splits image_size num_workers mean std preprocessing augmentation",
    "model": "library name pretrained timm_version pretrained_file pretrained_revision dropout drop_path_rate batchnorm use_age",
    "training": "optimizer optimizer_parameter_groups epochs batch_size eval_batch_size accumulation_steps learning_rate weight_decay betas eps loss pos_weight auto_pos_weight device precision amp grad_clip max_consecutive_skips scheduler schedule_total_epochs warmup_epochs warmup_start_lr min_lr monitor early_stopping_enabled min_epochs min_delta patience",
    "distributed": "enabled world_size gradient_transport cpus_per_rank backend timeout_seconds multiprocessing_context log_every_n_groups",
    "evaluation": "threshold precision expected_valid_count duplicate_logit_atol",
    "preflight": "max_sample_instances validation_samples gradient_relative_l2_max gradient_max_abs reload_atol reload_rtol",
    "submission": "mode id_column value_column label_column",
}


class UnresolvedRecipeError(ValueError):
    pass


def _exact(mapping, keys, where):
    expected = set(keys.split())
    if not isinstance(mapping, dict) or set(mapping) != expected:
        actual = set(mapping) if isinstance(mapping, dict) else set()
        raise ValueError(f"{where}: missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _number(value, name, positive=True):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"Invalid {name}: {value!r}")


def _safe_name(value, name):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(c in value for c in '/\\:<>"|?*')):
        raise ValueError(f"{name} must be one explicit safe directory name.")


def unresolved_recipe(config):
    train, distributed = config["training"], config["distributed"]
    missing = [f"training.{k}" for k in (
        "optimizer", "optimizer_parameter_groups", "eval_batch_size", "betas", "eps", "device",
        "scheduler", "schedule_total_epochs", "warmup_start_lr", "early_stopping_enabled") if train[k] is None]
    missing += [f"distributed.{k}" for k in ("backend", "timeout_seconds", "multiprocessing_context") if distributed[k] is None]
    if train["early_stopping_enabled"] is True:
        missing += [f"training.{k}" for k in ("min_epochs", "min_delta", "patience") if train[k] is None]
    return missing


def validate_convnext_config(config, *, require_resolved=True):
    _exact(config, " ".join(FIELDS), "config")
    for section, keys in FIELDS.items():
        _exact(config[section], keys, section)
    exp, data, model, train, distributed = (config[k] for k in ("experiment", "data", "model", "training", "distributed"))
    _exact(config["recipe"], FIELDS["recipe"], "recipe")
    _safe_name(exp["name"], "experiment.name")
    if exp["protocol"] != PROTOCOL or exp["seed"] != 42:
        raise ValueError("This protocol requires timm_convnext_official_trainvalid_v1 and seed 42.")
    for key in ("output_root",):
        if not isinstance(exp[key], str) or not exp[key]:
            raise ValueError(f"experiment.{key} is required.")
    if (model["library"] != "timm" or model["name"] != MODEL_NAME or model["pretrained"] is not True
            or model["use_age"] is not False or model["dropout"] != 0.0 or model["drop_path_rate"] != 0.0
            or model["batchnorm"] != "not_applicable" or not isinstance(model["timm_version"], str)):
        raise ValueError("Invalid timm model identity; no backbone aliases or alternate initialization are allowed.")
    if model["pretrained_file"] is not None and not isinstance(model["pretrained_file"], str):
        raise ValueError("model.pretrained_file must be a prepared local file or null for local HF cache lookup.")
    if not isinstance(model["pretrained_revision"], str) or not model["pretrained_revision"]:
        raise ValueError("model.pretrained_revision must be explicit.")
    if (data["image_size"] != 512 or data["mean"] != [0.485, 0.456, 0.406] or data["std"] != [0.229, 0.224, 0.225]
            or data["preprocessing"] != PREPROCESSING or data["augmentation"] != AUGMENTATION):
        raise ValueError("The complete fixed ConvNeXt RGB/FOV/BILINEAR/letterbox-512 recipe is required.")
    for key in ("raw_root", "manifest_dir", "image_column", "label_column", "age_column"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key} is required.")
    if type(data["num_workers"]) is not int or data["num_workers"] < 0:
        raise ValueError("data.num_workers must be nonnegative.")
    _exact(data["splits"], "train valid test", "data.splits")
    for split in EXPECTED_COUNTS:
        if data["splits"][split] != {"csv": f"{split}.csv", "images": split, "labeled": split != "test"}:
            raise ValueError("Only the official train/valid/test sources are allowed.")
    # The model/data protocol is frozen, but resource-facing training knobs are
    # intentionally configurable.  In particular, batch size, accumulation,
    # evaluation batch, workers, epochs, learning rate and precision may be
    # changed for a different server.  Keep only checks required by the
    # implementation so a custom throughput configuration fails clearly.
    for key in ("epochs", "batch_size", "eval_batch_size", "accumulation_steps",
                "schedule_total_epochs", "warmup_epochs"):
        if train[key] is not None:
            _positive_int(train[key], f"training.{key}")
    for key in ("min_epochs", "patience"):
        if train[key] is not None:
            _positive_int(train[key], f"training.{key}")
    for key in ("learning_rate", "weight_decay", "eps", "warmup_start_lr", "min_lr"):
        _number(train[key], f"training.{key}")
    if (train["loss"] != "bce" or train["pos_weight"] != 5.0
            or train["auto_pos_weight"] is not False or train["monitor"] != "roc_auc"):
        raise ValueError("This implementation currently fixes BCE with positive-class weight 5 and AUROC selection.")
    if train["precision"] not in ("fp32", "bf16") or type(train["amp"]) is not bool:
        raise ValueError("training.precision must be fp32 or bf16 and training.amp must be boolean.")
    if train["precision"] == "fp32" and train["amp"]:
        raise ValueError("FP32 training must set training.amp=false.")
    if train["precision"] == "bf16" and not train["amp"]:
        raise ValueError("BF16 training must set training.amp=true.")
    if (distributed["enabled"] is not True or distributed["world_size"] != 4
            or distributed["gradient_transport"] != "native_ddp" or distributed["cpus_per_rank"] != 6):
        raise ValueError("The official run uses four ranks and the project's existing native DDP backend.")
    if (config["evaluation"] != {"threshold": 0.5, "precision": "fp32", "expected_valid_count": 5849,
                                  "duplicate_logit_atol": 1e-6}
            or config["submission"] != {"mode": "competition", "id_column": "Image_name",
                                        "value_column": "Probability", "label_column": "Pred_Label"}):
        raise ValueError("Evaluation and submission must use the fixed protocol fields.")
    missing = unresolved_recipe(config)
    if train["grad_clip"] is not None and train["grad_clip"] is not False:
        _number(train["grad_clip"], "training.grad_clip")
    elif train["grad_clip"] is None:
        missing.append("training.grad_clip")
    flight = config["preflight"]
    _positive_int(flight["max_sample_instances"], "preflight.max_sample_instances")
    if flight["max_sample_instances"] > 64 or flight["validation_samples"] != 8:
        raise ValueError("Default preflight is limited to 64 global instances and eight validation records; larger budgets require an explicit CLI argument.")
    for key in ("gradient_relative_l2_max", "gradient_max_abs", "reload_atol", "reload_rtol"):
        _number(flight[key], f"preflight.{key}")
    # null is unresolved; false explicitly disables clipping for this chosen recipe.
    if train["max_consecutive_skips"] is not None:
        raise ValueError("BF16 has no scaler overflow/retry budget; max_consecutive_skips must be null.")
    if train["early_stopping_enabled"] is not None and type(train["early_stopping_enabled"]) is not bool:
        raise ValueError("early_stopping_enabled must be a boolean.")
    if train["early_stopping_enabled"] is False and any(train[k] is not None for k in ("min_epochs", "min_delta", "patience")):
        raise ValueError("Disabled early stopping requires null min_epochs/min_delta/patience.")
    if require_resolved and missing:
        raise UnresolvedRecipeError("Experiment recipe has unresolved required settings: " + "; ".join(missing))
    if missing:
        return config
    if train["optimizer"] != "adamw" or train["optimizer_parameter_groups"] != "all_trainable":
        raise ValueError("This implementation fixes AdamW over all trainable parameters.")
    if train["scheduler"] != "warmup_cosine_success_v1":
        raise ValueError("This implementation fixes the successful-update warmup/cosine schedule.")
    engine = config["recipe"]
    if any(not isinstance(engine[k], str) or not engine[k] for k in FIELDS["recipe"].split()):
        raise ValueError("The selected implementation and its parameter choices must be explicit.")
    for key in ("eval_batch_size", "schedule_total_epochs"):
        _positive_int(train[key], f"training.{key}")
    if train["schedule_total_epochs"] < train["epochs"]:
        raise ValueError("Schedule must cover all training epochs.")
    if not 0 < train["min_lr"] <= train["warmup_start_lr"] < train["learning_rate"]:
        raise ValueError("Invalid explicit learning-rate endpoints.")
    if (not isinstance(train["betas"], list) or len(train["betas"]) != 2
            or any(type(b) not in (int, float) or not 0 <= b < 1 for b in train["betas"])):
        raise ValueError("Explicit AdamW betas are required.")
    _number(train["eps"], "training.eps")
    if not isinstance(train["device"], str) or not train["device"].startswith("cuda"):
        raise ValueError("The PPU recipe must explicitly use a CUDA-compatible device.")
    if train["early_stopping_enabled"]:
        _positive_int(train["min_epochs"], "min_epochs")
        _positive_int(train["patience"], "patience")
        _number(train["min_delta"], "min_delta", positive=False)
        if train["min_epochs"] > train["epochs"]:
            raise ValueError("min_epochs exceeds epochs.")
    if distributed["backend"] != "nccl":
        raise ValueError("The ConvNeXt PPU runtime requires the existing NCCL-compatible backend.")
    if distributed["multiprocessing_context"] not in ("spawn", "forkserver"):
        raise ValueError("Data workers require an explicit safe spawn/forkserver context.")
    for key in ("timeout_seconds",):
        _positive_int(distributed[key], f"distributed.{key}")
    _positive_int(distributed["log_every_n_groups"], "distributed.log_every_n_groups")
    return config


def load_convnext_config(path: Path, *, require_resolved=True):
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_convnext_config(config, require_resolved=require_resolved)

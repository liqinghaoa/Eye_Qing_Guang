"""Validate the supported baseline configuration before costly work."""

import math


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    for section in ("experiment", "data", "model", "training", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    data, train, model = config["data"], config["training"], config["model"]
    for split in ("train", "valid", "test"):
        spec = data["splits"][split]
        for key in ("csv", "images", "labeled"):
            if key not in spec:
                raise ValueError(f"data.splits.{split}.{key} is required.")
        if not isinstance(spec["labeled"], bool) or spec["labeled"] != (split != "test"):
            raise ValueError("Train/valid must be labeled and test must be unlabeled.")
    if int(data["image_size"]) < 32 or int(data.get("num_workers", 0)) < 0:
        raise ValueError("image_size must be >=32; num_workers must be >=0.")
    if len(data["mean"]) != 3 or len(data["std"]) != 3 or any(
        not math.isfinite(float(v)) for v in data["mean"] + data["std"]
    ) or min(data["std"]) <= 0:
        raise ValueError("RGB normalization requires finite means and positive stds.")
    if model["backbone"] not in {"resnet18", "resnet50", "efficientnet_b0", "tiny_cnn"}:
        raise ValueError("Unsupported model backbone.")
    if not 0 <= float(model.get("dropout", 0)) < 1:
        raise ValueError("dropout must be in [0, 1).")
    for key in ("pretrained", "use_age"):
        if not isinstance(model[key], bool):
            raise ValueError(f"model.{key} must be a boolean.")
    if model["backbone"] == "tiny_cnn" and model["pretrained"]:
        raise ValueError("tiny_cnn has no pretrained weights.")
    if train["loss"] not in {"bce", "focal"}:
        raise ValueError("training.loss must be bce or focal.")
    for key in ("epochs", "batch_size", "patience"):
        if int(train[key]) < 1:
            raise ValueError(f"training.{key} must be positive.")
    for key in ("learning_rate", "weight_decay", "focal_gamma"):
        value = float(train[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative.")
    if float(train["learning_rate"]) == 0:
        raise ValueError("learning_rate must be positive.")
    if train["monitor"] not in {"roc_auc", "accuracy", "f1", "balanced_accuracy"}:
        raise ValueError("Unsupported early-stopping metric.")
    if not 0 <= float(config["evaluation"]["threshold"]) <= 1:
        raise ValueError("evaluation.threshold must be in [0, 1].")
    protocol = config["experiment"].get("protocol")
    if protocol is not None:
        if protocol != "b0_dev_v1":
            raise ValueError(f"Unsupported experiment protocol: {protocol}")
        validate_b0(config)
    elif data.get("preprocessing", {}).get("version", "legacy_resize_v1") != "legacy_resize_v1":
        raise ValueError("New preprocessing requires an explicitly supported experiment protocol.")


def validate_b0(config):
    allowed = {
        "experiment": "name protocol seed output_root",
        "data": "raw_root manifest_dir image_column label_column age_column splits image_size num_workers mean std preprocessing augmentation",
        "development": "split_seed associations_csv",
        "model": "backbone pretrained dropout use_age",
        "training": "epochs batch_size eval_batch_size accumulation_steps learning_rate weight_decay betas eps loss auto_pos_weight focal_gamma device amp grad_clip max_consecutive_skips scheduler schedule_total_epochs warmup_epochs warmup_start_lr min_lr monitor min_epochs min_delta patience",
        "evaluation": "threshold subgroup_fit_samples",
    }
    if set(config) != set(allowed):
        raise ValueError("B0 requires exactly the documented configuration sections.")
    for section, fields in allowed.items():
        if not isinstance(config[section], dict):
            raise ValueError(f"B0 section must be a mapping: {section}")
        if set(config[section]) != set(fields.split()):
            raise ValueError(f"B0 missing/unknown fields in {section}: {set(config[section]) ^ set(fields.split())}")
    data, train, model = config["data"], config["training"], config["model"]
    if set(data["splits"]) != {"train", "valid", "test"} or any(
        set(spec) != {"csv", "images", "labeled"} for spec in data["splits"].values()
    ):
        raise ValueError("Unknown B0 split settings.")
    if config["development"]["associations_csv"] is not None and not isinstance(config["development"]["associations_csv"], str):
        raise ValueError("associations_csv must be a path string or null.")
    if (model != {"backbone": "resnet50", "pretrained": False, "dropout": 0.0, "use_age": False}
            or data["image_size"] != 512 or data["mean"] != [0.5]*3 or data["std"] != [0.5]*3
            or train["loss"] != "bce" or train["auto_pos_weight"] is not False
            or train["monitor"] != "roc_auc" or config["evaluation"]["threshold"] != 0.5):
        raise ValueError("B0 identity requires scratch ResNet50-512, image-only, unweighted BCE and AUROC.")
    if config["experiment"]["seed"] != 42 or config["development"]["split_seed"] != 20261018:
        raise ValueError("This first B0-dev configuration requires seed 42 and split seed 20261018.")
    prep = {"version": "fov_letterbox_v1", "preview_size": 512, "threshold": 8,
            "min_line_fraction": 0.01, "margin": 0.02, "min_area": 0.5, "min_coverage": 0.995}
    aug = {"horizontal_flip": 0.5, "rotation_degrees": 10,
           "brightness": [0.9, 1.1], "contrast": [0.9, 1.1]}
    if data["preprocessing"] != prep or data["augmentation"] != aug:
        raise ValueError("B0 v1 requires its versioned FOV, letterbox and ordered augmentation recipe.")
    if train["scheduler"] != "warmup_cosine_success_v1":
        raise ValueError("Unsupported scheduler.")
    for key in ("epochs", "batch_size", "eval_batch_size", "accumulation_steps", "min_epochs",
                "max_consecutive_skips", "schedule_total_epochs", "warmup_epochs", "patience"):
        if type(train[key]) is not int or train[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if not train["min_epochs"] <= train["epochs"] <= train["schedule_total_epochs"]:
        raise ValueError("Require min_epochs <= epochs <= schedule_total_epochs.")
    if not 2 <= train["warmup_epochs"] < train["schedule_total_epochs"]:
        raise ValueError("Require 2 <= warmup_epochs < schedule_total_epochs.")
    if not isinstance(train["amp"], bool) or not str(train["device"]).startswith("cuda"):
        raise ValueError("B0 requires explicit CUDA and a boolean AMP setting.")
    for key in ("grad_clip", "eps", "warmup_start_lr", "min_lr", "min_delta"):
        if isinstance(train[key], bool) or not math.isfinite(float(train[key])) or float(train[key]) <= 0:
            raise ValueError(f"Invalid B0 numeric field: {key}")
    if not 0 < train["min_lr"] <= train["warmup_start_lr"] < train["learning_rate"]:
        raise ValueError("Invalid learning-rate endpoints.")
    if len(train["betas"]) != 2 or not all(0 <= beta < 1 for beta in train["betas"]):
        raise ValueError("AdamW betas must be in [0,1).")
    if type(config["evaluation"]["subgroup_fit_samples"]) is not int or config["evaluation"]["subgroup_fit_samples"] < 1:
        raise ValueError("subgroup_fit_samples must be positive.")

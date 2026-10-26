"""Fail-closed contract for the fixed-epoch train+valid final fit."""

from pathlib import Path
import copy
import yaml

from eye_qing_guang.contracts.convnext_config import (
    PROTOCOL as OFFICIAL_PROTOCOL,
    validate_convnext_config,
)

PROTOCOL = "timm_convnext_final_fit_v1"
FINAL_EPOCHS = 7
MERGED_TRAIN_COUNT = 57245
TEST_COUNT = 5855


def validate_convnext_final_config(config, *, require_resolved=True):
    if not isinstance(config, dict) or config.get("experiment", {}).get("protocol") != PROTOCOL:
        raise ValueError(f"This configuration requires protocol={PROTOCOL}.")
    training = config.get("training", {})
    if training.get("final_epochs") != FINAL_EPOCHS:
        raise ValueError("Final-fit requires training.final_epochs=7.")
    if training.get("epochs") != FINAL_EPOCHS:
        raise ValueError("Final-fit requires training.epochs=7.")
    if training.get("early_stopping_enabled") is not False:
        raise ValueError("Final-fit must disable early stopping.")
    if any(training.get(key) is not None for key in ("min_epochs", "min_delta", "patience")):
        raise ValueError("Final-fit early-stopping fields must be null.")
    if training.get("schedule_total_epochs") != 100 or training.get("warmup_epochs") != 5:
        raise ValueError("Final-fit must retain the 100-epoch schedule and five-epoch warmup curve.")
    if training.get("pos_weight") != 5.0:
        raise ValueError("The existing final-fit protocol is fixed at pos_weight=5.")
    if config["evaluation"].get("expected_valid_count") != 5849:
        raise ValueError("The source official valid count must remain recorded as 5849.")
    base = copy.deepcopy(config)
    base["experiment"]["protocol"] = OFFICIAL_PROTOCOL
    base["training"].pop("final_epochs", None)
    validate_convnext_config(base, require_resolved=require_resolved)
    return config


def load_convnext_final_config(path: Path, *, require_resolved=True):
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_convnext_final_config(config, require_resolved=require_resolved)

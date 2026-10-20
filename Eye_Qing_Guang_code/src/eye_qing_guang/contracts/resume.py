"""Validated epoch-boundary DDP continuation; legacy RNG state is not reconstructed."""

import copy
import math
from pathlib import Path

import torch

from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.contracts.precision import training_precision
from eye_qing_guang.trainers.schedules import DevelopmentStopping


def load_resume_checkpoint(path, config):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 2:
        raise ValueError("Resume requires a format-2 training checkpoint, not an inference checkpoint.")
    for key in ("model_state", "optimizer_state", "training_state", "epoch", "config", "metrics"):
        if key not in payload or payload[key] is None:
            raise ValueError(f"Resume checkpoint is missing {key}.")
    source = payload["config"]
    validate_config(source)
    validate_config(config)
    if source["experiment"].get("protocol") != "b0_dev_ddp_v1" or config["experiment"].get("protocol") != "b0_dev_ddp_v1":
        raise ValueError("Epoch-boundary resume currently supports four-rank B0 only.")
    # Resource changes are explicit; data identity, objective and optimizer stay fixed.
    allowed = {"experiment": {"name", "output_root"}, "data": {"num_workers"},
               "training": {"batch_size", "eval_batch_size", "accumulation_steps", "epochs", "precision"},
               "distributed": {"timeout_seconds", "log_every_n_groups"}}
    changes = {}
    for section in set(source) | set(config):
        if section == "diagnostics":
            continue
        if not isinstance(source.get(section), dict) or not isinstance(config.get(section), dict):
            if source.get(section) != config.get(section):
                raise ValueError(f"Resume changes unsupported section {section}.")
            continue
        for key in set(source[section]) | set(config[section]):
            before, after = source[section].get(key), config[section].get(key)
            if before != after:
                if key not in allowed.get(section, set()):
                    raise ValueError(f"Resume must preserve {section}.{key}.")
                changes[f"{section}.{key}"] = {"source": before, "target": after}
    if training_precision(source["training"]) != training_precision(config["training"]):
        raise ValueError("Resume must preserve training precision.")
    epoch = payload["epoch"]
    if type(epoch) is not int or not 1 <= epoch < config["training"]["epochs"]:
        raise ValueError("Resume checkpoint must end a completed epoch before the requested final epoch.")
    state = payload["training_state"]
    if not isinstance(state, dict) or not {"scheduler", "scaler", "early_stopping", "progress", "distributed"} <= set(state):
        raise ValueError("Resume checkpoint lacks optimizer/schedule/early-stopping progress.")
    if state["distributed"]["world_size"] != config["distributed"]["world_size"]:
        raise ValueError("Resume must preserve world size.")
    stopping = DevelopmentStopping(**state["early_stopping"])
    if not stopping.best_epoch or not 1 <= stopping.best_epoch <= epoch or not math.isfinite(stopping.best_auc):
        raise ValueError("Invalid saved best-checkpoint state.")
    if epoch >= stopping.min_epochs and stopping.stale >= stopping.patience:
        raise ValueError("The source already satisfies early stopping; do not silently extend it.")
    for name, tensor in payload["model_state"].items():
        if not isinstance(tensor, torch.Tensor) or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"Invalid/non-finite checkpoint model tensor: {name}")
    for values in payload["optimizer_state"]["state"].values():
        for tensor in values.values():
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("Non-finite saved optimizer state.")
    metadata = {"checkpoint": str(path), "source_epoch": epoch, "start_epoch": epoch + 1,
                "exact_replay": False, "rng_policy": "legacy RNG/DataLoader states unavailable; reseeded continuation",
                "resource_changes": changes, "legacy_resume_supported_flag": state.get("resume_supported", False)}
    world = config["distributed"]["world_size"]
    metadata["batch_transition"] = {
        "source_global_batch": world * source["training"]["batch_size"] * source["training"]["accumulation_steps"],
        "target_global_batch": world * config["training"]["batch_size"] * config["training"]["accumulation_steps"],
        "policy": "preserve_optimizer_steps_and_next_lr; retime_remaining_successful_updates_if_group_count_changes"}
    return payload, metadata


def restore_training_state(payload, model, optimizer, schedule, scaler, fitting_count, world_size):
    state = payload["training_state"]
    progress = copy.deepcopy(state["progress"])
    groups = math.ceil(math.ceil((fitting_count // world_size) / payload["config"]["training"]["batch_size"]) /
                       payload["config"]["training"]["accumulation_steps"])
    if fitting_count % world_size or progress["consumed_samples"] != payload["epoch"] * fitting_count:
        raise ValueError("Resume is only supported at a complete frozen-data epoch boundary.")
    origin_keys = ("update_origin_epoch", "update_origin_attempts", "update_origin_successes")
    if any(key in progress for key in origin_keys) and not all(key in progress for key in origin_keys):
        raise ValueError("Incomplete batch-change update origin.")
    origin_epoch, origin_attempts, origin_successes = (progress.get(key, 0) for key in origin_keys)
    if (any(type(value) is not int or value < 0 for value in (origin_epoch, origin_attempts, origin_successes))
            or origin_epoch > payload["epoch"] or origin_successes > origin_attempts
            or (origin_epoch == 0 and (origin_attempts or origin_successes))):
        raise ValueError("Invalid batch-change update origin.")
    if progress["attempted_groups"] != origin_attempts + (payload["epoch"] - origin_epoch) * groups:
        raise ValueError("Saved update count differs from frozen epoch length.")
    saved_schedule = state["scheduler"]
    expected = payload["config"]["training"]
    total_epochs = expected["schedule_total_epochs"]
    target_groups, remainder = divmod(schedule.total, total_epochs)
    if remainder or target_groups < 1:
        raise ValueError("Invalid target scheduler group count.")
    schedule.load_state_dict(saved_schedule)
    continuation = saved_schedule.get("continuation")
    if ((continuation is None and origin_epoch != 0)
            or (continuation is not None and (origin_epoch == 0 or continuation["anchor"] != origin_successes))):
        raise ValueError("Saved scheduler continuation differs from the update origin.")
    if schedule.total != origin_successes + (total_epochs - origin_epoch) * groups:
        raise ValueError("Resume schedule mismatch: total/update origin")
    base = saved_schedule
    while base.get("continuation") is not None:
        base = base["continuation"]["previous"]
    base_groups, remainder = divmod(base["total"], total_epochs)
    if (remainder or base_groups < 1 or base["warmup"] != expected["warmup_epochs"] * base_groups
            or ("continuation" not in saved_schedule and base_groups != groups)):
        raise ValueError("Resume schedule mismatch: original warmup/epoch plan")
    updates = saved_schedule["successful_updates"]
    if (type(updates) is not int or updates <= 0 or updates < origin_successes or updates != progress["successful_updates"] or
            progress["skipped_groups"] < 0 or progress["attempted_groups"] != updates + progress["skipped_groups"] or
            progress["skipped_groups"] < origin_attempts - origin_successes or
            not 0 <= progress["successful_samples"] <= progress["consumed_samples"]):
        raise ValueError("Inconsistent saved successful-update/skip counters.")
    for key in ("consumed_samples", "successful_samples"):
        if type(progress[key]) is not int or progress[key] < 0 or progress[key] % world_size:
            raise ValueError(f"Saved global sample count cannot be restored per rank: {key}")
        progress[key] //= world_size
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if len(optimizer.param_groups) != 1:
        raise ValueError("B0 resume requires the original single AdamW parameter group.")
    group = optimizer.param_groups[0]
    if (group["weight_decay"] != expected["weight_decay"] or tuple(group["betas"]) != tuple(expected["betas"])
            or group["eps"] != expected["eps"] or len(optimizer.state) != len(group["params"])):
        raise ValueError("Saved AdamW state/hyperparameters differ from the checkpoint recipe.")
    for parameter, values in optimizer.state.items():
        if int(values["step"].item()) != updates:
            raise ValueError("Saved AdamW step differs from scheduler progress.")
        for key in ("exp_avg", "exp_avg_sq"):
            if values[key].shape != parameter.shape:
                raise ValueError("Saved AdamW moments do not match model shapes.")
        if bool((values["exp_avg_sq"] < 0).any()):
            raise ValueError("Saved AdamW second moments must be nonnegative.")
    if not optimizer.state:
        raise ValueError("Saved AdamW state is empty.")
    if not math.isclose(optimizer.param_groups[0]["lr"], schedule.value(updates), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("Saved optimizer LR differs from the successful-update schedule.")
    if target_groups != groups:
        schedule.retime_remaining((total_epochs - payload["epoch"]) * target_groups)
        progress.update(update_origin_epoch=payload["epoch"], update_origin_attempts=progress["attempted_groups"],
                        update_origin_successes=updates)
    scaler.load_state_dict(state["scaler"])
    return progress, DevelopmentStopping(**state["early_stopping"])

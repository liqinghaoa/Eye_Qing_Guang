"""Independent official-train/valid workflow and best-only submission pipeline."""

import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from eye_qing_guang.contracts.timm_checkpoints import CheckpointStore, load_timm_checkpoint
from eye_qing_guang.contracts.timm_config import load_timm_config
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.timm_official import ensure_official_manifests, ordered_predictions
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.metrics.timm_official import compute_official_metrics
from eye_qing_guang.models.timm_efficientnet import build_timm_model
from eye_qing_guang.trainers.timm_official import (
    SelectionState, build_optimizer_schedule, pw5_loss, train_epoch, update_group,
)
from eye_qing_guang.utils.experiment_io import read_csv, run_manifest, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import seed_everything
from eye_qing_guang.utils.timm_artifacts import atomic_csv, atomic_json, publish_submission
from eye_qing_guang.utils.timm_source import create_source_runtime

VALID_FIELDS = ["image_id", "label", "logit", "probability"]
TEST_FIELDS = ["image_id", "logit", "probability"]
STAGES = ("training", "best_validation", "test_inference", "submission")


def _loader(records, config, root, device, runtime=None, *, training=False):
    sharding = runtime if config["distributed"]["enabled"] else None
    return make_loader(records, config, root, age_stats=None, training=training, device=device, distributed=sharding)


def _new_directory(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for part in ("checkpoints", "reports", "predictions/best", "submission"):
        (directory / part).mkdir(parents=True)


def preflight_environment(config, device):
    result = {"torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "device_type": device.type, "timm_version": config["model"]["timm_version"]}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(device_name=properties.name, total_memory=properties.total_memory)
    return result


def compare_predictions(actual, expected, records, config):
    actual = ordered_predictions(actual, records, labeled=True)
    expected = ordered_predictions(expected, records, labeled=True)
    maximum = {}
    for field in ("logit", "probability"):
        left = np.asarray([float(r[field]) for r in actual])
        right = np.asarray([float(r[field]) for r in expected])
        maximum[field] = float(np.max(np.abs(left-right)))
        if not np.allclose(left, right, atol=config["preflight"]["reload_atol"], rtol=config["preflight"]["reload_rtol"]):
            raise ValueError(f"Best reload {field} differs from selected-epoch predictions: max_abs={maximum[field]}")
    actual_auc = compute_official_metrics(actual, 0.0)["roc_auc"]
    expected_auc = compute_official_metrics(expected, 0.0)["roc_auc"]
    if actual_auc != expected_auc:
        raise ValueError("Best reload changed selection AUROC.")
    return maximum


def distributed_validation(model, records, config, root, device, runtime):
    loader = runtime.checked("validation_loader", lambda: _loader(records, config, root, device, runtime))
    eval_model = runtime.unwrap_model(model) if hasattr(runtime, "unwrap_model") else model
    with runtime.evaluation_state(model):
        local_rows, local_mean = runtime.checked("validation_forward", lambda: predict_batches(
            eval_model, loader, device, criterion=pw5_loss(device), allow_empty=True))
        gathered = runtime.gather_predictions(local_rows)
        numbers = runtime.sum_numbers([len(local_rows), len(local_rows)*local_mean if local_rows else 0.0])
    def finalize():
        if int(numbers[0]) != len(records):
            raise ValueError("Validation sharding padded or lost official records.")
        rows = ordered_predictions(gathered, records, labeled=True)
        return rows, compute_official_metrics(rows, numbers[1]/numbers[0])
    # Only metrics are broadcast; full ordered predictions stay on primary.
    finalized = runtime.checked("validation_metrics", lambda: finalize() if runtime.rank == 0 else None)
    metrics = runtime.broadcast(finalized[1] if runtime.rank == 0 else None)
    return finalized[0] if runtime.rank == 0 else None, metrics


def training_stage(model, metadata, records, config, root, device, runtime, directory, manifest):
    loader = runtime.checked("training_loader", lambda: _loader(records["train"], config, root, device, runtime, training=True))
    groups = math.ceil(len(loader)/config["training"]["accumulation_steps"])
    optimizer, schedule = runtime.checked("optimizer_and_schedule", lambda: build_optimizer_schedule(model, config["training"], groups))
    selection, history = SelectionState(), []
    store = runtime.checked("checkpoint_store", lambda: CheckpointStore(directory / "checkpoints") if runtime.rank == 0 else None)
    stop_reason = "epoch_budget_completed"
    for epoch in range(1, config["training"]["epochs"]+1):
        started = time.monotonic()
        train = train_epoch(model, loader, optimizer, schedule, device, config["training"], runtime, epoch,
                            log_every=config["distributed"]["log_every_n_groups"])
        if train["consumed_samples"] != len(records["train"]):
            raise ValueError("Training must consume every official sample exactly once per epoch.")
        rows, metrics = distributed_validation(model, records["valid"], config, root, device, runtime)
        def persist():
            improved, stop = selection.update(epoch, metrics["roc_auc"], config["training"])
            if improved:
                store.save("best.pt", model, epoch, config, metrics, metadata)
                atomic_csv(directory / "predictions/best/valid_predictions.csv", rows, VALID_FIELDS)
            store.save("last.pt", model, epoch, config, metrics, metadata)
            item = {"epoch": epoch, "train_weighted_bce_pw5": train["weighted_bce_pw5"],
                    "valid_weighted_bce_pw5": metrics["weighted_bce_pw5"], "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"], "lr": train["lr"],
                    "successful_updates": train["successful_updates"], "consumed_samples": train["consumed_samples"],
                    "seconds": time.monotonic()-started}
            history.append(item)
            atomic_csv(directory / "reports/history.csv", history, list(item))
            atomic_json(directory / "reports" / f"epoch_{epoch:04d}.json", {**metrics, **item})
            manifest["stages"]["training"].update(status="running", completed_epochs=epoch,
                                                     samples=epoch*len(records["train"]))
            atomic_json(directory / "run_manifest.json", manifest)
            print(f"Epoch {epoch}: train_pw5={train['weighted_bce_pw5']:.6f}, valid_pw5={metrics['weighted_bce_pw5']:.6f}, "
                  f"AUROC={metrics['roc_auc']:.8f}, AP={metrics['average_precision']:.8f}, best={selection.best_epoch}, "
                  f"samples={train['consumed_samples']}, elapsed={item['seconds']:.1f}s", flush=True)
            return {"stop": stop, "best_epoch": selection.best_epoch, "best_valid_auroc": selection.best_raw}
        decision = runtime.primary("epoch_artifacts", persist)
        if decision["stop"]:
            stop_reason = "early_stopping"
            break
    result = {"best_epoch": decision["best_epoch"], "best_valid_auroc": decision["best_valid_auroc"],
              "last_epoch": epoch, "completed_epochs": epoch, "stop_reason": stop_reason}
    runtime.primary("training_summary", lambda: atomic_json(directory / "reports/metrics.json", result))
    return result


def post_training(directory, records, config, root, device, manifest):
    """Only rank 0 calls this AFTER the training process group has closed."""
    stage = "best_validation"
    try:
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        model, payload = load_timm_checkpoint(directory / "checkpoints/best.pt", device, root)
        frozen = payload["config"]
        if frozen != config:
            raise ValueError("Best checkpoint configuration differs from this frozen run.")
        valid_loader = _loader(records["valid"], frozen, root, device)
        rows, mean = predict_batches(model, valid_loader, device, criterion=pw5_loss(device))
        rows = ordered_predictions(rows, records["valid"], labeled=True)
        saved = read_csv(directory / "predictions/best/valid_predictions.csv")
        differences = compare_predictions(rows, saved, records["valid"], frozen)
        metrics = compute_official_metrics(rows, mean)
        if metrics["roc_auc"] != payload["metrics"]["roc_auc"]:
            raise ValueError("Reloaded best AUROC differs from saved checkpoint metrics.")
        atomic_json(directory / "reports/valid_best_metrics.json", {**metrics, "reload_max_abs": differences})
        manifest["stages"][stage].update(status="completed", epoch=payload["epoch"], samples=len(rows))
        stage = "test_inference"
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        rows, _ = predict_batches(model, _loader(records["test"], frozen, root, device), device)
        rows = ordered_predictions(rows, records["test"], labeled=False)
        predictions = directory / "predictions/best/test_predictions.csv"
        if predictions.exists():
            raise FileExistsError(predictions)
        atomic_csv(predictions, rows, TEST_FIELDS)
        manifest["stages"][stage].update(status="completed", epoch=payload["epoch"], samples=len(rows))
        stage = "submission"
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        source = resolve_path(frozen["data"]["raw_root"], root) / frozen["data"]["splits"]["test"]["csv"]
        destination = publish_submission(predictions, source, directory / "submission/submission.csv",
                                          source_id_column=frozen["data"]["image_column"])
        manifest["stages"][stage].update(status="completed", samples=len(rows), path=str(destination))
        weights = sorted(p.name for p in (directory / "checkpoints").iterdir())
        if weights != ["best.pt", "last.pt"]:
            raise ValueError(f"Unexpected checkpoint files: {weights}")
        size = (directory / "checkpoints/best.pt").stat().st_size
        manifest.update(status="completed", deployment={"inference_model": "checkpoints/best.pt",
            "uncompressed_bytes": size, "MB": size/1e6, "MiB": size/(1024**2),
            "size_limit_bytes": 500_000_000, "model_size_status": "passed" if size <= 500_000_000 else "exceeded",
            "t4_latency_status": "not_measured"})
        atomic_json(directory / "run_manifest.json", manifest)
        return destination
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["stages"][stage].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "run_manifest.json", manifest)
        raise


def preflight_plan(config, maximum):
    world, micro, accumulation = config["distributed"]["world_size"], config["training"]["batch_size"], config["training"]["accumulation_steps"]
    valid = config["preflight"]["validation_samples"]
    target_cost = 2*world*micro*accumulation+3*valid
    if maximum >= target_cost:
        return {"micro_batch": micro, "accumulation_steps": accumulation, "samples": target_cost, "target_batch_memory": "verified"}
    small_cost = 2*world*min(micro, 4)+3*valid
    if maximum < small_cost:
        raise ValueError(f"Preflight budget {maximum} is insufficient; minimum={small_cost}; no automatic increase.")
    return {"micro_batch": min(micro, 4), "accumulation_steps": 1, "samples": small_cost, "target_batch_memory": "not_verified"}


def _balanced_subset(records, count):
    negative = [r for r in records if str(r["label"]) == "0"][:count//2]
    positive = [r for r in records if str(r["label"]) == "1"][:count-count//2]
    if len(negative)+len(positive) != count:
        raise ValueError("Insufficient positive/negative validation records for bounded preflight.")
    selected = {r["image_id"] for r in negative+positive}
    return [r for r in records if r["image_id"] in selected]


def execute_preflight(config, records, root, device, runtime, directory, maximum):
    plan = preflight_plan(config, maximum)
    runtime.primary("preflight_directory", lambda: _new_directory(directory))
    report = {"status": "running", "config": config, "budget": maximum, "plan": plan,
              "consumed_sample_instances": 0, "official_test_predicted": False,
              "target_device": str(device), "target_batch_memory": "not_verified",
              "mixed_precision_status": "not_verified", "long_run_stability": "not_verified", "t4_latency_status": "not_measured",
              "environment": preflight_environment(config, device)}
    runtime.primary("preflight_started", lambda: atomic_json(directory / "report.json", report))
    try:
        seed_everything(config["experiment"]["seed"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, metadata = runtime.checked("preflight_model", lambda: build_timm_model(config["model"], root))
        model = model.to(device)
        model = runtime.initialize_model(model)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        small = copy.deepcopy(config)
        small["training"].update(batch_size=plan["micro_batch"], accumulation_steps=plan["accumulation_steps"])
        take = plan["micro_batch"]*plan["accumulation_steps"]*runtime.world_size
        train_records = records["train"][:take]
        loader = _loader(train_records, small, root, device, runtime, training=True)
        cached = runtime.checked("preflight_data", lambda: list(loader))
        optimizer, schedule = build_optimizer_schedule(model, config["training"], 2)
        model.train()
        update_group(model, cached, optimizer, schedule, device, config["training"], runtime, step=False, precision="fp32")
        report["consumed_sample_instances"] += take
        reference = torch.cat([p.grad.detach().float().cpu().flatten() for p in model.parameters()])
        model.load_state_dict(initial, strict=True)
        model.train()
        update_group(model, cached, optimizer, schedule, device, config["training"], runtime)
        report["consumed_sample_instances"] += take
        actual = torch.cat([p.grad.detach().float().cpu().flatten() for p in model.parameters()])
        delta = actual-reference
        difference = {"relative_l2": float(delta.norm()/reference.norm().clamp_min(1e-12)), "max_abs": float(delta.abs().max())}
        def check_update():
            if (difference["relative_l2"] > config["preflight"]["gradient_relative_l2_max"]
                    or difference["max_abs"] > config["preflight"]["gradient_max_abs"]):
                raise FloatingPointError(f"BF16/FP32 gradient difference exceeds frozen tolerances: {difference}")
            if not any(not torch.equal(v.detach().cpu(), initial[k]) for k, v in model.named_parameters()):
                raise ValueError("Preflight optimizer update did not change parameters.")
        runtime.checked("preflight_update_check", check_update)
        runtime.verify_parameters(model)
        def local_memory():
            return ({"rank": runtime.rank,
                     "allocated_bytes": torch.cuda.max_memory_allocated(device),
                     "reserved_bytes": torch.cuda.max_memory_reserved(device)}
                    if device.type == "cuda" else {"rank": runtime.rank, "allocated_bytes": 0, "reserved_bytes": 0})
        memory = local_memory()
        report["peak_memory_local_rank"] = memory
        report["max_reserved_bytes"] = int(runtime.sum_numbers([memory["reserved_bytes"]])[0])
        valid = _balanced_subset(records["valid"], config["preflight"]["validation_samples"])
        rows, metrics = distributed_validation(model, valid, config, root, device, runtime)
        report["consumed_sample_instances"] += len(valid)
        def save():
            store = CheckpointStore(directory / "checkpoints")
            for name in ("best.pt", "last.pt"):
                store.save(name, model, 1, config, metrics, metadata)
        runtime.primary("preflight_checkpoint_save", save)
        expected = runtime.broadcast(rows)
        for name in ("best.pt", "last.pt"):
            reloaded, _ = runtime.checked("preflight_checkpoint_reload", lambda: load_timm_checkpoint(directory / "checkpoints" / name, device, root))
            reloaded_rows, _ = distributed_validation(reloaded, valid, config, root, device, runtime)
            runtime.primary("preflight_reload_comparison", lambda: compare_predictions(reloaded_rows, expected, valid, config))
            report["consumed_sample_instances"] += len(valid)
            del reloaded
        if report["consumed_sample_instances"] != plan["samples"] or plan["samples"] > maximum:
            raise ValueError("Preflight exceeded or miscounted its global real-instance budget.")
        report.update(status="completed", model_metadata=metadata, gradient_difference=difference,
                      mixed_precision_status="passed", target_batch_memory=plan["target_batch_memory"],
                      validation_subset={"samples": len(valid), "selection": "first four per class, restored official order"})
        runtime.primary("preflight_report", lambda: atomic_json(directory / "report.json", report))
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        runtime.primary("preflight_failure_report", lambda: atomic_json(directory / "report.json", report))
        raise


def run_timm_experiment(config_path, root, *, preflight_only=False, preflight_version="pf_v1",
                        max_sample_instances=None, skip_preflight=False):
    root = Path(root).resolve()
    config = load_timm_config(resolve_path(config_path, root))
    if (not preflight_version or any(c in preflight_version for c in '/\\:<>"|?*')
            or preflight_version in {".", ".."}):
        raise ValueError("Use an explicit safe preflight version.")
    maximum = config["preflight"]["max_sample_instances"] if max_sample_instances is None else max_sample_instances
    if type(maximum) is not int or maximum < 1:
        raise ValueError("An explicit positive global preflight budget is required.")
    output = resolve_path(config["experiment"]["output_root"], root)
    directory = output / config["experiment"]["name"]
    flight_dir = output / "_preflight" / f"{config['experiment']['name']}_{preflight_version}"
    if preflight_only and flight_dir.exists():
        raise FileExistsError(flight_dir)
    if not preflight_only and directory.exists():
        raise FileExistsError(directory)
    runtime, device = create_source_runtime(config, root)
    manifest = None
    started = time.monotonic()
    try:
        records, sources = runtime.primary("official_manifests", lambda: ensure_official_manifests(config, root))
        if config["data"].get("cache", {}).get("enabled", False):
            from eye_qing_guang.datasets.offline_cache import cached_records, cache_signature
            def validate_cache():
                cached_records([row for split in records.values() for row in split], config["data"], root)
            runtime.primary("offline_cache_validation", validate_cache)
            sources["offline_cache"] = {"root": config["data"]["cache"]["root"],
                                        "signature": cache_signature(config["data"], root)}
        if skip_preflight:
            preflight = {"status": "skipped", "target_batch_memory": "skipped",
                         "config": config, "target_device": str(device),
                         "environment": preflight_environment(config, device)}
        elif not preflight_only and flight_dir.exists():
            def reuse():
                with (flight_dir / "report.json").open(encoding="utf-8") as handle:
                    result = json.load(handle)
                if (result.get("status") != "completed" or result.get("config") != config
                        or result.get("target_device") != str(device) or result.get("mixed_precision_status") != "passed"
                        or result.get("environment") != preflight_environment(config, device)):
                    raise ValueError("Existing preflight is not a matching qualified frozen configuration.")
                return result
            preflight = runtime.primary("preflight_reuse", reuse)
        else:
            preflight = execute_preflight(config, records, root, device, runtime, flight_dir, maximum)
        if preflight_only:
            return flight_dir
        if preflight["target_batch_memory"] not in {"verified", "skipped"}:
            required = preflight_plan(config, 10**18)["samples"]
            raise RuntimeError(
                "Small-budget preflight passed, but target batch memory is not verified. "
                f"Formal run was not created. This configuration requires an explicit "
                f"target-group preflight budget of at least {required} instances and a new preflight version."
            )
        runtime.primary("formal_directory", lambda: _new_directory(directory))
        manifest = run_manifest("running", protocol=config["experiment"]["protocol"],
            stages={s: {"status": "not_started"} for s in STAGES}, sources=sources, config=config,
            preflight=("skipped_by_cli" if skip_preflight else str(flight_dir)),
            loss="bce", pos_weight=5.0, auto_pos_weight=False,
            focal_gamma="unused_for_this_run",
            effective_batch=(runtime.world_size * config["training"]["batch_size"]
                             * config["training"]["accumulation_steps"]),
            precision=config["training"]["precision"], scaler_enabled=False,
            gradient_transport=config["distributed"]["gradient_transport"], training_engine=config["recipe"],
            device=str(device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            micro_batch=config["training"]["batch_size"], accumulation_steps=config["training"]["accumulation_steps"],
            torch_version=str(torch.__version__), world_size=runtime.world_size, t4_latency_status="not_measured")
        def start_run():
            write_yaml(directory / "config_snapshot.yaml", config)
            manifest["stages"]["training"].update(status="running", completed_epochs=0, samples=0)
            atomic_json(directory / "run_manifest.json", manifest)
        runtime.primary("formal_start", start_run)
        seed_everything(config["experiment"]["seed"])
        model, metadata = runtime.checked("formal_model", lambda: build_timm_model(config["model"], root))
        model = model.to(device)
        model = runtime.initialize_model(model)
        manifest["model_metadata"] = metadata
        result = training_stage(model, metadata, records, config, root, device, runtime, directory, manifest)
        def complete_training():
            manifest["stages"]["training"].update(status="completed", **result)
            atomic_json(directory / "run_manifest.json", manifest)
        runtime.primary("training_complete", complete_training)
        runtime.checked("training_devices_complete", lambda: torch.cuda.synchronize(device))
        runtime.close()
        del model
        torch.cuda.empty_cache()
        if runtime.rank == 0:
            destination = post_training(directory, records, config, root, device, manifest)
            manifest["total_seconds"] = time.monotonic()-started
            atomic_json(directory / "run_manifest.json", manifest)
            print(f"timm experiment completed: {destination}", flush=True)
        return directory
    except BaseException as exc:
        if manifest is not None and runtime.rank == 0:
            manifest["status"] = "failed"
            # post_training already records its precise failed stage.
            if manifest["stages"]["training"]["status"] != "completed":
                manifest["stages"]["training"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(directory / "run_manifest.json", manifest)
        raise
    finally:
        runtime.close()

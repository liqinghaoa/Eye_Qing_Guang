"""One frozen KD variant per run, with student-only validation and deployment."""

import copy
import gc
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from eye_qing_guang.contracts.convnext_kd_config import load_kd_config, MANIFEST_PROTOCOL
from eye_qing_guang.contracts.convnext_kd_checkpoints import KDCheckpointStore, load_kd_student_checkpoint
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.convnext_official import ensure_official_manifests, ordered_predictions
from eye_qing_guang.datasets.offline_cache import cached_records, cache_signature
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.metrics.convnext_official import compute_official_metrics
from eye_qing_guang.models.frozen_teacher_ensemble import FrozenTeacherEnsemble
from eye_qing_guang.models.timm_convnext import build_convnext_model, prepared_pretrained_file
from eye_qing_guang.pipelines.convnext_official import (
    _loader, _new_directory, compare_predictions, distributed_validation, VALID_FIELDS, TEST_FIELDS, STAGES,
)
from eye_qing_guang.trainers.convnext_official import SelectionState, build_optimizer_schedule, pw5_loss, autocast_context
from eye_qing_guang.trainers.convnext_kd import train_epoch, make_targets, loss_terms, _check_gradients
from eye_qing_guang.utils.experiment_io import read_csv, run_manifest, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import seed_everything
from eye_qing_guang.utils.timm_artifacts import atomic_csv, atomic_json, publish_submission
from eye_qing_guang.utils.timm_source import create_source_runtime

EXPECTED_LABEL_COUNTS = {"train": (49739, 1657), "valid": (5658, 191)}
PREFLIGHT_REQUIRED_INSTANCES = 216


def prepare_sources(config, root):
    records, sources = ensure_official_manifests(config, root)
    for split, expected in EXPECTED_LABEL_COUNTS.items():
        labels = [str(row["label"]) for row in records[split]]
        if (labels.count("0"), labels.count("1")) != expected:
            raise ValueError(f"Official {split} label counts differ from the KD data protocol.")
    cached_records([row for rows in records.values() for row in rows], config["data"], root)
    sources["offline_cache"] = {"root": config["data"]["cache"]["root"],
                                "signature": cache_signature(config["data"], root)}
    return records, sources


def prepare_models(config, root, device):
    prepared_pretrained_file(config["model"], root)
    teachers = FrozenTeacherEnsemble.load(config, root, device)
    # Loading all task teachers precedes this reset, including their random architecture construction.
    seed_everything(config["experiment"]["seed"])
    student, metadata = build_convnext_model(config["model"], root, image_size=config["data"]["image_size"])
    return teachers, student.to(device).float(), metadata


def training_stage(student, metadata, teachers, records, config, root, device, runtime, directory, manifest):
    loader = _loader(records["train"], config, root, device, runtime, training=True)
    groups = math.ceil(len(loader) / config["training"]["accumulation_steps"])
    optimizer, schedule = build_optimizer_schedule(student, config["training"], groups)
    teachers.assert_student_optimizer(student, optimizer)
    selection, history = SelectionState(), []
    store = KDCheckpointStore(directory / "checkpoints")
    progress = manifest["training_progress"]
    stop_reason = "epoch_budget_completed"
    for epoch in range(1, config["training"]["epochs"] + 1):
        started = time.monotonic()
        progress["current_epoch"] = epoch
        train = train_epoch(student, loader, optimizer, schedule, device, config["training"],
                            config["distillation"], teachers, epoch,
                            log_every=config["distributed"]["log_every_n_groups"], progress=progress)
        if train["consumed_samples"] != len(records["train"]):
            raise ValueError("KD must consume every official train sample exactly once each epoch.")
        rows, metrics = distributed_validation(student, records["valid"], config, root, device, runtime)
        improved, stop = selection.update(epoch, metrics["roc_auc"], config["training"])
        if improved:
            store.save("best.pt", student, epoch, config, metrics, metadata, teachers.provenance)
            atomic_csv(directory / "predictions/best/valid_predictions.csv", rows, VALID_FIELDS)
        store.save("last.pt", student, epoch, config, metrics, metadata, teachers.provenance)
        item = {"epoch": epoch, **train, "valid_weighted_bce_pw5": metrics["weighted_bce_pw5"],
                **{key: metrics[key] for key in (
                    "roc_auc", "average_precision", "pr_auc_trapezoid", "brier", "balanced_accuracy",
                    "sensitivity", "specificity", "precision", "f1")},
                "seconds": time.monotonic() - started}
        history.append(item)
        atomic_csv(directory / "reports/history.csv", history, list(item))
        atomic_json(directory / "reports" / f"epoch_{epoch:04d}.json", {**metrics, **item})
        progress["completed_epochs"] = epoch
        manifest["stages"]["training"].update(status="running", completed_epochs=epoch,
                                               samples=progress["consumed_train_samples"])
        atomic_json(directory / "run_manifest.json", manifest)
        print(f"Epoch {epoch}: KD_total={train['train_total_loss']:.6f}, "
              f"valid_pw5={metrics['weighted_bce_pw5']:.6f}, AUROC={metrics['roc_auc']:.8f}, "
              f"AP={metrics['average_precision']:.8f}, best={selection.best_epoch}, "
              f"samples={train['consumed_samples']}, elapsed={item['seconds']:.1f}s", flush=True)
        if stop:
            stop_reason = "early_stopping"
            break
    teachers.verify_frozen()
    result = {"best_epoch": selection.best_epoch, "best_valid_auroc": selection.best_raw,
              "completed_epochs": epoch, "last_epoch": epoch, "stop_reason": stop_reason}
    atomic_json(directory / "reports/metrics.json", result)
    return result


def post_training(directory, records, config, root, device, manifest):
    stage = "best_validation"
    try:
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        student, payload = load_kd_student_checkpoint(directory / "checkpoints/best.pt", device, root)
        if payload["config"] != config:
            raise ValueError("Student best checkpoint differs from this frozen KD run.")
        rows, mean = predict_batches(student, _loader(records["valid"], config, root, device),
                                     device, criterion=pw5_loss(device))
        rows = ordered_predictions(rows, records["valid"], labeled=True)
        differences = compare_predictions(rows, read_csv(directory / "predictions/best/valid_predictions.csv"),
                                          records["valid"], config)
        metrics = compute_official_metrics(rows, mean)
        if metrics["roc_auc"] != payload["metrics"]["roc_auc"]:
            raise ValueError("Reloaded student AUROC differs from the selected epoch.")
        atomic_json(directory / "reports/valid_best_metrics.json", {**metrics, "reload_max_abs": differences})
        manifest["stages"][stage].update(status="completed", epoch=payload["epoch"], samples=len(rows))
        stage = "test_inference"
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        rows, _ = predict_batches(student, _loader(records["test"], config, root, device), device)
        rows = ordered_predictions(rows, records["test"], labeled=False)
        predictions = directory / "predictions/best/test_predictions.csv"
        if predictions.exists():
            raise FileExistsError(predictions)
        atomic_csv(predictions, rows, TEST_FIELDS)
        manifest["stages"][stage].update(status="completed", epoch=payload["epoch"], samples=len(rows))
        stage = "submission"
        manifest["stages"][stage]["status"] = "running"
        atomic_json(directory / "run_manifest.json", manifest)
        source = resolve_path(config["data"]["raw_root"], root) / config["data"]["splits"]["test"]["csv"]
        destination = publish_submission(predictions, source, directory / "submission/submission.csv",
                                          source_id_column=config["data"]["image_column"])
        if sorted(p.name for p in (directory / "checkpoints").iterdir()) != ["best.pt", "last.pt"]:
            raise ValueError("KD must retain only student best.pt and last.pt.")
        size = (directory / "checkpoints/best.pt").stat().st_size
        manifest["stages"][stage].update(status="completed", samples=len(rows), path=str(destination))
        manifest.update(status="completed", deployment={
            "inference_models": "student_only", "inference_model": "checkpoints/best.pt",
            "uncompressed_bytes": size, "MB": size / 1e6, "MiB": size / 1024**2,
            "size_limit_bytes": 500_000_000, "model_size_status": "passed" if size <= 500_000_000 else "exceeded",
            "t4_latency_status": "not_measured"})
        atomic_json(directory / "run_manifest.json", manifest)
        return destination
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["stages"][stage].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "run_manifest.json", manifest)
        raise


def run_kd_experiment(config_path, root):
    root = Path(root).expanduser().resolve()
    config = load_kd_config(resolve_path(config_path, root))
    directory = resolve_path(config["experiment"]["output_root"], root) / config["experiment"]["name"]
    if directory.exists():
        raise FileExistsError(f"KD run directory already exists: {directory}")
    runtime, device = create_source_runtime(config, root)
    manifest, started = None, time.monotonic()
    try:
        records, sources = prepare_sources(config, root)
        teachers, student, metadata = prepare_models(config, root, device)
        print("KD preparation complete: official manifests/cache, frozen teachers and ImageNet student ready.", flush=True)
        _new_directory(directory)
        kd = config["distillation"]
        manifest = run_manifest("running", protocol=config["experiment"]["protocol"],
            config=config, sources=sources, manifest_source_protocol=MANIFEST_PROTOCOL,
            stages={stage: {"status": "not_started"} for stage in STAGES},
            training_progress={"current_epoch": 0, "completed_epochs": 0, "consumed_train_samples": 0},
            official_valid_merged=False, teachers_frozen=True,
            teacher_target_source="official_train_same_augmented_view", student_initialization="ImageNet1K",
            distillation_variant=kd["variant"], teacher_ensemble=kd["fusion"], temperature=kd["temperature"],
            alpha=kd["alpha"], inference_models="student_only", teacher_provenance=teachers.provenance,
            model_metadata=metadata, effective_batch=config["training"]["batch_size"] * config["training"]["accumulation_steps"],
            precision=config["training"]["precision"], teacher_precision="fp32",
            scaler_enabled=False, device=str(device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=str(torch.__version__), preflight="separate_explicit_command", t4_latency_status="not_measured")
        manifest["stages"]["training"]["status"] = "running"
        write_yaml(directory / "config_snapshot.yaml", config)
        atomic_json(directory / "run_manifest.json", manifest)
        print(f"Full KD training started: {directory}; variant={kd['variant']}", flush=True)
        result = training_stage(student, metadata, teachers, records, config, root, device, runtime, directory, manifest)
        manifest["stages"]["training"].update(status="completed", **result)
        manifest["teachers_frozen_verified_after_training"] = True
        atomic_json(directory / "run_manifest.json", manifest)
        # The optimizer is local to training_stage; free teachers and all training activations before student reload.
        del student, teachers
        runtime.close()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        destination = post_training(directory, records, config, root, device, manifest)
        manifest["total_seconds"] = time.monotonic() - started
        atomic_json(directory / "run_manifest.json", manifest)
        print(f"KD experiment completed: {destination}", flush=True)
        return directory
    except BaseException as exc:
        if manifest is not None:
            manifest["status"], manifest["error"] = "failed", f"{type(exc).__name__}: {exc}"
            if manifest["stages"]["training"]["status"] != "completed":
                manifest["stages"]["training"].update(
                    status="failed", completed_epochs=manifest["training_progress"]["completed_epochs"],
                    samples=manifest["training_progress"]["consumed_train_samples"])
            atomic_json(directory / "run_manifest.json", manifest)
        progress = manifest["training_progress"] if manifest else {"completed_epochs": 0, "consumed_train_samples": 0}
        print(f"KD failed; formal_training_started={manifest is not None}; progress={progress}; "
              f"partial_output={directory if directory.exists() else 'none'}", flush=True)
        raise
    finally:
        runtime.close()


def predict_kd_student(checkpoint, manifest_path, root, output, device):
    """Standalone student prediction from a portable raw-image manifest."""
    root, output = Path(root).expanduser().resolve(), Path(output)
    output = resolve_path(output, root)
    if output.exists():
        raise FileExistsError(output)
    student, payload = load_kd_student_checkpoint(resolve_path(checkpoint, root), device, root)
    config = copy.deepcopy(payload["config"])
    config["data"]["cache"]["enabled"] = False
    records = load_manifest(resolve_path(manifest_path, root), root, labeled=False)
    rows, _ = predict_batches(student, _loader(records, config, root, device), device)
    rows = ordered_predictions(rows, records, labeled=False)
    atomic_csv(output, rows, TEST_FIELDS)
    return output


def _reference_predictions(config, root, valid_records):
    references = []
    for spec in config["distillation"]["teachers"]:
        path = resolve_path(spec["checkpoint"], root).parent.parent / "predictions/best/valid_predictions.csv"
        rows = read_csv(path)
        if (len(rows) != len(valid_records) or len({row["image_id"] for row in rows}) != len(rows)
                or any(set(row) != set(VALID_FIELDS) for row in rows)):
            raise ValueError(f"Invalid saved teacher validation predictions: {path}")
        references.append(ordered_predictions(rows, valid_records, labeled=True))
    return references


def run_kd_preflight(config_path, root, output, max_sample_instances):
    root = Path(root).expanduser().resolve()
    config = load_kd_config(resolve_path(config_path, root))
    if type(max_sample_instances) is not int or max_sample_instances < PREFLIGHT_REQUIRED_INSTANCES:
        raise ValueError("KD preflight requires an explicit budget of at least 216 model-forward sample instances; "
                         "no model forward was started.")
    output = resolve_path(output, root)
    formal = resolve_path(config["experiment"]["output_root"], root) / config["experiment"]["name"]
    if output == formal or output.is_relative_to(formal):
        raise ValueError("Preflight output must be independent of the formal run directory.")
    if output.exists():
        raise FileExistsError(output)
    runtime, device = create_source_runtime(config, root)
    output.mkdir(parents=True, exist_ok=False)
    teachers = None
    report = {"status": "running", "formal_training_started": False, "official_test_predicted": False,
              "config": config, "max_model_forward_sample_instances": max_sample_instances,
              "model_forward_sample_instances": 0, "student_forward_sample_instances": 0,
              "optimizer_step": False, "t4_latency_status": "not_measured"}
    atomic_json(output / "report.json", report)
    started = time.monotonic()
    try:
        records, _ = prepare_sources(config, root)
        references = _reference_predictions(config, root, records["valid"])
        teachers, student, _ = prepare_models(config, root, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        valid = records["valid"][:8]
        if len(valid) != 8:
            raise ValueError("Preflight requires eight official validation images.")
        valid_batch = next(iter(_loader(valid, config, root, device)))
        images = valid_batch["image"].to(device)
        if len(images) != 8:
            raise ValueError("Preflight validation must consume exactly eight images.")
        comparisons = []
        for spec, actual, saved in zip(config["distillation"]["teachers"],
                                        teachers.individual_predictions(images), references):
            by_id = {row["image_id"]: row for row in saved}
            maxima = {}
            for field, values, atol in (("logit", actual[0], 1e-4), ("probability", actual[1], 1e-5)):
                expected = np.asarray([float(by_id[row["image_id"]][field]) for row in valid])
                measured = values.detach().cpu().numpy().reshape(-1)
                maxima[field] = float(np.max(np.abs(measured - expected)))
                if not np.allclose(measured, expected, atol=atol, rtol=1e-4):
                    raise ValueError(f"{spec['id']} saved validation {field} differs: max_abs={maxima[field]}")
            comparisons.append({"id": spec["id"], "max_abs": maxima})
        chosen = set()
        for label in ("0", "1"):
            subset = [row for row in records["train"] if str(row["label"]) == label][:24]
            if len(subset) != 24:
                raise ValueError("Preflight needs 24 train samples from each class.")
            chosen.update(row["image_id"] for row in subset)
        train = [row for row in records["train"] if row["image_id"] in chosen]
        batch = next(iter(_loader(train, config, root, device, runtime, training=True)))
        images, labels = batch["image"].to(device), batch["label"].to(device)
        if len(images) != 48:
            raise ValueError("Preflight must retain the complete 48-sample student microbatch.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        teacher_started = time.monotonic()
        q = teachers.probabilities(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        teacher_seconds = time.monotonic() - teacher_started
        target, _ = make_targets(q, labels, config["distillation"]["variant"], config["distillation"]["epsilon"])
        student.train()
        optimizer, _ = build_optimizer_schedule(student, config["training"], 1)
        teachers.assert_student_optimizer(student, optimizer)
        optimizer.zero_grad(set_to_none=True)
        student_started = time.monotonic()
        report["student_forward_sample_instances"] = len(images)
        with autocast_context(device, config["training"]["precision"]):
            logits = student(images, None)
        terms = loss_terms(logits, labels, target, config["distillation"]["alpha"])
        terms["total_loss"].mean().backward()
        _check_gradients(student)
        if not any(torch.count_nonzero(p.grad) for p in student.parameters() if p.requires_grad):
            raise FloatingPointError("Preflight student gradients are all zero.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        student_seconds = time.monotonic() - student_started
        teachers.verify_frozen()
        consumed = teachers.forward_sample_instances + report["student_forward_sample_instances"]
        if consumed != PREFLIGHT_REQUIRED_INSTANCES or consumed > max_sample_instances:
            raise ValueError("KD preflight instance accounting differs from its fixed 216-instance plan.")
        report.update(status="completed", model_forward_sample_instances=consumed,
            teacher_forward_sample_instances=teachers.forward_sample_instances, student_backward_samples=48,
            distinct_images=56, teacher_validation_comparisons=comparisons, teachers_frozen=True,
            student_gradients="finite_nonzero", teacher_train_forward_seconds=teacher_seconds,
            student_forward_backward_seconds=student_seconds, seconds=time.monotonic() - started,
            max_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            max_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        atomic_json(output / "report.json", report)
        print(f"KD preflight completed: {output}; model-forward instances={consumed}/"
              f"{max_sample_instances}; formal training not started.", flush=True)
        return output
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
            model_forward_sample_instances=(teachers.forward_sample_instances if teachers else 0)
                                           + report["student_forward_sample_instances"])
        atomic_json(output / "report.json", report)
        print(f"KD preflight failed; formal_training_started=False; forward_instances="
              f"{report['model_forward_sample_instances']}; partial_output={output}", flush=True)
        raise
    finally:
        runtime.close()

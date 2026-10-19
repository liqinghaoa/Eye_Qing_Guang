"""B0-dev orchestration; official valid/test are never selected or opened here."""

import csv
import math
import subprocess
import time

import numpy as np
import PIL
import sklearn
import torch
import torchvision

from eye_qing_guang.contracts.checkpoints import save_checkpoint
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.datasets.development import load_development_manifests
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.development import prediction_metrics, write_development_analysis
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.trainers.binary import make_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import DevelopmentStopping, SuccessfulUpdateSchedule
from eye_qing_guang.utils.experiment_io import new_experiment, run_manifest, write_csv, write_json, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import choose_device, seed_everything


def environment(root, device):
    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "numpy": np.__version__, "Pillow": PIL.__version__, "sklearn": sklearn.__version__,
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "augmentation_backend": "Pillow RGB + torch RNG, ordered FOV/flip/rotate/brightness/contrast/letterbox",
            "evaluation_precision": "FP32", "resume_supported": False}


def new_progress():
    return {"attempted_groups": 0, "successful_updates": 0, "skipped_groups": 0,
            "consumed_samples": 0, "successful_samples": 0, "consecutive_skips": 0}


def train_development(config, root, name=None):
    validate_config(config)
    settings = config["training"]
    output_root = resolve_path(config["experiment"]["output_root"], root)
    name = name or config["experiment"]["name"]
    if (output_root / name).exists():
        raise FileExistsError(output_root / name)
    device = choose_device(settings["device"])
    if device.type != "cuda":
        raise RuntimeError("B0 requires explicit CUDA; no automatic CPU fallback.")
    records, provenance = load_development_manifests(config, root)
    seed_everything(config["experiment"]["seed"])
    loaders = {role: make_loader(rows, config, root, training=role == "train", device=device)
               for role, rows in records.items()}
    for loader in loaders.values():
        loader.dataset[0]
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["eps"])
    criterion = build_loss(settings, pos_weight=1.0).to(device)
    scaler = make_scaler(settings["amp"])
    groups = math.ceil(len(loaders["train"]) / settings["accumulation_steps"])
    schedule = SuccessfulUpdateSchedule(optimizer, groups, settings)
    stopping = DevelopmentStopping(settings["min_epochs"], settings["patience"], settings["min_delta"])
    directory = new_experiment(output_root, name)
    config = {**config, "experiment": {**config["experiment"], "name": name}}
    progress = new_progress()
    history, best_predictions = [], None
    error, status, stage = None, "failed", "initializing"
    updates_handle = None
    started = time.perf_counter()

    def state():
        return {"scheduler": schedule.state_dict(), "scaler": scaler.state_dict(),
                "early_stopping": stopping.state_dict(), "progress": dict(progress),
                "resume_supported": False}

    try:
        write_yaml(directory / "config_snapshot.yaml", config)
        write_json(directory / "run_started.json", run_manifest(
            "running", protocol="b0_dev_v1", environment=environment(root, device),
            provenance=provenance, microbatches_per_epoch=len(loaders["train"]),
            expected_groups_per_epoch=groups, schedule=schedule.state_dict(),
            pos_weight=1.0, train_amp=settings["amp"], nominal_effective_batch=settings["batch_size"]*settings["accumulation_steps"],
            parameter_count=sum(p.numel() for p in model.parameters()),
            inference_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
        ))
        torch.cuda.reset_peak_memory_stats(device)
        updates_handle = (directory / "reports" / "updates.csv").open("x", encoding="utf-8", newline="")
        update_fields = ["epoch", "attempted_group", "attempted_update_index", "successful_update_index", "group_samples", "success",
                         "lr", "scale_before", "scale_after", "grad_norm", "consumed_samples",
                         "successful_samples", "skipped_groups"]
        updates_writer = csv.DictWriter(updates_handle, fieldnames=update_fields)
        updates_writer.writeheader()

        def report_group(row):
            updates_writer.writerow({"epoch": epoch, **row})
            updates_handle.flush()

        print(f"B0-dev training started: {directory}; official valid/test excluded", flush=True)
        for epoch in range(1, settings["epochs"]+1):
            stage = f"epoch_{epoch}_training"
            epoch_started = time.perf_counter()
            train_metrics = train_accumulated_epoch(model, loaders["train"], optimizer, criterion,
                                                     device, settings, scaler, schedule, progress, report_group)
            stage = f"epoch_{epoch}_internal_development"
            predictions, loss = predict_batches(model, loaders["valid"], device, criterion)
            if [row["image_id"] for row in predictions] != [row["image_id"] for row in records["valid"]]:
                raise RuntimeError("Internal development evaluation order/count changed.")
            metrics = prediction_metrics(predictions)
            improved, stop = stopping.update(epoch, metrics["roc_auc"])
            result = {"epoch": epoch, **train_metrics, "valid_loss": loss, **metrics,
                      "epoch_seconds": time.perf_counter()-epoch_started,
                      "best_epoch": stopping.best_epoch, "reference_auc": stopping.reference_auc,
                      "stale": stopping.stale, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            history.append(result)
            write_json(directory / "reports" / f"epoch_{epoch:04d}.json", result)
            if improved:
                best_predictions = predictions
                save_checkpoint(directory / "checkpoints" / f"epoch_{epoch:04d}.pt", model, optimizer,
                                epoch, config, None, 1.0, metrics, state())
            print(f"Epoch {epoch}: BCE={train_metrics['train_loss']:.6f} dev_BCE={loss:.6f} "
                  f"dev_AUROC={metrics['roc_auc']:.8f} AP={metrics['average_precision']:.8f} "
                  f"updates={train_metrics['successful_updates']} skipped={train_metrics['skipped_groups']} "
                  f"lr={train_metrics['last_attempt_lr']:.9g}", flush=True)
            if stop:
                break
        stage = "saving_final_artifacts"
        save_checkpoint(directory / "checkpoints" / "last.pt", model, optimizer,
                        epoch, config, None, 1.0, metrics, state())
        from eye_qing_guang.pipelines.training import _copy_checkpoint

        _copy_checkpoint(directory / "checkpoints" / f"epoch_{stopping.best_epoch:04d}.pt",
                         directory / "checkpoints" / "best.pt")
        # Inference-only payload deliberately excludes optimizer and training state.
        payload = torch.load(directory / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
        for key in ("optimizer_state", "training_state"):
            payload.pop(key)
        with (directory / "checkpoints" / "best_inference.pt").open("xb") as handle:
            torch.save(payload, handle)
        write_csv(directory / "reports" / "history.csv", history)
        write_json(directory / "reports" / "metrics.json", {
            **prediction_metrics(best_predictions), "best_epoch": stopping.best_epoch,
            "selection_source": "internal_development", "completed_epochs": len(history),
            "convergence": "not_automatically_certified; inspect curves",
        })
        write_csv(directory / "predictions" / "dev_best.csv", best_predictions,
                  ["image_id", "label", "logit", "probability"])
        stage = "development_error_analysis"
        write_development_analysis(directory, records["train"], records["valid"], best_predictions, root, config)
        status, stage = "completed", "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"B0 failed at {stage}; completed epochs={len(history)}; progress={progress}; outputs={directory}", flush=True)
        raise
    finally:
        if updates_handle is not None:
            updates_handle.close()
        write_json(directory / "run_manifest.json", run_manifest(
            status, protocol="b0_dev_v1", stage=stage, completed_epochs=len(history),
            best_epoch=stopping.best_epoch, progress=progress, error=error,
            elapsed_seconds=time.perf_counter()-started, resume_supported=False,
        ))
    return directory

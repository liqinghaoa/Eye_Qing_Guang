"""Four-rank B0-dev, with global validation and a single artifact writer."""

import csv
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from eye_qing_guang.contracts.checkpoints import save_checkpoint
from eye_qing_guang.contracts.precision import training_precision
from eye_qing_guang.contracts.resume import load_resume_checkpoint, restore_training_state
from eye_qing_guang.datasets.development import load_development_manifests
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.development import prediction_metrics, write_development_analysis
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.pipelines.b0_development import environment, new_progress
from eye_qing_guang.trainers.binary import check_runtime_precision, make_training_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import DevelopmentStopping, SuccessfulUpdateSchedule
from eye_qing_guang.utils.distributed import close_distributed, initialize_distributed, wrap_model
from eye_qing_guang.utils.experiment_io import new_experiment, run_manifest, write_csv, write_json, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import seed_everything
from eye_qing_guang.utils.training_diagnostics import TrainingDiagnostics


def global_progress(progress, world_size):
    result = dict(progress)
    for key in ("consumed_samples", "successful_samples", "reference_samples"):
        if key in result:
            result[key] *= world_size
    return result


def global_epoch_metrics(local, context):
    context.agree([local[key] for key in ("attempted_groups", "successful_updates", "skipped_groups",
                                        "global_successful_updates")], "epoch update counts")
    loss_sum, consumed, successful_samples = context.reduce([
        local["train_loss"] * local["consumed_samples"], local["consumed_samples"], local["successful_samples"]])
    seconds, norm = context.reduce([local["train_seconds"], local["max_grad_norm"]], dist.ReduceOp.MAX)
    return {**local, "train_loss": loss_sum / consumed, "consumed_samples": int(consumed),
            "successful_samples": int(successful_samples), "train_seconds": seconds,
            "max_grad_norm": norm, "consumed_samples_per_second": consumed / seconds}


def ordered_predictions(shards, records):
    rows = [row for shard in shards for row in shard]
    by_id = {row["image_id"]: row for row in rows}
    expected = [row["image_id"] for row in records]
    if len(by_id) != len(rows) or len(rows) != len(expected) or set(by_id) != set(expected):
        raise RuntimeError("Distributed evaluation has duplicate, missing or unexpected image IDs.")
    result = [by_id[image_id] for image_id in expected]
    if any(int(prediction["label"]) != int(record["label"]) for prediction, record in zip(result, records)):
        raise RuntimeError("Distributed evaluation label mismatch.")
    return result


def distributed_predictions(model, loader, records, criterion, context, diagnostics=None):
    """Uneven/empty shards use the unwrapped FP32 model; collect only afterwards."""
    if diagnostics is not None:
        diagnostics.mark("evaluation_buffer_sync")
    context.synchronize_buffers(model)
    predictions, loss = predict_batches(model, loader, context.device, criterion, allow_empty=True, diagnostics=diagnostics)
    if diagnostics is not None:
        diagnostics.mark("evaluation_gather")
    totals = context.reduce([0.0 if loss is None else loss * len(predictions), len(predictions)])
    shards = context.gather_json(predictions)
    ordered = None

    def check():
        nonlocal ordered
        ordered = ordered_predictions(shards, records)
        if int(totals[1]) != len(records):
            raise RuntimeError("Distributed evaluation sample count changed.")

    context.on_primary(check)
    return ordered, totals[0] / totals[1]


def failure_record(directory, context, stage, progress, error, formal_training_started):
    """Best-effort local failure output: no collectives after a peer failure."""
    print(f"DDP failed: rank={context.rank}, stage={stage}, formal_training_started={formal_training_started}, "
          f"local_progress={progress}, outputs={directory}, error={error}", flush=True)
    if directory is not None:
        try:
            write_json(directory / "reports" / f"failure_rank_{context.rank:04d}.json", run_manifest(
                "failed", rank=context.rank, stage=stage, error=error, local_progress=progress,
                formal_training_started=formal_training_started,
                progress_scope="local samples; globally synchronized completed update indices"))
        except (OSError, ValueError):
            pass


def train_distributed(config, root, name=None, resume_from=None):
    resume_payload, resume_info = load_resume_checkpoint(resume_from, config) if resume_from is not None else (None, None)
    context = initialize_distributed(config)
    directory, handle, history_handle, history_writer = None, None, None, None
    history, best_predictions = [], None
    progress = new_progress()
    status, stage, error = "failed", "initializing", None
    formal_training_started = False
    diagnostics = None
    start_epoch = resume_info["start_epoch"] if resume_info else 1
    inherited_epochs = start_epoch - 1
    started = time.perf_counter()
    config = {**config, "experiment": {**config["experiment"], "name": name or config["experiment"]["name"]}}
    settings, options = config["training"], config["distributed"]
    stopping = DevelopmentStopping(settings["min_epochs"], settings["patience"], settings["min_delta"])
    try:
        directory = Path(context.on_primary(lambda: str(new_experiment(
            resolve_path(config["experiment"]["output_root"], root), config["experiment"]["name"]))))
        context.on_primary(lambda: write_yaml(directory / "config_snapshot.yaml", config))
        if config.get("diagnostics", {}).get("enabled", False):
            diagnostics = TrainingDiagnostics(directory, context, config["diagnostics"])
        if context.primary:
            print(f"Four-card setup: {directory}; resume_from={resume_from}; start_epoch={start_epoch}", flush=True)
        stage = "training_precision_check"
        if diagnostics is not None:
            diagnostics.mark(stage)
        check_runtime_precision(settings, context.device)
        stage = "frozen_manifests_and_loaders"
        if diagnostics is not None:
            diagnostics.mark(stage)
        records, provenance = load_development_manifests(config, root)
        loaders = {role: make_loader(rows, config, root, training=role == "train", device=context.device,
                                     distributed=context) for role, rows in records.items()}
        for loader in loaders.values():
            loader.dataset[context.rank % len(loader.dataset)]
        stage = "model_initialization"
        if diagnostics is not None:
            diagnostics.mark(stage)
        seed_everything(config["experiment"]["seed"])
        model = build_model(config["model"]).to(context.device)
        model = wrap_model(model, context)
        seed_everything(config["experiment"]["seed"] + context.rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                     weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]), eps=settings["eps"])
        criterion = build_loss(settings, 1.0).to(context.device)
        scaler = make_training_scaler(settings)
        groups = math.ceil(len(loaders["train"]) / settings["accumulation_steps"])
        schedule = SuccessfulUpdateSchedule(optimizer, groups, settings)
        if resume_payload is not None:
            stage = "restore_training_state"
            if diagnostics is not None:
                diagnostics.mark(stage)
            progress, stopping = restore_training_state(resume_payload, model.module, optimizer, schedule, scaler,
                                                        len(records["train"]), context.world_size)
            resume_info["batch_transition"].update(
                source_groups_per_epoch=math.ceil(math.ceil((len(records["train"]) // context.world_size) /
                    resume_payload["config"]["training"]["batch_size"]) /
                    resume_payload["config"]["training"]["accumulation_steps"]),
                target_groups_per_epoch=groups,
                source_schedule_total=resume_payload["training_state"]["scheduler"]["total"],
                target_schedule_total=schedule.total,
                next_lr=schedule.value(schedule.successful_updates + 1))
            context.agree([schedule.successful_updates, progress["attempted_groups"], progress["consumed_samples"]],
                          "restored training progress")
            if context.primary:
                print(f"Checkpoint state restored: epoch={inherited_epochs}, successful_updates={schedule.successful_updates}; "
                      "checking inherited best predictions on internal development data", flush=True)
            best_path = Path(resume_from) if stopping.best_epoch == resume_payload["epoch"] else (
                Path(resume_from).parent / f"epoch_{stopping.best_epoch:04d}.pt")

            def inherit_best():
                from eye_qing_guang.pipelines.training import _copy_checkpoint

                best = torch.load(best_path, map_location="cpu", weights_only=True)
                if best["epoch"] != stopping.best_epoch or best["metrics"]["roc_auc"] != stopping.best_auc:
                    raise ValueError("Inherited best checkpoint does not match saved early-stopping state.")
                _copy_checkpoint(best_path, directory / "checkpoints" / f"epoch_{stopping.best_epoch:04d}.pt")

            context.on_primary(inherit_best)
            stage = "restore_best_predictions"
            if diagnostics is not None:
                diagnostics.mark(stage)
            # Even if no later epoch improves, final best predictions/artifacts remain usable.
            if stopping.best_epoch == resume_payload["epoch"]:
                best_model = model.module
            else:
                from eye_qing_guang.contracts.checkpoints import load_checkpoint

                best_model, _ = load_checkpoint(best_path, context.device)
            best_predictions, _ = distributed_predictions(best_model, loaders["valid"], records["valid"], criterion, context, diagnostics)
            del best_model
            del resume_payload
        topology = {"world_size": context.world_size, "backend": dist.get_backend(),
                    "visible_physical_devices": "4,5,6,7", "local_rank_to_physical": [4, 5, 6, 7],
                    "per_rank_batch": settings["batch_size"], "accumulation_steps": settings["accumulation_steps"],
                    "precision": training_precision(settings), "loss_scaling": scaler.is_enabled(),
                    "global_effective_batch": context.world_size * settings["batch_size"] * settings["accumulation_steps"],
                    "batch_norm": "ordinary_per_rank; rank0_buffers_broadcast_before_evaluation",
                    "workers_per_rank": config["data"]["num_workers"],
                    "total_workers": context.world_size * config["data"]["num_workers"]}

        def state():
            return {"scheduler": schedule.state_dict(), "scaler": scaler.state_dict(),
                    "early_stopping": stopping.state_dict(), "progress": global_progress(progress, context.world_size),
                    "distributed": topology, "resume_supported": True, "checkpoint_boundary": "completed_epoch",
                    "resume": resume_info, "exact_replay": False}

        def initialize_outputs():
            nonlocal handle, writer
            write_json(directory / "run_started.json", run_manifest(
                "running", protocol="b0_dev_ddp_v1", provenance=provenance,
                environment={**environment(root, context.device), "resume_supported": True,
                             "resume_scope": "completed_epoch_non_bitwise"},
                distributed=topology, expected_groups_per_epoch=groups, schedule=schedule.state_dict(),
                per_rank_microbatches_per_epoch=len(loaders["train"]), pos_weight=1.0, train_amp=settings["amp"],
                train_precision=training_precision(settings),
                resume=resume_info, start_epoch=start_epoch, resume_supported=True,
                parameter_count=sum(p.numel() for p in model.parameters()),
                inference_tensor_bytes=sum(t.numel()*t.element_size() for t in model.module.state_dict().values())))
            handle = (directory / "reports" / "updates.csv").open("x", encoding="utf-8", newline="")
            fields = ["epoch", "attempted_group", "attempted_update_index", "successful_update_index", "group_samples",
                      "success", "lr", "scale_before", "scale_after", "grad_norm", "consumed_samples", "successful_samples", "skipped_groups"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            print(f"Four-card B0-dev ready: {directory}; {topology}; official valid/test excluded", flush=True)
            if resume_info:
                print(f"Resumed epoch {inherited_epochs}; next epoch={start_epoch}; successful_updates={schedule.successful_updates}; "
                      f"next_lr={schedule.value(schedule.successful_updates + 1):.12g}; exact_replay=False", flush=True)
                print(f"Resume batch/schedule: {resume_info['batch_transition']}", flush=True)

        writer = None
        context.on_primary(initialize_outputs)
        context.reset_peak_memory()
        for epoch in range(start_epoch, settings["epochs"] + 1):
            loaders["train"].sampler.set_epoch(epoch)
            stage = f"epoch_{epoch}_training"
            formal_training_started = True
            if epoch == start_epoch and context.primary:
                print("Full four-card development training started", flush=True)
            if diagnostics is not None:
                diagnostics.mark(stage, epoch=epoch)
            epoch_started = time.perf_counter()

            def report_group(row):
                if not context.primary:
                    return
                row = dict(row)
                for key in ("group_samples", "consumed_samples", "successful_samples"):
                    row[key] *= context.world_size
                writer.writerow({"epoch": epoch, **row})
                handle.flush()
                if row["attempted_group"] % options["log_every_n_groups"] == 0 or (row["attempted_group"] - 1) % groups == 0:
                    epoch_samples = row["consumed_samples"] - (epoch - 1) * len(records["train"])
                    elapsed = time.perf_counter() - epoch_started
                    print(f"Epoch {epoch}: samples={epoch_samples}/{len(records['train'])}, "
                          f"global_updates={row['successful_update_index']}, skips={row['skipped_groups']}, "
                          f"lr={row['lr']:.8g}, elapsed={elapsed:.1f}s, consumed/s={epoch_samples/max(elapsed, 1e-9):.2f}", flush=True)

            local = train_accumulated_epoch(model, loaders["train"], optimizer, criterion, context.device, settings,
                                             scaler, schedule, progress, report_group, context, diagnostics=diagnostics)
            if diagnostics is not None:
                diagnostics.mark("epoch_metrics_gather")
            train_metrics = global_epoch_metrics(local, context)
            if train_metrics["consumed_samples"] != len(records["train"]):
                raise RuntimeError("Global training sample count differs from frozen manifest.")
            stage = f"epoch_{epoch}_internal_development"
            if diagnostics is not None:
                diagnostics.mark(stage)
            validation_started = time.perf_counter()
            predictions, loss = distributed_predictions(model.module, loaders["valid"], records["valid"], criterion, context, diagnostics)
            validation_seconds = context.reduce([time.perf_counter() - validation_started], dist.ReduceOp.MAX)[0]

            def select():
                metrics = prediction_metrics(predictions)
                improved, stop = stopping.update(epoch, metrics["roc_auc"])
                return {"metrics": metrics, "improved": improved, "stop": stop, "stopping": stopping.state_dict()}

            decision = context.on_primary(select)
            stopping = DevelopmentStopping(**decision["stopping"])
            metrics = decision["metrics"]
            peaks = context.gather_json(context.peak_memory())
            stage = f"epoch_{epoch}_saving"
            if diagnostics is not None:
                diagnostics.mark(stage, memory=True)

            def save_epoch():
                nonlocal best_predictions
                saving_started = time.perf_counter()
                if decision["improved"]:
                    best_predictions = predictions
                # Keep every completed epoch resumable; best selection remains AUROC-only.
                save_checkpoint(directory / "checkpoints" / f"epoch_{epoch:04d}.pt", model, optimizer,
                                epoch, config, None, 1.0, metrics, state())
                result = {"epoch": epoch, **train_metrics, "valid_loss": loss, **metrics,
                          "validation_and_gather_seconds": validation_seconds,
                          "checkpoint_save_seconds": time.perf_counter() - saving_started,
                          "best_epoch": stopping.best_epoch, "reference_auc": stopping.reference_auc, "stale": stopping.stale,
                          "per_rank_peak_memory": peaks, "peak_allocated_bytes": max(p["allocated_bytes"] for p in peaks),
                          "peak_reserved_bytes": max(p["reserved_bytes"] for p in peaks)}
                history.append(result)
                return result

            context.on_primary(save_epoch)
            # Includes checkpoint writing and selection; final report write is separately measured below.
            wall_seconds = context.reduce([time.perf_counter() - epoch_started], dist.ReduceOp.MAX)[0]

            def report_epoch():
                report_started = time.perf_counter()
                history[-1]["epoch_seconds"] = wall_seconds
                write_json(directory / "reports" / f"epoch_{epoch:04d}.json", history[-1])
                print(f"Epoch {epoch}: BCE={train_metrics['train_loss']:.6f}, dev_BCE={loss:.6f}, "
                      f"AUROC={metrics['roc_auc']:.8f}, AP={metrics['average_precision']:.8f}, "
                      f"updates={train_metrics['successful_updates']}, train={train_metrics['train_seconds']:.1f}s, "
                      f"throughput={train_metrics['consumed_samples_per_second']:.2f}/s", flush=True)
                return time.perf_counter() - report_started

            report_seconds = context.on_primary(report_epoch)
            complete_seconds = context.reduce([time.perf_counter() - epoch_started], dist.ReduceOp.MAX)[0]

            def persist_history():
                nonlocal history_handle, history_writer
                history[-1]["report_write_seconds"] = report_seconds
                history[-1]["complete_epoch_seconds"] = complete_seconds
                if history_handle is None:
                    history_handle = (directory / "reports" / "history.csv").open("x", encoding="utf-8", newline="")
                    history_writer = csv.DictWriter(history_handle, fieldnames=list(history[-1]))
                    history_writer.writeheader()
                history_writer.writerow(history[-1])
                history_handle.flush()

            context.on_primary(persist_history)
            if decision["stop"]:
                break
        stage = "saving_final_artifacts"
        if diagnostics is not None:
            diagnostics.mark(stage)

        def finalize():
            from eye_qing_guang.pipelines.training import _copy_checkpoint

            save_checkpoint(directory / "checkpoints" / "last.pt", model, optimizer, epoch, config, None, 1.0, metrics, state())
            _copy_checkpoint(directory / "checkpoints" / f"epoch_{stopping.best_epoch:04d}.pt", directory / "checkpoints" / "best.pt")
            payload = torch.load(directory / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
            for key in ("optimizer_state", "training_state"):
                payload.pop(key)
            with (directory / "checkpoints" / "best_inference.pt").open("xb") as destination:
                torch.save(payload, destination)
            write_json(directory / "reports" / "metrics.json", {**prediction_metrics(best_predictions),
                "best_epoch": stopping.best_epoch, "selection_source": "internal_development", "completed_epochs": epoch,
                "convergence": "not_automatically_certified; inspect curves"})
            write_csv(directory / "predictions" / "dev_best.csv", best_predictions, ["image_id", "label", "logit", "probability"])

        context.on_primary(finalize)
        # CPU image-proxy analysis can take longer than a collective timeout.
        # No rank needs the communicator afterwards; release it before root-only analysis.
        close_distributed()
        if diagnostics is not None:
            diagnostics.mark("training_device_work_complete")
            diagnostics.close()
            diagnostics = None
        stage = "development_error_analysis"
        if context.primary:
            write_development_analysis(directory, records["train"], records["valid"], best_predictions, root, config)
        status, stage = "completed", "completed"
        if context.primary:
            print(f"Four-card training completed: {directory}", flush=True)
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        if diagnostics is not None:
            diagnostics.capture_failure()
        failure_record(directory, context, stage, progress, error, formal_training_started)
        raise
    finally:
        if diagnostics is not None:
            diagnostics.close()
        if handle is not None:
            handle.close()
        if history_handle is not None:
            history_handle.close()
        try:
            if directory is not None and context.primary:
                write_json(directory / "run_manifest.json", run_manifest(
                    status, protocol="b0_dev_ddp_v1", stage=stage, error=error, completed_epochs=inherited_epochs + len(history),
                    newly_completed_epochs=len(history), resume=resume_info,
                    best_epoch=stopping.best_epoch, progress=global_progress(progress, context.world_size),
                    formal_training_started=formal_training_started, elapsed_seconds=time.perf_counter()-started,
                    progress_caveat="failure may interrupt ranks between groups; see per-rank failure records",
                    resume_supported=True, exact_replay=False))
        finally:
            close_distributed()
    return directory

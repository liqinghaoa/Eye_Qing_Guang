"""Explicitly budgeted four-rank numerical smoke check, never formal training."""

import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from eye_qing_guang.contracts.checkpoints import load_checkpoint, save_checkpoint
from eye_qing_guang.contracts.precision import training_precision
from eye_qing_guang.datasets.development import load_development_manifests
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.pipelines.b0_development import environment, new_progress
from eye_qing_guang.pipelines.b0_distributed import failure_record, global_epoch_metrics, global_progress
from eye_qing_guang.pipelines.preflight import ReplayLoader, clipped_gradients, compare_precision, cpu_state
from eye_qing_guang.trainers.binary import check_runtime_precision, make_training_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import SuccessfulUpdateSchedule
from eye_qing_guang.utils.distributed import close_distributed, initialize_distributed, wrap_model
from eye_qing_guang.utils.experiment_io import new_experiment, run_manifest, write_json, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import seed_everything


def validate_budget(config, max_samples, max_groups):
    if type(max_samples) is not int or type(max_groups) is not int or min(max_samples, max_groups) <= 0:
        raise ValueError("Preflight sample-instance and global-group limits must be positive integers.")
    capacity = config["distributed"]["world_size"] * config["training"]["batch_size"] * config["training"]["accumulation_steps"]
    required_groups = 1 if training_precision(config["training"]) == "fp32" else 2
    if capacity * required_groups > max_samples or required_groups > max_groups:
        raise ValueError(f"One configured global group requires {capacity} instances. This precision check requires "
                         f"{capacity * required_groups} instances/{required_groups} groups, exceeding the explicit "
                         f"{max_samples}/{max_groups} budget. No compute or automatic budget extension was started.")
    return capacity


def distributed_fp32_reference(model, replay, criterion, context, settings, progress):
    """Independent weighted backward on identical local microbatches; no optimizer step."""
    model.train()
    model.zero_grad(set_to_none=True)
    local_samples = len(replay.dataset)
    progress["attempted_groups"] += 1
    progress["reference_groups"] = progress.get("reference_groups", 0) + 1
    total = 0.0
    for index, batch in enumerate(replay):
        size = len(batch["image"])
        progress["consumed_samples"] += size
        progress["reference_samples"] = progress.get("reference_samples", 0) + size
        images, labels, ages = (batch[key].to(context.device).float() for key in ("image", "label", "age"))
        if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
            raise FloatingPointError("Non-finite DDP FP32 reference inputs.")
        synchronization = model.no_sync() if index < len(replay) - 1 else nullcontext()
        with synchronization:
            with torch.autocast(device_type=context.device.type, enabled=False):
                logits = model(images, ages).float()
                loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not math.isfinite(loss.item()):
                raise FloatingPointError("Non-finite DDP FP32 reference logits/loss.")
            (loss * size / local_samples).backward()
        total += loss.item() * size
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    if not context.all_true(finite):
        raise FloatingPointError("Non-finite/absent DDP FP32 reference gradients.")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"]))
    if not context.all_true(math.isfinite(norm)):
        raise FloatingPointError("Non-finite DDP reference gradient norm.")
    loss_sum, count = context.reduce([total, local_samples])
    return {"loss": loss_sum / count, "gradients": clipped_gradients(model)}


def preflight_distributed(config, root, name, max_samples=64, max_groups=4):
    capacity = validate_budget(config, max_samples, max_groups)
    precision = training_precision(config["training"])
    mixed = precision != "fp32"
    if name == config["experiment"]["name"]:
        raise ValueError("Preflight must not occupy the formal experiment name.")
    context = initialize_distributed(config)
    directory = None
    progress = new_progress()
    stage, status, error = "initializing", "failed", None
    result = {"preflight_protocol": "fixed_precision_reference_first_v2", "training_precision": precision,
              "precision_comparison": {"status": "not_run"}, "update_attempts": [], "stage_seconds": {},
              "planned_sample_instances": capacity * (2 if mixed else 1), "planned_groups": 2 if mixed else 1}
    started = time.perf_counter()

    @contextmanager
    def phase(label):
        nonlocal stage
        stage = label
        phase_started = time.perf_counter()
        if context.primary:
            print(f"Preflight [{precision}] {label}: started", flush=True)
        yield
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        # Device communication waits for prior device work; report the slowest rank.
        seconds = context.reduce([time.perf_counter() - phase_started], torch.distributed.ReduceOp.MAX)[0]
        result["stage_seconds"][label] = seconds
        if context.primary:
            print(f"Preflight [{precision}] {label}: done ({seconds:.2f}s)", flush=True)

    try:
        settings = config["training"]
        with phase("output_initialization"):
            directory = Path(context.on_primary(lambda: str(new_experiment(resolve_path(config["experiment"]["output_root"], root), name))))
            context.on_primary(lambda: write_yaml(directory / "config_snapshot.yaml", config))
            context.on_primary(lambda: write_json(directory / "run_started.json", run_manifest(
                "running", purpose="four_rank_preflight_only", formal_training_started=False,
                environment=environment(root, context.device), training_precision=precision,
                preflight_protocol=result["preflight_protocol"],
                max_global_sample_instances=max_samples, max_global_groups=max_groups, world_size=context.world_size,
                backend=torch.distributed.get_backend(), configured_global_group_samples=capacity,
                planned_sample_instances=result["planned_sample_instances"], planned_groups=result["planned_groups"],
                fp32_reference_counts_toward_budget=True)))
        with phase("precision_operator_check"):
            result["runtime_precision_check"] = check_runtime_precision(settings, context.device)
        with phase("frozen_manifest_validation"):
            records, provenance = load_development_manifests(config, root)
            result["provenance"] = provenance
        fit = records["train"]
        if len(fit) % context.world_size:
            raise ValueError("Frozen fitting sample count is not divisible by four.")
        selected = [next(row for row in fit if row["label"] == label) for label in ("0", "1")]
        selected_ids = {row["image_id"] for row in selected}
        selected += [row for row in fit if row["image_id"] not in selected_ids][:capacity-2]
        if len(selected) < capacity:
            raise ValueError("Not enough distinct fitting images for one complete global debug group.")
        rows = selected[:capacity]
        with phase("model_initialization"):
            seed_everything(config["experiment"]["seed"])
            model = wrap_model(build_model(config["model"]).to(context.device), context)
            seed_everything(config["experiment"]["seed"] + context.rank)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                                         betas=tuple(settings["betas"]), eps=settings["eps"])
            criterion = build_loss(settings, 1.0).to(context.device)
            scaler = make_training_scaler(settings)
            groups = math.ceil(math.ceil((len(fit) // context.world_size) / settings["batch_size"]) / settings["accumulation_steps"])
            schedule = SuccessfulUpdateSchedule(optimizer, groups, settings)
        with phase("load_and_transform_one_group"):
            loader = make_loader(rows, config, root, training=True, device=context.device, distributed=context)
            replay = ReplayLoader(list(loader), settings["batch_size"])
        with phase("snapshot_initial_state"):
            initial = cpu_state(model.module)

        # Reference first: normal FP16 calibration can never consume its allowance.
        # Both paths restore identical initial parameters/BN and replay identical tensors.
        reference = None
        if mixed:
            with phase("fp32_reference"):
                context.reset_peak_memory()
                reference = distributed_fp32_reference(model, replay, criterion, context, settings, progress)
                result["fp32_reference"] = {"status": "passed", "loss": reference["loss"]}
                result["fp32_reference_peak_memory_per_rank"] = context.gather_json(context.peak_memory())
                model.zero_grad(set_to_none=True)
                model.module.load_state_dict(initial)
                seed_everything(config["experiment"]["seed"] + context.rank)
        else:
            result["precision_comparison"] = {"status": "not_applicable", "reason": "fp32_baseline_has_no_mixed_precision_path"}

        target_gradients = {}

        def record(row):
            if mixed and row["success"]:
                target_gradients.update(clipped_gradients(model))
            if context.primary:
                result["update_attempts"].append({**row, "group_samples": row["group_samples"]*context.world_size,
                                                  "consumed_samples": row["consumed_samples"]*context.world_size,
                                                  "successful_samples": row["successful_samples"]*context.world_size})

        with phase(f"{precision}_update"):
            context.reset_peak_memory()
            local = train_accumulated_epoch(model, replay, optimizer, criterion, context.device, settings,
                                             scaler, schedule, progress, record, context)
            metrics = global_epoch_metrics(local, context)
            result["training_peak_memory_per_rank"] = context.gather_json(context.peak_memory())
            if mixed:
                result["amp_peak_memory_per_rank"] = result["training_peak_memory_per_rank"]
        if not metrics["successful_updates"]:
            status = "inconclusive"
            result["precision_comparison"] = {"status": "inconclusive", "reason": "fp16_gradient_overflow_after_completed_fp32_reference",
                                              "next_action": "Use the explicit BF16 or FP32 configuration; no automatic retries or budget increases."}
        else:
            result["successful_training_group"] = metrics
            if mixed:
                result["successful_amp_group"] = metrics
                with phase("precision_comparison"):
                    comparison = compare_precision(metrics["train_loss"], target_gradients, reference)
                    result["precision_comparison"] = comparison
                    if not context.all_true(comparison["status"] == "passed"):
                        raise FloatingPointError("Four-rank mixed precision/FP32 discrepancy exceeds frozen preflight tolerances.")
            else:
                result["fp32_baseline"] = "passed"
            with phase("parameter_consistency"):
                changed = any(not torch.equal(initial[key], parameter.detach().cpu()) for key, parameter in model.module.named_parameters())
                if not context.all_true(changed):
                    raise RuntimeError("Successful DDP update did not change parameters on every rank.")
                for parameter in model.module.parameters():
                    source = parameter.detach().clone()
                    torch.distributed.broadcast(source, src=0)
                    if not context.all_true(torch.equal(parameter.detach(), source)):
                        raise RuntimeError("DDP parameters differ after update.")
                result["parameter_consistency"] = "passed"
            with phase("checkpoint_save"):
                context.synchronize_buffers(model.module)
                context.on_primary(lambda: save_checkpoint(directory / "checkpoints" / "preflight.pt", model, optimizer, 0,
                                                           config, None, 1.0, {}, {"progress": global_progress(progress, context.world_size),
                                                           "training_precision": precision, "world_size": context.world_size, "resume_supported": False}))

            def check_load():
                loaded, _ = load_checkpoint(directory / "checkpoints" / "preflight.pt", context.device)
                # A tiny deterministic evaluation does not need another spawned worker pool.
                evaluation_config = {**config, "data": {**config["data"], "num_workers": 0}}
                evaluation_loader = make_loader(rows[:settings["eval_batch_size"]], evaluation_config, root, device=context.device)
                before = cpu_state(loaded)
                expected, _ = predict_batches(model.module, evaluation_loader, context.device)
                actual, _ = predict_batches(loaded, evaluation_loader, context.device)
                repeated, _ = predict_batches(loaded, evaluation_loader, context.device)
                if expected != actual or actual != repeated or any(not torch.equal(before[key], value.cpu()) for key, value in loaded.state_dict().items()):
                    raise RuntimeError("DDP checkpoint roundtrip/repeated FP32 eval or buffers changed.")
                result["checkpoint_single_process_load"] = "passed"

            with phase("single_process_checkpoint_load"):
                context.on_primary(check_load)
            status = "completed"
        stage = "completed" if status == "completed" else "precision_inconclusive"
        if progress["consumed_samples"] * context.world_size > max_samples or progress["attempted_groups"] > max_groups:
            raise RuntimeError("Global DDP preflight budget exceeded.")
        if context.primary:
            print(f"Four-card preflight {status}: {directory}; global_progress={global_progress(progress, context.world_size)}", flush=True)
    except BaseException as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure_record(directory, context, stage, progress, error, False)
        raise
    finally:
        try:
            if directory is not None and context.primary:
                summary = global_progress(progress, context.world_size)
                summary["reference_samples"] = progress.get("reference_samples", 0)*context.world_size
                result.update(status=status, progress=summary, max_global_sample_instances=max_samples,
                              max_global_groups=max_groups, formal_training_started=False, elapsed_seconds=time.perf_counter()-started,
                              throughput_scope="cached transformed debug tensors, not online training performance")
                write_json(directory / "reports" / "preflight.json", result)
                write_json(directory / "run_manifest.json", run_manifest(status, purpose="four_rank_preflight_only", stage=stage,
                           error=error, progress=summary, training_precision=precision, formal_training_started=False))
        finally:
            close_distributed()
    return directory

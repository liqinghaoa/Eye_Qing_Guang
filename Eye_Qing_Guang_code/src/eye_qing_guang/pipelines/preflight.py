"""Bounded fitting-only B0 numerical/resource preflight, never formal training."""

import math
import time

import torch
from PIL import Image, ImageDraw

from eye_qing_guang.contracts.checkpoints import load_checkpoint, save_checkpoint
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.datasets.development import load_development_manifests
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.transforms import fov_crop, letterbox
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.pipelines.b0_development import environment, new_progress
from eye_qing_guang.trainers.binary import make_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import SuccessfulUpdateSchedule
from eye_qing_guang.utils.experiment_io import new_experiment, run_manifest, write_csv, write_json, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import choose_device, seed_everything

MAX_SAMPLES = 64
MAX_GROUPS = 4


def budget_available(progress, samples):
    return (samples > 0 and progress["consumed_samples"] + samples <= MAX_SAMPLES
            and progress["attempted_groups"] < MAX_GROUPS)


def cpu_state(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def clipped_gradients(model):
    return {name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.grad is not None}


class ReplayLoader:
    """Replay transformed CPU tensors so AMP/FP32 see identical physical batches."""

    def __init__(self, batches, batch_size):
        self.batches = batches
        self.batch_size = batch_size
        self.dataset = range(sum(len(batch["image"]) for batch in batches))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def fp32_reference(model, loader, criterion, device, settings, progress):
    """Independent sample-weighted FP32 backward, with the SAME microbatches/BN."""
    samples = len(loader.dataset)
    if not budget_available(progress, samples):
        raise RuntimeError("FP32 reference would exceed the frozen preflight budget.")
    progress["attempted_groups"] += 1
    progress["reference_groups"] = progress.get("reference_groups", 0) + 1
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in loader:
        size = len(batch["image"])
        progress["consumed_samples"] += size
        progress["reference_samples"] = progress.get("reference_samples", 0) + size
        images, labels, ages = (batch[key].to(device).float() for key in ("image", "label", "age"))
        if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
            raise FloatingPointError("Non-finite FP32 reference inputs.")
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(images, ages).float()
            loss = criterion(logits, labels)
        if not torch.isfinite(logits).all() or not math.isfinite(loss.item()):
            raise FloatingPointError("Non-finite FP32 reference logits/BCE.")
        (loss * size / samples).backward()
        total_loss += loss.item() * size
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()):
        raise FloatingPointError("Non-finite or absent FP32 reference gradients.")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"]).item())
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite FP32 reference gradient norm.")
    return {"loss": total_loss / samples, "preclip_grad_norm": norm,
            "gradients": clipped_gradients(model)}


def compare_precision(amp_loss, amp_gradients, reference):
    """Visible smoke-test tolerances, not a tuned accuracy or convergence claim."""
    if set(amp_gradients) != set(reference["gradients"]):
        raise RuntimeError("AMP/FP32 gradient parameter sets differ.")
    difference_squared = reference_squared = 0.0
    for name, expected in reference["gradients"].items():
        actual = amp_gradients[name].double()
        expected = expected.double()
        difference_squared += float((actual - expected).square().sum())
        reference_squared += float(expected.square().sum())
    difference, reference_norm = math.sqrt(difference_squared), math.sqrt(reference_squared)
    loss_difference = abs(amp_loss - reference["loss"])
    loss_allowance = 0.005 + 0.02 * abs(reference["loss"])
    gradient_allowance = 0.005 + 0.1 * reference_norm
    return {"status": "passed" if loss_difference <= loss_allowance and difference <= gradient_allowance else "failed",
            "amp_loss": amp_loss, "fp32_loss": reference["loss"],
            "loss_abs_difference": loss_difference, "loss_abs_allowance": loss_allowance,
            "clipped_gradient_l2_difference": difference, "reference_clipped_gradient_l2": reference_norm,
            "clipped_gradient_relative_l2": difference / max(reference_norm, 1e-12),
            "clipped_gradient_l2_allowance": gradient_allowance,
            "tolerances": {"loss_atol": 0.005, "loss_rtol": 0.02, "gradient_l2_atol": 0.005, "gradient_l2_rtol": 0.1},
            "same_physical_microbatches": True, "same_initial_weights_and_bn": True,
            "reference_optimizer_step": False}


def preflight(config, root, name):
    validate_config(config)
    output_root = resolve_path(config["experiment"]["output_root"], root)
    if name == config["experiment"]["name"] or (output_root / name).exists():
        raise ValueError("Preflight needs a new, separate experiment directory.")
    device = choose_device(config["training"]["device"])
    records, provenance = load_development_manifests(config, root)
    settings = config["training"]
    group_capacity = settings["batch_size"] * settings["accumulation_steps"]
    if group_capacity > MAX_SAMPLES:
        raise ValueError("A full configured accumulation group exceeds the 64-instance preflight budget.")
    fit = records["train"]
    # Both classes in the first debug group; later attempts use fresh fitting IDs.
    selected = [next(row for row in fit if row["label"] == label) for label in ("0", "1")]
    selected_ids = {row["image_id"] for row in selected}
    selected += [row for row in fit if row["image_id"] not in selected_ids][:MAX_SAMPLES-2]
    if group_capacity < 2 or len(selected) < group_capacity:
        raise ValueError("Preflight requires both classes and a full configured accumulation group.")
    seed_everything(config["experiment"]["seed"])
    directory = new_experiment(output_root, name)
    progress = new_progress()
    status, error, stage = "failed", None, "initializing"
    result = {"precision_comparison": {"status": "not_run"}, "update_attempts": []}
    used_rows = []
    try:
        write_yaml(directory / "config_snapshot.yaml", config)
        write_json(directory / "run_started.json", run_manifest(
            "running", purpose="bounded_preflight_not_performance", provenance=provenance,
            environment=environment(root, device), max_sample_instances=MAX_SAMPLES,
            max_groups=MAX_GROUPS, reference_backward_counts_toward_budget=True,
        ))
        previews = []
        for index, row in enumerate(selected[:12]):
            with Image.open(resolve_path(row["image_path"], root)) as source:
                image = source.convert("RGB")
            cropped, info = fov_crop(image, config["data"]["preprocessing"])
            comparison = Image.new("RGB", (512, 280), "white")
            comparison.paste(letterbox(image, 256), (0, 24))
            comparison.paste(letterbox(cropped, 256), (256, 24))
            ImageDraw.Draw(comparison).text((4, 5), f"{row['image_id']} | original / FOV", fill="black")
            with (directory / "figures" / f"fov_{index:02d}.png").open("xb") as handle:
                comparison.save(handle, format="PNG")
            previews.append({"image_id": row["image_id"], **info})
        write_json(directory / "reports" / "fov_previews.json", previews)
        stage = "model_initialization"
        model = build_model(config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                     weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]), eps=settings["eps"])
        criterion = build_loss(settings, 1.0).to(device)
        scaler = make_scaler(settings["amp"])
        full_groups = math.ceil(math.ceil(len(fit) / settings["batch_size"]) / settings["accumulation_steps"])
        schedule = SuccessfulUpdateSchedule(optimizer, full_groups, settings)
        torch.cuda.reset_peak_memory_stats(device)
        offset = 0
        success = False
        while budget_available(progress, group_capacity) and offset + group_capacity <= len(selected):
            rows = selected[offset:offset+group_capacity]
            offset += group_capacity
            used_rows.extend(rows)
            loader = make_loader(rows, config, root, training=True, device=device)
            replay = ReplayLoader(list(loader), settings["batch_size"])
            initial = cpu_state(model)
            amp_gradients = {}

            def record_group(row):
                result["update_attempts"].append({**row, "image_ids": [entry["image_id"] for entry in rows]})
                if row["success"]:
                    amp_gradients.update(clipped_gradients(model))

            stage = "bounded_amp_update_group"
            train_metrics = train_accumulated_epoch(model, replay, optimizer, criterion, device, settings,
                                                     scaler, schedule, progress, record_group)
            if train_metrics["successful_updates"] == 0:
                # Normal scale backoff: no data replay and no recipe modification.
                continue
            success = True
            if not any(not torch.equal(initial[key], parameter.detach().cpu())
                       for key, parameter in model.named_parameters()):
                raise RuntimeError("A reported successful update did not change model parameters.")
            result["successful_amp_group"] = train_metrics
            result["amp_training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            result["amp_training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            stage = "checkpoint_before_fp32_reference"
            save_checkpoint(directory / "checkpoints" / "preflight.pt", model, optimizer, 0, config,
                            None, 1.0, {}, {"scheduler": schedule.state_dict(), "scaler": scaler.state_dict(),
                                           "progress": dict(progress), "resume_supported": False})
            if budget_available(progress, len(replay.dataset)):
                # Reuse one GPU model and release optimizer tensors, so a second
                # model/optimizer does not inflate the FP32 reference memory test.
                amp_state = cpu_state(model)
                optimizer.state.clear()
                stage = "fp32_sample_weighted_reference"
                try:
                    model.load_state_dict(initial)
                    reference = fp32_reference(model, replay, criterion, device, settings, progress)
                    result["precision_comparison"] = compare_precision(train_metrics["train_loss"], amp_gradients, reference)
                finally:
                    model.zero_grad(set_to_none=True)
                    model.load_state_dict(amp_state)
                if result["precision_comparison"]["status"] != "passed":
                    raise FloatingPointError("AMP/FP32 loss or clipped-gradient discrepancy exceeds recorded smoke-test tolerances.")
            else:
                result["precision_comparison"] = {"status": "inconclusive",
                                                  "reason": "budget_exhausted_before_fp32_reference"}
            break
        if progress["consumed_samples"] > MAX_SAMPLES or progress["attempted_groups"] > MAX_GROUPS:
            raise RuntimeError("Preflight budget invariant violated.")
        if not success:
            status = "inconclusive"
            result["precision_comparison"] = {"status": "inconclusive",
                                              "reason": "budget_or_fresh_samples_exhausted_before_successful_update"}
        else:
            stage = "checkpoint_and_deterministic_evaluation"
            reloaded, _ = load_checkpoint(directory / "checkpoints" / "preflight.pt", device)
            evaluation_rows = used_rows[:min(len(used_rows), settings["eval_batch_size"])]
            evaluation_loader = make_loader(evaluation_rows, config, root, device=device)
            original_predictions, _ = predict_batches(model, evaluation_loader, device, criterion)
            buffers = {key: tensor.clone() for key, tensor in reloaded.named_buffers()}
            predictions, _ = predict_batches(reloaded, evaluation_loader, device, criterion)
            repeated, _ = predict_batches(reloaded, evaluation_loader, device, criterion)
            if original_predictions != predictions or predictions != repeated:
                raise RuntimeError("Checkpoint roundtrip or repeated eval predictions changed.")
            if any(not torch.equal(buffers[key], tensor) for key, tensor in reloaded.named_buffers()):
                raise RuntimeError("Evaluation changed BatchNorm buffers.")
            write_csv(directory / "predictions" / "fit_debug.csv", predictions)
            one_loader = make_loader(used_rows[:1], config, root, device=device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            predict_batches(reloaded, one_loader, device)
            torch.cuda.synchronize(device)
            result["single_path_unwarmed_seconds"] = time.perf_counter() - start
            result["evaluation_batch_checked"] = len(evaluation_rows)
            status = "completed" if result["precision_comparison"]["status"] == "passed" else "inconclusive"
        stage = "completed" if status == "completed" else "budget_inconclusive"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"Preflight failed; formal training not started; stage={stage}; progress={progress}; outputs={directory}", flush=True)
        raise
    finally:
        result.update(status=status, progress=dict(progress), fitting_preview_count=len(previews) if "previews" in locals() else 0,
                      requires_manual_fov_review=True, formal_timing_or_convergence_certified=False)
        write_json(directory / "reports" / "preflight.json", result)
        write_json(directory / "run_manifest.json", run_manifest(
            status, stage=stage, purpose="preflight_only", formal_training_started=False,
            progress=progress, error=error, outputs=str(directory),
        ))
    print(f"Preflight {status}: {directory}; progress={progress}", flush=True)
    return directory

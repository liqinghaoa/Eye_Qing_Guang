"""One binary-classification training epoch."""

import math
import time
from contextlib import nullcontext

import torch

from eye_qing_guang.contracts.precision import training_precision


def make_scaler(enabled):
    # The fixed standard single-optimizer scaler protocol also supports torch 2.1.
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=65536,
                                    growth_factor=2, backoff_factor=0.5, growth_interval=2000)
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=65536,
                                    growth_factor=2, backoff_factor=0.5, growth_interval=2000)


def make_training_scaler(settings):
    # BF16 uses autocast without loss scaling; amp=true alone must not enable it.
    return make_scaler(training_precision(settings) == "fp16")


def training_autocast(settings, device):
    precision = training_precision(settings)
    return torch.autocast(device_type=device.type,
                          dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                          enabled=precision != "fp32" and device.type == "cuda")


def check_runtime_precision(settings, device):
    """Probe actual BF16 convolution/backward; a device capability flag is insufficient."""
    precision = training_precision(settings)
    if precision != "bf16":
        return {"precision": precision, "bf16_probe": "not_required"}
    if device.type != "cuda":
        raise RuntimeError("BF16 production training requires the assigned CUDA-compatible PPU devices.")
    probe = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
    with training_autocast(settings, device):
        output = probe(torch.ones(2, 3, 8, 8, device=device))
    if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all()):
        raise RuntimeError("PPU BF16 convolution probe did not produce finite BF16 output; use the explicit FP32 config.")
    output.float().square().mean().backward()
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in probe.parameters()):
        raise FloatingPointError("PPU BF16 backward probe has non-finite or missing gradients.")
    torch.cuda.synchronize(device)
    return {"precision": precision, "bf16_probe": "passed"}


def finish_update(model, optimizer, scaler, clip, distributed=None):
    """Unscale once; let GradScaler handle element overflow, not the clipper."""
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    has_gradients = bool(gradients)
    if distributed is not None:
        has_gradients = distributed.all_true(has_gradients)
    if not has_gradients:
        raise RuntimeError("No model gradients were produced.")
    finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())
    before = float(scaler.get_scale())
    if distributed is not None:
        distributed.agree([int(finite), before, scaler.state_dict().get("_growth_tracker", 0)],
                          "unscaled gradient finiteness/scaler state before step")
    norm = None
    if finite:
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item())
        if distributed is not None and not distributed.all_true(math.isfinite(norm)):
            raise FloatingPointError("Non-finite global gradient norm on a DDP rank.")
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite global gradient norm.")
    elif not scaler.is_enabled():
        raise FloatingPointError("Non-finite unscaled FP32/BF16 gradient with loss scaling disabled.")
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    if distributed is not None:
        distributed.agree([int(after >= before), after, scaler.state_dict().get("_growth_tracker", 0)],
                          "step/skip and updated scaler state")
    return after >= before, norm, before, after


def train_accumulated_epoch(model, loader, optimizer, criterion, device, settings,
                            scaler, schedule, progress, report_group=None, distributed=None, diagnostics=None):
    """Consume natural batches once; normalize each group by its actual samples."""
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if diagnostics is not None:
        diagnostics.bind_loader(loader)
        diagnostics.mark("data_iterator_initialization")
    iterator = iter(loader)
    total_loss = 0.0
    consumed = successful_samples = attempted = successful = skipped = 0
    first_lr = last_lr = None
    max_norm = 0.0
    expected_samples = len(loader.sampler) if distributed is not None else len(loader.dataset)
    accumulation = settings["accumulation_steps"]
    precision = training_precision(settings)
    if precision in {"fp32", "bf16"} and scaler.is_enabled():
        raise ValueError("FP32/BF16 must not use an enabled GradScaler.")
    if precision == "fp16" and device.type == "cuda" and not scaler.is_enabled():
        raise ValueError("FP16 CUDA training requires an enabled GradScaler.")
    for group_start in range(0, len(loader), accumulation):
        microbatches = min(accumulation, len(loader) - group_start)
        group_samples = min(microbatches * loader.batch_size, expected_samples - consumed)
        optimizer.zero_grad(set_to_none=True)
        lr = schedule.prepare()
        update_index = schedule.successful_updates + 1
        progress["attempted_groups"] += 1
        attempted += 1
        if diagnostics is not None:
            diagnostics.mark("group_agreement", group=progress["attempted_groups"], successful_updates=schedule.successful_updates)
        if distributed is not None:
            distributed.agree([group_samples, schedule.successful_updates, progress["attempted_groups"]],
                              "group size/update index")
        for microbatch_index in range(microbatches):
            if diagnostics is not None:
                diagnostics.waiting_for_batch(consumed, loader.batch_size,
                                               microbatch=microbatch_index + 1, local_sample_offset=consumed)
            batch = next(iterator)
            if diagnostics is not None:
                diagnostics.mark("batch_received", actual_image_ids=list(batch.get("image_id", [])))
            # Count fetched sample instances before transfer/forward/backward:
            # any of these operations may fail after the batch was consumed.
            size = len(batch["image"])
            consumed += size
            progress["consumed_samples"] += size
            if diagnostics is not None:
                diagnostics.mark("input_transfer")
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
            ages = batch["age"].to(device, non_blocking=True)
            if diagnostics is not None:
                diagnostics.mark("input_finite_check")
            if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
                raise FloatingPointError("Non-finite training inputs or labels.")
            synchronization = (model.no_sync() if distributed is not None and microbatch_index < microbatches - 1
                               else nullcontext())
            with synchronization:
                if diagnostics is not None:
                    diagnostics.mark("forward", memory=True)
                with training_autocast(settings, device):
                    logits = model(images, ages)
                if diagnostics is not None:
                    diagnostics.mark("logit_finite_check")
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite training logits.")
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = criterion(logits.float(), labels)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError("Non-finite unscaled training BCE.")
                if diagnostics is not None:
                    diagnostics.mark("backward")
                scaler.scale(loss * size / group_samples).backward()
            total_loss += loss.item() * size
        if diagnostics is not None:
            diagnostics.mark("optimizer_step", memory=True)
        if distributed is None:
            success, norm, before, after = finish_update(model, optimizer, scaler, settings["grad_clip"])
        else:
            success, norm, before, after = finish_update(model, optimizer, scaler, settings["grad_clip"], distributed)
        schedule.record(success)
        if success:
            successful += 1
            successful_samples += group_samples
            progress["successful_samples"] += group_samples
            progress["consecutive_skips"] = 0
            first_lr = lr if first_lr is None else first_lr
            last_lr = lr
        else:
            skipped += 1
            progress["skipped_groups"] += 1
            progress["consecutive_skips"] += 1
        progress.update(successful_updates=schedule.successful_updates, scale_before=before,
                        scale_after=after, last_attempt_lr=lr, last_grad_norm=norm)
        if diagnostics is not None:
            diagnostics.mark("update_complete", memory=True, successful_updates=schedule.successful_updates,
                             local_consumed_samples=progress["consumed_samples"], success=success)
        if report_group is not None:
            report_group({"attempted_group": progress["attempted_groups"],
                          "attempted_update_index": update_index,
                          "successful_update_index": schedule.successful_updates,
                          "group_samples": group_samples, "success": success, "lr": lr,
                          "scale_before": before, "scale_after": after, "grad_norm": norm,
                          "consumed_samples": progress["consumed_samples"],
                          "successful_samples": progress["successful_samples"],
                          "skipped_groups": progress["skipped_groups"]})
        if norm is not None:
            max_norm = max(max_norm, norm)
        optimizer.zero_grad(set_to_none=True)
        if progress["consecutive_skips"] >= settings["max_consecutive_skips"]:
            raise FloatingPointError("Consecutive AMP overflow group limit reached; recipe was not changed.")
    if consumed != expected_samples:
        raise RuntimeError("Training loader failed to consume the frozen fitting sample count.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"train_loss": total_loss / consumed, "consumed_samples": consumed,
            "successful_samples": successful_samples, "attempted_groups": attempted,
            "successful_updates": successful, "skipped_groups": skipped,
            "global_successful_updates": schedule.successful_updates,
            "first_success_lr": first_lr, "last_success_lr": last_lr,
            "last_attempt_lr": progress["last_attempt_lr"], "max_grad_norm": max_norm,
            "scale_before": progress["scale_before"], "scale_after": progress["scale_after"],
            "train_seconds": elapsed, "consumed_samples_per_second": consumed / elapsed}


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        images, ages, labels = (batch[key].to(device) for key in ("image", "age", "label"))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images, ages), labels)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
        count += len(images)
    if not count:
        raise ValueError("Empty training loader.")
    return total / count

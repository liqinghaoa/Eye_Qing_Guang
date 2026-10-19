"""One binary-classification training epoch."""

import math
import time

import torch


def make_scaler(enabled):
    # The fixed standard single-optimizer scaler protocol also supports torch 2.1.
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=65536,
                                    growth_factor=2, backoff_factor=0.5, growth_interval=2000)


def finish_update(model, optimizer, scaler, clip):
    """Unscale once; let GradScaler handle element overflow, not the clipper."""
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("No model gradients were produced.")
    finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())
    norm = None
    if finite:
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item())
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite global gradient norm.")
    elif not scaler.is_enabled():
        raise FloatingPointError("Non-finite FP32 gradient.")
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    return after >= before, norm, before, after


def train_accumulated_epoch(model, loader, optimizer, criterion, device, settings,
                            scaler, schedule, progress, report_group=None):
    """Consume natural batches once; normalize each group by its actual samples."""
    model.train()
    iterator = iter(loader)
    total_loss = 0.0
    consumed = successful_samples = attempted = successful = skipped = 0
    first_lr = last_lr = None
    max_norm = 0.0
    started = time.perf_counter()
    accumulation = settings["accumulation_steps"]
    for group_start in range(0, len(loader), accumulation):
        microbatches = min(accumulation, len(loader) - group_start)
        group_samples = min(microbatches * loader.batch_size, len(loader.dataset) - consumed)
        optimizer.zero_grad(set_to_none=True)
        lr = schedule.prepare()
        update_index = schedule.successful_updates + 1
        progress["attempted_groups"] += 1
        attempted += 1
        for _ in range(microbatches):
            batch = next(iterator)
            # Count fetched sample instances before transfer/forward/backward:
            # any of these operations may fail after the batch was consumed.
            size = len(batch["image"])
            consumed += size
            progress["consumed_samples"] += size
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
            ages = batch["age"].to(device, non_blocking=True)
            if not torch.isfinite(images).all() or not torch.isfinite(labels).all():
                raise FloatingPointError("Non-finite training inputs or labels.")
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=scaler.is_enabled() and device.type == "cuda"):
                logits = model(images, ages)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite training logits.")
            with torch.autocast(device_type=device.type, enabled=False):
                loss = criterion(logits.float(), labels)
            if not math.isfinite(loss.item()):
                raise FloatingPointError("Non-finite unscaled training BCE.")
            # criterion is a per-sample mean; multiply by microbatch size before
            # dividing by the actual accumulation-group sample count (including tail).
            scaler.scale(loss * size / group_samples).backward()
            total_loss += loss.item() * size
        success, norm, before, after = finish_update(model, optimizer, scaler, settings["grad_clip"])
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
    if consumed != len(loader.dataset):
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

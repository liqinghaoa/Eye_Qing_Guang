"""Sample-correct accumulation on the project's four-rank native DDP runtime."""

import math
import time
from dataclasses import dataclass
from itertools import islice
from contextlib import nullcontext

import torch

from eye_qing_guang.trainers.schedules import SuccessfulUpdateSchedule


def build_optimizer_schedule(model, settings, groups_per_epoch):
    if settings["optimizer"] != "adamw" or settings["optimizer_parameter_groups"] != "all_trainable":
        raise ValueError("This recipe requires AdamW over all trainable parameters.")
    if settings["scheduler"] != "warmup_cosine_success_v1":
        raise ValueError("This recipe requires the successful-update warmup/cosine schedule.")
    parameters = list(model.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError("Encoder and head must both be fully trainable.")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                                  betas=tuple(settings["betas"]), eps=settings["eps"])
    return optimizer, SuccessfulUpdateSchedule(optimizer, groups_per_epoch, settings)


def pw5_loss(device):
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], dtype=torch.float32, device=device), reduction="mean")


def autocast_context(device, precision):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision != "bf16":
        raise ValueError("This BF16 protocol does not use FP16 or GradScaler.")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def update_group(model, group, optimizer, schedule, device, settings, runtime, *, step=True, precision=None):
    """Normalize by true local sample count, then apply rank MEAN exactly once."""
    count = sum(len(batch["image"]) for batch in group)
    if count <= 0:
        raise ValueError("Empty accumulation group.")
    optimizer.zero_grad(set_to_none=True)
    criterion = pw5_loss(device)
    weighted_sum = 0.0
    for batch_index, batch in enumerate(group):
        def forward_backward():
            images, labels = batch["image"].to(device), batch["label"].to(device)
            if labels.dtype != torch.float32 or labels.shape != (len(images), 1):
                raise ValueError("Labels must be float32 [B,1].")
            if not torch.isfinite(images).all() or not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
                raise FloatingPointError("Non-finite inputs or invalid binary labels.")
            with runtime.accumulation_context(model, synchronize=(batch_index == len(group)-1)) if hasattr(runtime, "accumulation_context") else nullcontext():
                with autocast_context(device, precision or settings["precision"]):
                    logits = model(images, None)
                if logits.shape != labels.shape or not torch.isfinite(logits).all():
                    raise FloatingPointError("Invalid [B,1] logits.")
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = criterion(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite weighted BCE.")
                (loss * (len(images) / count)).backward()
            return float(loss.detach()) * len(images)
        weighted_sum += runtime.checked("forward_backward", forward_backward)
    def check_gradients():
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Missing or non-finite gradients; BF16 run stops without retrying.")
    runtime.checked("local_gradient_check", check_gradients)
    runtime.average_gradients(model, count)
    runtime.checked("aggregated_gradient_check", check_gradients)
    if settings["grad_clip"] is not False:
        runtime.checked("gradient_clipping", lambda: torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["grad_clip"], error_if_nonfinite=True))
    lr = None
    if step:
        lr = schedule.prepare()
        runtime.checked("optimizer_step", optimizer.step)
        schedule.record(True)
    return {"weighted_loss_sum": weighted_sum, "local_samples": count, "lr": lr}


def train_epoch(model, loader, optimizer, schedule, device, settings, runtime, epoch, log_every=0):
    if loader.sampler is not None and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    model.train()
    iterator = runtime.checked("training_iterator", lambda: iter(loader))
    samples, total, groups = 0, 0.0, 0
    started = time.monotonic()
    expected_groups = math.ceil(len(loader)/settings["accumulation_steps"])
    while True:
        group = runtime.checked("training_data_wait", lambda: list(islice(iterator, settings["accumulation_steps"])))
        if not group:
            break
        result = update_group(model, group, optimizer, schedule, device, settings, runtime)
        samples += result["local_samples"]
        total += result["weighted_loss_sum"]
        groups += 1
        if log_every and runtime.rank == 0 and (groups % log_every == 0):
            elapsed = max(time.monotonic()-started, 1e-9)
            print(f"Epoch {epoch}: groups={groups}/{expected_groups} local_samples={samples} "
                  f"loss={total/samples:.6f} lr={optimizer.param_groups[0]['lr']:.8g} "
                  f"elapsed={elapsed:.1f}s local_samples/s={samples/elapsed:.1f}", flush=True)
    summed = runtime.sum_numbers([samples, total, groups])
    return {"consumed_samples": int(summed[0]), "weighted_bce_pw5": summed[1]/summed[0],
            "successful_updates": schedule.successful_updates, "groups": groups,
            "lr": optimizer.param_groups[0]["lr"]}


@dataclass
class SelectionState:
    best_raw: float = -float("inf")
    best_epoch: int | None = None
    best_significant: float = -float("inf")
    stale: int = 0

    def update(self, epoch, score, settings):
        if score is None or not math.isfinite(score):
            raise ValueError("Validation AUROC must be finite.")
        improved = score > self.best_raw
        if improved:
            self.best_raw, self.best_epoch = score, epoch
        stop = False
        if settings["early_stopping_enabled"]:
            if score > self.best_significant + settings["min_delta"]:
                self.best_significant, self.stale = score, 0
            else:
                self.stale += 1
            stop = epoch >= settings["min_epochs"] and self.stale >= settings["patience"]
        return improved, stop

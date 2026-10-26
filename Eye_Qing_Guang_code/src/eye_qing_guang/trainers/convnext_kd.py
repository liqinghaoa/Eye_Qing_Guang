"""Sample-weighted binary distillation; only the student receives gradients."""

import math
import time
from itertools import islice

import torch
import torch.nn.functional as F

from eye_qing_guang.trainers.convnext_official import autocast_context

SUM_FIELDS = (
    "total_loss", "supervised_bce_pw5", "soft_bce", "kd_kl_diagnostic",
    "raw_entropy", "target_entropy", "wrong", "wrong_positive", "wrong_negative",
    "corrected", "lambda_on_wrong_sum",
)


def check_binary_tensor(value, name, *, shape=None, labels=False):
    if value.ndim != 2 or value.shape[1] != 1 or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} must have matching [B,1] shape.")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}.")
    if labels and not ((value == 0) | (value == 1)).all():
        raise ValueError("Labels must be binary 0/1.")


def binary_entropy(probability):
    return -(probability * probability.log() + (1 - probability) * (1 - probability).log())


@torch.no_grad()
def make_targets(fused_probability, labels, variant, epsilon=1e-6):
    check_binary_tensor(fused_probability, "teacher probability")
    check_binary_tensor(labels, "labels", shape=fused_probability.shape, labels=True)
    if not ((fused_probability >= 0) & (fused_probability <= 1)).all():
        raise ValueError("Teacher probabilities must be in [0,1] before clipping.")
    if variant not in {"standard", "rectified"} or not 0 < epsilon < .5:
        raise ValueError("Invalid KD variant or epsilon.")
    with torch.autocast(device_type=labels.device.type, enabled=False):
        q, y = fused_probability.detach().float().clamp(epsilon, 1 - epsilon), labels.detach().float()
        wrong = (q >= .5) != (y >= .5)
        entropy = binary_entropy(q)
        p_true = y * q + (1 - y) * (1 - q)
        p_pred = torch.maximum(q, 1 - q)
        lam = p_true * torch.exp(-entropy) / (1 + p_pred - p_true)
        target = (torch.where(wrong, lam * q + (1 - lam) * y, q)
                  if variant == "rectified" else q)
        target = target.clamp(epsilon, 1 - epsilon).detach()
        if not torch.isfinite(target).all() or not torch.isfinite(lam).all():
            raise FloatingPointError("Non-finite corrected soft target.")
        return target, {"raw_entropy": entropy, "target_entropy": binary_entropy(target),
                        "wrong": wrong, "lambda": lam}


def loss_terms(student_logits, labels, target, alpha=1.0):
    check_binary_tensor(student_logits, "student logits")
    check_binary_tensor(labels, "labels", shape=student_logits.shape, labels=True)
    check_binary_tensor(target, "soft target", shape=student_logits.shape)
    if not ((target > 0) & (target < 1)).all() or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Soft targets must be clipped into (0,1), alpha must be finite and nonnegative.")
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        z, y, q = student_logits.float(), labels.float(), target.detach().float()
        supervised = F.binary_cross_entropy_with_logits(z, y, pos_weight=z.new_tensor([5.0]), reduction="none")
        soft = F.binary_cross_entropy_with_logits(z, q, reduction="none")
        terms = {"total_loss": supervised + alpha * soft, "supervised_bce_pw5": supervised,
                 "soft_bce": soft, "kd_kl_diagnostic": soft - binary_entropy(q)}
        if any(not torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError("Non-finite KD loss; stop before backward or optimizer update.")
        return terms


def _check_gradients(student):
    gradients = [p.grad for p in student.parameters() if p.requires_grad]
    if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Missing or non-finite student gradients; no retries or skipped updates.")


def update_group(student, group, optimizer, schedule, device, training, distillation,
                 teachers, *, step=True, progress=None):
    count = sum(len(batch["image"]) for batch in group)
    if not count:
        raise ValueError("Empty KD accumulation group.")
    teachers.assert_student_optimizer(student, optimizer)
    optimizer.zero_grad(set_to_none=True)
    sums = {key: 0.0 for key in SUM_FIELDS}
    for batch in group:
        images, labels = batch["image"].to(device), batch["label"].to(device)
        if images.dtype != torch.float32 or labels.dtype != torch.float32:
            raise ValueError("KD requires the original FP32 view and FP32 labels.")
        check_binary_tensor(labels, "labels", shape=(len(images), 1), labels=True)
        # Teacher forward finishes without an autograd graph before student activations exist.
        q = teachers.probabilities(images)
        target, diagnostics = make_targets(q, labels, distillation["variant"], distillation["epsilon"])
        with autocast_context(device, training["precision"]):
            logits = student(images, None)
        terms = loss_terms(logits, labels, target, distillation["alpha"])
        (terms["total_loss"].mean() * (len(images) / count)).backward()
        if progress is not None:
            progress["consumed_train_samples"] += len(images)
        for key, value in terms.items():
            sums[key] += float(value.detach().sum())
        wrong = diagnostics["wrong"]
        sums["raw_entropy"] += float(diagnostics["raw_entropy"].sum())
        sums["target_entropy"] += float(diagnostics["target_entropy"].sum())
        sums["wrong"] += int(wrong.sum())
        sums["wrong_positive"] += int((wrong & (labels == 1)).sum())
        sums["wrong_negative"] += int((wrong & (labels == 0)).sum())
        if distillation["variant"] == "rectified":
            sums["corrected"] += int(wrong.sum())
            sums["lambda_on_wrong_sum"] += float(diagnostics["lambda"][wrong].sum())
    _check_gradients(student)
    if training["grad_clip"] is not False:
        torch.nn.utils.clip_grad_norm_(student.parameters(), training["grad_clip"], error_if_nonfinite=True)
    if step:
        schedule.prepare()
        optimizer.step()
        schedule.record(True)
    return {"samples": count, "sums": sums}


def train_epoch(student, loader, optimizer, schedule, device, training, distillation,
                teachers, epoch, *, log_every=10, progress=None):
    student.train()
    if any(model.training for model in teachers.models):
        raise ValueError("Calling student.train() must not switch a teacher to train mode.")
    iterator, samples, groups = iter(loader), 0, 0
    sums, started = {key: 0.0 for key in SUM_FIELDS}, time.monotonic()
    expected_groups = math.ceil(len(loader) / training["accumulation_steps"])
    while True:
        group = list(islice(iterator, training["accumulation_steps"]))
        if not group:
            break
        result = update_group(student, group, optimizer, schedule, device, training,
                              distillation, teachers, progress=progress)
        samples += result["samples"]
        groups += 1
        for key in sums:
            sums[key] += result["sums"][key]
        if log_every and groups % log_every == 0:
            print(f"Epoch {epoch}: groups={groups}/{expected_groups}, samples={samples}, "
                  f"KD_total={sums['total_loss']/samples:.6f}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    if not samples:
        raise ValueError("Empty KD training epoch.")
    return {
        "train_total_loss": sums["total_loss"] / samples,
        "train_supervised_bce_pw5": sums["supervised_bce_pw5"] / samples,
        "train_soft_bce": sums["soft_bce"] / samples,
        "train_kd_kl_diagnostic": sums["kd_kl_diagnostic"] / samples,
        "teacher_raw_entropy_mean": sums["raw_entropy"] / samples,
        "target_entropy_mean": sums["target_entropy"] / samples,
        "teacher_wrong_samples": int(sums["wrong"]), "teacher_wrong_fraction": sums["wrong"] / samples,
        "teacher_wrong_positive": int(sums["wrong_positive"]), "teacher_wrong_negative": int(sums["wrong_negative"]),
        "corrected_samples": int(sums["corrected"]),
        "correction_lambda_mean_on_wrong": (sums["lambda_on_wrong_sum"] / sums["corrected"]
                                          if sums["corrected"] else None),
        "consumed_samples": samples, "groups": groups, "successful_updates": schedule.successful_updates,
        "lr": optimizer.param_groups[0]["lr"],
    }

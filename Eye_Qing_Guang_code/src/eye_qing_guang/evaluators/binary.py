"""Ordered probabilities for labeled validation or unlabeled test data."""

import math

import torch


@torch.inference_mode()
def predict_batches(model, loader, device, criterion=None, allow_empty=False, diagnostics=None):
    model.eval()
    rows = []
    total, count = 0.0, 0
    if diagnostics is not None:
        diagnostics.mark("evaluation_iterator_initialization")
    iterator = iter(loader)
    for index in range(len(loader)):
        if diagnostics is not None:
            diagnostics.mark("evaluation_data_wait", evaluation_batch=index + 1)
        batch = next(iterator)
        if diagnostics is not None:
            diagnostics.mark("evaluation_forward", actual_image_ids=list(batch["image_id"]))
        images, ages = batch["image"].to(device), batch["age"].to(device)
        if not torch.isfinite(images).all():
            raise FloatingPointError("Non-finite evaluation inputs.")
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(images, ages).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model logits.")
        probabilities = torch.sigmoid(logits).flatten().cpu().tolist()
        if criterion is not None:
            labels = batch["label"].to(device)
            loss = criterion(logits, labels).item()
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss.")
            total += loss * len(images)
        for index, probability in enumerate(probabilities):
            row = {"image_id": batch["image_id"][index],
                   "logit": float(logits.flatten()[index].item()), "probability": probability}
            if "label" in batch:
                row["label"] = int(batch["label"][index].item())
            rows.append(row)
        count += len(images)
    if diagnostics is not None:
        diagnostics.mark("evaluation_complete", evaluated_samples=count, memory=True)
    if not count and not allow_empty:
        raise ValueError("Empty evaluation loader.")
    return rows, total / count if criterion is not None and count else None

"""Ordered probabilities for labeled validation or unlabeled test data."""

import math

import torch


@torch.inference_mode()
def predict_batches(model, loader, device, criterion=None):
    model.eval()
    rows = []
    total, count = 0.0, 0
    for batch in loader:
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
    if not count:
        raise ValueError("Empty evaluation loader.")
    return rows, total / count if criterion is not None else None

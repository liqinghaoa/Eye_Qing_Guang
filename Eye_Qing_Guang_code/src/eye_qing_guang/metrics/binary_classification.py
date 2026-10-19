"""Probability validation and binary metrics with an explicit threshold."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, roc_auc_score,
)


def compute_binary_metrics(labels, probabilities, threshold=0.5) -> dict:
    target = np.asarray(labels)
    scores = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or not len(target) or len(target) != len(scores):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors.")
    if not np.isin(target, [0, 1]).all() or not np.isfinite(scores).all() or (
        (scores < 0) | (scores > 1)
    ).any() or not 0 <= threshold <= 1:
        raise ValueError("Invalid labels, probabilities, or threshold.")
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(target)) == 2
    recalls = []
    if tn + fp:
        recalls.append(float(tn / (tn + fp)))
    if tp + fn:
        recalls.append(float(tp / (tp + fn)))
    return {
        "n_samples": len(target), "threshold": float(threshold),
        "positives": int(target.sum()), "positive_rate": float(target.mean()),
        "brier": float(np.mean((scores - target) ** 2)),
        "roc_auc": float(roc_auc_score(target, scores)) if both else None,
        "average_precision": float(average_precision_score(target, scores)) if both else None,
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

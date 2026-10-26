"""Whole-validation metrics, including tied-score specificity operating points."""

import math

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from eye_qing_guang.metrics.binary_classification import compute_binary_metrics
from eye_qing_guang.contracts.convnext_config import weighted_bce_key


def recall_at_specificity(labels, probabilities, minimum):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if not 0 <= minimum <= 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Specificity operating points require both classes and a valid constraint.")
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    positives, negatives = int(labels.sum()), int(len(labels)-labels.sum())
    best = {"threshold": None, "endpoint": "all_negative", "sensitivity": 0.0, "specificity": 1.0,
            "tp": 0, "fp": 0, "tn": negatives, "fn": positives}
    cumulative_tp = labels.cumsum()
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    best_key = (0.0, 1.0, float("inf"))
    for end in ends:
        tp = int(cumulative_tp[end])
        fp = int(end+1-tp)
        sensitivity, specificity = tp / positives, (negatives-fp) / negatives
        key = sensitivity, specificity, float(scores[end])
        if specificity >= minimum and key > best_key:
            best_key = key
            best = {"threshold": float(scores[end]), "endpoint": None, "sensitivity": sensitivity,
                    "specificity": specificity, "tp": tp, "fp": fp, "tn": negatives-fp, "fn": positives-tp}
    return best


def compute_official_metrics(predictions, weighted_bce, pos_weight=5.0):
    labels = [r["label"] for r in predictions]
    scores = [r["probability"] for r in predictions]
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)
    if metrics["roc_auc"] is None or not math.isfinite(metrics["roc_auc"]):
        raise ValueError("Official validation AUROC must be finite with both classes.")
    if not math.isfinite(weighted_bce) or weighted_bce < 0:
        raise ValueError("Invalid validation weighted BCE.")
    precision, recall, _ = precision_recall_curve(labels, scores)
    metrics.update({weighted_bce_key(pos_weight): float(weighted_bce),
                   "pr_auc_trapezoid": float(auc(recall, precision)),
                   "hard_label_roc_auc": float(roc_auc_score(labels, np.asarray(scores) >= 0.5)),
                   "recall_at_specificity_90": recall_at_specificity(labels, scores, 0.90),
                   "recall_at_specificity_95": recall_at_specificity(labels, scores, 0.95)})
    return metrics

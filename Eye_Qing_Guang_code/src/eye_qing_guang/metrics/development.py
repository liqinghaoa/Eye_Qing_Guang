"""Development-only prediction diagnostics and fit-defined image proxy strata."""

from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from eye_qing_guang.datasets.transforms import fov_crop
from eye_qing_guang.metrics.binary_classification import compute_binary_metrics
from eye_qing_guang.utils.experiment_io import write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path


def prediction_metrics(predictions, threshold=0.5):
    labels = [row["label"] for row in predictions]
    probabilities = [row["probability"] for row in predictions]
    logits = [row["logit"] for row in predictions]
    metrics = compute_binary_metrics(labels, probabilities, threshold)
    both = len(set(labels)) == 2
    metrics.update(
        logit_roc_auc=float(roc_auc_score(labels, logits)) if both else None,
        logit_average_precision=float(average_precision_score(labels, logits)) if both else None,
        probability_min=float(min(probabilities)), probability_max=float(max(probabilities)),
        probability_mean=float(np.mean(probabilities)),
        probability_p01=float(np.quantile(probabilities, 0.01)),
        probability_p50=float(np.quantile(probabilities, 0.5)),
        probability_p99=float(np.quantile(probabilities, 0.99)),
        saturated_zero=sum(value == 0 for value in probabilities),
        saturated_one=sum(value == 1 for value in probabilities),
    )
    return metrics


def image_proxies(row, root, data):
    with Image.open(resolve_path(row["image_path"], root)) as source:
        image = source.convert("RGB")
    width, height = image.size
    cropped, info = fov_crop(image, data["preprocessing"])
    cropped.thumbnail((256, 256), Image.Resampling.BILINEAR)
    gray = np.asarray(cropped.convert("L"), dtype=np.float32) / 255
    differences = []
    if gray.shape[0] > 1:
        differences.append(float(np.square(np.diff(gray, axis=0)).mean()))
    if gray.shape[1] > 1:
        differences.append(float(np.square(np.diff(gray, axis=1)).mean()))
    return {"image_id": row["image_id"], "short_side": min(width, height),
            "exposure": float(gray.mean()), "sharpness_proxy": float(np.mean(differences)) if differences else 0.0,
            **info}


def write_development_analysis(directory, fit_rows, dev_rows, predictions, root, config):
    if [row["image_id"] for row in dev_rows] != [row["image_id"] for row in predictions]:
        raise ValueError("Development predictions are missing, duplicated or out of manifest order.")
    data = config["data"]
    # Evenly spaced IDs after stable sorting: deterministic, no label/score-based selection.
    ordered = sorted(fit_rows, key=lambda row: row["image_id"].casefold())
    count = min(len(ordered), config["evaluation"]["subgroup_fit_samples"])
    indices = np.linspace(0, len(ordered)-1, count, dtype=int)
    fit_features = [image_proxies(ordered[index], root, data) for index in indices]
    features = [image_proxies(row, root, data) for row in dev_rows]
    boundaries = {key: float(np.median([row[key] for row in fit_features]))
                  for key in ("short_side", "exposure", "sharpness_proxy")}
    write_json(directory / "reports" / "subgroup_rules.json", {
        "source": "internal_fit_only", "fit_sample_ids": [row["image_id"] for row in fit_features],
        "boundary": "median", "boundaries": boundaries,
        "proxy_definition": "RGB FOV thumbnail<=256, grayscale mean and mean squared adjacent differences; not clinical quality",
    })
    groups = []
    for key, boundary in boundaries.items():
        for name, predicate in (("low", lambda value: value <= boundary), ("high", lambda value: value > boundary)):
            subset = [prediction for feature, prediction in zip(features, predictions) if predicate(feature[key])]
            groups.append({"proxy": key, "group": name, "boundary": boundary,
                           **(prediction_metrics(subset) if subset else {"n_samples": 0, "roc_auc": None})})
    write_json(directory / "reports" / "subgroups.json", groups)
    write_csv(directory / "reports" / "image_proxies.csv", features)
    write_json(directory / "reports" / "fov_summary.json", {
        "source": "internal_development_deterministic_transform",
        "samples": len(features),
        "fallback_reasons": dict(Counter(row["fallback_reason"] or "accepted" for row in features)),
        "mean_area_ratio": float(np.mean([row["area_ratio"] for row in features])),
    })
    errors = []
    for row in predictions:
        decision = int(row["probability"] >= config["evaluation"]["threshold"])
        if decision != row["label"]:
            errors.append({**row, "error": "false_negative" if row["label"] else "false_positive"})
    write_csv(directory / "reports" / "errors.csv", errors,
              ["image_id", "logit", "probability", "label", "error"])
    prior = sum(int(row["label"]) for row in fit_rows) / len(fit_rows)
    write_json(directory / "reports" / "prior_reference.json", {
        "prior_source": "internal_fit", "constant_probability": prior,
        **compute_binary_metrics([int(row["label"]) for row in dev_rows], [prior]*len(dev_rows)),
    })

"""Unlabeled inference and explicit submission-format conversion."""

from pathlib import Path

import numpy as np

from eye_qing_guang.contracts.checkpoints import load_checkpoint
from eye_qing_guang.contracts.config import submission_options
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.utils.experiment_io import read_csv, run_manifest, write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import choose_device


def predict_test(checkpoint: Path, root: Path, output_dir: Path,
                 device_name="auto", manifest: Path | None = None) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    config = payload["config"]
    if config["experiment"].get("protocol") == "b0_dev_v1" and manifest is None:
        raise ValueError("B0 prediction requires --manifest from the separate test-manifest command; "
                         "internal development manifests do not contain test data.")
    path = manifest or resolve_path(config["data"]["manifest_dir"], root) / "test.csv"
    rows = load_manifest(path, root, labeled=False)
    loader = make_loader(rows, config, root, payload["age_stats"], device=device)
    predictions, _ = predict_batches(model, loader, device)
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "test_predictions.csv"
    write_csv(destination, predictions)
    write_json(output_dir / "run_manifest.json", run_manifest(
        "completed", checkpoint=str(checkpoint), source_manifest=str(path),
        checkpoint_epoch=payload["epoch"], samples=len(rows),
    ))
    return destination


def build_submission(predictions: Path, test_csv: Path, output: Path,
                     id_column="Image_name", value_column="Probability",
                     source_id_column="Image_name", mode="competition", threshold=0.5,
                     label_column="Pred_Label") -> Path:
    if output.exists():
        raise FileExistsError(output)
    submission_options({"submission": {"id_column": id_column, "value_column": value_column,
                                      "label_column": label_column, "mode": mode}})
    if mode == "competition" and output.name != "submission.csv":
        raise ValueError("Competition output must be named submission.csv.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    rows, source = read_csv(predictions), read_csv(test_csv)
    if not rows or not source:
        raise ValueError("Prediction and test tables must be nonempty.")
    if not {"image_id", "probability"} <= rows[0].keys() or source_id_column not in source[0]:
        raise ValueError("Missing prediction or official test identifier/probability columns.")
    scores = {row["image_id"]: float(row["probability"]) for row in rows}
    identifiers = [row[source_id_column] for row in source]
    if any(not row["image_id"].strip() for row in rows) or any(not name.strip() for name in identifiers):
        raise ValueError("Prediction and test identifiers must be nonempty.")
    if len(scores) != len(rows) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate prediction or test identifiers.")
    if set(scores) != set(identifiers):
        raise ValueError("Prediction identifiers do not match the official test table.")
    if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Invalid probability values.")
    result = []
    for identifier in identifiers:
        probability = scores[identifier]
        row = {id_column: identifier,
               value_column: int(probability >= threshold) if mode == "label" else probability}
        if mode == "competition":
            row[label_column] = int(probability >= threshold)
        result.append(row)
    fields = [id_column, value_column] + ([label_column] if mode == "competition" else [])
    write_csv(output, result, fields)
    return output

"""Unlabeled inference and explicit submission-format conversion."""

from pathlib import Path

import numpy as np

from eye_qing_guang.contracts.checkpoints import load_checkpoint
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
                     id_column="Image_name", value_column="probability",
                     source_id_column="Image_name", mode="probability", threshold=0.5) -> Path:
    if output.exists():
        raise FileExistsError(output)
    if id_column == value_column or mode not in {"probability", "label"}:
        raise ValueError("Submission needs distinct ID/value columns and a supported mode.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    rows, source = read_csv(predictions), read_csv(test_csv)
    if not rows or not source:
        raise ValueError("Prediction and test tables must be nonempty.")
    scores = {row["image_id"]: float(row["probability"]) for row in rows}
    identifiers = [row[source_id_column] for row in source]
    if len(scores) != len(rows) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate prediction or test identifiers.")
    if set(scores) != set(identifiers):
        raise ValueError("Prediction identifiers do not match the official test table.")
    if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Invalid probability values.")
    result = [{id_column: identifier, value_column: scores[identifier] if mode == "probability"
               else int(scores[identifier] >= threshold)} for identifier in identifiers]
    write_csv(output, result, [id_column, value_column])
    return output

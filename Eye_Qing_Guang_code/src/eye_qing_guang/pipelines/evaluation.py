"""Evaluate a versioned checkpoint using its saved preprocessing settings."""

from pathlib import Path

from eye_qing_guang.contracts.checkpoints import load_checkpoint
from eye_qing_guang.contracts.config import is_b0
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.binary_classification import compute_binary_metrics
from eye_qing_guang.utils.experiment_io import run_manifest, write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import choose_device


def evaluate_checkpoint(checkpoint: Path, root: Path, output_dir: Path,
                        split="valid", device_name="auto", manifest: Path | None = None,
                        evaluation_source: str | None = None) -> dict:
    if split not in {"train", "valid"}:
        raise ValueError("Evaluation is restricted to labeled train/valid splits.")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    config = payload["config"]
    path = manifest or resolve_path(config["data"]["manifest_dir"], root) / f"{split}.csv"
    b0 = is_b0(config)
    if b0:
        default_path = resolve_path(config["data"]["manifest_dir"], root) / f"{split}.csv"
        if path.resolve() == default_path.resolve():
            expected = "internal_development" if split == "valid" else "internal_fit"
            if evaluation_source is not None and evaluation_source != expected:
                raise ValueError("Evaluation source conflicts with the saved B0 development manifest.")
            evaluation_source = expected
        elif evaluation_source is None:
            raise ValueError("B0 external manifest evaluation requires an explicit --evaluation-source.")
    else:
        evaluation_source = evaluation_source or f"configured_{split}"
    rows = load_manifest(path, root, labeled=True)
    loader = make_loader(rows, config, root, payload["age_stats"], device=device)
    criterion = build_loss(config["training"], payload["pos_weight"]).to(device)
    predictions, loss = predict_batches(model, loader, device, criterion)
    if b0:
        from eye_qing_guang.metrics.development import prediction_metrics

        metrics = prediction_metrics(predictions, float(config["evaluation"]["threshold"]))
    else:
        metrics = compute_binary_metrics(
            [row["label"] for row in predictions], [row["probability"] for row in predictions],
            float(config["evaluation"]["threshold"]),
        )
    metrics["loss"] = loss
    metrics["evaluation_source"] = evaluation_source
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "predictions.csv", predictions)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run_manifest.json", run_manifest(
        "completed", checkpoint=str(checkpoint), split=split, source_manifest=str(path),
        samples=len(rows), checkpoint_epoch=payload["epoch"],
        evaluation_source=evaluation_source,
    ))
    return metrics

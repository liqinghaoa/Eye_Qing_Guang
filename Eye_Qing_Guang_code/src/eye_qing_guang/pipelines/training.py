"""Train on the official train split and select checkpoints on valid only."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from eye_qing_guang.contracts.checkpoints import save_checkpoint
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.metadata import fit_age_stats
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.binary_classification import compute_binary_metrics
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.trainers.binary import train_one_epoch
from eye_qing_guang.utils.experiment_io import (
    new_experiment, run_manifest, write_csv, write_json, write_yaml,
)
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import choose_device, seed_everything


def _copy_checkpoint(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def train(config: dict, root: Path, name: str | None = None) -> Path:
    if config["experiment"].get("protocol") == "b0_dev_v1":
        from eye_qing_guang.pipelines.b0_development import train_development

        return train_development(config, root, name)
    output_root = resolve_path(config["experiment"]["output_root"], root)
    experiment_name = name or config["experiment"]["name"]
    if (output_root / experiment_name).exists():
        raise FileExistsError(output_root / experiment_name)
    seed_everything(int(config["experiment"]["seed"]))
    device = choose_device(config["training"]["device"])
    manifest_dir = resolve_path(config["data"]["manifest_dir"], root)
    records = {split: load_manifest(manifest_dir / f"{split}.csv", root, True)
               for split in ("train", "valid")}
    if {row["image_id"].casefold() for row in records["train"]} & {
        row["image_id"].casefold() for row in records["valid"]
    }:
        raise ValueError("Training and validation identifiers overlap.")
    if {int(row["label"]) for row in records["train"]} != {0, 1}:
        raise ValueError("Training requires both label classes.")
    monitor = config["training"]["monitor"]
    if monitor == "roc_auc" and len({row["label"] for row in records["valid"]}) != 2:
        raise ValueError("ROC AUC checkpoint selection requires both validation classes.")
    age_stats = fit_age_stats(records["train"]) if config["model"]["use_age"] else None
    positive = sum(int(row["label"]) for row in records["train"])
    pos_weight = ((len(records["train"]) - positive) / positive
                  if config["training"].get("auto_pos_weight", False) else 1.0)
    loaders = {split: make_loader(rows, config, root, age_stats, split == "train")
               for split, rows in records.items()}
    # Decode representative inputs and construct the model before creating outputs.
    for loader in loaders.values():
        loader.dataset[0]
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = build_loss(config["training"], pos_weight).to(device)
    directory = new_experiment(output_root, experiment_name)
    config = {**config, "experiment": {**config["experiment"], "name": directory.name}}
    history = []
    best_score, best_epoch, stale = -float("inf"), 0, 0
    samples = {split: len(rows) for split, rows in records.items()}
    status, error = "failed", None
    started = run_manifest("running", experiment=directory.name, device=str(device),
                           samples=samples, torch=str(torch.__version__),
                           age_stats=age_stats, pos_weight=pos_weight)
    try:
        write_yaml(directory / "config_snapshot.yaml", config)
        write_json(directory / "run_started.json", started)
        print(f"Training started: {directory} | device={device} | samples={samples}", flush=True)
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            training_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            predictions, validation_loss = predict_batches(model, loaders["valid"], device, criterion)
            metrics = compute_binary_metrics(
                [row["label"] for row in predictions], [row["probability"] for row in predictions],
                float(config["evaluation"]["threshold"]),
            )
            result = {"epoch": epoch, "train_loss": training_loss,
                      "valid_loss": validation_loss, **metrics}
            history.append(result)
            write_json(directory / "reports" / f"epoch_{epoch:04d}.json", result)
            score = metrics[monitor]
            if score is None:
                raise ValueError(f"Validation metric is undefined: {monitor}")
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                save_checkpoint(directory / "checkpoints" / f"epoch_{epoch:04d}.pt",
                                model, optimizer, epoch, config, age_stats, pos_weight, metrics)
            else:
                stale += 1
            print(f"Epoch {epoch}: train_loss={training_loss:.5f} "
                  f"valid_loss={validation_loss:.5f} {monitor}={score:.5f}", flush=True)
            if stale >= int(config["training"]["patience"]):
                break
        save_checkpoint(directory / "checkpoints" / "last.pt", model, optimizer,
                        history[-1]["epoch"], config, age_stats, pos_weight, metrics)
        _copy_checkpoint(directory / "checkpoints" / f"epoch_{best_epoch:04d}.pt",
                         directory / "checkpoints" / "best.pt")
        write_csv(directory / "reports" / "history.csv", history)
        write_json(directory / "reports" / "metrics.json", {
            "best_epoch": best_epoch, "monitor": monitor, "best_score": best_score,
            **next(row for row in history if row["epoch"] == best_epoch),
        })
        status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(directory / "run_manifest.json", run_manifest(
            status, experiment=directory.name, device=str(device), samples=samples,
            completed_epochs=len(history), best_epoch=best_epoch, error=error,
        ))
    return directory

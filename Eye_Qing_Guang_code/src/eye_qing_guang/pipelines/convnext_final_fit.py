"""Fixed-epoch official train+valid final fit; no validation or best selection."""

import copy
import math
import time
from pathlib import Path

import torch

from eye_qing_guang.contracts.convnext_final_checkpoints import FinalCheckpointStore, load_final_checkpoint
from eye_qing_guang.contracts.convnext_final_config import (
    MERGED_TRAIN_COUNT, TEST_COUNT, load_convnext_final_config,
)
from eye_qing_guang.datasets.convnext_official import official_records, ordered_predictions
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.offline_cache import cache_signature, cached_records
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.models.timm_convnext import build_convnext_model
from eye_qing_guang.trainers.convnext_official import build_optimizer_schedule, pw5_loss, train_epoch
from eye_qing_guang.utils.experiment_io import read_csv, run_manifest, write_csv, write_json, write_yaml
from eye_qing_guang.utils.paths import resolve_path
from eye_qing_guang.utils.reproducibility import seed_everything
from eye_qing_guang.utils.timm_artifacts import atomic_csv, atomic_json, publish_submission
from eye_qing_guang.utils.timm_source import create_source_runtime

MERGED_NEGATIVE_COUNT = 55397
MERGED_POSITIVE_COUNT = 1848

def _new_directory(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for part in ("checkpoints", "reports", "predictions", "submission"):
        (directory / part).mkdir(parents=True)


def _write_final_manifest(root, config, records):
    directory = resolve_path(config["data"]["manifest_dir"], root)
    if directory.exists():
        raise FileExistsError(f"Final-fit manifest directory exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    merged = [dict(row, split="train") for row in records["merged_train"]]
    write_csv(directory / "train.csv", merged, ["image_id", "image_path", "label", "age", "split"])
    write_csv(directory / "test.csv", records["test"], ["image_id", "image_path", "label", "age", "split"])
    write_json(directory / "summary.json", {
        "protocol": config["experiment"]["protocol"],
        "official_valid_merged": True,
        "validation_used_after_merge": False,
        "train_samples": len(merged), "test_samples": len(records["test"]),
        "labels": {"negative": sum(r["label"] == "0" for r in merged),
                    "positive": sum(r["label"] == "1" for r in merged)},
        "source_splits": {"train": 51396, "official_valid": 5849},
    })
    return directory


def _bind_cached(records, data, root, *, training):
    bound = cached_records(records, data, root)
    if training:
        # Official valid images retain their source cache index, but become
        # training samples after the merge and therefore receive augmentation.
        bound = [dict(row, _cache_split="train") for row in bound]
    return bound


def _train_loader(records, config, root, device):
    data = copy.deepcopy(config["data"])
    data["cache"] = {**data["cache"], "enabled": False}
    train_config = copy.deepcopy(config)
    train_config["data"] = data
    return make_loader(records, train_config, root, training=True, device=device)


def run_convnext_final_fit(config_path, root):
    root = Path(root).resolve()
    config = load_convnext_final_config(resolve_path(config_path, root))
    output = resolve_path(config["experiment"]["output_root"], root)
    directory = output / config["experiment"]["name"]
    if directory.exists():
        raise FileExistsError(directory)
    runtime, device = create_source_runtime(config, root)
    started = time.monotonic()
    manifest = None
    try:
        official, _ = runtime.primary("official_source_records", lambda: official_records(config, root))
        merged = official["train"] + official["valid"]
        if len(merged) != MERGED_TRAIN_COUNT or len({r["image_id"].casefold() for r in merged}) != MERGED_TRAIN_COUNT:
            raise ValueError("Merged train+valid manifest count or uniqueness check failed.")
        if (sum(r["label"] == "0" for r in merged) != MERGED_NEGATIVE_COUNT
                or sum(r["label"] == "1" for r in merged) != MERGED_POSITIVE_COUNT):
            raise ValueError("Merged train+valid label counts do not match the frozen protocol.")
        if len(official["test"]) != TEST_COUNT:
            raise ValueError("Official test count changed.")
        records = {"merged_train": merged, "test": official["test"]}
        manifest_dir = runtime.primary("final_fit_manifest", lambda: _write_final_manifest(root, config, records))
        all_for_cache = merged + official["test"]
        if config["data"]["cache"].get("enabled", False):
            runtime.primary("offline_cache_validation", lambda: cached_records(all_for_cache, config["data"], root))
            cache_source = {"root": config["data"]["cache"]["root"], "signature": cache_signature(config["data"], root)}
        else:
            cache_source = None
        runtime.primary("formal_directory", lambda: _new_directory(directory))
        manifest = run_manifest("running", protocol=config["experiment"]["protocol"], config=config,
            sources={"official_train": 51396, "official_valid_merged": 5849, "offline_cache": cache_source,
                     "manifest_dir": str(manifest_dir)},
            train_samples=MERGED_TRAIN_COUNT, official_valid_merged=True,
            validation_used_after_merge=False, checkpoint_selection="fixed_epoch",
            final_epoch=config["training"]["final_epochs"], test_samples=TEST_COUNT,
            effective_batch=config["training"]["batch_size"]*config["training"]["accumulation_steps"],
            stages={"training": {"status": "running", "completed_epochs": 0},
                    "final_reload": {"status": "not_started"}, "test_inference": {"status": "not_started"},
                    "submission": {"status": "not_started"}},
            validation_metrics="not_used", early_stopping="disabled", preflight="not_run",
            device=str(device), cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
        runtime.primary("formal_start", lambda: (write_yaml(directory / "config_snapshot.yaml", config), atomic_json(directory / "run_manifest.json", manifest)))
        seed_everything(config["experiment"]["seed"])
        model, metadata = runtime.checked("formal_model", lambda: build_convnext_model(config["model"], root))
        model = runtime.initialize_model(model.to(device))
        manifest["model_metadata"] = metadata
        atomic_json(directory / "run_manifest.json", manifest)
        train_records = _bind_cached(merged, config["data"], root, training=True)
        loader = runtime.checked("training_loader", lambda: _train_loader(train_records, config, root, device))
        groups = math.ceil(len(loader) / config["training"]["accumulation_steps"])
        optimizer, schedule = runtime.checked("optimizer_and_schedule", lambda: build_optimizer_schedule(model, config["training"], groups))
        store = runtime.checked("checkpoint_store", lambda: FinalCheckpointStore(directory / "checkpoints"))
        history = []
        for epoch in range(1, config["training"]["final_epochs"] + 1):
            started_epoch = time.monotonic()
            train = train_epoch(model, loader, optimizer, schedule, device, config["training"], runtime, epoch,
                                log_every=config["distributed"]["log_every_n_groups"])
            if train["consumed_samples"] != MERGED_TRAIN_COUNT:
                raise ValueError("Final-fit epoch did not consume all merged train+valid samples.")
            item = {"epoch": epoch, "train_weighted_bce_pw5": train["weighted_bce_pw5"],
                    "lr": train["lr"], "successful_updates": train["successful_updates"],
                    "consumed_samples": train["consumed_samples"], "seconds": time.monotonic()-started_epoch}
            history.append(item)
            store.save("last.pt", model, epoch, config, item, metadata)
            atomic_csv(directory / "reports/history.csv", history, list(item))
            atomic_json(directory / "reports" / f"epoch_{epoch:04d}.json", item)
            manifest["stages"]["training"].update(completed_epochs=epoch, samples=epoch*MERGED_TRAIN_COUNT)
            atomic_json(directory / "run_manifest.json", manifest)
            print(f"Epoch {epoch}: train_pw5={train['weighted_bce_pw5']:.6f}, samples={train['consumed_samples']}, elapsed={item['seconds']:.1f}s", flush=True)
        store.save("final.pt", model, config["training"]["final_epochs"], config, history[-1], metadata)
        atomic_json(directory / "reports/metrics.json", {
            "train_samples": MERGED_TRAIN_COUNT, "official_valid_merged": True,
            "validation_used_after_merge": False, "checkpoint_selection": "fixed_epoch",
            "final_epoch": config["training"]["final_epochs"], "test_samples": TEST_COUNT,
            "validation_metrics": "not_used", "history_final": history[-1],
        })
        manifest["stages"]["training"].update(status="completed")
        atomic_json(directory / "run_manifest.json", manifest)
        runtime.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        manifest["stages"]["final_reload"]["status"] = "running"
        reloaded, payload = load_final_checkpoint(directory / "checkpoints/final.pt", device, root)
        if payload["epoch"] != config["training"]["final_epochs"]:
            raise ValueError("Reloaded final checkpoint epoch mismatch.")
        manifest["stages"]["final_reload"].update(status="completed", epoch=payload["epoch"])
        manifest["stages"]["test_inference"]["status"] = "running"
        test_loader = make_loader(official["test"], config, root, training=False, device=device)
        rows, _ = predict_batches(reloaded, test_loader, device)
        rows = ordered_predictions(rows, official["test"], labeled=False)
        predictions = directory / "predictions/test_predictions.csv"
        atomic_csv(predictions, rows, ["image_id", "logit", "probability"])
        manifest["stages"]["test_inference"].update(status="completed", samples=len(rows))
        manifest["stages"]["submission"]["status"] = "running"
        source = resolve_path(config["data"]["raw_root"], root) / config["data"]["splits"]["test"]["csv"]
        submission = publish_submission(predictions, source, directory / "submission/submission.csv", source_id_column=config["data"]["image_column"])
        manifest["stages"]["submission"].update(status="completed", samples=len(rows), path=str(submission))
        weights = sorted(p.name for p in (directory / "checkpoints").iterdir())
        if weights != ["final.pt", "last.pt"]:
            raise ValueError(f"Unexpected final-fit checkpoints: {weights}")
        manifest.update(status="completed", total_seconds=time.monotonic()-started)
        atomic_json(directory / "run_manifest.json", manifest)
        print(f"ConvNeXt final-fit completed: {submission}", flush=True)
        return directory
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(directory / "run_manifest.json", manifest)
        raise
    finally:
        runtime.close()

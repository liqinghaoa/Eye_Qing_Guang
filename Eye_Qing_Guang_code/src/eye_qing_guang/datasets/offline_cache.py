"""Versioned deterministic image cache; official manifests always retain raw sources."""

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path

VERSION = "fov_presized_letterbox_v2"
INDEX_FIELDS = ["image_id", "split", "label", "age", "source_path", "source_size",
                "source_mtime_ns", "cache_path", "width", "height"]


def validate_cache_settings(data):
    cache = data.get("cache")
    offline = data["preprocessing"]["version"] == VERSION
    if cache is None:
        if offline:
            raise ValueError("The offline recipe requires explicit data.cache settings.")
        return
    if (not isinstance(cache, dict) or set(cache) != {"enabled", "root", "version"}
            or type(cache["enabled"]) is not bool or cache["version"] != VERSION
            or not isinstance(cache["root"], str) or not cache["root"]):
        raise ValueError("data.cache requires enabled(boolean), root, version=fov_presized_letterbox_v2.")
    if not offline:
        raise ValueError("Cache settings require the explicit offline preprocessing version.")


def cache_signature(data, root):
    return {"version": VERSION, "raw_root": str(resolve_path(data["raw_root"], root)),
            "preprocessing": data["preprocessing"], "image_size": data["image_size"],
            "format": "PNG_RGB_uint8", "train_stage": "fov_resize_longest_edge_no_padding",
            "evaluation_stage": "fov_letterbox"}


def cached_records(records, data, root):
    """Validate complete cache and bind by exact ID/split, never replace official IDs."""
    validate_cache_settings(data)
    directory = resolve_path(data["cache"]["root"], root)
    with (directory / "metadata.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("status") != "completed" or meta.get("signature") != cache_signature(data, root):
        raise ValueError(f"Incomplete or incompatible offline cache: {directory}")
    index = read_csv(directory / "index.csv")
    by_key = {}
    counts = {}
    for row in index:
        key = (row["split"], row["image_id"])
        if key in by_key or row["split"] not in ("train", "valid", "test"):
            raise ValueError("Duplicate or invalid cache index entry.")
        by_key[key] = row
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    if counts != meta.get("counts"):
        raise ValueError("Cache index counts disagree with completion metadata.")
    bound = []
    for row in records:
        entry = by_key.get((row["split"], row["image_id"]))
        if entry is None or entry["label"] != str(row["label"]) or entry["age"] != str(row["age"]):
            raise ValueError(f"Cache ID/label/age mismatch: {row['image_id']}")
        source = resolve_path(row["image_path"], root)
        stat = source.stat()
        if (entry["source_path"] != str(source) or int(entry["source_size"]) != stat.st_size
                or int(entry["source_mtime_ns"]) != stat.st_mtime_ns):
            raise ValueError(f"Raw source changed since cache build: {source}")
        image_path = (directory / entry["cache_path"]).resolve()
        if not image_path.is_relative_to(directory) or not image_path.is_file():
            raise FileNotFoundError(f"Missing/unsafe cache path: {image_path}")
        size = data["image_size"]
        dimensions = (int(entry["width"]), int(entry["height"]))
        if (min(dimensions) < 1 or (max(dimensions) != size if row["split"] == "train" else dimensions != (size, size))):
            raise ValueError("Cache index dimensions do not match recipe.")
        bound.append({**row, "image_path": str(image_path), "_cache_split": row["split"]})
    return bound


def _prepare_one(task):
    from PIL import Image
    from eye_qing_guang.datasets.transforms import prepare_offline_image

    row, data, source, destination, relative = task
    source, destination = Path(source), Path(destination)
    before = source.stat()
    try:
        with Image.open(source) as image:
            prepared, _ = prepare_offline_image(image.convert("RGB"), data, row["split"])
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw image changed while decoding")
        with destination.open("xb") as handle:
            prepared.save(handle, format="PNG", compress_level=1)
    except Exception as exc:
        raise RuntimeError(f"Cache generation failed at {row['split']}/{row['image_id']}: {exc}") from exc
    return {"image_id": row["image_id"], "split": row["split"], "label": row["label"],
            "age": row["age"], "source_path": str(source), "source_size": before.st_size,
            "source_mtime_ns": before.st_mtime_ns, "cache_path": relative,
            "width": prepared.width, "height": prepared.height}


def _representative_check(records, data, root):
    """Exercise decode, deterministic transform and lossless PNG round-trip before full build."""
    import io
    import numpy as np
    from PIL import Image
    from eye_qing_guang.datasets.transforms import letterbox, fov_crop, prepare_offline_image

    checked = 0
    for split, rows in records.items():
        for row in rows[:2]:
            with Image.open(resolve_path(row["image_path"], root)) as image:
                rgb = image.convert("RGB")
                prepared, _ = prepare_offline_image(rgb, data, split)
                if split != "train":
                    old, _ = fov_crop(rgb, data["preprocessing"])
                    if not np.array_equal(prepared, letterbox(old, data["image_size"])):
                        raise ValueError("Validation/test preprocessing changed pixels.")
            buffer = io.BytesIO()
            prepared.save(buffer, format="PNG", compress_level=1)
            buffer.seek(0)
            with Image.open(buffer) as reloaded:
                if reloaded.mode != "RGB" or not np.array_equal(prepared, reloaded):
                    raise ValueError("Lossless cache round-trip failed.")
            checked += 1
    print(f"Cache preflight passed: {checked} representative images", flush=True)


def build_offline_cache(config, root, *, workers=4):
    from PIL import __version__ as pillow_version
    if config["experiment"]["protocol"] in {"timm_convnext_official_trainvalid_v1", "timm_convnext_final_fit_v1"}:
        from eye_qing_guang.datasets.convnext_official import official_records
    elif config["experiment"]["protocol"] == "timm_official_trainvalid_v1":
        from eye_qing_guang.datasets.timm_official import official_records
    else:
        raise ValueError("Offline cache build supports only the official V2-S and ConvNeXt protocols.")
    from eye_qing_guang.utils.timm_artifacts import atomic_json

    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be nonnegative; 0 runs serially.")
    data = config["data"]
    validate_cache_settings(data)
    if not data.get("cache", {}).get("enabled", False):
        raise ValueError("Use a new cache-enabled configuration to build offline images.")
    directory = resolve_path(data["cache"]["root"], root)
    raw = resolve_path(data["raw_root"], root)
    if directory == raw or directory.is_relative_to(raw):
        raise ValueError("Cache must be outside the raw dataset directory.")
    if directory.exists():
        raise FileExistsError(f"Cache exists; choose a new versioned directory: {directory}")
    records, _ = official_records(config, root)
    _representative_check(records, data, root)
    counts = {split: len(rows) for split, rows in records.items()}
    total = sum(counts.values())
    meta = {"status": "building", "signature": cache_signature(data, root), "counts": counts,
            "pillow_version": pillow_version, "processed": 0}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "metadata.json", meta)
    tasks = []
    for split, rows in records.items():
        (directory / split).mkdir()
        for number, row in enumerate(rows):
            relative = f"{split}/{number:06d}.png"
            tasks.append((row, data, str(resolve_path(row["image_path"], root)), str(directory / relative), relative))
    started, index, executor = time.monotonic(), [], None
    print(f"Full cache build started: {total} images; workers={workers}; output={directory}", flush=True)
    try:
        if workers:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            results = executor.map(_prepare_one, tasks, chunksize=8)
        else:
            results = map(_prepare_one, tasks)
        for entry in results:
            index.append(entry)
            if len(index) % 500 == 0 or len(index) == total:
                elapsed = time.monotonic() - started
                print(f"Cache: {len(index)}/{total}; elapsed={elapsed:.1f}s; images/s={len(index)/max(elapsed, 1e-9):.2f}", flush=True)
        write_csv(directory / "index.csv", index, INDEX_FIELDS)
        meta.update(status="completed", processed=len(index), seconds=time.monotonic()-started)
        atomic_json(directory / "metadata.json", meta)
    except BaseException as exc:
        meta.update(status="failed", processed=len(index), error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory / "metadata.json", meta)
        print(f"Cache failed after {len(index)}/{total} ordered results; partial files retained at {directory}", flush=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    print(f"Cache completed: {directory}", flush=True)
    return directory

"""Convert competition tables into explicit, portable image manifests."""

from __future__ import annotations

import math
import copy
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_json
from eye_qing_guang.utils.paths import find_image, image_index, resolve_path

MANIFEST_FIELDS = ["image_id", "image_path", "label", "age", "split"]


def parse_age(value: str | float | None) -> float:
    if value is None or str(value).strip().lower() in {"", "nan", "na", "none"}:
        return float("nan")
    age = float(value)
    # Preserve the official CSV value.  The supplied valid split contains two
    # finite non-negative values of 137, and age is not a model input here.
    if not math.isfinite(age) or age < 0:
        raise ValueError(f"Invalid age: {value}")
    return age


def collect_records(config: dict, root: Path, decode: bool = False) -> tuple[dict, dict]:
    """Validate table fields, names and paths; optionally decode every image."""
    data = config["data"]
    raw_root = resolve_path(data["raw_root"], root)
    records: dict[str, list[dict]] = {}
    summary = {}
    seen: set[str] = set()
    for split, spec in data["splits"].items():
        source = raw_root / spec["csv"]
        rows = read_csv(source)
        if not rows:
            raise ValueError(f"Empty table: {source}")
        required = {data["image_column"], data["age_column"]}
        if spec["labeled"]:
            required.add(data["label_column"])
        if not required <= rows[0].keys():
            raise ValueError(f"Missing columns in {source}: {required - rows[0].keys()}")
        directory = raw_root / spec["images"]
        index = image_index(directory)
        converted = []
        case_changes = 0
        for row in rows:
            name = row[data["image_column"]].strip()
            key = name.casefold()
            if not name or key in seen:
                raise ValueError(f"Empty or duplicate image identifier across splits: {name}")
            seen.add(key)
            path = find_image(name, directory, index)
            case_changes += int(path.relative_to(directory).as_posix() != name)
            if decode:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    # verify() can accept a JPEG with truncated pixel data.
                    # Reopen after verification, then exercise the same RGB
                    # decoding path as FundusDataset (without tolerating truncation).
                    with Image.open(path) as image:
                        image.load()
                        with image.convert("RGB") as rgb:
                            rgb.load()
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Image decode failed for {name}: {path}: {exc}") from exc
            label = row[data["label_column"]].strip() if spec["labeled"] else ""
            if spec["labeled"] and label not in {"0", "1"}:
                raise ValueError(f"Invalid binary label for {name}: {label}")
            age = parse_age(row[data["age_column"]])
            try:
                portable_path = path.relative_to(root).as_posix()
            except ValueError:
                portable_path = str(path)
            converted.append({
                "image_id": name, "image_path": portable_path, "label": label,
                "age": "" if math.isnan(age) else age, "split": split,
            })
        records[split] = converted
        summary[split] = {
            "samples": len(converted),
            "labels": dict(Counter(row["label"] for row in converted if row["label"] != "")),
            "missing_age": sum(row["age"] == "" for row in converted),
            "filename_case_resolved": case_changes,
            "decoded": decode,
        }
    return records, summary


def build_manifests(config: dict, root: Path, output_dir: Path) -> dict:
    """Check all records before creating a new manifest version."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    records, summary = collect_records(config, root)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in records.items():
        write_csv(output_dir / f"{split}.csv", rows, MANIFEST_FIELDS)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_test_manifest(config: dict, root: Path, output_dir: Path, decode: bool = False) -> dict:
    """Build a separate inference-only manifest; never open train or valid."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    spec = config["data"]["splits"]["test"]
    if spec["labeled"] is not False:
        raise ValueError("Official test must be unlabeled.")
    test_config = copy.deepcopy(config)
    test_config["data"]["splits"] = {"test": spec}
    records, counts = collect_records(test_config, root, decode)
    summary = {"source": "official_test", "purpose": "inference_only",
               "source_csv": spec["csv"], "splits": counts}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "test.csv", records["test"], MANIFEST_FIELDS)
    write_json(output_dir / "summary.json", summary)
    return summary


def fit_age_stats(rows: list[dict]) -> dict[str, float]:
    values = np.array([parse_age(row["age"]) for row in rows], dtype=np.float64)
    observed = values[np.isfinite(values)]
    if not len(observed):
        raise ValueError("Age fusion requires at least one observed training age.")
    median = float(np.median(observed))
    imputed = np.where(np.isfinite(values), values, median)
    return {"median": median, "mean": float(imputed.mean()),
            "std": max(float(imputed.std()), 1.0)}


def normalize_age(value: str | float, stats: dict | None) -> float:
    if stats is None:
        return 0.0
    age = parse_age(value)
    if math.isnan(age):
        age = stats["median"]
    return (age - stats["mean"]) / stats["std"]

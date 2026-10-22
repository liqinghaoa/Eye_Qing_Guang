"""One-to-one official manifests; no internal split, padding, or test labels."""

import json
import math
from pathlib import Path

from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.contracts.timm_config import EXPECTED_COUNTS, PROTOCOL
from eye_qing_guang.datasets.metadata import MANIFEST_FIELDS, collect_records, parse_age
from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path


def _same_age(left, right):
    a, b = parse_age(left), parse_age(right)
    return a == b or math.isnan(a) and math.isnan(b)


def official_records(config, root):
    records, counts = collect_records(config, root, decode=False)
    data = config["data"]
    raw = resolve_path(data["raw_root"], root)
    for split, expected in EXPECTED_COUNTS.items():
        source = read_csv(raw / data["splits"][split]["csv"])
        rows = records[split]
        if len(rows) != expected:
            raise ValueError(f"Official {split} count changed: {len(rows)} != {expected}; inspect dataset version.")
        if [r["image_id"] for r in rows] != [r[data["image_column"]] for r in source]:
            raise ValueError("Official IDs must be preserved exactly, including case, extension and order.")
        if split == "test":
            if any(r["label"] != "" for r in rows):
                raise ValueError("Official test labels must never be consumed.")
        elif {r["label"] for r in rows} != {"0", "1"}:
            raise ValueError(f"Official {split} must contain both binary classes.")
        counts[split]["source"] = f"official_{split}"
        counts[split]["source_csv"] = str(raw / data["splits"][split]["csv"])
    return records, {"protocol": PROTOCOL, "raw_root": str(raw), "splits": counts}


def ensure_official_manifests(config, root):
    records, summary = official_records(config, root)
    directory = resolve_path(config["data"]["manifest_dir"], root)
    if directory.exists():
        with (directory / "summary.json").open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol") != PROTOCOL or existing.get("raw_root") != summary["raw_root"]:
            raise ValueError("Existing manifests lack this official source provenance; choose a new manifest_dir.")
        for split in EXPECTED_COUNTS:
            rows = load_manifest(directory / f"{split}.csv", root, labeled=split != "test")
            if len(rows) != len(records[split]) or existing.get("splits", {}).get(split, {}).get("source") != f"official_{split}":
                raise ValueError(f"Existing official {split} manifest differs.")
            for row, expected in zip(rows, records[split]):
                if (any(row[k] != expected[k] for k in ("image_id", "label", "split"))
                        or resolve_path(row["image_path"], root) != resolve_path(expected["image_path"], root)
                        or not _same_age(row["age"], expected["age"])):
                    raise ValueError(f"Existing official manifest differs at {split}/{expected['image_id']}.")
        return records, summary
    directory.mkdir(parents=True, exist_ok=False)
    for split, rows in records.items():
        write_csv(directory / f"{split}.csv", rows, MANIFEST_FIELDS)
    write_json(directory / "summary.json", summary)
    return records, summary


def ordered_predictions(predictions, records, *, labeled):
    expected = [r["image_id"] for r in records]
    keys = [str(r["image_id"]) for r in predictions]
    if len(set(k.casefold() for k in keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Predictions contain duplicate, missing or unexpected official IDs.")
    by_id = {r["image_id"]: r for r in predictions}
    result = []
    for record in records:
        row = dict(by_id[record["image_id"]])
        logit, probability = float(row["logit"]), float(row["probability"])
        if not math.isfinite(logit) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Non-finite logit or invalid probability.")
        row.update(logit=logit, probability=probability)
        if labeled:
            if str(row.get("label")) != str(record["label"]):
                raise ValueError("Prediction label does not match official source.")
            row["label"] = int(record["label"])
        elif "label" in row:
            raise ValueError("Test predictions must not contain labels.")
        result.append(row)
    return result

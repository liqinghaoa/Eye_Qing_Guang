"""Validate normalized manifests without requiring image decoding."""

from pathlib import Path

from eye_qing_guang.datasets.metadata import MANIFEST_FIELDS, parse_age
from eye_qing_guang.utils.experiment_io import read_csv
from eye_qing_guang.utils.paths import resolve_path


def load_manifest(path: Path, root: Path, labeled: bool) -> list[dict]:
    rows = read_csv(path)
    if not rows or not set(MANIFEST_FIELDS) <= rows[0].keys():
        raise ValueError(f"Missing or empty manifest fields: {path}")
    seen = set()
    for row in rows:
        key = row["image_id"].casefold()
        if not key or key in seen:
            raise ValueError(f"Duplicate/empty image identifier in {path}")
        seen.add(key)
        if labeled and row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid binary label in {path}: {row['label']}")
        if not labeled and row["label"] != "":
            raise ValueError("Test manifests must not contain labels.")
        parse_age(row["age"])
        if not resolve_path(row["image_path"], root).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows

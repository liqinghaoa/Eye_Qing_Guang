"""Read configurations and persist non-overwriting experiment artifacts."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def load_config(path: Path) -> dict:
    from eye_qing_guang.contracts.config import validate_config

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def new_experiment(root: Path, name: str) -> Path:
    if not name or name in {".", ".."} or Path(name).name != name or any(
        char in name for char in '/\\:<>"|?*'
    ):
        raise ValueError("Experiment name must be a single safe directory name.")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    for part in ("checkpoints", "reports", "predictions", "figures"):
        (directory / part).mkdir()
    return directory


def run_manifest(status: str, **details: Any) -> dict:
    return {
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        **details,
    }

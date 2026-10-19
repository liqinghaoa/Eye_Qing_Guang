"""Summarize completed evaluation reports without selecting a new threshold."""

import json
from pathlib import Path

from eye_qing_guang.utils.experiment_io import write_csv


def summarize_experiments(experiments: list[Path], output: Path) -> list[dict]:
    rows = []
    for experiment in experiments:
        for path in sorted((experiment / "reports").rglob("metrics.json")):
            with path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            rows.append({"experiment": experiment.name,
                         "report": path.relative_to(experiment).as_posix(), **metrics})
    if not rows:
        raise ValueError("No metrics.json reports found.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output, rows, fields)
    return rows

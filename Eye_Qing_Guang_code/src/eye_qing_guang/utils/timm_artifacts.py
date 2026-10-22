"""Atomic artifacts owned by one newly allocated timm run."""

import csv
import json
import os
import tempfile
from pathlib import Path

from eye_qing_guang.pipelines.inference import build_submission
from eye_qing_guang.utils.experiment_io import read_csv


def atomic_write(path, writer, *, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Same-filesystem link publishes atomically and fails if final already exists.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value, *, replace=True):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(encoded), replace=replace)


def atomic_csv(path, rows, fields):
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode("utf-8")
    atomic_write(path, lambda handle: handle.write(encoded))


def publish_submission(predictions, official_csv, destination, *, source_id_column="Image_name"):
    destination = Path(destination)
    if destination.name != "submission.csv":
        raise ValueError("Competition output must be named submission.csv.")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "submission.csv"
        build_submission(predictions, official_csv, candidate, source_id_column=source_id_column,
                         mode="competition", threshold=0.5)
        source = read_csv(official_csv)
        scores = {r["image_id"]: float(r["probability"]) for r in read_csv(predictions)}
        with candidate.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["Image_name", "Probability", "Pred_Label"]:
                raise ValueError("Invalid submission header or unexpected index.")
            rows = list(reader)
        if len(rows) != len(source):
            raise ValueError("Submission count changed during serialization.")
        for row, official in zip(rows, source):
            expected = official[source_id_column]
            if (row["Image_name"] != expected or float(row["Probability"]) != scores[expected]
                    or row["Pred_Label"] not in {"0", "1"}
                    or int(row["Pred_Label"]) != int(float(row["Probability"]) >= 0.5)):
                raise ValueError("Submission round-trip changed ID/probability binding or threshold labels.")
        atomic_write(destination, lambda handle: handle.write(candidate.read_bytes()), replace=False)
    return destination

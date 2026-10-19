"""Freeze an internal development split using official TRAIN images only."""

import copy
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.metadata import MANIFEST_FIELDS, collect_records
from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_json
from eye_qing_guang.utils.paths import resolve_path


def connected_groups(rows, associations):
    """Union all trusted patient/exam/confirmed-duplicate edges simultaneously."""
    ids = {row["image_id"].casefold() for row in rows}
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    anchors = {}
    for association in associations:
        key = association["image_id"].casefold()
        kind, group = association["association_type"], association["group_id"].strip()
        if key not in ids or kind not in {"patient", "examination", "confirmed_duplicate"} or not group:
            raise ValueError("Associations require known train IDs, trusted types and nonempty group IDs.")
        anchor = anchors.setdefault((kind, group), key)
        a, b = find(anchor), find(key)
        parent[max(a, b)] = min(a, b)
    return {key: find(key) for key in ids}


def development_records(config, root):
    source_config = copy.deepcopy(config)
    source_config["data"]["splits"] = {"train": config["data"]["splits"]["train"]}
    records, _ = collect_records(source_config, root)
    rows = sorted(records["train"], key=lambda row: row["image_id"].casefold())
    labels = np.array([int(row["label"]) for row in rows])
    path = config["development"]["associations_csv"]
    associations = read_csv(resolve_path(path, root)) if path else []
    if path and not associations:
        raise ValueError("The explicitly supplied associations table is empty.")
    groups = connected_groups(rows, associations)
    group_ids = [groups[row["image_id"].casefold()] for row in rows]
    grouped = len(set(group_ids)) < len(rows)
    seed = config["development"]["split_seed"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        if grouped:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            fit, dev = next(splitter.split(np.zeros(len(rows)), labels, group_ids))
            method = "StratifiedGroupKFold_5_fold0"
        else:
            fit, dev = train_test_split(np.arange(len(rows)), test_size=0.2,
                                        random_state=seed, shuffle=True, stratify=labels)
            method = "stratified_train_test_split_0.2"
    result = {}
    counts = {}
    for role, indices in (("train", fit), ("valid", dev)):
        subset = [{**rows[i], "split": role, "source_split": "official_train",
                   "group_id": group_ids[i]} for i in sorted(indices)]
        if {row["label"] for row in subset} != {"0", "1"}:
            raise ValueError("Both internal subsets must contain both classes; no ungrouped fallback.")
        fraction = len(subset) / len(rows)
        if role == "valid" and not 0.1 <= fraction <= 0.3:
            raise ValueError("Internal development fraction outside frozen usability range [0.1,0.3].")
        result[role] = subset
        positive = sum(int(row["label"]) for row in subset)
        counts[role] = {"samples": len(subset), "positives": positive,
                        "positive_rate": positive / len(subset)}
    overlap = {row["group_id"] for row in result["train"]} & {row["group_id"] for row in result["valid"]}
    if overlap:
        raise ValueError("Internal association-group overlap.")
    summary = {
        "protocol": "b0_dev_v1", "source": "official_train", "split_seed": seed,
        "source_csv": config["data"]["splits"]["train"]["csv"],
        "method": method, "scikit_learn": sklearn.__version__, "counts": counts,
        "association_types": dict(Counter(row["association_type"] for row in associations)),
        "associations_csv": path, "multi_sample_groups": grouped,
        "group_overlap": 0, "usability_dev_fraction_range": [0.1, 0.3],
        "limitations": ["Patient/eye linkage is only as complete as supplied trusted associations.",
                        "Without complete patient IDs, same-patient eyes across subsets cannot be excluded.",
                        "No automatic near-duplicate or patient-identity detection is performed."],
    }
    return result, summary


def build_development_manifests(config, root, output):
    if output.exists():
        raise FileExistsError(output)
    records, summary = development_records(config, root)
    output.mkdir(parents=True, exist_ok=False)
    for role, rows in records.items():
        write_csv(output / f"{role}.csv", rows, MANIFEST_FIELDS + ["source_split", "group_id"])
    write_json(output / "summary.json", summary)
    return summary


def load_development_manifests(config, root):
    """Recheck the frozen split against current official-train metadata/associations."""
    directory = resolve_path(config["data"]["manifest_dir"], root)
    with (directory / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected, expected_summary = development_records(config, root)
    if summary != expected_summary:
        raise ValueError("Development provenance/version changed; regenerate a new manifest version.")
    records = {}
    for role in ("train", "valid"):
        rows = load_manifest(directory / f"{role}.csv", root, True)
        if [{key: str(value) for key, value in row.items()} for row in expected[role]] != rows:
            raise ValueError("Frozen manifest differs from the official-train development protocol.")
        records[role] = rows
    return records, summary

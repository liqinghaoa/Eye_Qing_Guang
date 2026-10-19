"""Image dataset shared by training, evaluation and unlabeled inference."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from eye_qing_guang.datasets.metadata import normalize_age
from eye_qing_guang.utils.paths import resolve_path


class FundusDataset(Dataset):
    def __init__(self, records: list[dict], root: Path, transform, age_stats=None):
        self.records = records
        self.root = root
        self.transform = transform
        self.age_stats = age_stats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(resolve_path(row["image_path"], self.root)) as source:
            image = self.transform(source.convert("RGB"))
        sample = {
            "image": image,
            "age": torch.tensor([normalize_age(row["age"], self.age_stats)], dtype=torch.float32),
            "image_id": row["image_id"],
        }
        if row["label"] != "":
            sample["label"] = torch.tensor([float(row["label"])], dtype=torch.float32)
        return sample

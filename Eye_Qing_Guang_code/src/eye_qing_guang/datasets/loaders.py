"""Reproducible data loading without dropping validation/test samples."""

import torch
from torch.utils.data import DataLoader

from eye_qing_guang.datasets.fundus import FundusDataset
from eye_qing_guang.datasets.transforms import build_transforms
from eye_qing_guang.utils.reproducibility import seed_worker


def make_loader(records, config, root, age_stats=None, training=False, device=None):
    dataset = FundusDataset(
        records, root, build_transforms(config["data"], training), age_stats,
    )
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    if device is None:
        from eye_qing_guang.utils.reproducibility import choose_device

        device = choose_device(config["training"]["device"])
    batch_size = config["training"]["batch_size"] if training else config["training"].get(
        "eval_batch_size", config["training"]["batch_size"])
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=training,
        num_workers=int(config["data"].get("num_workers", 0)), drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda",
    )

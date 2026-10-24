"""Reproducible data loading without dropping validation/test samples."""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from eye_qing_guang.datasets.fundus import FundusDataset
from eye_qing_guang.datasets.transforms import build_transforms
from eye_qing_guang.utils.reproducibility import seed_worker


def make_loader(records, config, root, age_stats=None, training=False, device=None,
                distributed=None):
    if config["data"].get("cache", {}).get("enabled", False):
        from eye_qing_guang.datasets.offline_cache import cached_records
        records = cached_records(records, config["data"], root)
    dataset = FundusDataset(
        records, root, build_transforms(config["data"], training), age_stats,
    )
    seed = int(config["experiment"]["seed"])
    rank = distributed.rank if distributed is not None else 0
    generator = torch.Generator().manual_seed(seed + rank)
    if device is None:
        from eye_qing_guang.utils.reproducibility import choose_device

        device = choose_device(config["training"]["device"])
    batch_size = config["training"]["batch_size"] if training else config["training"].get(
        "eval_batch_size", config["training"]["batch_size"])
    sampler, options = None, {}
    if distributed is not None:
        if training:
            if not records or len(records) % distributed.world_size:
                raise ValueError("DDP training sample count must be positive and divisible by four; no padding/dropping.")
            sampler = DistributedSampler(dataset, num_replicas=distributed.world_size,
                                         rank=rank, shuffle=True, seed=seed, drop_last=False)
        else:
            sampler = list(range(rank, len(dataset), distributed.world_size))
    # Worker safety is independent of sharding: root-only checkpoint evaluation
    # also runs while the DDP process group is alive, without a distributed sampler.
    single_timm = (config["experiment"].get("protocol") in {"timm_official_trainvalid_v1", "timm_convnext_official_trainvalid_v1"}
                   and config.get("distributed", {}).get("enabled") is False)
    if config["data"].get("num_workers", 0) and (single_timm or config.get("distributed", {}).get("enabled", False)):
        options["multiprocessing_context"] = config["distributed"]["multiprocessing_context"]
        if config.get("diagnostics", {}).get("enabled", False):
            options["timeout"] = config["diagnostics"]["data_timeout_seconds"]
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=training and sampler is None, sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 0)), drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda",
        **options,
    )

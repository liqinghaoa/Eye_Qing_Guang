"""Four CPU/Gloo synthetic workers; NOT the PPU source-engine acceptance test."""

import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn

from eye_qing_guang.pipelines.timm_official import distributed_validation, execute_preflight
from eye_qing_guang.trainers.timm_official import build_optimizer_schedule, pw5_loss, update_group
from eye_qing_guang.utils.distributed import DistributedContext
from eye_qing_guang.utils.timm_source import B0DDPRuntime
from test_timm_official import synthetic_config


class BufferedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.register_buffer("offset", torch.zeros(1))

    def forward(self, images, ages=None):
        return self.linear(images.mean(dim=tuple(range(1, images.ndim))).reshape(-1, 1))+self.offset


def worker(rank, rendezvous, root_string):
    torch.set_num_threads(1)
    root, device = Path(root_string), torch.device("cpu")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    context = DistributedContext(rank, 4, device)
    runtime = B0DDPRuntime(synthetic_config(), context=context)
    config = synthetic_config()
    settings = {**config["training"], "precision": "fp32"}
    try:
        model = BufferedModel()
        model = runtime.initialize_model(model)
        initial = {k: v.clone() for k, v in runtime.unwrap_model(model).state_dict().items()}
        group = [{"image": torch.tensor([[rank+.2], [rank+.7]]), "label": torch.tensor([[0.0], [1.0]])},
                 {"image": torch.tensor([[rank+.4]]), "label": torch.tensor([[float(rank % 2)]])}]
        optimizer, schedule = build_optimizer_schedule(model, settings, 2)
        update_group(model, group, optimizer, schedule, device, settings, runtime, step=False)
        model = runtime.unwrap_model(model)
        expected = BufferedModel()
        expected.load_state_dict(initial)
        all_images = torch.tensor([[r+.2] for r in range(4)]+[[r+.7] for r in range(4)]+[[r+.4] for r in range(4)])
        all_labels = torch.tensor([[0.0]]*4+[[1.0]]*4+[[float(r % 2)] for r in range(4)])
        pw5_loss(device)(expected(all_images), all_labels).backward()
        for a, b in zip(model.parameters(), expected.parameters()):
            assert torch.allclose(a.grad, b.grad, atol=2e-6), (rank, a.grad, b.grad)
        model.offset.fill_(rank+.5)
        records = [{"image_id": f"valid_{i}.JPG", "image_path": f"valid_{i}.JPG",
                    "label": str(i % 2), "age": "", "split": "valid"} for i in range(5)]
        rows, metrics = distributed_validation(model, records, config, root, device, runtime)
        assert model.offset.item() == rank+.5, "Rank-local BN buffers were not restored after evaluation"
        assert metrics["n_samples"] == 5
        if rank == 0:
            assert [r["image_id"] for r in rows] == [r["image_id"] for r in records]
        try:
            runtime.primary("synthetic_disk_failure", lambda: (_ for _ in ()).throw(OSError("disk full")))
        except RuntimeError as exc:
            assert "disk full" in str(exc)
        else:
            raise AssertionError("Primary write failure was not broadcast to all ranks")
        class NoTestRecords(dict):
            def __getitem__(self, key):
                if key == "test":
                    raise AssertionError("Preflight tried to read official test records")
                return super().__getitem__(key)
        all_records = NoTestRecords({split: [{"image_id": f"synthetic_{i}.JPG", "image_path": f"synthetic_{i}.JPG",
            "label": str(i % 2), "age": "", "split": split} for i in range(count)]
            for split, count in (("train", 16), ("valid", 8))})
        metadata = {"timm_version": "1.0.22", "num_features": 1280}
        with patch("eye_qing_guang.pipelines.timm_official.build_timm_model", side_effect=lambda *a, **k: (BufferedModel(), metadata)), \
             patch("eye_qing_guang.contracts.timm_checkpoints.build_timm_model", side_effect=lambda *a, **k: (BufferedModel(), metadata)):
            report = execute_preflight(config, all_records, root, device, runtime, root / "preflight_only", 64)
        assert report["consumed_sample_instances"] == 56
        assert report["status"] == "completed"
        assert report["target_batch_memory"] == "not_verified"
        assert report["official_test_predicted"] is False
        (root / f"rank_{rank}.json").write_text(json.dumps({"global_metrics": metrics, "rank": rank,
            "native_buffer_restored": True, "primary_error_received": True}), encoding="utf-8")
    finally:
        runtime.close()


class TimmDistributedTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU/Gloo is unavailable")
    def test_four_workers_gradient_mean_unequal_validation_and_io_failure(self):
        with tempfile.TemporaryDirectory(prefix="eqg_timm_gloo_synthetic_") as directory:
            root = Path(directory)
            for i in range(5):
                Image.new("RGB", (16, 18), (30+i*20, 40, 50)).save(root / f"valid_{i}.JPG")
            for i in range(16):
                Image.new("RGB", (16, 18), (30+i*8, 40, 50)).save(root / f"synthetic_{i}.JPG")
            rendezvous = (root / "rendezvous").as_uri()
            mp.spawn(worker, args=(rendezvous, directory), nprocs=4, join=True)
            results = [json.loads((root / f"rank_{i}.json").read_text()) for i in range(4)]
            self.assertTrue(all(r["primary_error_received"] and r["native_buffer_restored"] for r in results))
            self.assertTrue(all(r["global_metrics"] == results[0]["global_metrics"] for r in results))


if __name__ == "__main__":
    unittest.main()

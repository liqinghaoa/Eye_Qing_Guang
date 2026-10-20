"""Four-process synthetic DDP correctness checks; no competition data or PPU run."""

import copy
import json
import math
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from eye_qing_guang.contracts.checkpoints import load_checkpoint, save_checkpoint
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.pipelines.b0_development import new_progress
from eye_qing_guang.pipelines.b0_distributed import (
    distributed_predictions, global_epoch_metrics, global_progress, ordered_predictions,
)
from eye_qing_guang.pipelines.distributed_preflight import validate_budget, distributed_fp32_reference
from eye_qing_guang.pipelines.preflight import ReplayLoader, clipped_gradients
from eye_qing_guang.trainers.binary import finish_update, make_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import DevelopmentStopping, SuccessfulUpdateSchedule
from eye_qing_guang.utils.distributed import DistributedContext, close_distributed, initialize_distributed
from eye_qing_guang.utils.experiment_io import load_config, new_experiment, read_csv
from eye_qing_guang.pipelines.inference import predict_test
from eye_qing_guang.pipelines.evaluation import evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.register_buffer("shared_buffer", torch.tensor([0.0]))

    def forward(self, image, age=None):
        return self.linear(image)


class PixelModel(LinearModel):
    def forward(self, image, age=None):
        return self.linear(image.mean(dim=(1, 2, 3)).reshape(-1, 1))


def samples(count):
    return [{"image": torch.tensor([index / 30]), "age": torch.zeros(1), "label": torch.tensor([float(index % 2)]),
             "image_id": f"image_{index:03d}"} for index in range(count)]


def four_rank_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=90))
    context = DistributedContext(rank, 4, torch.device("cpu"))
    try:
        config = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml")
        settings = {**config["training"], "batch_size": 2, "accumulation_steps": 2, "amp": False}
        data = samples(28)  # seven/rank -> groups of four and three/rank, both have two microbatches
        sampler = DistributedSampler(data, num_replicas=4, rank=rank, seed=42)
        sampler.set_epoch(3)
        indices = list(sampler)
        all_indices = context.gather_json(indices)
        if context.primary:
            assert sorted(index for shard in all_indices for index in shard) == list(range(28))
        torch.manual_seed(123)
        model = DistributedDataParallel(LinearModel())
        initial = copy.deepcopy(model.module.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 2, settings)
        progress = new_progress()
        reductions = {"count": 0}

        def communication_hook(state, bucket):
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

            state["count"] += 1
            return allreduce_hook(None, bucket)

        model.register_comm_hook(reductions, communication_hook)
        loader = DataLoader(data, batch_size=2, sampler=sampler)
        criterion = nn.BCEWithLogitsLoss()
        local = train_accumulated_epoch(model, loader, optimizer, criterion, context.device, settings,
                                         make_scaler(False), schedule, progress, distributed=context)
        metrics = global_epoch_metrics(local, context)
        assert reductions["count"] == 2, reductions
        assert metrics["consumed_samples"] == 28 and metrics["successful_samples"] == 28
        assert metrics["successful_updates"] == 2 and metrics["attempted_groups"] == 2
        assert progress["consumed_samples"] == 7 and schedule.successful_updates == 2

        def compare_global_reference():
            reference = LinearModel()
            reference.load_state_dict(initial)
            opt = torch.optim.SGD(reference.parameters(), lr=0.001)
            for group in range(2):
                group_indices = [index for shard in all_indices for index in shard[group*4:(group+1)*4]]
                x = torch.stack([data[index]["image"] for index in group_indices])
                y = torch.stack([data[index]["label"] for index in group_indices])
                opt.zero_grad()
                criterion(reference(x), y).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), settings["grad_clip"])
                opt.param_groups[0]["lr"] = schedule.value(group + 1)
                opt.step()
            for actual, expected in zip(model.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

        context.on_primary(compare_global_reference)
        # No padding in an uneven validation split; rank 3 gets an empty shard.
        validation = samples(3)
        model.module.shared_buffer.fill_(rank)
        validation_loader = DataLoader(validation, batch_size=2, sampler=list(range(rank, 3, 4)))
        predictions, loss = distributed_predictions(model.module, validation_loader, validation, criterion, context)
        assert model.module.shared_buffer.item() == 0
        if context.primary:
            assert [row["image_id"] for row in predictions] == [row["image_id"] for row in validation]
            assert math.isfinite(loss)
        stopping = DevelopmentStopping(30, 15, 1e-4)

        def stop_decision():
            for epoch in range(1, 31):
                improved, stop = stopping.update(epoch, 0.7)
            return {"state": stopping.state_dict(), "stop": stop}

        decision = context.on_primary(stop_decision)
        stopping = DevelopmentStopping(**decision["state"])
        assert decision["stop"] and stopping.stale == 29 and stopping.best_epoch == 1

        # One rank injects overflow BEFORE DDP gradient synchronization. All ranks skip the first group.
        overflow_model = DistributedDataParallel(LinearModel())
        overflow_optimizer = torch.optim.SGD(overflow_model.parameters(), lr=0.001)
        overflow_schedule = SuccessfulUpdateSchedule(overflow_optimizer, 2, settings)
        overflow_progress = new_progress()
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        hook = overflow_model.module.linear.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf")) if rank == 2 else gradient)

        def remove_hook(row):
            if row["attempted_group"] == 1:
                assert not row["success"] and row["scale_after"] == 4
                hook.remove()

        local = train_accumulated_epoch(overflow_model, loader, overflow_optimizer, criterion, context.device, {**settings, "amp": True},
                                         scaler, overflow_schedule, overflow_progress, remove_hook, context)
        metrics = global_epoch_metrics(local, context)
        assert metrics["skipped_groups"] == 1 and metrics["successful_updates"] == 1
        assert metrics["successful_samples"] == 12 and metrics["consumed_samples"] == 28
        assert overflow_schedule.successful_updates == 1 and scaler.get_scale() == 4

        # A post-synchronization local corruption must fail BEFORE any optimizer step.
        overflow_optimizer.zero_grad(set_to_none=True)
        scaler.scale(criterion(overflow_model(torch.ones(2, 1)), torch.zeros(2, 1))).backward()
        before = [p.detach().clone() for p in overflow_model.parameters()]
        if rank == 1:
            overflow_model.module.linear.weight.grad.fill_(float("inf"))
        try:
            finish_update(overflow_model, overflow_optimizer, scaler, 5, context)
        except RuntimeError as exc:
            assert "disagree" in str(exc)
        else:
            raise AssertionError("Divergent finite/overflow decisions were accepted")
        for actual, expected in zip(overflow_model.parameters(), before):
            assert torch.equal(actual.detach(), expected)

        # Independent FP32 weighted reference retains identical distributed gradients.
        replay = ReplayLoader(list(loader), 2)
        reference = distributed_fp32_reference(model, replay, criterion, context, settings, new_progress())
        assert math.isfinite(reference["loss"]) and reference["gradients"]

        def save_and_reload():
            directory = new_experiment(Path(output), "rank0_only")
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            save_checkpoint(directory / "checkpoints/model.pt", model, opt, 1, config, None, 1, {}, {"resume_supported": False})
            with patch("eye_qing_guang.contracts.checkpoints.build_model", return_value=LinearModel()):
                loaded, payload = load_checkpoint(directory / "checkpoints/model.pt", torch.device("cpu"))
            assert all(not name.startswith("module.") for name in payload["model_state"])
            for actual, expected in zip(loaded.parameters(), model.module.parameters()):
                torch.testing.assert_close(actual, expected)
            # Single-process prediction rejects missing B0 test manifest before opening any data.
            with patch("eye_qing_guang.pipelines.inference.load_checkpoint", return_value=(loaded, payload)):
                try:
                    predict_test(directory / "checkpoints/model.pt", Path(output), directory / "predictions/missing", "cpu")
                except ValueError as exc:
                    assert "--manifest" in str(exc)
                else:
                    raise AssertionError("DDP checkpoint bypassed explicit test-manifest protection")
            with patch("eye_qing_guang.pipelines.evaluation.load_checkpoint", return_value=(loaded, payload)):
                try:
                    evaluate_checkpoint(directory / "checkpoints/model.pt", Path(output), directory / "reports/external",
                                        device_name="cpu", manifest=Path(output) / "external.csv")
                except ValueError as exc:
                    assert "evaluation-source" in str(exc)
                else:
                    raise AssertionError("DDP checkpoint bypassed evaluation provenance protection")
            (Path(output) / "result.json").write_text(json.dumps({"ranks": 4, "tail_global_samples": 12,
                "global_reference": "passed", "overflow_consensus": "passed", "empty_validation_shard": "passed"}))

        context.on_primary(save_and_reload)
        try:
            context.on_primary(lambda: (_ for _ in ()).throw(FileExistsError("occupied_output")))
        except (FileExistsError, RuntimeError):
            pass
        else:
            raise AssertionError("Root I/O failure was not propagated")
        close_distributed()

        # Exercise both orchestration paths with tiny synthetic PNGs and patched CPU-only initialization.
        # This is NOT a target ResNet50/PPU/FP16 acceptance run.
        from eye_qing_guang.pipelines.b0_distributed import train_distributed
        from eye_qing_guang.pipelines.distributed_preflight import preflight_distributed

        output_root = Path(output)
        records = {"train": [], "valid": []}
        for index in range(11):
            path = output_root / f"synthetic_{index}.png"
            if rank == 0:
                Image.new("RGB", (8, 8), (60 + index*10, 80, 100)).save(path)
            role = "train" if index < 8 else "valid"
            records[role].append({"image_id": path.name, "image_path": str(path), "age": "50", "label": str(index % 2)})
        config["experiment"]["output_root"] = str(output_root)
        # Exercise nonzero spawn workers with a live process group. Tiny root-only
        # checkpoint evaluation now deliberately avoids spawning a second pool.
        config["data"]["num_workers"] = 1
        config["training"].update(batch_size=1, accumulation_steps=2, eval_batch_size=2, epochs=2, min_epochs=1, patience=15)
        provenance = {"purpose": "synthetic_DDP_unit_test"}
        dist.init_process_group("gloo", init_method=(output_root / "preflight_rendezvous").as_uri(),
                                rank=rank, world_size=4, timeout=timedelta(seconds=90))
        with patch("eye_qing_guang.pipelines.distributed_preflight.initialize_distributed", return_value=context), patch(
            "eye_qing_guang.pipelines.distributed_preflight.load_development_manifests", return_value=(records, provenance)), patch(
            "eye_qing_guang.pipelines.distributed_preflight.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.contracts.checkpoints.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.pipelines.distributed_preflight.make_training_scaler", side_effect=lambda settings: torch.amp.GradScaler("cpu", init_scale=8)):
            directory = preflight_distributed(config, output_root, "synthetic_ddp_preflight", max_samples=16, max_groups=2)
        if rank == 0:
            report = json.loads((directory / "reports/preflight.json").read_text())
            assert report["status"] == "completed", report
            assert report["progress"]["consumed_samples"] == 16 and report["progress"]["reference_samples"] == 8
            assert report["progress"]["attempted_groups"] == 2
            assert list(report["stage_seconds"]).index("fp32_reference") < list(report["stage_seconds"]).index("fp16_update")

        # Explicit FP32 and BF16 orchestration, including checkpoint reload. CPU BF16
        # here tests control flow only; the actual BF16 arithmetic is tested on CUDA.
        config["data"]["num_workers"] = 0
        for precision in ("fp32", "bf16", "fp16_overflow"):
            config["training"].update(precision="fp16" if precision == "fp16_overflow" else precision,
                                      amp=precision != "fp32")
            dist.init_process_group("gloo", init_method=(output_root / f"{precision}_rendezvous").as_uri(),
                                    rank=rank, world_size=4, timeout=timedelta(seconds=90))

            def target_update(*args, **kwargs):
                assert args[8]["reference_samples"] == 2  # FP32 reference ran before the target.
                injected = args[0].module.linear.weight.register_hook(
                    lambda gradient: torch.full_like(gradient, float("inf")) if rank == 2 else gradient)
                try:
                    return train_accumulated_epoch(*args, **kwargs)
                finally:
                    injected.remove()

            with patch("eye_qing_guang.pipelines.distributed_preflight.initialize_distributed", return_value=context), patch(
                "eye_qing_guang.pipelines.distributed_preflight.load_development_manifests", return_value=(records, provenance)), patch(
                "eye_qing_guang.pipelines.distributed_preflight.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
                "eye_qing_guang.contracts.checkpoints.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
                "eye_qing_guang.pipelines.distributed_preflight.check_runtime_precision",
                return_value={"precision": precision, "probe": "CPU_test_control_flow_only"}), patch(
                "eye_qing_guang.pipelines.distributed_preflight.make_training_scaler",
                side_effect=lambda settings: torch.amp.GradScaler("cpu", enabled=precision == "fp16_overflow", init_scale=8)), patch(
                "eye_qing_guang.pipelines.distributed_preflight.train_accumulated_epoch",
                side_effect=target_update if precision == "fp16_overflow" else train_accumulated_epoch):
                directory = preflight_distributed(config, output_root, f"synthetic_{precision}",
                                                  max_samples=8 if precision == "fp32" else 24,
                                                  max_groups=1 if precision == "fp32" else 3)
            if rank == 0:
                report = json.loads((directory / "reports/preflight.json").read_text())
                if precision == "fp16_overflow":
                    assert report["status"] == "inconclusive", report
                    assert report["fp32_reference"]["status"] == "passed"
                    assert report["precision_comparison"]["reason"] == "fp16_gradient_overflow_after_completed_fp32_reference"
                    assert report["progress"]["consumed_samples"] == 16 and len(report["update_attempts"]) == 1
                    assert not (directory / "checkpoints/preflight.pt").exists()
                else:
                    assert report["status"] == "completed", report
                    assert report["checkpoint_single_process_load"] == "passed"
                    assert report["progress"]["consumed_samples"] == (8 if precision == "fp32" else 16)
                    assert report["progress"]["successful_updates"] == 1

        dist.init_process_group("gloo", init_method=(output_root / "train_rendezvous").as_uri(),
                                rank=rank, world_size=4, timeout=timedelta(seconds=90))
        config["data"]["num_workers"] = 0
        config["training"]["amp"] = False
        config["training"]["precision"] = "fp32"
        with patch("eye_qing_guang.pipelines.b0_distributed.initialize_distributed", return_value=context), patch(
            "eye_qing_guang.pipelines.b0_distributed.load_development_manifests", return_value=(records, provenance)), patch(
            "eye_qing_guang.pipelines.b0_distributed.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.pipelines.b0_distributed.write_development_analysis"):
            directory = train_distributed(config, output_root, "synthetic_ddp_training")
        if rank == 0:
            report = json.loads((directory / "run_manifest.json").read_text())
            assert report["status"] == "completed", report
            assert report["completed_epochs"] == 2 and report["progress"]["consumed_samples"] == 16
            assert (directory / "checkpoints/best_inference.pt").exists()
            history = (directory / "reports/history.csv").read_text()
            assert "complete_epoch_seconds" in history and "report_write_seconds" in history
            rows = read_csv(directory / "reports/history.csv")
            assert len(rows) == 2 and len({row["epoch"] for row in rows}) == 2

        # Resume a completed epoch in a new directory, with the same global batch
        # but a changed physical batch. The inherited best must survive a tie.
        source = output_root / "synthetic_ddp_training/checkpoints/epoch_0001.pt"
        source_payload = torch.load(source, map_location="cpu", weights_only=True)
        resumed = copy.deepcopy(config)
        resumed["training"].update(batch_size=2, accumulation_steps=1)
        resumed["diagnostics"] = {"enabled": True, "data_timeout_seconds": 10, "stack_timeout_seconds": 5,
                                  "device_poll_seconds": 10, "device_poll_timeout_seconds": 1}
        dist.init_process_group("gloo", init_method=(output_root / "resume_rendezvous").as_uri(),
                                rank=rank, world_size=4, timeout=timedelta(seconds=90))

        def verify_restored_state(*args, **kwargs):
            assert args[7].successful_updates == 1
            assert args[8]["consumed_samples"] == 2 and args[8]["attempted_groups"] == 1
            assert args[1].batch_size == 2
            for key, value in args[0].module.state_dict().items():
                torch.testing.assert_close(value, source_payload["model_state"][key], rtol=0, atol=0)
            return train_accumulated_epoch(*args, **kwargs)

        with patch("eye_qing_guang.pipelines.b0_distributed.initialize_distributed", return_value=context), patch(
            "eye_qing_guang.pipelines.b0_distributed.load_development_manifests", return_value=(records, provenance)), patch(
            "eye_qing_guang.pipelines.b0_distributed.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.contracts.checkpoints.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.pipelines.b0_distributed.train_accumulated_epoch", side_effect=verify_restored_state), patch(
            "eye_qing_guang.pipelines.b0_distributed.write_development_analysis"):
            directory = train_distributed(resumed, output_root, "synthetic_ddp_resumed", resume_from=source)
        if rank == 0:
            report = json.loads((directory / "run_manifest.json").read_text())
            assert report["status"] == "completed" and report["completed_epochs"] == 2
            assert report["newly_completed_epochs"] == 1 and report["resume"]["source_epoch"] == 1
            rows = read_csv(directory / "reports/updates.csv")
            assert len(rows) == 1 and rows[0]["epoch"] == "2" and rows[0]["successful_update_index"] == "2"
            assert (directory / "checkpoints/epoch_0001.pt").exists()
            assert (directory / "checkpoints/epoch_0002.pt").exists()
            assert (directory / "reports/diagnostics_rank_0000.jsonl").exists()
            assert torch.load(directory / "checkpoints/best.pt", weights_only=True)["epoch"] == 1

        # Change GLOBAL batch, then resume the resulting checkpoint again. The
        # counter origin must include the one old group, not epoch * new groups.
        changed = copy.deepcopy(config)
        changed["training"].update(batch_size=1, accumulation_steps=1)
        next_source = source
        for final_epoch in (2, 3):
            changed["training"]["epochs"] = final_epoch
            dist.init_process_group("gloo", init_method=(output_root / f"batch_change_{final_epoch}_rendezvous").as_uri(),
                                    rank=rank, world_size=4, timeout=timedelta(seconds=90))
            with patch("eye_qing_guang.pipelines.b0_distributed.initialize_distributed", return_value=context), patch(
                "eye_qing_guang.pipelines.b0_distributed.load_development_manifests", return_value=(records, provenance)), patch(
                "eye_qing_guang.pipelines.b0_distributed.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
                "eye_qing_guang.contracts.checkpoints.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
                "eye_qing_guang.pipelines.b0_distributed.write_development_analysis"):
                directory = train_distributed(changed, output_root, f"synthetic_changed_batch_{final_epoch}",
                                              resume_from=next_source)
            next_source = directory / "checkpoints" / f"epoch_{final_epoch:04d}.pt"
            if rank == 0:
                saved = torch.load(next_source, weights_only=True)
                report = json.loads((directory / "run_manifest.json").read_text())
                assert report["status"] == "completed", report
                assert saved["training_state"]["progress"]["attempted_groups"] == 1 + (final_epoch - 1) * 2
                assert saved["training_state"]["scheduler"]["successful_updates"] == 1 + (final_epoch - 1) * 2
                assert saved["training_state"]["progress"]["consumed_samples"] == final_epoch * 8
                assert saved["training_state"]["scheduler"]["total"] == 1 + 79 * 2
                assert report["resume"]["batch_transition"]["target_groups_per_epoch"] == 2
                assert len(read_csv(directory / "reports/history.csv")) == 1
                for values in saved["optimizer_state"]["state"].values():
                    assert int(values["step"].item()) == 1 + (final_epoch - 1) * 2

        # Interrupt before epoch 2: epoch 1 timing must already be on disk,
        # without relying on final artifact generation or a CSV rewrite.
        dist.init_process_group("gloo", init_method=(output_root / "interrupted_rendezvous").as_uri(),
                                rank=rank, world_size=4, timeout=timedelta(seconds=90))
        attempts = 0

        def interrupt_second_epoch(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise KeyboardInterrupt("synthetic interruption before epoch 2")
            return train_accumulated_epoch(*args, **kwargs)

        with patch("eye_qing_guang.pipelines.b0_distributed.initialize_distributed", return_value=context), patch(
            "eye_qing_guang.pipelines.b0_distributed.load_development_manifests", return_value=(records, provenance)), patch(
            "eye_qing_guang.pipelines.b0_distributed.build_model", side_effect=lambda *a, **k: PixelModel()), patch(
            "eye_qing_guang.pipelines.b0_distributed.train_accumulated_epoch", side_effect=interrupt_second_epoch):
            try:
                train_distributed(config, output_root, "synthetic_ddp_interrupted")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Expected synthetic interruption")
        if rank == 0:
            interrupted = output_root / "synthetic_ddp_interrupted"
            rows = read_csv(interrupted / "reports/history.csv")
            assert len(rows) == 1 and rows[0]["epoch"] == "1"
            row = rows[0]
            assert float(row["report_write_seconds"]) >= 0
            assert float(row["complete_epoch_seconds"]) >= float(row["epoch_seconds"])
            assert not (interrupted / "checkpoints/last.pt").exists()
            assert json.loads((interrupted / "run_manifest.json").read_text())["status"] == "failed"
    finally:
        close_distributed()


class DDPTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml")

    def test_strict_config_and_budget(self):
        for section, key, value in [("distributed", "world_size", 2), ("distributed", "enabled", False),
                                    ("distributed", "backend", "gloo"), ("distributed", "multiprocessing_context", "fork"),
                                    ("distributed", "log_every_n_groups", 0), ("training", "device", "cuda:0")]:
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config["distributed"]["ignored_option"] = True
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, "128"):
            validate_budget(self.config, 64, 4)
        self.assertEqual(validate_budget(self.config, 256, 2), 128)
        with self.assertRaises(ValueError):
            validate_budget(self.config, 256, 1)
        with self.assertRaises(ValueError):
            validate_budget(self.config, 128, 2)
        fp32 = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_fp32_v2.yaml")
        self.assertEqual(validate_budget(fp32, 128, 1), 128)
        self.assertEqual(global_progress({"consumed_samples": 7, "successful_samples": 3, "successful_updates": 1}, 4),
                         {"consumed_samples": 28, "successful_samples": 12, "successful_updates": 1})

    def test_no_padding_loader_and_validation_shards(self):
        context = DistributedContext(0, 4, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_loader([{}] * 7, self.config, ROOT, training=True, device=context.device, distributed=context)
        config = copy.deepcopy(self.config)
        config["data"]["num_workers"] = 0
        shards = [list(make_loader([{}] * 7, config, ROOT, device=context.device,
                                  distributed=DistributedContext(rank, 4, context.device)).sampler) for rank in range(4)]
        self.assertEqual(shards, [[0, 4], [1, 5], [2, 6], [3]])
        context.rank = 3
        self.assertEqual(list(make_loader([{}] * 3, config, ROOT, device=context.device, distributed=context).sampler), [])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "torchrun"):
                initialize_distributed(self.config)

    def test_prediction_id_contract(self):
        expected = [{"image_id": "b", "label": "1"}, {"image_id": "a", "label": "0"}]
        a, b = {"image_id": "a", "label": 0}, {"image_id": "b", "label": 1}
        self.assertEqual(ordered_predictions([[a], [b]], expected), [b, a])
        for shards in ([[a], [a]], [[b]], [[a], [{**b, "label": 0}]]):
            with self.assertRaises(RuntimeError):
                ordered_predictions(shards, expected)

    def test_unsharded_ddp_evaluation_keeps_safe_worker_context(self):
        context = DistributedContext(0, 4, torch.device("cpu"))
        for method in ("spawn", "forkserver"):
            if method not in mp.get_all_start_methods():
                continue
            config = copy.deepcopy(self.config)
            config["data"]["num_workers"] = 1
            config["distributed"]["multiprocessing_context"] = method
            for distributed in (None, context):
                loader = make_loader([{}] * 4, config, ROOT, device=context.device, distributed=distributed)
                self.assertEqual(loader.multiprocessing_context.get_start_method(), method)
                self.assertEqual(len(loader.sampler), 4 if distributed is None else 1)
        config["data"]["num_workers"] = 0
        loader = make_loader([{}] * 4, config, ROOT, device=context.device)
        self.assertIsNone(loader.multiprocessing_context)
        single = load_config(ROOT / "configs/b0_resnet50_512_dev_v1.yaml")
        single["data"]["num_workers"] = 1
        loader = make_loader([{}] * 4, single, ROOT, device=context.device)
        self.assertIsNone(loader.multiprocessing_context)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires CPU/Gloo multiprocessing")
    def test_four_process_training_and_failure_contracts(self):
        with tempfile.TemporaryDirectory(prefix="eqg_ddp_synthetic_") as temporary:
            rendezvous = (Path(temporary) / "rendezvous").as_uri()
            mp.spawn(four_rank_worker, args=(rendezvous, temporary), nprocs=4, join=True)
            result = json.loads((Path(temporary) / "result.json").read_text())
            self.assertEqual(result["ranks"], 4)
            self.assertEqual(result["overflow_consensus"], "passed")


if __name__ == "__main__":
    unittest.main()

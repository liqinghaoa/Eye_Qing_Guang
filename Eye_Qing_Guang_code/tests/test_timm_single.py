"""Synthetic single-device checks; no real dataset, weights or GPU required."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from test_timm_official import PixelClassifier, synthetic_config
from eye_qing_guang.contracts.timm_config import load_timm_config, validate_timm_config
from eye_qing_guang.pipelines.timm_official import _loader, execute_preflight, run_timm_experiment
from eye_qing_guang.trainers.timm_official import build_optimizer_schedule, pw5_loss, train_epoch, update_group
from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_yaml
from eye_qing_guang.utils.timm_source import SingleDeviceRuntime, create_source_runtime

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090.yaml"


class TimmSingleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = synthetic_config()
        self.config["distributed"].update(enabled=False, world_size=1, gradient_transport="none", backend="none")
        self.config["training"].update(precision="fp32", amp=False, batch_size=2, accumulation_steps=2)
        self.device = torch.device("cpu")
        self.runtime = SingleDeviceRuntime(self.config, self.device)
        self.metadata = {"timm_version": "1.0.22", "num_features": 1280}

    def records(self, root, split, count):
        rows = []
        for i in range(count):
            name = f"{split}_{i}.jpg"
            Image.new("RGB", (18, 22), (30+i*10, 40, 60)).save(root / name)
            rows.append({"image_id": name, "image_path": name, "label": str(i % 2) if split != "test" else "",
                         "age": "", "split": split})
        return rows

    def test_config_modes_and_effective_batch(self):
        single = load_timm_config(CONFIG)
        old = load_timm_config(ROOT / "configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml")
        self.assertEqual(single["training"]["batch_size"] * single["training"]["accumulation_steps"], 128)
        self.assertNotEqual(single["experiment"]["name"], old["experiment"]["name"])
        self.assertEqual(old["distributed"]["world_size"], 4)
        for changes in ({"world_size": 4}, {"gradient_transport": "native_ddp"}, {"backend": "nccl"}):
            bad = copy.deepcopy(single)
            bad["distributed"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_timm_config(bad)

    def test_factory_binds_only_selected_device_without_communication(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}, clear=True), \
             patch("torch.distributed.is_initialized", return_value=False), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.set_device") as bind, \
             patch("torch.cuda.is_bf16_supported") as bf16, \
             patch("torch.distributed.init_process_group", side_effect=AssertionError("unexpected NCCL")), \
             patch("eye_qing_guang.utils.timm_source.B0DDPRuntime", side_effect=AssertionError("unexpected DDP")):
            runtime, device = create_source_runtime(self.config)
            self.assertEqual(str(device), "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            bind.assert_called_once_with(device)
            bf16.assert_not_called()
            model = PixelClassifier()
            self.assertIs(runtime.initialize_model(model), model)
            runtime.verify_parameters(model)
            self.assertEqual(runtime.sum_numbers([5]), [5])
            runtime.close()

    def test_wrong_launcher_and_unavailable_precision_fail_before_training(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True), \
             patch("torch.cuda.set_device") as bind:
            with self.assertRaisesRegex(RuntimeError, "one Python process"):
                create_source_runtime(self.config)
            bind.assert_not_called()
        config = copy.deepcopy(self.config)
        config["training"].update(precision="bf16", amp=True)
        with patch("eye_qing_guang.utils.timm_source._single_device_runtime", return_value=self.runtime), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "BF16"):
                create_source_runtime(config)
        self.assertTrue(self.runtime.closed)

    def test_single_sampler_workers_and_tail_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.records(root, "train", 9)
            loader = _loader(records, self.config, root, self.device, self.runtime, training=True)
            self.assertIsInstance(loader.sampler, RandomSampler)
            self.assertFalse(loader.drop_last)
            model = PixelClassifier()
            optimizer, schedule = build_optimizer_schedule(model, self.config["training"], 3)
            with patch("torch.distributed.all_reduce", side_effect=AssertionError("unexpected collective")):
                result = train_epoch(model, loader, optimizer, schedule, self.device, self.config["training"], self.runtime, 1)
            self.assertEqual(result["consumed_samples"], 9)
            self.assertEqual(result["groups"], 3)
            self.assertEqual(result["successful_updates"], 3)
            config = copy.deepcopy(self.config)
            config["data"]["num_workers"] = 1
            eval_loader = _loader(records, config, root, self.device, self.runtime)
            self.assertIsInstance(eval_loader.sampler, SequentialSampler)
            self.assertEqual(eval_loader.multiprocessing_context.get_start_method(), "spawn")
            seen = [name for batch in eval_loader for name in batch["image_id"]]
            self.assertEqual(seen, [row["image_id"] for row in records])

    def test_tail_gradients_match_true_sample_mean(self):
        model = PixelClassifier()
        reference = copy.deepcopy(model)
        images, labels = torch.tensor([[.1], [.6], [.9]]), torch.tensor([[0.], [1.], [1.]])
        group = [{"image": images[:2], "label": labels[:2]}, {"image": images[2:], "label": labels[2:]}]
        optimizer, schedule = build_optimizer_schedule(model, self.config["training"], 2)
        update_group(model, group, optimizer, schedule, self.device, self.config["training"], self.runtime, step=False)
        pw5_loss(self.device)(reference(images), labels).backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)

    def test_synthetic_preflight_and_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {split: self.records(root, split, count) for split, count in (("train", 9), ("valid", 8), ("test", 5))}
            raw = root / self.config["data"]["raw_root"]
            raw.mkdir(parents=True)
            write_csv(raw / "test.csv", [{"Image_name": r["image_id"]} for r in records["test"]])
            config_path = root / "config.yaml"
            self.config["training"]["epochs"] = 1
            write_yaml(config_path, self.config)
            factory = lambda *a, **k: (PixelClassifier(), self.metadata)
            with patch("eye_qing_guang.pipelines.timm_official.build_timm_model", side_effect=factory), \
                 patch("eye_qing_guang.contracts.timm_checkpoints.build_timm_model", side_effect=factory), \
                 patch("eye_qing_guang.pipelines.timm_official.create_source_runtime", return_value=(self.runtime, self.device)), \
                 patch("eye_qing_guang.pipelines.timm_official.ensure_official_manifests", return_value=(records, {})), \
                 patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"), \
                 patch("torch.distributed.all_reduce", side_effect=AssertionError("unexpected collective")):
                report = execute_preflight(self.config, records, root, self.device, self.runtime, root / "preflight", 64)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["consumed_sample_instances"], 32)
                output = run_timm_experiment(config_path, root, skip_preflight=True)
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["world_size"], 1)
            self.assertEqual(manifest["gradient_transport"], "none")
            self.assertEqual(manifest["effective_batch"], 4)
            self.assertEqual(read_csv(output / "reports/history.csv")[0]["consumed_samples"], "9")
            submitted = read_csv(output / "submission/submission.csv")
            self.assertEqual([r["Image_name"] for r in submitted], [r["image_id"] for r in records["test"]])
            for row in submitted:
                self.assertEqual(int(row["Pred_Label"]), int(float(row["Probability"]) >= .5))
            for name in ("best.pt", "last.pt"):
                checkpoint = torch.load(output / "checkpoints" / name, weights_only=True)
                self.assertFalse(any(k.startswith("module.") for k in checkpoint["model_state"]))
                validate_timm_config(checkpoint["config"])


if __name__ == "__main__":
    unittest.main()

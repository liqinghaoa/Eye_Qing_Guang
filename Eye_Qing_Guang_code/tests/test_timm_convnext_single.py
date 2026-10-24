"""ConvNeXt single-device/cache integration on synthetic data; no GPU or downloads."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler
from test_timm_official import PixelClassifier
from eye_qing_guang.contracts.convnext_config import load_convnext_config, validate_convnext_config, MODEL_NAME
from eye_qing_guang.contracts.convnext_checkpoints import load_convnext_checkpoint
from eye_qing_guang.contracts.timm_config import load_timm_config
from eye_qing_guang.contracts.timm_checkpoints import load_timm_checkpoint
from eye_qing_guang.datasets.convnext_official import official_records
from eye_qing_guang.datasets.offline_cache import build_offline_cache, cached_records, cache_signature
from eye_qing_guang.pipelines.convnext_official import _loader, execute_preflight, run_convnext_experiment
from eye_qing_guang.trainers.convnext_official import build_optimizer_schedule, pw5_loss, update_group, train_epoch
from eye_qing_guang.utils.experiment_io import write_csv, read_csv, write_yaml
from eye_qing_guang.utils.timm_source import SingleDeviceRuntime, create_source_runtime

ROOT = Path(__file__).resolve().parents[1]
BASE = "timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official"


class ConvNeXtSingleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_convnext_config(ROOT / f"configs/{BASE}_single3090_cache_v1.yaml")
        self.config["data"].update(raw_root="raw", manifest_dir="manifests/convnext", num_workers=0)
        self.config["data"]["cache"]["root"] = "cache/v1"
        self.config["training"].update(epochs=1, batch_size=2, eval_batch_size=3, accumulation_steps=2,
            precision="fp32", amp=False, early_stopping_enabled=False, min_epochs=None, patience=None, min_delta=None)
        self.device = torch.device("cpu")
        self.runtime = SingleDeviceRuntime(self.config, self.device)
        self.metadata = {"timm_version": "1.0.22", "num_features": 768}
        self.counts = {"train": 9, "valid": 8, "test": 5}
        self.count_patch = patch("eye_qing_guang.datasets.convnext_official.EXPECTED_COUNTS", self.counts)
        self.count_patch.start()
        for split, count in self.counts.items():
            directory = self.root / "raw" / split
            directory.mkdir(parents=True)
            rows = []
            for i in range(count):
                name = f"{split}_{i}.JPG"
                y, x = np.indices((641, 801) if i == 0 else (29, 43))
                pixels = np.stack(((x+i*15)%256, (y*3)%256, (x+y)%256), axis=-1).astype(np.uint8)
                Image.fromarray(pixels).save(directory / name)
                rows.append({"Image_name": name, "True_Label": str(i%2) if split != "test" else "IGNORE", "Age": "42"})
            write_csv(self.root / f"raw/{split}.csv", rows)
        self.records, _ = official_records(self.config, self.root)

    def tearDown(self):
        self.count_patch.stop()
        self.temp.cleanup()

    def test_new_configs_preserve_old_mode_and_model_protocol(self):
        old = load_convnext_config(ROOT / f"configs/{BASE}.yaml")
        raw = load_convnext_config(ROOT / f"configs/{BASE}_single3090.yaml")
        cached = load_convnext_config(ROOT / f"configs/{BASE}_single3090_cache_v1.yaml")
        self.assertEqual(old["distributed"]["world_size"], 4)
        self.assertTrue(old["distributed"]["enabled"])
        self.assertNotIn("cache", old["data"])
        self.assertNotIn("cache", raw["data"])
        for config in (raw, cached):
            self.assertFalse(config["distributed"]["enabled"])
            self.assertEqual(config["model"], old["model"])
            self.assertTrue(config["training"]["early_stopping_enabled"])
            self.assertEqual(config["training"]["batch_size"]*config["training"]["accumulation_steps"], 128)
            self.assertEqual(config["data"]["mean"], [.485, .456, .406])
            self.assertNotEqual(config["experiment"]["name"], old["experiment"]["name"])
        for changes in ({"world_size": 4}, {"backend": "nccl"}, {"gradient_transport": "native_ddp"}):
            bad = copy.deepcopy(raw)
            bad["distributed"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_convnext_config(bad)

    def test_single_factory_device_binding_and_four_rank_routing(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True), \
             patch("torch.distributed.is_initialized", return_value=False), \
             patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.set_device") as bind, patch("torch.cuda.is_bf16_supported") as bf16, \
             patch("eye_qing_guang.utils.timm_source.B0DDPRuntime", side_effect=AssertionError("unexpected DDP")), \
             patch("torch.distributed.init_process_group", side_effect=AssertionError("unexpected NCCL")):
            runtime, device = create_source_runtime(self.config)
            self.assertEqual(str(device), "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
            bind.assert_called_once_with(device)
            bf16.assert_not_called()
            self.assertIs(runtime.initialize_model(PixelClassifier()).__class__, PixelClassifier)
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True), patch("torch.cuda.set_device") as bind:
            with self.assertRaisesRegex(RuntimeError, "one Python process"):
                create_source_runtime(self.config)
            bind.assert_not_called()
        bad_precision = copy.deepcopy(self.config)
        bad_precision["training"].update(precision="bf16", amp=True)
        with patch("eye_qing_guang.utils.timm_source._single_device_runtime", return_value=self.runtime), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "BF16"):
                create_source_runtime(bad_precision)
        old = load_convnext_config(ROOT / f"configs/{BASE}.yaml")
        fake = SimpleNamespace(rank=0, local_rank=0, world_size=4, device=torch.device("cuda:0"))
        with patch("eye_qing_guang.utils.timm_source.launch_coordinates", return_value=(0, 0, 4)), \
             patch("eye_qing_guang.utils.timm_source.B0DDPRuntime", return_value=fake) as ddp, \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIs(create_source_runtime(old)[0], fake)
            ddp.assert_called_once_with(old)

    def test_raw_sampler_spawn_workers_and_tail_gradients(self):
        config = load_convnext_config(ROOT / f"configs/{BASE}_single3090.yaml")
        config["data"].update(raw_root="raw", num_workers=1)
        config["training"] = copy.deepcopy(self.config["training"])
        loader = _loader(self.records["train"], config, self.root, self.device, self.runtime, training=True)
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), "spawn")
        ids = [name for batch in loader for name in batch["image_id"]]
        self.assertCountEqual(ids, [r["image_id"] for r in self.records["train"]])
        config["data"]["num_workers"] = 0
        loader = _loader(self.records["train"], config, self.root, self.device, self.runtime, training=True)
        model = PixelClassifier()
        optimizer, schedule = build_optimizer_schedule(model, config["training"], 3)
        result = train_epoch(model, loader, optimizer, schedule, self.device, config["training"], self.runtime, 1)
        self.assertEqual((result["consumed_samples"], result["groups"]), (9, 3))
        model, reference = PixelClassifier(), PixelClassifier()
        reference.load_state_dict(model.state_dict())
        images, labels = torch.tensor([[.1], [.6], [.9]]), torch.tensor([[0.], [1.], [1.]])
        optimizer, schedule = build_optimizer_schedule(model, config["training"], 2)
        update_group(model, [{"image": images[:2], "label": labels[:2]}, {"image": images[2:], "label": labels[2:]}],
                     optimizer, schedule, self.device, config["training"], self.runtime, step=False)
        pw5_loss(self.device)(reference(images), labels).backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        rank = SimpleNamespace(rank=1, world_size=4)
        old = load_convnext_config(ROOT / f"configs/{BASE}.yaml")
        old["data"]["num_workers"] = 0
        old["training"]["eval_batch_size"] = 3
        shard = _loader(self.records["valid"], old, self.root, self.device, rank)
        self.assertEqual(list(shard.sampler), [1, 5])

    def test_v2s_cache_reuse_and_convnext_normalization(self):
        v2s = load_timm_config(ROOT / "configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml")
        v2s["data"] = {**copy.deepcopy(self.config["data"]), "mean": [.5]*3, "std": [.5]*3}
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", self.counts):
            build_offline_cache(v2s, self.root, workers=0)
        self.assertEqual(cache_signature(v2s["data"], self.root), cache_signature(self.config["data"], self.root))
        bound = cached_records(self.records["valid"], self.config["data"], self.root)
        loader = _loader(self.records["valid"], self.config, self.root, self.device, self.runtime)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        with Image.open(bound[0]["image_path"]) as image:
            raw = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()/255
        mean, std = (torch.tensor(self.config["data"][key])[:, None, None] for key in ("mean", "std"))
        torch.testing.assert_close(next(iter(loader))["image"][0], (raw-mean)/std, rtol=0, atol=0)
        self.assertFalse(torch.equal((raw-mean)/std, (raw-.5)/.5))
        with self.assertRaises(FileExistsError):
            build_offline_cache(self.config, self.root, workers=0)

    def test_preflight_training_best_reload_and_submission_raw_and_cache(self):
        build_offline_cache(self.config, self.root, workers=0)
        factory = lambda *a, **kw: (PixelClassifier(), self.metadata)
        with patch("eye_qing_guang.pipelines.convnext_official.build_convnext_model", side_effect=factory), \
             patch("eye_qing_guang.contracts.convnext_checkpoints.build_convnext_model", side_effect=factory), \
             patch("eye_qing_guang.pipelines.convnext_official.create_source_runtime", return_value=(self.runtime, self.device)), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"), \
             patch("torch.distributed.all_reduce", side_effect=AssertionError("unexpected collective")):
            report = execute_preflight(self.config, self.records, self.root, self.device, self.runtime, self.root / "preflight", 64)
            self.assertEqual((report["status"], report["consumed_sample_instances"]), ("completed", 32))
            for enabled in (True, False):
                with self.subTest(cache=enabled):
                    config = copy.deepcopy(self.config)
                    config["data"]["cache"]["enabled"] = enabled
                    if not enabled:
                        config["data"].pop("cache")
                        config["data"]["preprocessing"]["version"] = "fov_letterbox_v1"
                    config["experiment"]["name"] += "_cached" if enabled else "_raw"
                    path = self.root / f"config_{enabled}.yaml"
                    write_yaml(path, config)
                    output = run_convnext_experiment(path, self.root, skip_preflight=True)
                    manifest = json.loads((output / "run_manifest.json").read_text())
                    self.assertEqual((manifest["status"], manifest["world_size"], manifest["gradient_transport"]),
                                     ("completed", 1, "none"))
                    self.assertEqual(manifest["effective_batch"], 4)
                    self.assertEqual("offline_cache" in manifest["sources"], enabled)
                    self.assertEqual(read_csv(output / "reports/history.csv")[0]["consumed_samples"], "9")
                    self.assertEqual([r["Image_name"] for r in read_csv(output / "submission/submission.csv")],
                                     [r["image_id"] for r in self.records["test"]])
                    for name in ("best.pt", "last.pt"):
                        _, payload = load_convnext_checkpoint(output / "checkpoints" / name, self.device, self.root)
                        self.assertEqual((payload["model_name"], payload["num_features"]), (MODEL_NAME, 768))
                        validate_convnext_config(payload["config"])
                    with self.assertRaises(ValueError):
                        load_timm_checkpoint(output / "checkpoints/best.pt", self.device, self.root)

    def test_incomplete_cache_blocks_training_before_model_creation(self):
        path = self.root / "config.yaml"
        write_yaml(path, self.config)
        with patch("eye_qing_guang.pipelines.convnext_official.create_source_runtime", return_value=(self.runtime, self.device)), \
             patch("eye_qing_guang.pipelines.convnext_official.build_convnext_model") as model:
            with self.assertRaises(FileNotFoundError):
                run_convnext_experiment(path, self.root, skip_preflight=True)
            model.assert_not_called()
        self.assertFalse((self.root / self.config["experiment"]["output_root"] / self.config["experiment"]["name"]).exists())


if __name__ == "__main__":
    unittest.main()

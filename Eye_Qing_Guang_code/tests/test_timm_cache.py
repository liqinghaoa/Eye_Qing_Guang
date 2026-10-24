"""Offline cache integration on generated images only."""

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

import numpy as np
import torch
from PIL import Image
from test_timm_official import synthetic_config, PixelClassifier
from eye_qing_guang.contracts.timm_config import load_timm_config, validate_timm_config
from eye_qing_guang.datasets.offline_cache import build_offline_cache, cached_records
from eye_qing_guang.datasets.timm_official import official_records
from eye_qing_guang.datasets.transforms import build_transforms, OFFLINE_VERSION
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.pipelines.timm_official import run_timm_experiment
from eye_qing_guang.utils.experiment_io import write_csv, write_yaml, read_csv
from eye_qing_guang.utils.timm_source import SingleDeviceRuntime

ROOT = Path(__file__).resolve().parents[1]


class OfflineCacheTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = synthetic_config()
        self.config["distributed"].update(enabled=False, world_size=1, gradient_transport="none", backend="none")
        self.config["training"].update(epochs=1, precision="fp32", amp=False, batch_size=2, accumulation_steps=2)
        self.config["data"]["preprocessing"]["version"] = OFFLINE_VERSION
        self.config["data"]["cache"] = {"enabled": True, "root": "cache/v1", "version": OFFLINE_VERSION}
        self.counts = {"train": 5, "valid": 8, "test": 3}
        self.count_patch = patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", self.counts)
        self.count_patch.start()
        raw = self.root / self.config["data"]["raw_root"]
        for split, count in self.counts.items():
            (raw / split).mkdir(parents=True)
            rows = []
            for i in range(count):
                name = f"{split}_{i}.JPG"
                y, x = np.indices((643, 871) if i == 0 else (43, 71))
                pixels = np.stack(((x*3+i*11)%256, (y*5)%256, (x+y)%256), axis=-1).astype(np.uint8)
                Image.fromarray(pixels).save(raw / split / name)
                rows.append({"Image_name": name, "True_Label": str(i%2) if split != "test" else "IGNORE", "Age": "42"})
            write_csv(raw / f"{split}.csv", rows)
        self.records, _ = official_records(self.config, self.root)

    def tearDown(self):
        self.count_patch.stop()
        self.temp.cleanup()

    def test_opt_in_and_config_compatibility(self):
        for name in ("timm_v2s_in21k_ft_in1k_512_pw5_official.yaml",
                     "timm_v2s_in21k_ft_in1k_512_pw5_official_single3090.yaml"):
            config = load_timm_config(ROOT / "configs" / name)
            self.assertNotIn("cache", config["data"])
        new = load_timm_config(ROOT / "configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml")
        self.assertTrue(new["data"]["cache"]["enabled"])
        validate_timm_config(self.config)
        old = copy.deepcopy(self.config)
        old["data"]["preprocessing"]["version"] = "fov_letterbox_v1"
        with self.assertRaises(ValueError):
            validate_timm_config(old)

    def test_parallel_build_pixel_equality_augmentation_and_spawn_loader(self):
        directory = build_offline_cache(self.config, self.root, workers=2)
        meta = json.loads((directory / "metadata.json").read_text())
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["processed"], 16)
        old = copy.deepcopy(self.config["data"])
        old.pop("cache")
        old["preprocessing"]["version"] = "fov_letterbox_v1"
        for split, rows in self.records.items():
            bound = cached_records(rows, self.config["data"], self.root)
            self.assertEqual([r["image_id"] for r in bound], [r["image_id"] for r in rows])
            transform = build_transforms(self.config["data"], split == "train")
            for row, cached in zip(rows, bound):
                with Image.open(self.root / row["image_path"]) as source, Image.open(cached["image_path"]) as prepared:
                    torch.manual_seed(41)
                    raw_tensor = transform(source.convert("RGB"))
                    torch.manual_seed(41)
                    cache_tensor = transform.from_cache(prepared, split)
                    torch.testing.assert_close(raw_tensor, cache_tensor, rtol=0, atol=0)
                    if split != "train":
                        torch.testing.assert_close(build_transforms(old)(source.convert("RGB")), cache_tensor, rtol=0, atol=0)
            if split == "train":
                with Image.open(bound[0]["image_path"]) as prepared:
                    torch.manual_seed(1)
                    first = transform.from_cache(prepared, split)
                    torch.manual_seed(2)
                    self.assertFalse(torch.equal(first, transform.from_cache(prepared, split)))
        config = copy.deepcopy(self.config)
        config["data"]["num_workers"] = 1
        loader = make_loader(self.records["train"], config, self.root, training=True, device=torch.device("cpu"))
        with patch("eye_qing_guang.datasets.transforms.fov_crop", side_effect=AssertionError("No online FOV")):
            # Spawn workers are independently imported; the in-process loader below checks routing too.
            ids = [item for batch in loader for item in batch["image_id"]]
            direct = make_loader(self.records["valid"], self.config, self.root, device=torch.device("cpu"))
            next(iter(direct))
        self.assertCountEqual(ids, [r["image_id"] for r in self.records["train"]])
        with self.assertRaises(FileExistsError):
            build_offline_cache(self.config, self.root, workers=0)

    def test_missing_changed_and_incomplete_caches_rejected(self):
        directory = build_offline_cache(self.config, self.root, workers=0)
        path = directory / "metadata.json"
        meta = json.loads(path.read_text())
        path.write_text(json.dumps({**meta, "status": "building"}))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            cached_records(self.records["train"], self.config["data"], self.root)
        path.write_text(json.dumps(meta))
        source = self.root / self.records["train"][0]["image_path"]
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns+1_000_000))
        with self.assertRaisesRegex(ValueError, "Raw source changed"):
            cached_records(self.records["train"], self.config["data"], self.root)
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        (directory / "train/000000.png").unlink()
        with self.assertRaises(FileNotFoundError):
            cached_records(self.records["train"], self.config["data"], self.root)

    def test_failed_build_is_never_usable(self):
        with patch("eye_qing_guang.datasets.offline_cache._prepare_one", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                build_offline_cache(self.config, self.root, workers=0)
        meta = json.loads((self.root / "cache/v1/metadata.json").read_text())
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["processed"], 0)
        with self.assertRaises(ValueError):
            cached_records(self.records["train"], self.config["data"], self.root)

    def test_cache_training_reload_and_submission(self):
        build_offline_cache(self.config, self.root, workers=0)
        config_path = self.root / "config.yaml"
        write_yaml(config_path, self.config)
        runtime = SingleDeviceRuntime(self.config, torch.device("cpu"))
        factory = lambda *a, **kw: (PixelClassifier(), {"timm_version": "1.0.22", "num_features": 1280})
        with patch("eye_qing_guang.pipelines.timm_official.create_source_runtime", return_value=(runtime, runtime.device)), \
             patch("eye_qing_guang.pipelines.timm_official.build_timm_model", side_effect=factory), \
             patch("eye_qing_guang.contracts.timm_checkpoints.build_timm_model", side_effect=factory), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
            output = run_timm_experiment(config_path, self.root, skip_preflight=True)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("offline_cache", manifest["sources"])
        self.assertEqual([r["Image_name"] for r in read_csv(output / "submission/submission.csv")],
                         [r["image_id"] for r in self.records["test"]])
        payload = torch.load(output / "checkpoints/best.pt", weights_only=True)
        self.assertEqual(payload["config"]["data"]["preprocessing"]["version"], OFFLINE_VERSION)


if __name__ == "__main__":
    unittest.main()

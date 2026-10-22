"""Small synthetic tests: data contracts, model shape and the executable workflow."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from eye_qing_guang.cli import main
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets import metadata
from eye_qing_guang.datasets.metadata import build_manifests, collect_records, fit_age_stats, normalize_age
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.binary_classification import compute_binary_metrics
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.pipelines.inference import build_submission
from eye_qing_guang.utils.experiment_io import read_csv, write_csv, write_yaml
from config_fixtures import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="eye_qing_guang_test_")
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(load_config(PROJECT_ROOT / "configs/resnet18.yaml"))
        self.config["data"]["raw_root"] = "data/raw/glaucoma_first"  # Synthetic fixtures stay under self.root.
        self.config["data"]["image_size"] = 32
        self.config["data"]["augmentation"] = {"horizontal_flip": 0.0, "rotation_degrees": 0}
        self.config["model"].update(backbone="tiny_cnn", pretrained=False, use_age=True)
        self.config["training"].update(epochs=1, batch_size=2, device="cpu")
        raw = self.root / self.config["data"]["raw_root"]
        for split, count in (("train", 4), ("valid", 2), ("test", 2)):
            directory = raw / split
            directory.mkdir(parents=True)
            rows = []
            for index in range(count):
                name = f"{split}_{index:05d}"
                Image.new("RGB", (36, 40), color=(40 + index * 30, 50, 70)).save(directory / f"{name}.JPG")
                row = {"Image_name": f"{name}.jpg", "Age": "" if index == 1 else str(30 + index * 10)}
                if split != "test":
                    row["True_Label"] = str(index % 2)
                rows.append(row)
            write_csv(raw / f"{split}.csv", rows)
        write_yaml(self.root / "config.yaml", self.config)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, command, *arguments):
        common = [command, "--project-root", str(self.root)]
        if command in {"audit", "manifest", "train", "submit"}:
            common += ["--config", "config.yaml"]
        main([*common, *arguments])

    def manifests(self):
        directory = self.root / self.config["data"]["manifest_dir"]
        build_manifests(self.config, self.root, directory)
        return directory

    def test_case_resolution_and_unlabeled_test(self):
        records, summary = collect_records(self.config, self.root, decode=True)
        self.assertTrue(records["train"][0]["image_path"].endswith(".JPG"))
        self.assertEqual(records["test"][0]["label"], "")
        self.assertEqual(summary["train"]["filename_case_resolved"], 4)
        directory = self.manifests()
        self.assertEqual(len(load_manifest(directory / "test.csv", self.root, False)), 2)
        with self.assertRaises(FileExistsError):
            build_manifests(self.config, self.root, directory)

    def test_missing_image_stops_before_output_creation(self):
        with patch("eye_qing_guang.datasets.metadata.find_image", side_effect=FileNotFoundError("missing")):
            directory = self.root / "invalid_manifest"
            with self.assertRaises(FileNotFoundError):
                build_manifests(self.config, self.root, directory)
            self.assertFalse(directory.exists())

    def test_invalid_label_rejected(self):
        rows = read_csv(self.root / self.config["data"]["raw_root"] / "train.csv")
        rows[0]["True_Label"] = "2"
        original = metadata.read_csv
        with patch("eye_qing_guang.datasets.metadata.read_csv",
                   side_effect=lambda path: rows if path.name == "train.csv" else original(path)):
            with self.assertRaisesRegex(ValueError, "Invalid binary label"):
                collect_records(self.config, self.root)

    def test_age_statistics_are_fit_on_training_records(self):
        records, _ = collect_records(self.config, self.root)
        stats = fit_age_stats(records["train"])
        self.assertAlmostEqual(stats["median"], 50.0)
        self.assertAlmostEqual(stats["mean"], 47.5)
        self.assertAlmostEqual(normalize_age("", stats), (50 - 47.5) / stats["std"])
        records["valid"][0]["age"] = 119
        self.assertEqual(stats, fit_age_stats(records["train"]))

    def test_metrics_and_invalid_scores(self):
        metrics = compute_binary_metrics([0, 1, 0, 1], [0.1, 0.8, 0.6, 0.9])
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["fp"], 1)
        self.assertIsNone(compute_binary_metrics([0, 0], [0.1, 0.2])["roc_auc"])
        for scores in ([float("nan")], [1.1], []):
            with self.assertRaises(ValueError):
                compute_binary_metrics([0], scores)

    def test_model_shapes_and_focal_loss(self):
        for use_age in (False, True):
            config = {**self.config["model"], "use_age": use_age}
            model = build_model(config)
            output = model(torch.zeros(2, 3, 32, 32), torch.zeros(2, 1))
            self.assertEqual(tuple(output.shape), (2, 1))
            loss = build_loss({**self.config["training"], "loss": "focal"})(output, torch.zeros(2, 1))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_supported_torchvision_backbones(self):
        for backbone in ("resnet18", "resnet50", "efficientnet_b0"):
            with self.subTest(backbone=backbone):
                config = {**self.config["model"], "backbone": backbone, "use_age": False}
                model = build_model(config).eval()
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, 32, 32))
                self.assertEqual(tuple(output.shape), (1, 1))

    def test_config_preflight(self):
        invalid = copy.deepcopy(self.config)
        invalid["training"]["monitor"] = "unknown"
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_submission_order_duplicates_and_overwrite(self):
        predictions = self.root / "predictions.csv"
        source = self.root / self.config["data"]["raw_root"] / "test.csv"
        write_csv(predictions, [{"image_id": "test_00001.jpg", "probability": 0.9},
                                {"image_id": "test_00000.jpg", "probability": 0.1}])
        output = self.root / "submission.csv"
        build_submission(predictions, source, output)
        self.assertEqual(read_csv(output)[0]["Image_name"], "test_00000.jpg")
        with self.assertRaises(FileExistsError):
            build_submission(predictions, source, output)
        duplicate = self.root / "duplicates.csv"
        write_csv(duplicate, [{"image_id": "test_00000.jpg", "probability": 0.1}] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_submission(duplicate, source, self.root / "bad" / "submission.csv")

    def test_full_synthetic_cli_workflow(self):
        self.command("audit", "--output", "audit.json", "--decode-images")
        self.command("manifest")
        self.command("train", "--name", "smoke_v1")
        experiment = self.root / "experiments/smoke_v1"
        checkpoint = "experiments/smoke_v1/checkpoints/best.pt"
        self.assertTrue((experiment / "checkpoints/last.pt").is_file())
        with (experiment / "run_manifest.json").open() as handle:
            self.assertEqual(json.load(handle)["status"], "completed")
        self.command("evaluate", "--checkpoint", checkpoint, "--device", "cpu",
                     "--output-dir", "experiments/smoke_v1/reports/valid_best")
        self.command("predict", "--checkpoint", checkpoint, "--device", "cpu",
                     "--output-dir", "experiments/smoke_v1/predictions/test_best")
        self.command("submit", "--predictions",
                     "experiments/smoke_v1/predictions/test_best/test_predictions.csv",
                     "--output", "submission.csv")
        self.command("curves", "--history", "experiments/smoke_v1/reports/history.csv",
                     "--output", "curves.png")
        self.command("confusion", "--metrics", "experiments/smoke_v1/reports/valid_best/metrics.json",
                     "--output", "confusion.png")
        self.command("gradcam", "--checkpoint", checkpoint, "--device", "cpu", "--age", "30",
                     "--image", "data/raw/glaucoma_first/valid/valid_00000.JPG", "--output", "gradcam.png")
        self.command("summarize", "--experiments", "experiments/smoke_v1", "--output", "summary.csv")
        self.assertEqual(len(read_csv(self.root / "submission.csv")), 2)
        for name in ("curves.png", "confusion.png", "gradcam.png"):
            with Image.open(self.root / name) as image:
                image.verify()
        with self.assertRaises(FileExistsError):
            self.command("train", "--name", "smoke_v1")


if __name__ == "__main__":
    unittest.main()

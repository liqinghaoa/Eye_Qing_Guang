"""Competition submission regression tests use temporary synthetic data only."""

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
from torch import nn

from eye_qing_guang.cli import main
from eye_qing_guang.contracts.checkpoints import load_checkpoint, save_checkpoint
from eye_qing_guang.contracts.config import submission_options, validate_config
from eye_qing_guang.contracts.dataset import load_manifest
from eye_qing_guang.datasets.metadata import build_test_manifest
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.pipelines.inference import build_submission, predict_test
from eye_qing_guang.utils.experiment_io import load_config, read_csv, write_csv, write_yaml

ROOT = Path(__file__).resolve().parents[1]


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eqg_submission_synthetic_")
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/b0_resnet50_512_dev_v1.yaml")
        self.config["data"]["raw_root"] = "data/raw/glaucoma_first"  # Synthetic fixtures stay under self.root.
        raw = self.root / self.config["data"]["raw_root"]
        (raw / "test").mkdir(parents=True)
        self.ids = ["test_00002.jpg", "test_00000.jpg", "test_眼底.jpg", "test_00001.jpg"]
        for name in self.ids:
            Image.new("RGB", (36, 40), (60, 40, 30)).save(raw / "test" / (name[:-4] + ".JPG"))
        self.source = raw / "test.csv"
        write_csv(self.source, [{"Image_name": name, "Age": ""} for name in self.ids])
        self.predictions = self.root / "predictions.csv"
        self.scores = dict(zip(self.ids, (0.0, 0.4999999999, 0.5, 1.0)))
        self.rows = [{"image_id": name, "probability": self.scores[name]} for name in reversed(self.ids)]
        write_csv(self.predictions, self.rows)
        write_yaml(self.root / "config.yaml", self.config)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def command(self, command, *args):
        main([command, "--project-root", str(self.root), "--config", "config.yaml", *args])

    def test_competition_header_utf8_order_threshold_and_overwrite(self):
        output = self.root / "submission.csv"
        build_submission(self.predictions, self.source, output)
        text = output.read_bytes().decode("utf-8")
        self.assertEqual(text.splitlines()[0], "Image_name,Probability,Pred_Label")
        rows = read_csv(output)
        self.assertEqual([row["Image_name"] for row in rows], self.ids)
        self.assertEqual([int(row["Pred_Label"]) for row in rows], [0, 0, 1, 1])
        self.assertEqual([float(row["Probability"]) for row in rows], list(self.scores.values()))
        with self.assertRaises(FileExistsError):
            build_submission(self.predictions, self.source, output)

    def test_competition_rejects_wrong_filename_or_headers(self):
        with self.assertRaisesRegex(ValueError, "submission.csv"):
            build_submission(self.predictions, self.source, self.root / "submission_v1.csv")
        with self.assertRaisesRegex(ValueError, "columns"):
            build_submission(self.predictions, self.source, self.root / "submission.csv", value_column="probability")
        self.assertFalse((self.root / "submission.csv").exists())

    def test_invalid_predictions_never_create_outputs(self):
        variants = {
            "missing": self.rows[:-1],
            "extra": self.rows + [{"image_id": "extra.jpg", "probability": 0.2}],
            "duplicate": self.rows + [self.rows[0]],
            "empty_id": [{**row, "image_id": ""} for row in self.rows],
        }
        for name, score in (("nan", float("nan")), ("inf", float("inf")), ("negative", -0.1), ("above_one", 1.1)):
            variants[name] = [{**row, "probability": score} for row in self.rows]
        for name, rows in variants.items():
            with self.subTest(name=name):
                path = self.root / f"{name}.csv"
                write_csv(path, rows)
                output = self.root / name / "submission.csv"
                with self.assertRaises(ValueError):
                    build_submission(path, self.source, output)
                self.assertFalse(output.parent.exists())
        duplicate_source = self.root / "duplicate_test.csv"
        write_csv(duplicate_source, [{"Image_name": self.ids[0]}] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_submission(self.predictions, duplicate_source, self.root / "submission.csv")

    def test_explicit_legacy_two_column_modes(self):
        for mode in ("probability", "label"):
            output = self.root / f"{mode}.csv"
            build_submission(self.predictions, self.source, output, value_column="probability", mode=mode)
            rows = read_csv(output)
            self.assertEqual(list(rows[0]), ["Image_name", "probability"])
            expected = [0, 0, 1, 1] if mode == "label" else list(self.scores.values())
            self.assertEqual([float(row["probability"]) for row in rows], expected)

    def test_test_manifest_reads_only_test_and_preserves_development_config(self):
        before = copy.deepcopy(self.config)
        self.command("test-manifest", "--output-dir", "manifests/test_v1", "--decode-images")
        directory = self.root / "manifests/test_v1"
        rows = load_manifest(directory / "test.csv", self.root, labeled=False)
        self.assertEqual([row["image_id"] for row in rows], self.ids)
        self.assertTrue(all(row["label"] == "" and row["split"] == "test" for row in rows))
        self.assertTrue(all(row["image_path"].endswith(".JPG") for row in rows))
        self.assertEqual(self.config, before)
        self.assertFalse((self.root / self.config["data"]["manifest_dir"]).exists())
        with (directory / "summary.json").open() as handle:
            summary = json.load(handle)
        self.assertEqual(summary["source"], "official_test")
        self.assertEqual(summary["splits"]["test"]["samples"], 4)
        with self.assertRaises(FileExistsError):
            self.command("test-manifest", "--output-dir", "manifests/test_v1")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.command("test-manifest", "--output-dir", self.config["data"]["manifest_dir"])

    def test_test_manifest_failure_does_not_create_directory(self):
        output = self.root / "bad_manifest"
        with patch("eye_qing_guang.datasets.metadata.find_image", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(FileNotFoundError):
                build_test_manifest(self.config, self.root, output)
        self.assertFalse(output.exists())

    def test_submission_config_is_optional_and_typo_checked(self):
        old = copy.deepcopy(self.config)
        old.pop("submission")
        validate_config(old)
        self.assertEqual(submission_options(old)["mode"], "competition")
        for options in ({"probablity_column": "Probability"}, {"mode": "unknown"},
                        {"value_column": "probability"}, {"label_column": "Image_name"}):
            candidate = copy.deepcopy(old)
            candidate["submission"] = options
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_config(candidate)

    def test_b0_checkpoint_test_manifest_prediction_and_submission_cli(self):
        directory = self.root / "manifests/test_v1"
        build_test_manifest(self.config, self.root, directory)
        # A synthetic encoder tests orchestration/checkpoint loading, not ResNet50 performance.
        with patch("eye_qing_guang.models.factory.build_encoder", side_effect=lambda *args: (
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3,
        )):
            model = build_model(self.config["model"])
            optimizer = torch.optim.AdamW(model.parameters())
            for version in ("new", "old"):
                config = copy.deepcopy(self.config)
                if version == "old":
                    config.pop("submission")
                checkpoint = self.root / f"{version}.pt"
                save_checkpoint(checkpoint, model, optimizer, 0, config, None, 1.0, {})
                _, payload = load_checkpoint(checkpoint, torch.device("cpu"))
                self.assertEqual(payload["config"], config)
                with self.assertRaisesRegex(ValueError, "--manifest"):
                    predict_test(checkpoint, self.root, self.root / f"{version}_missing", "cpu")
                self.assertFalse((self.root / f"{version}_missing").exists())
                prediction_dir = self.root / f"{version}_predictions"
                main(["predict", "--project-root", str(self.root), "--checkpoint", str(checkpoint),
                      "--device", "cpu", "--manifest", str(directory / "test.csv"),
                      "--output-dir", str(prediction_dir)])
                write_yaml(self.root / f"{version}.yaml", config)
                main(["submit", "--project-root", str(self.root), "--config", f"{version}.yaml",
                      "--predictions", str(prediction_dir / "test_predictions.csv"),
                      "--output", f"{version}/submission.csv"])
                rows = read_csv(self.root / version / "submission.csv")
                self.assertEqual([row["Image_name"] for row in rows], self.ids)
                self.assertEqual(list(rows[0]), ["Image_name", "Probability", "Pred_Label"])
                self.assertTrue(all(int(row["Pred_Label"]) == int(float(row["Probability"]) >= 0.5) for row in rows))


if __name__ == "__main__":
    unittest.main()

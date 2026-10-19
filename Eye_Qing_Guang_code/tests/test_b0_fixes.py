"""Regression coverage for strict decoding, failure accounting and bounded precision checks."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from eye_qing_guang.datasets.development import build_development_manifests
from eye_qing_guang.datasets.metadata import collect_records
from eye_qing_guang.pipelines.b0_development import new_progress
from eye_qing_guang.pipelines.preflight import (
    ReplayLoader, budget_available, clipped_gradients, compare_precision, fp32_reference, preflight,
)
from eye_qing_guang.trainers.binary import finish_update, make_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import SuccessfulUpdateSchedule
from eye_qing_guang.utils.experiment_io import load_config, write_csv, write_json
from eye_qing_guang.cli import main

ROOT = Path(__file__).resolve().parents[1]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, image, age=None):
        return self.linear(image)


class FixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="b0_fixes_")
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/b0_resnet50_512_dev_v1.yaml")
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def samples(self, count=7):
        return [{"image": torch.tensor([index / 10]), "age": torch.zeros(1),
                 "label": torch.tensor([float(index % 2)]), "image_id": str(index)} for index in range(count)]

    def audit_bytes(self, encoded):
        original_open = Image.open
        config = copy.deepcopy(self.config)
        config["data"]["splits"] = {"train": config["data"]["splits"]["train"]}
        path = self.root / "data/raw/glaucoma_first/train/synthetic.jpg"
        with patch("eye_qing_guang.datasets.metadata.read_csv", return_value=[
            {"Image_name": "synthetic.jpg", "Age": "50", "True_Label": "0"},
        ]), patch("eye_qing_guang.datasets.metadata.image_index", return_value={}), patch(
            "eye_qing_guang.datasets.metadata.find_image", return_value=path,
        ), patch("eye_qing_guang.datasets.metadata.Image.open", side_effect=lambda path: original_open(io.BytesIO(encoded))):
            return collect_records(config, self.root, decode=True)

    def test_truncated_jpeg_is_rejected_even_when_verify_accepts_it(self):
        pixels = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG")
        damaged = buffer.getvalue()[:-100]
        with Image.open(io.BytesIO(damaged)) as image:
            image.verify()  # This is the previously missed corruption.
        with self.assertRaisesRegex(ValueError, "Image decode failed for synthetic.jpg"):
            self.audit_bytes(damaged)
        _, summary = self.audit_bytes(buffer.getvalue())
        self.assertTrue(summary["train"]["decoded"])

    def test_valid_png_rgb_conversion(self):
        buffer = io.BytesIO()
        Image.new("L", (12, 15), 42).save(buffer, format="PNG")
        _, summary = self.audit_bytes(buffer.getvalue())
        self.assertEqual(summary["train"]["samples"], 1)

    def test_failure_counter_covers_input_forward_loss_and_backward_errors(self):
        settings = {**self.config["training"], "batch_size": 2}
        for failure in ("input", "forward", "logits", "loss", "backward"):
            with self.subTest(failure=failure):
                model = Model()
                samples = self.samples(2)
                criterion = nn.BCEWithLogitsLoss()
                if failure == "input":
                    samples[0]["image"].fill_(float("nan"))
                elif failure == "forward":
                    model.forward = lambda *args: (_ for _ in ()).throw(RuntimeError("forward failure"))
                elif failure == "logits":
                    with torch.no_grad():
                        model.linear.bias.fill_(float("inf"))
                elif failure == "loss":
                    criterion = lambda logits, labels: logits.mean() * float("nan")
                else:
                    model.linear.weight.register_hook(lambda grad: (_ for _ in ()).throw(RuntimeError("backward failure")))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                schedule = SuccessfulUpdateSchedule(optimizer, 1, settings)
                progress = new_progress()
                with self.assertRaises((FloatingPointError, RuntimeError)):
                    train_accumulated_epoch(model, DataLoader(samples, batch_size=2), optimizer, criterion,
                                             torch.device("cpu"), settings, make_scaler(False), schedule, progress)
                self.assertEqual(progress["consumed_samples"], 2)
                self.assertEqual(progress["successful_samples"], 0)
                self.assertEqual(progress["attempted_groups"], 1)

    def test_reference_weighting_with_unequal_microbatches_and_budget(self):
        torch.manual_seed(42)
        model = Model()
        reference = copy.deepcopy(model)
        loader = ReplayLoader(list(DataLoader(self.samples(7), batch_size=3)), 3)
        settings = self.config["training"]
        progress = new_progress()
        result = fp32_reference(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), settings, progress)
        batch = next(iter(DataLoader(self.samples(7), batch_size=7)))
        loss = nn.BCEWithLogitsLoss()(reference(batch["image"]), batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), settings["grad_clip"])
        metrics = compare_precision(loss.item(), clipped_gradients(reference), result)
        self.assertEqual(metrics["status"], "passed")
        self.assertLess(metrics["clipped_gradient_l2_difference"], 1e-6)
        self.assertEqual(progress["consumed_samples"], 7)
        self.assertEqual(progress["reference_samples"], 7)
        self.assertEqual(progress["attempted_groups"], 1)
        self.assertFalse(budget_available({"consumed_samples": 60, "attempted_groups": 1}, 7))
        self.assertFalse(budget_available({"consumed_samples": 8, "attempted_groups": 4}, 7))
        exhausted = {"consumed_samples": 64, "attempted_groups": 3}
        with self.assertRaisesRegex(RuntimeError, "budget"):
            fp32_reference(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), settings, exhausted)
        self.assertEqual(exhausted["consumed_samples"], 64)

    def test_numerical_mismatch_is_not_reported_as_pass(self):
        reference = {"loss": 1.0, "gradients": {"weight": torch.ones(3)}}
        self.assertEqual(compare_precision(1.0, {"weight": torch.ones(3)}, reference)["status"], "passed")
        self.assertEqual(compare_precision(2.0, {"weight": torch.ones(3)}, reference)["status"], "failed")
        self.assertEqual(compare_precision(1.0, {"weight": torch.full((3,), 5.0)}, reference)["status"], "failed")

    def source(self):
        raw = self.root / self.config["data"]["raw_root"]
        (raw / "train").mkdir(parents=True)
        rows = []
        for index in range(80):
            name = f"train_{index:05d}.jpg"
            Image.new("RGB", (36, 40), (80, 50, 40)).save(raw / "train" / name)
            rows.append({"Image_name": name, "Age": "50", "True_Label": str(index%2)})
        write_csv(raw / "train.csv", rows)
        build_development_manifests(self.config, self.root, self.root / self.config["data"]["manifest_dir"])

    def run_preflight(self, name, overflow=False):
        actual_finish = finish_update
        calls = []

        def overflow_then_recover(model, optimizer, scaler, clip):
            calls.append(float(optimizer.param_groups[0]["lr"]))
            skips = 1 if overflow is True else overflow if type(overflow) is int else 0
            if len(calls) <= skips or overflow == "all":
                # Inject a real non-finite scaled gradient before unscale, so
                # the standard CUDA GradScaler must skip and lower its scale.
                next(parameter for parameter in model.parameters() if parameter.grad is not None).grad.fill_(float("inf"))
            return actual_finish(model, optimizer, scaler, clip)

        with patch("eye_qing_guang.models.factory.build_encoder", side_effect=lambda *args: (
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3,
        )), patch("eye_qing_guang.trainers.binary.finish_update", side_effect=overflow_then_recover):
            directory = preflight(self.config, self.root, name)
        with (directory / "run_manifest.json").open() as handle:
            manifest = json.load(handle)
        with (directory / "reports/preflight.json").open() as handle:
            report = json.load(handle)
        return manifest, report, calls

    def test_inconclusive_cli_exits_nonzero(self):
        write_json(self.root / "run_manifest.json", {"status": "inconclusive"})
        with patch("eye_qing_guang.utils.experiment_io.load_config", return_value=self.config), patch(
            "eye_qing_guang.pipelines.preflight.preflight", return_value=self.root,
        ):
            with self.assertRaises(SystemExit) as result:
                main(["preflight", "--project-root", str(self.root)])
        self.assertEqual(result.exception.code, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration uses only synthetic images and a tiny substituted encoder")
    def test_default_budget_includes_fp32_reference_and_target_eval_batch(self):
        self.source()
        manifest, report, calls = self.run_preflight("default_comparison")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["progress"]["consumed_samples"], 64)
        self.assertEqual(manifest["progress"]["reference_samples"], 32)
        self.assertEqual(manifest["progress"]["attempted_groups"], 2)
        self.assertEqual(report["precision_comparison"]["status"], "passed")
        self.assertEqual(report["evaluation_batch_checked"], 16)
        self.assertEqual(len(calls), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration uses only synthetic images and a tiny substituted encoder")
    def test_first_real_scaler_overflow_can_recover_without_false_failure(self):
        self.source()
        manifest, report, calls = self.run_preflight("overflow_recovery", overflow=True)
        self.assertEqual(manifest["status"], "inconclusive")
        self.assertIsNone(manifest["error"])
        self.assertEqual(manifest["progress"]["consumed_samples"], 64)
        self.assertEqual(manifest["progress"]["successful_updates"], 1)
        self.assertEqual(manifest["progress"]["skipped_groups"], 1)
        self.assertEqual(calls, [0.0001, 0.0001])
        self.assertEqual(report["precision_comparison"]["reason"], "budget_exhausted_before_fp32_reference")
        first, second = report["update_attempts"]
        self.assertLess(first["scale_after"], first["scale_before"])
        self.assertFalse(set(first["image_ids"]) & set(second["image_ids"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration uses only synthetic images and a tiny substituted encoder")
    def test_all_skips_are_budget_inconclusive_not_a_recipe_failure(self):
        self.source()
        manifest, report, calls = self.run_preflight("all_overflow", overflow="all")
        self.assertEqual(manifest["status"], "inconclusive")
        self.assertIsNone(manifest["error"])
        self.assertEqual(manifest["progress"]["successful_updates"], 0)
        self.assertEqual(manifest["progress"]["consumed_samples"], 64)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["precision_comparison"]["status"], "inconclusive")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration uses only synthetic images and a tiny substituted encoder")
    def test_two_skips_recover_and_reference_fits_at_exact_group_budget(self):
        self.config["training"]["batch_size"] = 8
        self.source()
        manifest, report, calls = self.run_preflight("two_backoffs_and_reference", overflow=2)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["progress"]["consumed_samples"], 64)
        self.assertEqual(manifest["progress"]["attempted_groups"], 4)
        self.assertEqual(manifest["progress"]["skipped_groups"], 2)
        self.assertEqual(manifest["progress"]["reference_samples"], 16)
        self.assertEqual(calls, [0.0001]*3)
        self.assertEqual(report["precision_comparison"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()

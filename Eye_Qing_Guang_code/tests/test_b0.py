"""B0 protocol tests use synthetic data; no competition image is consumed."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from eye_qing_guang.contracts.checkpoints import load_checkpoint, save_checkpoint
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.datasets.development import (
    build_development_manifests, connected_groups, development_records, load_development_manifests,
)
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.transforms import build_transforms, fov_crop, letterbox, scaled_size
from eye_qing_guang.evaluators.binary import predict_batches
from eye_qing_guang.losses.classification import build_loss
from eye_qing_guang.metrics.development import prediction_metrics, write_development_analysis
from eye_qing_guang.models.factory import build_model
from eye_qing_guang.trainers.binary import finish_update, make_scaler, train_accumulated_epoch
from eye_qing_guang.trainers.schedules import DevelopmentStopping, SuccessfulUpdateSchedule, fit_length
from eye_qing_guang.utils.experiment_io import load_config, read_csv, write_csv
from eye_qing_guang.pipelines.b0_development import new_progress
from eye_qing_guang.pipelines.b0_development import train_development
from eye_qing_guang.pipelines.preflight import preflight
from eye_qing_guang.pipelines.evaluation import evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, image, age=None):
        return self.linear(image)


class B0Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="b0_synthetic_")
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/b0_resnet50_512_dev_v1.yaml")
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, n=100):
        raw = self.root / self.config["data"]["raw_root"]
        (raw / "train").mkdir(parents=True)
        rows = []
        for index in range(n):
            name = f"train_{index:05d}"
            Image.new("RGB", (36+index%3, 40), (60, 40, 30)).save(raw / "train" / f"{name}.JPG")
            rows.append({"Image_name": name+".jpg", "Age": "50", "True_Label": str(index%2)})
        write_csv(raw / "train.csv", rows)
        return rows

    def test_config_identity_and_unknown_fields(self):
        for section, key, value in (("model", "pretrained", True), ("training", "auto_pos_weight", True),
                                    ("training", "accumulation_step", 2), ("data", "image_size", 224),
                                    ("training", "device", "auto"), ("experiment", "protocol", "b0_fit")):
            candidate = copy.deepcopy(self.config)
            candidate[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_config(candidate)

    def test_no_official_valid_test_access_and_frozen_split(self):
        self.source()
        directory = self.root / self.config["data"]["manifest_dir"]
        summary = build_development_manifests(self.config, self.root, directory)
        self.assertEqual(summary["counts"]["train"]["samples"], 80)
        self.assertEqual(summary["counts"]["valid"]["samples"], 20)
        self.assertEqual({path.name for path in directory.iterdir()}, {"train.csv", "valid.csv", "summary.json"})
        records, _ = load_development_manifests(self.config, self.root)
        again, _ = development_records(self.config, self.root)
        self.assertEqual([row["image_id"] for row in records["train"]], [row["image_id"] for row in again["train"]])
        self.assertTrue(all(row["source_split"] == "official_train" for row in records["valid"]))
        with self.assertRaises(FileExistsError):
            build_development_manifests(self.config, self.root, directory)
        # A changed source label cannot silently keep the same versioned split.
        with patch("eye_qing_guang.datasets.development.development_records", return_value=(again, {"source": "official_valid"})):
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_development_manifests(self.config, self.root)

    def test_union_of_patient_and_duplicate_constraints(self):
        rows = [{"image_id": name} for name in ("a", "b", "c", "d")]
        links = [{"image_id": name, "association_type": kind, "group_id": group}
                 for name, kind, group in (("a", "patient", "p"), ("b", "patient", "p"),
                                           ("b", "confirmed_duplicate", "x"), ("c", "confirmed_duplicate", "x"))]
        groups = connected_groups(rows, links)
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotEqual(groups["a"], groups["d"])
        with self.assertRaises(ValueError):
            connected_groups(rows, [{"image_id": "e", "association_type": "patient", "group_id": "p"}])

    def test_group_split_isolated_and_invalid_grouping_stops(self):
        self.source()
        links = [{"image_id": f"train_{index:05d}.jpg", "association_type": "patient", "group_id": str(index//2)}
                 for index in range(100)]
        write_csv(self.root / "links.csv", links)
        self.config["development"]["associations_csv"] = "links.csv"
        records, summary = development_records(self.config, self.root)
        self.assertEqual(summary["method"], "StratifiedGroupKFold_5_fold0")
        self.assertFalse({row["group_id"] for row in records["train"]} & {row["group_id"] for row in records["valid"]})
        for link in links:
            link["group_id"] = "all"
        with patch("eye_qing_guang.datasets.development.read_csv", return_value=links):
            with self.assertRaises((ValueError, UserWarning)):
                development_records(self.config, self.root)

    def test_fov_letterbox_safety_and_rounding(self):
        settings = self.config["data"]["preprocessing"]
        for color, expected in ((0, "empty_foreground"), (8, "empty_foreground"), (9, "")):
            image = Image.new("RGB", (200, 100), (color,)*3)
            cropped, info = fov_crop(image, settings)
            self.assertEqual(info["fallback_reason"], expected)
            self.assertEqual(cropped.size, image.size)
        image = Image.new("RGB", (200, 100))
        ImageDraw.Draw(image).rectangle((20, 10, 179, 89), fill=(100, 50, 20))
        cropped, info = fov_crop(image, settings)
        self.assertEqual(info["crop_box"], [16, 8, 184, 92])
        self.assertGreaterEqual(info["candidate_coverage"], 0.995)
        self.assertGreaterEqual(info["area_ratio"], 0.5)
        ImageDraw.Draw(image).point((0, 0), fill="white")
        _, edge = fov_crop(image, settings)
        self.assertGreaterEqual(edge["candidate_coverage"], 0.995)
        sparse = Image.new("RGB", (200, 100))
        ImageDraw.Draw(sparse).rectangle((50, 25, 149, 74), fill="white")
        self.assertEqual(fov_crop(sparse, settings)[1]["fallback_reason"], "small_area")
        self.assertEqual(scaled_size(4, 1, 10), (10, 3))  # 2.5 rounds UP, not to even.
        boxed = np.asarray(letterbox(Image.new("RGB", (4, 1), "white"), 10))
        self.assertTrue((boxed[3:6] == 255).all())
        self.assertTrue((boxed[:3] == 0).all())

    def test_foreground_loss_fallback(self):
        image = Image.new("RGB", (512, 512))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 80, 431, 431), fill="white")
        # Five pixels/row and <=5 pixels/column remain below the 1% line rule,
        # but collectively exceed the 0.5% allowed foreground loss.
        for index in range(512):
            for offset in range(5):
                draw.point((index, (index+offset)%512), fill="white")
        self.assertEqual(fov_crop(image, self.config["data"]["preprocessing"])[1]["fallback_reason"], "foreground_loss")

    def test_transforms_deterministic_legacy_and_seeded_aug(self):
        image = Image.new("RGB", (140, 80), "black")
        ImageDraw.Draw(image).rectangle((0, 10, 100, 69), fill=(160, 90, 40))
        transform = build_transforms(self.config["data"])
        one, two = transform(image), transform(image)
        self.assertTrue(torch.equal(one, two))
        self.assertEqual(tuple(one.shape), (3, 512, 512))
        self.assertEqual(float(one[:, 0, 0].min()), -1)
        augment = build_transforms(self.config["data"], True)
        torch.manual_seed(42)
        first = augment(image)
        torch.manual_seed(42)
        self.assertTrue(torch.equal(first, augment(image)))
        self.assertFalse(torch.equal(first, augment(image)))
        legacy = load_config(ROOT / "configs/resnet18.yaml")
        self.assertEqual(tuple(build_transforms(legacy["data"])(image).shape), (3, 224, 224))

    def test_schedule_endpoints_and_skip_reuses_index(self):
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        schedule = SuccessfulUpdateSchedule(optimizer, 3, self.config["training"])
        self.assertAlmostEqual(schedule.prepare(), 1e-4)
        schedule.record(False)
        self.assertAlmostEqual(schedule.prepare(), 1e-4)
        self.assertAlmostEqual(schedule.value(15), 1e-3)
        self.assertAlmostEqual(schedule.value(240), 1e-6)
        schedule.record(True)
        self.assertGreater(schedule.prepare(), 1e-4)
        with self.assertRaises(ValueError):
            schedule.value(241)

    def test_best_patience_independence_and_minimum_epoch(self):
        stop = DevelopmentStopping(30, 15, 1e-4)
        self.assertEqual(stop.update(1, 0.8), (True, False))
        self.assertEqual(stop.update(2, 0.8), (False, False))
        self.assertEqual(stop.best_epoch, 1)
        self.assertEqual(stop.update(3, 0.80005), (True, False))
        self.assertEqual(stop.stale, 2)
        self.assertEqual(stop.reference_auc, 0.8)
        stop.update(4, 0.80011)
        self.assertEqual(stop.stale, 0)
        for epoch in range(5, 30):
            self.assertFalse(stop.update(epoch, 0.80011)[1])
        self.assertTrue(stop.update(30, 0.80011)[1])
        with self.assertRaises(ValueError):
            stop.update(31, float("nan"))
        self.assertEqual(fit_length([8])["fit_epochs"], 30)
        self.assertEqual(fit_length([40, 41])["fit_epochs"], 41)
        self.assertEqual(fit_length([90])["fit_epochs"], 80)

    def samples(self):
        return [{"image": torch.tensor([float(i)/10]), "age": torch.zeros(1),
                 "label": torch.tensor([float(i%2)]), "image_id": str(i)} for i in range(7)]

    def test_sample_weighted_accumulation_and_tail(self):
        torch.manual_seed(42)
        model = LinearModel()
        reference = copy.deepcopy(model)
        settings = {**self.config["training"], "batch_size": 3, "grad_clip": 1e6}
        samples = self.samples()
        loader = DataLoader(samples, batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        schedule = SuccessfulUpdateSchedule(optimizer, 2, settings)
        reference_schedule = SuccessfulUpdateSchedule(reference_optimizer, 2, settings)
        criterion = nn.BCEWithLogitsLoss()
        result = train_accumulated_epoch(model, loader, optimizer, criterion, torch.device("cpu"),
                                         settings, make_scaler(False), schedule, new_progress())
        for group in (samples[:6], samples[6:]):
            reference_optimizer.zero_grad()
            reference_schedule.prepare()
            image = torch.stack([row["image"] for row in group])
            label = torch.stack([row["label"] for row in group])
            criterion(reference(image), label).backward()
            reference_optimizer.step()
            reference_schedule.record(True)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(result["consumed_samples"], 7)
        self.assertEqual(result["successful_samples"], 7)
        self.assertEqual(result["successful_updates"], 2)

    def test_bce_exact_and_zero_positive_batch(self):
        logits = torch.tensor([[-3.0], [1.0], [0.0]], requires_grad=True)
        labels = torch.zeros(3, 1)
        criterion = build_loss(self.config["training"], 1.0)
        self.assertTrue(torch.equal(criterion(logits, labels), nn.BCEWithLogitsLoss()(logits, labels)))
        self.assertEqual(criterion.pos_weight.item(), 1)

    def test_overflow_is_skipped_without_clipping_and_fp32_fails(self):
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.linear.weight.grad = torch.full_like(model.linear.weight, float("inf"))

        class OverflowScaler:
            scale_value = 65536
            def is_enabled(self): return True
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.scale_value
            def step(self, optimizer): return None  # No optimizer step on overflow.
            def update(self): self.scale_value /= 2

        before = model.linear.weight.clone()
        with patch("torch.nn.utils.clip_grad_norm_") as clip:
            success, norm, scale_before, scale_after = finish_update(model, optimizer, OverflowScaler(), 5)
            clip.assert_not_called()
        self.assertFalse(success)
        self.assertIsNone(norm)
        self.assertEqual(scale_after, scale_before/2)
        self.assertTrue(torch.equal(model.linear.weight, before))
        with self.assertRaises(FloatingPointError):
            finish_update(model, optimizer, make_scaler(False), 5)

    def test_overflow_index_recovery_and_consecutive_limit(self):
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        settings = {**self.config["training"], "batch_size": 2, "accumulation_steps": 1,
                    "max_consecutive_skips": 2}
        loader = DataLoader(self.samples()[:6], batch_size=2)
        schedule = SuccessfulUpdateSchedule(optimizer, 3, settings)
        progress = new_progress()
        outcomes = iter([(False, None, 65536, 32768), (True, 1, 32768, 32768), (True, 1, 32768, 32768)])
        with patch("eye_qing_guang.trainers.binary.finish_update", side_effect=lambda *args: next(outcomes)):
            result = train_accumulated_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss(), torch.device("cpu"),
                                             settings, make_scaler(False), schedule, progress)
        self.assertEqual(result["skipped_groups"], 1)
        self.assertEqual(result["successful_samples"], 4)
        self.assertEqual(schedule.successful_updates, 2)
        schedule = SuccessfulUpdateSchedule(optimizer, 3, settings)
        progress = new_progress()
        with patch("eye_qing_guang.trainers.binary.finish_update", return_value=(False, None, 2, 1)):
            with self.assertRaisesRegex(FloatingPointError, "Consecutive"):
                train_accumulated_epoch(model, loader, optimizer, nn.BCEWithLogitsLoss(), torch.device("cpu"),
                                        settings, make_scaler(False), schedule, progress)
        self.assertEqual(progress["attempted_groups"], 2)
        self.assertEqual(progress["consumed_samples"], 4)

    def test_target_512_tail_one_shape_bce_gradient(self):
        # One synthetic target-size training forward/backward, not a resource certification.
        model = build_model(self.config["model"]).train()
        output = model(torch.zeros(1, 3, 512, 512))
        self.assertEqual(tuple(output.shape), (1, 1))
        loss = build_loss(self.config["training"])(output, torch.zeros(1, 1))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_old_new_checkpoint_and_deterministic_eval_bn(self):
        for filename in ("resnet18.yaml", "b0_resnet50_512_dev_v1.yaml"):
            config = load_config(ROOT / "configs" / filename)
            with patch(
                "eye_qing_guang.models.factory.build_encoder",
                return_value=(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3),
            ):
                model = build_model(config["model"], initialize_pretrained=False)
                optimizer = torch.optim.AdamW(model.parameters())
                checkpoint = self.root / (filename+".pt")
                extra = {"resume_supported": False} if "b0" in filename else None
                save_checkpoint(checkpoint, model, optimizer, 1, config, None, 1, {}, extra)
                restored, payload = load_checkpoint(checkpoint, torch.device("cpu"))
                self.assertEqual(payload["config"]["data"].get("preprocessing"), config["data"].get("preprocessing"))
                torch.testing.assert_close(model.classifier[-1].weight, restored.classifier[-1].weight)
        model = nn.Sequential(nn.BatchNorm1d(1), nn.Linear(1, 1))

        class Model(nn.Module):
            def __init__(self): super().__init__(); self.network = model
            def forward(self, image, age=None): return self.network(image)

        wrapper = Model()
        original = {key: value.clone() for key, value in wrapper.named_buffers()}
        loader = DataLoader(self.samples(), batch_size=2)
        first, _ = predict_batches(wrapper, loader, torch.device("cpu"))
        second, _ = predict_batches(wrapper, loader, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertEqual([row["image_id"] for row in first], [str(i) for i in range(7)])
        self.assertTrue(all(torch.equal(original[key], value) for key, value in wrapper.named_buffers()))

    def test_development_reports_fit_only_boundaries(self):
        self.source()
        records, _ = development_records(self.config, self.root)
        directory = self.root / "analysis"
        directory.mkdir()
        predictions = [{"image_id": row["image_id"], "label": int(row["label"]),
                        "logit": -2.0, "probability": float(torch.sigmoid(torch.tensor(-2.0)))}
                       for row in records["valid"]]
        write_development_analysis(directory, records["train"], records["valid"], predictions, self.root, self.config)
        with (directory / "reports/subgroup_rules.json").open() as handle:
            rules = json.load(handle)
        self.assertFalse(set(rules["fit_sample_ids"]) & {row["image_id"] for row in records["valid"]})
        self.assertAlmostEqual(prediction_metrics(predictions)["roc_auc"], 0.5)
        with (directory / "reports/prior_reference.json").open() as handle:
            self.assertEqual(json.load(handle)["prior_source"], "internal_fit")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration only; numerical CPU tests remain available")
    def test_b0_orchestration_and_preflight_on_synthetic_tiny_encoder(self):
        # Exercise the actual orchestration/scaler/artifact path with a cheap
        # substituted encoder, not a target-model throughput or memory claim.
        self.source(20)
        config = copy.deepcopy(self.config)
        config["training"].update(epochs=2, min_epochs=2, schedule_total_epochs=3,
                                  warmup_epochs=2, batch_size=3, eval_batch_size=2)
        directory = self.root / config["data"]["manifest_dir"]
        build_development_manifests(config, self.root, directory)
        with patch("eye_qing_guang.models.factory.build_encoder", side_effect=lambda *args: (
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), 3,
        )):
            debug = preflight(config, self.root, "synthetic_preflight")
            with (debug / "run_manifest.json").open() as handle:
                record = json.load(handle)
            self.assertEqual(record["status"], "completed")
            self.assertLessEqual(record["progress"]["consumed_samples"], 64)
            self.assertEqual(record["progress"]["attempted_groups"], 2)
            with (debug / "reports/preflight.json").open() as handle:
                check = json.load(handle)
            self.assertEqual(check["precision_comparison"]["status"], "passed")
            self.assertEqual(record["progress"]["reference_samples"], 6)
            experiment = train_development(config, self.root, "synthetic_dev")
            self.assertTrue((experiment / "checkpoints/best_inference.pt").is_file())
            self.assertEqual(len(read_csv(experiment / "predictions/dev_best.csv")), 4)
            updates = read_csv(experiment / "reports/updates.csv")
            self.assertEqual(len(updates), 6)
            self.assertEqual(sum(int(row["group_samples"]) for row in updates), 32)
            self.assertTrue((experiment / "reports/subgroups.json").is_file())
            metrics = evaluate_checkpoint(experiment / "checkpoints/best.pt", self.root,
                                          experiment / "reports/eval_roundtrip", device_name="cpu")
            self.assertEqual(metrics["evaluation_source"], "internal_development")
            with self.assertRaisesRegex(ValueError, "explicit"):
                evaluate_checkpoint(experiment / "checkpoints/best.pt", self.root,
                                    experiment / "reports/rejected", device_name="cpu",
                                    manifest=self.root / "not_official_valid.csv")
            with self.assertRaises(FileExistsError):
                train_development(config, self.root, "synthetic_dev")


if __name__ == "__main__":
    unittest.main()

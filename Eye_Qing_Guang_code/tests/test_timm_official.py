"""Offline synthetic checks; no competition images, pretrained downloads or PPU run."""

import copy
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from eye_qing_guang.contracts.timm_checkpoints import CheckpointStore, load_timm_checkpoint
from eye_qing_guang.contracts.timm_config import UnresolvedRecipeError, load_timm_config, validate_timm_config
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.datasets.timm_official import ensure_official_manifests, ordered_predictions
from eye_qing_guang.metrics.timm_official import compute_official_metrics, recall_at_specificity
from eye_qing_guang.models.timm_efficientnet import build_timm_model, prepared_pretrained_file
from eye_qing_guang.pipelines.timm_official import (
    compare_predictions, post_training, preflight_plan, run_timm_experiment, training_stage,
)
from eye_qing_guang.trainers.timm_official import (
    SelectionState, build_optimizer_schedule, pw5_loss, update_group,
)
from eye_qing_guang.utils.experiment_io import read_csv, write_csv
from eye_qing_guang.utils.timm_artifacts import atomic_json, publish_submission
from eye_qing_guang.utils.timm_source import launch_coordinates

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml"


def synthetic_config():
    """These optimizer/transport choices are test fixtures, NEVER recovered source evidence."""
    config = load_timm_config(TEMPLATE, require_resolved=False)
    config["data"]["raw_root"] = "data/raw/glaucoma_first"
    config["recipe"].update(implementation="synthetic-test", implementation_path="synthetic.py",
        parameter_notes="SYNTHETIC TEST ONLY")
    config["training"].update(optimizer="adamw", optimizer_parameter_groups="all_trainable",
        eval_batch_size=3, betas=[0.9, 0.999], eps=1e-8, device="cuda", grad_clip=False,
        scheduler="warmup_cosine_success_v1", schedule_total_epochs=100, warmup_start_lr=1e-5,
        early_stopping_enabled=False, min_epochs=None, min_delta=None, patience=None)
    config["distributed"].update(backend="nccl", timeout_seconds=30, multiprocessing_context="spawn",
        log_every_n_groups=10)
    config["data"]["num_workers"] = 0
    return config


class LocalRuntime:
    """Single-process control double; it does not certify source gradient transport."""

    rank = local_rank = 0
    world_size = 1

    def checked(self, phase, action):
        return action()

    def primary(self, phase, action):
        return action()

    def broadcast(self, value):
        return value

    def sum_numbers(self, values):
        return values

    def average_gradients(self, model, samples):
        pass

    def initialize_model(self, model):
        pass

    def close(self):
        pass


class PixelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, images, ages=None):
        return self.linear(images.mean(dim=tuple(range(1, images.ndim))).reshape(-1, 1))


class FakeEncoder(nn.Module):
    num_features = 1280

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 1280)

    def forward(self, images):
        return self.projection(images.mean((2, 3)))


class TimmOfficialTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="eqg_timm_synthetic_")
        self.root = Path(self.temporary.name)
        self.config = synthetic_config()
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.metadata = {"timm_version": "1.0.22", "model_name": self.config["model"]["name"], "num_features": 1280}

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, counts=None):
        counts = counts or {"train": 8, "valid": 5, "test": 8}
        raw = self.root / self.config["data"]["raw_root"]
        for split, count in counts.items():
            directory = raw / split
            directory.mkdir(parents=True)
            rows = []
            for index in range(count):
                name = f"{split}_{index:03d}.JPG"
                Image.new("RGB", (18, 22), (20+index*10, 40, 60)).save(directory / name)
                # Deliberately present test labels must never become dataset targets.
                rows.append({"Image_name": name, "True_Label": str(index % 2) if split != "test" else "DO_NOT_READ",
                             "Age": ""})
            write_csv(raw / f"{split}.csv", rows)
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", counts):
            return ensure_official_manifests(self.config, self.root), counts

    def predictions(self, records, labeled):
        result = []
        for index, row in enumerate(records):
            logit = (index-2)/2
            value = {"image_id": row["image_id"], "logit": logit, "probability": 1/(1+math.exp(-logit))}
            if labeled:
                value["label"] = int(row["label"])
            result.append(value)
        return result

    def test_known_identity_and_b0_recipe_are_resolved(self):
        config = load_timm_config(TEMPLATE, require_resolved=False)
        self.assertEqual(config["training"]["epochs"], 100)
        self.assertEqual(config["training"]["batch_size"]*config["training"]["accumulation_steps"]*4, 128)
        validate_timm_config(self.config)

    def test_missing_and_ambiguous_fields_fail(self):
        changes = [("model", "backbone", "resnet50"), ("model", "name", "tf_efficientnetv2_s.in21k"),
            ("training", "pos_weight", 1.0), ("training", "precision", "fp16"),
            ("distributed", "gradient_transport", "gloo_staged")]
        for section, key, value in changes:
            config = copy.deepcopy(self.config)
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_timm_config(config)
        config = copy.deepcopy(self.config)
        del config["data"]["preprocessing"]["min_coverage"]
        with self.assertRaisesRegex(ValueError, "FOV"):
            validate_timm_config(config)

    def test_resource_and_optimizer_settings_are_configurable(self):
        config = copy.deepcopy(self.config)
        config["training"].update(batch_size=32, eval_batch_size=64, accumulation_steps=1,
                                  epochs=80, schedule_total_epochs=80, learning_rate=0.0005,
                                  precision="fp32", amp=False)
        config["data"]["num_workers"] = 8
        validate_timm_config(config)
        plan = preflight_plan(config, 280)
        self.assertEqual((plan["micro_batch"], plan["accumulation_steps"], plan["target_batch_memory"]),
                         (32, 1, "verified"))

    def test_inapplicable_and_required_parameters_are_distinct(self):
        config = copy.deepcopy(self.config)
        config["training"]["grad_clip"] = None
        with self.assertRaisesRegex(UnresolvedRecipeError, "grad_clip"):
            validate_timm_config(config)
        config = copy.deepcopy(self.config)
        config["training"].update(early_stopping_enabled=True, min_epochs=5, min_delta=0.0, patience=3)
        validate_timm_config(config)
        config["training"]["patience"] = None
        with self.assertRaisesRegex(UnresolvedRecipeError, "patience"):
            validate_timm_config(config)

    def test_launch_requires_four_ranks(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "torchrun"):
            launch_coordinates()

    def test_official_manifests_preserve_every_source_and_test_is_unlabeled(self):
        (records, summary), counts = self.dataset()
        self.assertEqual({k: len(v) for k, v in records.items()}, counts)
        self.assertTrue(all(row["label"] == "" for row in records["test"]))
        self.assertEqual(summary["splits"]["valid"]["source"], "official_valid")
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", counts):
            before = (self.root / self.config["data"]["manifest_dir"] / "train.csv").read_bytes()
            reused, _ = ensure_official_manifests(self.config, self.root)
            self.assertEqual(records, reused)
            self.assertEqual(before, (self.root / self.config["data"]["manifest_dir"] / "train.csv").read_bytes())

    def test_count_mismatch_and_internal_manifest_reuse_fail(self):
        (records, _), counts = self.dataset()
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", {**counts, "train": 9}):
            with self.assertRaisesRegex(ValueError, "count changed"):
                ensure_official_manifests(self.config, self.root)
        path = self.root / self.config["data"]["manifest_dir"] / "summary.json"
        atomic_json(path, {"protocol": "b0_dev_v1"})
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", counts):
            with self.assertRaisesRegex(ValueError, "provenance"):
                ensure_official_manifests(self.config, self.root)

    def test_existing_manifest_wrong_labels_order_and_paths_fail(self):
        (_, _), counts = self.dataset()
        path = self.root / self.config["data"]["manifest_dir"] / "valid.csv"
        rows = read_csv(path)
        rows[0]["label"] = str(1-int(rows[0]["label"]))
        from eye_qing_guang.utils.timm_artifacts import atomic_csv
        atomic_csv(path, rows, list(rows[0]))
        with patch("eye_qing_guang.datasets.timm_official.EXPECTED_COUNTS", counts):
            with self.assertRaisesRegex(ValueError, "differs"):
                ensure_official_manifests(self.config, self.root)

    def test_casefold_cross_split_duplicate_fails(self):
        self.dataset()
        source = self.root / self.config["data"]["raw_root"] / "valid.csv"
        rows = read_csv(source)
        rows[0]["Image_name"] = "TRAIN_000.jpg"
        from eye_qing_guang.utils.timm_artifacts import atomic_csv
        atomic_csv(source, rows, list(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ensure_official_manifests(self.config, self.root)

    def test_validation_shards_are_unequal_and_never_padded(self):
        (records, _), _ = self.dataset()
        loaders = [make_loader(records["valid"], self.config, self.root, device=self.device,
            distributed=SimpleNamespace(rank=r, world_size=4)) for r in range(4)]
        indices = [list(loader.sampler) for loader in loaders]
        self.assertEqual([len(part) for part in indices], [2, 1, 1, 1])
        self.assertEqual(sorted(i for part in indices for i in part), list(range(5)))
        training = make_loader(records["train"], self.config, self.root, training=True, device=self.device,
                               distributed=SimpleNamespace(rank=0, world_size=4))
        self.assertFalse(training.drop_last)
        training.sampler.set_epoch(3)
        self.assertEqual(training.sampler.epoch, 3)

    def test_model_constructor_exact_name_shape_and_full_training(self):
        fake = SimpleNamespace(get_pretrained_cfg=lambda name: SimpleNamespace(to_dict=lambda: {
            "mean": (0.5,)*3, "std": (0.5,)*3, "num_classes": 1000, "fixed_input_size": False}),
            create_model=lambda *a, **kw: FakeEncoder())
        with patch("eye_qing_guang.models.timm_efficientnet._timm", return_value=(fake, "1.0.22")):
            with patch.object(fake, "create_model", wraps=fake.create_model) as create:
                model, metadata = build_timm_model(self.config["model"], self.root, initialize_pretrained=False)
                self.assertEqual(create.call_args.args, (self.config["model"]["name"],))
                self.assertEqual(create.call_args.kwargs, {"pretrained": False, "num_classes": 0,
                    "global_pool": "avg", "drop_rate": 0.0, "drop_path_rate": 0.0})
        self.assertEqual(model(torch.zeros(2, 3, 16, 16), torch.ones(2, 1)).shape, (2, 1))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model(torch.ones(2, 3, 16, 16)).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_missing_pretrained_file_fails_without_download(self):
        settings = {**self.config["model"], "pretrained_file": "missing_weights.bin"}
        with self.assertRaises(FileNotFoundError):
            prepared_pretrained_file(settings, self.root)

    def test_pw5_is_positive_term_only(self):
        criterion = pw5_loss(self.device)
        self.assertEqual(criterion.pos_weight.tolist(), [5.0])
        actual = criterion(torch.zeros(2, 1), torch.tensor([[0.0], [1.0]]))
        self.assertAlmostEqual(actual.item(), 3*math.log(2), places=6)

    def test_accumulation_tail_matches_one_true_sample_mean(self):
        source = PixelClassifier()
        reference = copy.deepcopy(source)
        group = [{"image": torch.full((n, 1), float(i+1)), "label": torch.tensor([[float(j % 2)] for j in range(n)])}
                 for i, n in enumerate((3, 1))]
        settings = {**self.config["training"], "precision": "fp32"}
        optimizer, schedule = build_optimizer_schedule(source, settings, 2)
        result = update_group(source, group, optimizer, schedule, self.device, settings, LocalRuntime(), step=False)
        images = torch.cat([r["image"] for r in group])
        labels = torch.cat([r["label"] for r in group])
        pw5_loss(self.device)(reference(images), labels).backward()
        for a, b in zip(source.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-7))
        self.assertEqual(result["local_samples"], 4)
        self.assertEqual(schedule.successful_updates, 0)
        update_group(source, group, optimizer, schedule, self.device, settings, LocalRuntime())
        self.assertEqual(schedule.successful_updates, 1)

    def test_invalid_gradient_or_shape_does_not_advance_schedule(self):
        model = PixelClassifier()
        settings = {**self.config["training"], "precision": "fp32"}
        optimizer, schedule = build_optimizer_schedule(model, settings, 2)
        bad = [{"image": torch.zeros(2, 1), "label": torch.zeros(2)}]
        with self.assertRaisesRegex(ValueError, "Labels"):
            update_group(model, bad, optimizer, schedule, self.device, settings, LocalRuntime())
        self.assertEqual(schedule.successful_updates, 0)

    def test_metrics_ties_endpoints_hard_auc_and_pr_area(self):
        rows = [{"label": label, "probability": score} for label, score in zip([0, 1, 0, 1], [.1, .8, .8, .9])]
        metrics = compute_official_metrics(rows, 0.3)
        self.assertGreater(metrics["pr_auc_trapezoid"], 0)
        self.assertEqual(metrics["hard_label_roc_auc"], metrics["balanced_accuracy"])
        point = metrics["recall_at_specificity_95"]
        self.assertEqual((point["threshold"], point["sensitivity"], point["specificity"]), (.9, .5, 1.0))
        endpoint = recall_at_specificity([0, 1], [.5, .5], 1.0)
        self.assertIsNone(endpoint["threshold"])
        json.dumps(endpoint, allow_nan=False)
        with self.assertRaises(ValueError):
            compute_official_metrics([{"label": 0, "probability": .5}], .1)

    def test_selection_strict_improvement_and_independent_min_delta(self):
        settings = {**self.config["training"], "early_stopping_enabled": True,
                    "min_epochs": 3, "patience": 2, "min_delta": .01}
        state = SelectionState()
        self.assertEqual(state.update(1, .8, settings), (True, False))
        self.assertEqual(state.update(2, .805, settings), (True, False))
        self.assertEqual(state.update(3, .805, settings), (False, True))
        self.assertEqual(state.best_epoch, 2)
        with self.assertRaises(ValueError):
            state.update(4, float("nan"), settings)

    def test_three_scores_keep_exactly_best_and_last_with_strict_reload(self):
        model = PixelClassifier()
        store = CheckpointStore(self.root / "checkpoints")
        state = SelectionState()
        for epoch, score in enumerate((.8, .805, .805), 1):
            improved, _ = state.update(epoch, score, self.config["training"])
            with torch.no_grad():
                model.linear.bias.fill_(epoch)
            if improved:
                store.save("best.pt", model, epoch, self.config, {"roc_auc": score}, self.metadata)
            store.save("last.pt", model, epoch, self.config, {"roc_auc": score}, self.metadata)
        self.assertEqual(sorted(p.name for p in store.directory.iterdir()), ["best.pt", "last.pt"])
        with patch("eye_qing_guang.contracts.timm_checkpoints.build_timm_model", side_effect=lambda *a, **k: (PixelClassifier(), {})) as build:
            best, first = load_timm_checkpoint(store.directory / "best.pt", self.device, self.root)
            last, final = load_timm_checkpoint(store.directory / "last.pt", self.device, self.root)
            self.assertTrue(all(call.kwargs["initialize_pretrained"] is False for call in build.call_args_list))
        self.assertEqual((first["epoch"], final["epoch"]), (2, 3))
        self.assertEqual(best.linear.bias.item(), 2)
        self.assertEqual(last.linear.bias.item(), 3)
        self.assertFalse({"optimizer_state", "training_state"} & first.keys())
        with self.assertRaises(FileExistsError):
            CheckpointStore(store.directory)
        with self.assertRaises(ValueError):
            store.save("epoch_0004.pt", model, 4, self.config, {"roc_auc": .9}, self.metadata)

    def test_checkpoint_failure_keeps_previous_and_cleans_temporary(self):
        model = PixelClassifier()
        store = CheckpointStore(self.root / "checkpoints")
        store.save("best.pt", model, 1, self.config, {"roc_auc": .8}, self.metadata)
        original = (store.directory / "best.pt").read_bytes()
        with patch("eye_qing_guang.contracts.timm_checkpoints.torch.save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                store.save("best.pt", model, 2, self.config, {"roc_auc": .9}, self.metadata)
        self.assertEqual(original, (store.directory / "best.pt").read_bytes())
        self.assertEqual([p.name for p in store.directory.iterdir()], ["best.pt"])

    def test_submission_eight_synthetic_images_order_and_round_trip(self):
        (records, _), _ = self.dataset()
        rows = self.predictions(records["test"], False)
        rows[0]["probability"] = 0.5
        rows[1]["probability"] = float(np.nextafter(np.float32(.5), np.float32(0)))
        predictions = self.root / "synthetic_predictions.csv"
        write_csv(predictions, list(reversed(rows)))
        destination = self.root / "submission" / "submission.csv"
        publish_submission(predictions, self.root / self.config["data"]["raw_root"] / "test.csv", destination)
        actual = read_csv(destination)
        self.assertEqual([r["Image_name"] for r in actual], [r["image_id"] for r in rows])
        self.assertEqual([r["Pred_Label"] for r in actual[:2]], ["1", "0"])
        self.assertEqual(list(actual[0]), ["Image_name", "Probability", "Pred_Label"])
        with self.assertRaises(FileExistsError):
            publish_submission(predictions, self.root / self.config["data"]["raw_root"] / "test.csv", destination)

    def test_submission_bad_scores_or_ids_never_publish_final(self):
        (records, _), _ = self.dataset()
        original = self.predictions(records["test"], False)
        cases = [original[:-1], [*original, original[0]], [{**r, "probability": float("nan")} for r in original],
                 [{**r, "probability": 1.1} for r in original]]
        for i, rows in enumerate(cases):
            predictions = self.root / f"bad_{i}.csv"
            write_csv(predictions, rows)
            destination = self.root / f"bad_{i}" / "submission.csv"
            with self.assertRaises(ValueError):
                publish_submission(predictions, self.root / self.config["data"]["raw_root"] / "test.csv", destination)
            self.assertFalse(destination.exists())

    def test_ordering_rejects_duplicates_label_changes_and_nonfinite_logits(self):
        (records, _), _ = self.dataset()
        rows = self.predictions(records["valid"], True)
        self.assertEqual(ordered_predictions(list(reversed(rows)), records["valid"], labeled=True), rows)
        for wrong in ([*rows, rows[0]], [{**r, "logit": float("inf")} for r in rows],
                      [{**r, "label": 1-r["label"]} for r in rows]):
            with self.assertRaises(ValueError):
                ordered_predictions(wrong, records["valid"], labeled=True)

    def test_reload_mismatch_blocks_test_inference(self):
        (records, _), _ = self.dataset()
        rows = self.predictions(records["valid"], True)
        wrong = copy.deepcopy(rows)
        wrong[0]["logit"] += .01
        with self.assertRaisesRegex(ValueError, "differs"):
            compare_predictions(wrong, rows, records["valid"], self.config)

    def test_preflight_budget_and_no_resume_cli(self):
        small = preflight_plan(self.config, 64)
        self.assertEqual((small["samples"], small["target_batch_memory"]), (56, "not_verified"))
        self.assertEqual(preflight_plan(self.config, 280)["target_batch_memory"], "verified")
        with self.assertRaises(ValueError):
            preflight_plan(self.config, 55)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run/run_timm_v2s_experiment.py"),
                                 "--resume-from", "anything.pt"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_cli_unresolved_config_creates_no_formal_outputs(self):
        temporary_config = load_timm_config(TEMPLATE, require_resolved=False)
        temporary_config["training"]["eval_batch_size"] = None
        temporary_config["experiment"]["output_root"] = str(self.root / "outputs")
        path = self.root / "unresolved.yaml"
        path.write_text(yaml.safe_dump(temporary_config), encoding="utf-8")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run/run_timm_v2s_experiment.py"),
                                 "--project-root", str(self.root), "--config", "unresolved.yaml"],
                                 capture_output=True, text=True, cwd=self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training.eval_batch_size", result.stderr)
        self.assertFalse((self.root / "outputs").exists())

    def test_missing_timm_dependency_is_explicit(self):
        from eye_qing_guang.models.timm_efficientnet import _timm
        with patch.dict(sys.modules, {"timm": None}):
            with self.assertRaisesRegex(RuntimeError, "pinned timm"):
                _timm(self.config["model"])

    @unittest.skipUnless(importlib.util.find_spec("timm"), "timm is not installed in this test environment")
    def test_real_timm_architecture_with_synthetic_pixels_no_pretrained_download(self):
        model, metadata = build_timm_model(self.config["model"], self.root, initialize_pretrained=False)
        model.eval()
        logits = model(torch.zeros(2, 3, 32, 32), None)
        self.assertEqual(logits.shape, (2, 1))
        loss = pw5_loss(self.device)(logits, torch.tensor([[0.0], [1.0]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metadata["num_features"], 1280)
        self.assertTrue(all(p.requires_grad and p.grad is not None for p in model.parameters()))

    def test_three_epoch_orchestration_and_complete_submission(self):
        (records, _), _ = self.dataset()
        directory = self.root / "new_run"
        directory.mkdir()
        config = copy.deepcopy(self.config)
        # Deliberately bounded unit fixture; never passed to the public config loader.
        config["training"].update(epochs=3, precision="fp32")
        manifest = {"status": "running", "stages": {s: {"status": "not_started"} for s in
                    ("training", "best_validation", "test_inference", "submission")}}
        rows = self.predictions(records["valid"], True)
        score_values = [.70, .71, .705]
        fake_train = {"consumed_samples": len(records["train"]), "weighted_bce_pw5": .3,
                      "lr": .0003, "successful_updates": 1}
        model = PixelClassifier()
        with patch("eye_qing_guang.pipelines.timm_official.train_epoch", return_value=fake_train), \
             patch("eye_qing_guang.pipelines.timm_official.distributed_validation", side_effect=[
                 (rows, {**compute_official_metrics(rows, .3), "roc_auc": score}) for score in score_values]):
            result = training_stage(model, self.metadata, records, config, self.root, self.device,
                                    LocalRuntime(), directory, manifest)
        self.assertEqual((result["best_epoch"], result["last_epoch"]), (2, 3))
        self.assertEqual(sorted(p.name for p in (directory / "checkpoints").iterdir()), ["best.pt", "last.pt"])
        self.assertEqual(len(read_csv(directory / "reports/history.csv")), 3)
        # The post-training test uses a safe synthetic loader result, not the incomplete source recipe.
        metrics = compute_official_metrics(rows, .3)
        payload = {"config": self.config, "epoch": 2, "metrics": metrics}
        manifest["stages"]["training"].update(status="completed", **result)
        test_rows = self.predictions(records["test"], False)
        with patch("eye_qing_guang.pipelines.timm_official.load_timm_checkpoint", return_value=(model, payload)), \
             patch("eye_qing_guang.pipelines.timm_official.predict_batches", side_effect=[(rows, .3), (test_rows, None)]):
            submission = post_training(directory, records, self.config, self.root, self.device, manifest)
        self.assertTrue(submission.is_file())
        self.assertEqual(manifest["status"], "completed")
        self.assertTrue(all(s["status"] == "completed" for s in manifest["stages"].values()))
        self.assertEqual(manifest["deployment"]["t4_latency_status"], "not_measured")

    def test_post_training_mismatch_or_submission_failure_never_claims_completed(self):
        (records, _), _ = self.dataset()
        rows = self.predictions(records["valid"], True)
        for failure in ("reload", "submission"):
            directory = self.root / failure
            (directory / "predictions/best").mkdir(parents=True)
            write_csv(directory / "predictions/best/valid_predictions.csv", rows)
            manifest = {"status": "running", "stages": {s: {"status": "not_started"} for s in
                        ("training", "best_validation", "test_inference", "submission")}}
            manifest["stages"]["training"]["status"] = "completed"
            actual = copy.deepcopy(rows)
            if failure == "reload":
                actual[0]["logit"] += .01
            payload = {"config": self.config, "epoch": 1, "metrics": compute_official_metrics(rows, .3)}
            with patch("eye_qing_guang.pipelines.timm_official.load_timm_checkpoint", return_value=(PixelClassifier(), payload)), \
                 patch("eye_qing_guang.pipelines.timm_official.predict_batches", side_effect=[
                     (actual, .3), (self.predictions(records["test"], False), None)]) as predict, \
                 patch("eye_qing_guang.pipelines.timm_official.publish_submission", side_effect=OSError("disk full")):
                with self.assertRaises((ValueError, OSError)):
                    post_training(directory, records, self.config, self.root, self.device, manifest)
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse((directory / "submission/submission.csv").exists())
            if failure == "reload":
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(manifest["stages"]["test_inference"]["status"], "not_started")
            else:
                self.assertEqual(manifest["stages"]["submission"]["status"], "failed")

    def test_training_failure_never_enters_test_or_submission(self):
        (records, sources), _ = self.dataset()
        config_path = self.root / "synthetic_config.yaml"
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        with patch("eye_qing_guang.pipelines.timm_official.create_source_runtime", return_value=(LocalRuntime(), self.device)), \
             patch("eye_qing_guang.pipelines.timm_official.ensure_official_manifests", return_value=(records, sources)), \
             patch("eye_qing_guang.pipelines.timm_official.execute_preflight", return_value={"target_batch_memory": "verified"}), \
             patch("eye_qing_guang.pipelines.timm_official.build_timm_model", return_value=(PixelClassifier(), self.metadata)), \
             patch("eye_qing_guang.pipelines.timm_official.training_stage", side_effect=RuntimeError("synthetic training failure")), \
             patch("eye_qing_guang.pipelines.timm_official.post_training") as post:
            with self.assertRaisesRegex(RuntimeError, "synthetic training failure"):
                run_timm_experiment(config_path, self.root)
        post.assert_not_called()
        directory = self.root / "experiments" / self.config["experiment"]["name"]
        manifest = json.loads((directory / "run_manifest.json").read_text())
        self.assertEqual(manifest["stages"]["training"]["status"], "failed")
        self.assertEqual(manifest["stages"]["training"]["completed_epochs"], 0)
        self.assertEqual(manifest["stages"]["test_inference"]["status"], "not_started")
        self.assertFalse((directory / "submission/submission.csv").exists())

    def test_small_preflight_does_not_create_formal_run(self):
        (records, sources), _ = self.dataset()
        config_path = self.root / "synthetic_config.yaml"
        config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        with patch("eye_qing_guang.pipelines.timm_official.create_source_runtime", return_value=(LocalRuntime(), self.device)), \
             patch("eye_qing_guang.pipelines.timm_official.ensure_official_manifests", return_value=(records, sources)), \
             patch("eye_qing_guang.pipelines.timm_official.execute_preflight", return_value={"target_batch_memory": "not_verified"}):
            with self.assertRaisesRegex(RuntimeError, "target batch memory"):
                run_timm_experiment(config_path, self.root)
        self.assertFalse((self.root / "experiments" / self.config["experiment"]["name"]).exists())


if __name__ == "__main__":
    unittest.main()

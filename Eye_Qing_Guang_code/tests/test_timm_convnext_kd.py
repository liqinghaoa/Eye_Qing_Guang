"""KD mathematics and isolated synthetic workflows; no task weights, GPU or real images."""

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import eye_qing_guang.contracts.convnext_kd_config as contract
from eye_qing_guang.contracts.convnext_config import load_convnext_config
from eye_qing_guang.contracts.convnext_checkpoints import load_convnext_checkpoint
from eye_qing_guang.contracts.convnext_kd_checkpoints import load_kd_student_checkpoint, validate_student_payload
from eye_qing_guang.datasets.convnext_official import official_records
from eye_qing_guang.datasets.offline_cache import build_offline_cache
from eye_qing_guang.models.frozen_teacher_ensemble import FrozenTeacherEnsemble, teacher_input
from eye_qing_guang.pipelines.convnext_official import _loader
from eye_qing_guang.pipelines.convnext_kd import (
    run_kd_experiment, run_kd_preflight, prepare_models, predict_kd_student,
)
from eye_qing_guang.trainers.convnext_official import build_optimizer_schedule, update_group as supervised_update
from eye_qing_guang.trainers.convnext_kd import make_targets, loss_terms, update_group
from eye_qing_guang.utils.experiment_io import write_csv, write_yaml, read_csv
from eye_qing_guang.utils.reproducibility import seed_everything
from eye_qing_guang.utils.timm_source import SingleDeviceRuntime, create_source_runtime

ROOT = Path(__file__).resolve().parents[1]


def config_for(variant="standard"):
    return contract.load_kd_config(ROOT / "configs" / (contract.EXPERIMENT_NAMES[variant] + ".yaml"))


def student_metadata():
    return {"timm_version": "1.0.22", "model_name": contract.MODEL_NAME, "num_features": 768,
            "num_classes": 0, "global_pool": "avg", "initialization": "pretrained",
            "normalization": "layernorm", "preprocessing_difference": "512 RGB/FOV/BILINEAR/letterbox; no timm center crop",
            "source": {"repo_id": "timm/" + contract.MODEL_NAME, "requested_revision": "main",
                       "local_file": "fixture.safetensors", "sha256": "0" * 64}}


def provenance():
    return [{**contract.teacher_identity(i), "size_bytes": 7,
             "pretrained_source": {"repo_id": "timm/" + spec["model_name"]}}
            for i, spec in enumerate(contract.TEACHERS)]


class ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 1)

    def forward(self, images, ages=None):
        return self.head(images.mean(dim=(-2, -1)))


class ToyTeacher(ToyStudent):
    def __init__(self, index):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.inputs = []
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[.2 + index * .1, -.3, .4]]))
            self.head.bias.fill_(index * .2)

    def forward(self, images, ages=None):
        self.inputs.append((images.dtype, self.training, len(images)))
        return self.head(self.bn(images.mean(dim=(-2, -1))))


def ensemble(chunk=2):
    return FrozenTeacherEnsemble([ToyTeacher(i) for i in range(3)], provenance(), chunk=chunk)


class KDMathTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = torch.device("cpu")

    def test_frozen_configs_and_wrong_method_settings(self):
        standard, rectified = config_for(), config_for("rectified")
        stripped = copy.deepcopy(rectified)
        stripped["experiment"]["name"] = standard["experiment"]["name"]
        stripped["distillation"].update(variant="standard", correction="none")
        self.assertEqual(standard, stripped)
        invalid = [
            ("training", "batch_size", 24), ("training", "pos_weight", 10.0),
            ("data", "image_size", 224), ("model", "name", "convnext_tiny.fb_in22k_ft_in1k"),
            ("distillation", "alpha", 0.0), ("distillation", "temperature", 2.0),
            ("distillation", "correction", "arm_binary_v1"),
            ("distillation", "teacher_forward_chunk", 4),
        ]
        for section, key, value in invalid:
            bad = copy.deepcopy(standard)
            bad[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_kd_config(bad)
        for key, value in (("expected_epoch", 30), ("weight", .5), ("loader", "timm_v2s_format3"),
                           ("checkpoint", "checkpoints/last.pt"), ("model_name", "wrong")):
            bad = copy.deepcopy(standard)
            bad["distillation"]["teachers"][0][key] = value
            with self.subTest(teacher_key=key), self.assertRaises(ValueError):
                contract.validate_kd_config(bad)
        for size in (224, 288, 512):
            load_convnext_config(ROOT / f"configs/timm_convnext_tiny_fb_in1k_{size}_pw5_official_single3090_cache_v1.yaml")
        load_convnext_config(ROOT / "configs/timm_convnext_tiny_fb_in1k_224_pw10_official_single3090_cache_v1.yaml")

    def test_arm_examples_boundaries_extremes_and_nonfinite_rejection(self):
        q = torch.tensor([[.8], [.2], [.8], [.5], [.5], [0.], [1.]], requires_grad=True)
        y = torch.tensor([[0.], [1.], [1.], [0.], [1.], [1.], [0.]])
        standard, _ = make_targets(q, y, "standard")
        target, diag = make_targets(q, y, "rectified")
        torch.testing.assert_close(target[:5], torch.tensor([[.06062866], [.93937134], [.8], [.125], [.5]]))
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.all((target > 0) & (target < 1)))
        torch.testing.assert_close(standard[2:5], q.detach()[2:5])
        self.assertEqual(int(diag["wrong"].sum()), 5)
        for bad in (float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(bad=bad), self.assertRaises((ValueError, FloatingPointError)):
                make_targets(torch.tensor([[bad]]), torch.zeros(1, 1), "rectified")
        with self.assertRaises(ValueError):
            make_targets(torch.tensor([[.2]]), torch.tensor([[.3]]), "standard")
        with self.assertRaises(ValueError):
            make_targets(torch.tensor([.2]), torch.zeros(1, 1), "standard")

    def test_soft_bce_gradient_matches_forward_bernoulli_kl(self):
        z = torch.tensor([[-2.], [.3], [2.]], requires_grad=True)
        q = torch.tensor([[.15], [.75], [.95]], requires_grad=True)
        y = torch.tensor([[0.], [1.], [1.]])
        terms = loss_terms(z, y, q)
        kl = q.detach() * (q.detach().log() - F.logsigmoid(z)) + (
            1 - q.detach()) * ((1 - q.detach()).log() - F.logsigmoid(-z))
        torch.testing.assert_close(torch.autograd.grad(terms["soft_bce"].mean(), z, retain_graph=True)[0],
                                   torch.autograd.grad(kl.mean(), z, retain_graph=True)[0])
        torch.testing.assert_close(terms["supervised_bce_pw5"],
                                   F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor([5.]), reduction="none"))
        torch.testing.assert_close(terms["soft_bce"], F.binary_cross_entropy_with_logits(z, q.detach(), reduction="none"))
        terms["total_loss"].mean().backward()
        self.assertIsNone(q.grad)
        self.assertTrue(torch.isfinite(z.grad).all())
        with self.assertRaises(FloatingPointError):
            loss_terms(torch.tensor([[-3e38]]), torch.ones(1, 1), torch.tensor([[.5]]))

    def test_normalization_chunk_order_and_frozen_bn(self):
        u = torch.rand(5, 3, 7, 9)
        mean, std = (u.new_tensor(v).view(1, 3, 1, 1) for v in (contract.IMAGENET_MEAN, contract.IMAGENET_STD))
        x = (u - mean) / std
        torch.testing.assert_close(teacher_input(x, 2), (u - .5) / .5)
        teachers = ensemble(2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            q = teachers.probabilities(x)
        expected = torch.stack([torch.sigmoid(m(teacher_input(x, i))) for i, m in enumerate(teachers.models)]).mean(0)
        torch.testing.assert_close(q, expected)
        self.assertEqual(teachers.forward_sample_instances, 15)
        self.assertFalse(q.requires_grad)
        self.assertTrue(all(dtype == torch.float32 and not training and count <= 2
                            for m in teachers.models for dtype, training, count in m.inputs[:-1]))
        teachers.verify_frozen()
        with torch.no_grad():
            teachers.models[0].bn.running_mean.add_(1)
        with self.assertRaisesRegex(ValueError, "state changed"):
            teachers.verify_frozen()
        with torch.no_grad():
            teachers.models[0].head.bias.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            teachers.probabilities(x)

    def test_unequal_accumulation_and_alpha_zero_match_supervised_update(self):
        config = config_for()
        settings = {**config["training"], "precision": "fp32", "amp": False}
        images, labels = torch.rand(3, 3, 5, 5), torch.tensor([[0.], [1.], [1.]])
        uneven = [{"image": images[:2], "label": labels[:2]}, {"image": images[2:], "label": labels[2:]}]
        full = [{"image": images, "label": labels}]
        for alpha in (0., 1.):
            left, right = ToyStudent(), ToyStudent()
            right.load_state_dict(left.state_dict())
            opt_l, sch_l = build_optimizer_schedule(left, settings, 2)
            opt_r, sch_r = build_optimizer_schedule(right, settings, 2)
            teachers = ensemble()
            kd = {**config["distillation"], "alpha": alpha}
            update_group(left, uneven, opt_l, sch_l, self.device, settings, kd, teachers)
            if alpha == 0:
                supervised_update(right, uneven, opt_r, sch_r, self.device, settings,
                                  SingleDeviceRuntime(config, self.device))
            else:
                update_group(right, full, opt_r, sch_r, self.device, settings, kd, teachers)
            for a, b in zip(left.parameters(), right.parameters()):
                torch.testing.assert_close(a, b)
            self.assertEqual(sch_l.successful_updates, 1)
            teachers.verify_frozen()
            contaminated = torch.optim.AdamW(list(left.parameters()) + list(teachers.models[0].parameters()))
            with self.assertRaisesRegex(ValueError, "student parameters"):
                teachers.assert_student_optimizer(left, contaminated)

    def test_single_runtime_registration_and_budget_rejection_precedes_forward(self):
        config = config_for()
        fake = SingleDeviceRuntime(config, self.device)
        with patch("eye_qing_guang.utils.timm_source._single_device_runtime", return_value=fake), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIs(create_source_runtime(config)[0], fake)
        raw = copy.deepcopy(config)
        raw["data"]["cache"]["enabled"] = False
        loader = _loader([{"image_id": "fixture", "image_path": "fixture.png"}], raw, ROOT, self.device)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), "spawn")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("eye_qing_guang.pipelines.convnext_kd.create_source_runtime", side_effect=AssertionError("GPU touched")):
            with self.assertRaisesRegex(ValueError, "216"):
                run_kd_preflight(ROOT / "configs" / (config["experiment"]["name"] + ".yaml"), ROOT,
                                 Path(tmp) / "no_output", 215)
            self.assertFalse((Path(tmp) / "no_output").exists())


class KDWorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root, self.device = Path(self.temp.name), torch.device("cpu")
        self.stack = ExitStack()
        self.counts = {"train": 9, "valid": 8, "test": 5}
        self.stack.enter_context(patch("eye_qing_guang.datasets.convnext_official.EXPECTED_COUNTS", self.counts))
        self.stack.enter_context(patch("eye_qing_guang.pipelines.convnext_kd.EXPECTED_LABEL_COUNTS",
                                       {"train": (5, 4), "valid": (4, 4)}))
        self.config = copy.deepcopy(config_for())
        settings = {**contract.FROZEN_TRAINING, "epochs": 3, "batch_size": 2, "accumulation_steps": 2,
                    "eval_batch_size": 3, "precision": "fp32", "amp": False, "min_epochs": 1, "patience": 2}
        self.stack.enter_context(patch.dict(contract.FROZEN_TRAINING, settings, clear=True))
        self.config["training"] = copy.deepcopy(settings)
        self.config["data"].update(raw_root="raw", manifest_dir="manifests", num_workers=6)
        self.config["data"]["cache"]["root"] = "cache/512"
        # Worker count remains frozen at six; patch the loader in this fixture to avoid spawning six processes.
        self.stack.enter_context(patch("eye_qing_guang.pipelines.convnext_kd._loader", side_effect=self.local_loader))
        self.stack.enter_context(patch("eye_qing_guang.pipelines.convnext_official._loader", side_effect=self.local_loader))
        self.stack.enter_context(patch("eye_qing_guang.pipelines.convnext_kd.create_source_runtime",
                                       side_effect=lambda c, r=None: (SingleDeviceRuntime(c, self.device), self.device)))
        for split, count in self.counts.items():
            folder = self.root / "raw" / split
            folder.mkdir(parents=True)
            rows = []
            for i in range(count):
                image_id = f"{split}_{i}.png"
                pixels = np.full((21, 29, 3), 20 + i * 20, dtype=np.uint8)
                Image.fromarray(pixels).save(folder / image_id)
                rows.append({"Image_name": image_id, "True_Label": str(i % 2) if split != "test" else "IGNORE", "Age": "40"})
            write_csv(self.root / "raw" / f"{split}.csv", rows)
        self.records, _ = official_records(self.config, self.root)
        weight = self.root / self.config["model"]["pretrained_file"]
        weight.parent.mkdir()
        weight.write_bytes(b"fixture")
        for spec in contract.TEACHERS:
            path = self.root / spec["checkpoint"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
        self.stack.enter_context(patch("eye_qing_guang.models.frozen_teacher_ensemble.load_convnext_checkpoint",
                                       side_effect=self.teacher_loader))
        self.stack.enter_context(patch("eye_qing_guang.models.frozen_teacher_ensemble.load_timm_checkpoint",
                                       side_effect=self.teacher_loader))
        self.stack.enter_context(patch("eye_qing_guang.pipelines.convnext_kd.build_convnext_model", side_effect=self.student_builder))
        self.stack.enter_context(patch("eye_qing_guang.contracts.convnext_kd_checkpoints.build_convnext_model", side_effect=self.student_builder))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def local_loader(self, records, config, root, device, runtime=None, *, training=False):
        from eye_qing_guang.datasets.loaders import make_loader
        local = copy.deepcopy(config)
        local["data"]["num_workers"] = 0
        return make_loader(records, local, root, training=training, device=device)

    def student_builder(self, *args, **kwargs):
        return ToyStudent(), student_metadata()

    def teacher_loader(self, path, device, root):
        index = next(i for i, spec in enumerate(contract.TEACHERS) if resolve_fixture(spec["checkpoint"], root) == path)
        expected = contract.teacher_identity(index)
        frozen = copy.deepcopy(self.config)
        frozen.pop("distillation")
        frozen["experiment"]["protocol"] = expected["protocol"]
        frozen["model"]["name"] = expected["model_name"]
        frozen["data"].update(mean=expected["mean"], std=expected["std"])
        return ToyTeacher(index).eval(), {
            "config": frozen, "format_version": 3, "checkpoint_kind": "inference_weight",
            **{key: expected[key] for key in ("model_name", "num_features", "epoch", "timm_version", "pos_weight")},
            "model_metadata": {"source": {"repo_id": "timm/" + expected["model_name"]}},
        }

    def test_teacher_identity_missing_file_and_random_state_isolation(self):
        build_offline_cache(self.config, self.root, workers=0)
        initialized, draws, batches = [], [], []
        for variant, extra in (("standard", 3), ("rectified", 17)):
            config = copy.deepcopy(self.config)
            config["distillation"].update(variant=variant, correction="none" if variant == "standard" else "arm_binary_v1")
            config["experiment"]["name"] = contract.EXPERIMENT_NAMES[variant]
            original = FrozenTeacherEnsemble.load
            def noisy_load(c, r, d):
                torch.rand(extra)
                return original(c, r, d)
            with patch("eye_qing_guang.pipelines.convnext_kd.FrozenTeacherEnsemble.load", side_effect=noisy_load):
                teachers, student, _ = prepare_models(config, self.root, self.device)
            initialized.append(copy.deepcopy(student.state_dict()))
            draws.append(torch.rand(5))
            batch = next(iter(self.local_loader(self.records["train"], config, self.root, self.device, training=True)))
            batches.append(batch)
            teachers.verify_frozen()
        for key in initialized[0]:
            torch.testing.assert_close(initialized[0][key], initialized[1][key], rtol=0, atol=0)
        torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
        self.assertEqual(batches[0]["image_id"], batches[1]["image_id"])
        torch.testing.assert_close(batches[0]["image"], batches[1]["image"], rtol=0, atol=0)
        original_loader = self.teacher_loader
        def wrong_loader(*args):
            model, payload = original_loader(*args)
            payload["epoch"] = 30
            return model, payload
        with patch("eye_qing_guang.models.frozen_teacher_ensemble.load_convnext_checkpoint", side_effect=wrong_loader):
            with self.assertRaisesRegex(ValueError, "identity"):
                FrozenTeacherEnsemble.load(self.config, self.root, self.device)
        (self.root / contract.TEACHERS[0]["checkpoint"]).unlink()
        with self.assertRaises(FileNotFoundError):
            FrozenTeacherEnsemble.load(self.config, self.root, self.device)

    def test_both_variants_student_only_checkpoint_reload_and_submission(self):
        build_offline_cache(self.config, self.root, workers=0)
        for variant in ("standard", "rectified"):
            config = copy.deepcopy(self.config)
            config["experiment"]["name"] = contract.EXPERIMENT_NAMES[variant]
            config["distillation"].update(variant=variant, correction="none" if variant == "standard" else "arm_binary_v1")
            path = self.root / f"{variant}.yaml"
            write_yaml(path, config)
            output = run_kd_experiment(path, self.root)
            history = read_csv(output / "reports/history.csv")
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["manifest_source_protocol"], contract.MANIFEST_PROTOCOL)
            self.assertTrue(manifest["teachers_frozen_verified_after_training"])
            self.assertTrue(all(int(row["consumed_samples"]) == 9 for row in history))
            self.assertTrue(all(int(row["corrected_samples"]) == 0 for row in history) if variant == "standard"
                            else all(int(row["corrected_samples"]) == int(row["teacher_wrong_samples"]) for row in history))
            self.assertEqual(len(read_csv(output / "submission/submission.csv")), 5)
            self.assertEqual(sorted(p.name for p in (output / "checkpoints").iterdir()), ["best.pt", "last.pt"])
            for name in ("best.pt", "last.pt"):
                _, payload = load_kd_student_checkpoint(output / "checkpoints" / name, self.device, self.root)
                self.assertEqual((payload["format_version"], payload["distillation_variant"]), (4, variant))
                self.assertEqual(set(payload["model_state"]), {"head.weight", "head.bias"})
            _, best = load_kd_student_checkpoint(output / "checkpoints/best.pt", self.device, self.root)
            maximum = max(float(row["roc_auc"]) for row in history)
            first_best = next(int(row["epoch"]) for row in history if float(row["roc_auc"]) == maximum)
            self.assertEqual(best["epoch"], first_best)
            with self.assertRaises(ValueError):
                load_convnext_checkpoint(output / "checkpoints/best.pt", self.device, self.root)
            bad = copy.deepcopy(best)
            bad["distillation_variant"] = "invalid"
            with self.assertRaises(ValueError):
                validate_student_payload(bad)
            # Simulate deployment with every teacher and the ImageNet source inaccessible.
            with patch("eye_qing_guang.models.frozen_teacher_ensemble.FrozenTeacherEnsemble.load",
                       side_effect=AssertionError("teacher access")), \
                 patch("eye_qing_guang.models.timm_convnext.prepared_pretrained_file",
                       side_effect=AssertionError("pretrained access")):
                def reload_builder(*args, **kwargs):
                    self.assertIs(kwargs.get("initialize_pretrained"), False)
                    return self.student_builder(*args, **kwargs)
                with patch("eye_qing_guang.contracts.convnext_kd_checkpoints.build_convnext_model", side_effect=reload_builder):
                    student, _ = load_kd_student_checkpoint(output / "checkpoints/best.pt", self.device, self.root)
                    self.assertEqual(tuple(student(torch.zeros(1, 3, 8, 8)).shape), (1, 1))
                    predicted = predict_kd_student(output / "checkpoints/best.pt", self.root / "manifests/test.csv",
                                                   self.root, output / "standalone.csv", self.device)
                    self.assertEqual(len(read_csv(predicted)), 5)
            with self.assertRaises(FileExistsError):
                run_kd_experiment(path, self.root)
        empty = self.root / "experiments" / contract.EXPERIMENT_NAMES["standard"]
        self.assertTrue(empty.exists())
        for spec in contract.TEACHERS:
            (self.root / spec["checkpoint"]).unlink()
        (self.root / self.config["model"]["pretrained_file"]).unlink()
        student, _ = load_kd_student_checkpoint(empty / "checkpoints/best.pt", self.device, self.root)
        self.assertEqual(tuple(student(torch.zeros(1, 3, 8, 8)).shape), (1, 1))
        config = copy.deepcopy(self.config)
        config["experiment"]["output_root"] = "existing_empty"
        (self.root / "existing_empty" / config["experiment"]["name"]).mkdir(parents=True)
        path = self.root / "empty.yaml"
        write_yaml(path, config)
        with self.assertRaises(FileExistsError):
            run_kd_experiment(path, self.root)

    def test_preflight_exact_216_instances_no_optimizer_step_or_formal_output(self):
        config = copy.deepcopy(self.config)
        config["training"]["batch_size"] = 48
        config["training"]["eval_batch_size"] = 16
        train_records = [{"image_id": f"tr{i}", "label": str(i % 2)} for i in range(48)]
        valid_records = self.records["valid"]
        images_valid, images_train = torch.rand(8, 3, 5, 5), torch.rand(48, 3, 5, 5)
        teachers = ensemble(8)
        for index, spec in enumerate(contract.TEACHERS):
            with torch.no_grad():
                logits = teachers.models[index](teacher_input(images_valid, index))
            rows = [{"image_id": row["image_id"], "label": row["label"], "logit": float(logits[i, 0]),
                     "probability": float(torch.sigmoid(logits[i, 0]))} for i, row in enumerate(valid_records)]
            write_csv((self.root / spec["checkpoint"]).parent.parent / "predictions/best/valid_predictions.csv", rows)
        batch_train = {"image": images_train, "label": torch.tensor([[float(i % 2)] for i in range(48)])}
        batch_valid = {"image": images_valid}
        path = self.root / "preflight.yaml"
        write_yaml(path, self.config)
        with patch("eye_qing_guang.pipelines.convnext_kd.load_kd_config", return_value=config), \
             patch("eye_qing_guang.pipelines.convnext_kd.prepare_sources",
                   return_value=({"train": train_records, "valid": valid_records, "test": []}, {})), \
             patch("eye_qing_guang.pipelines.convnext_kd.prepare_models", return_value=(teachers, ToyStudent(), student_metadata())), \
             patch("eye_qing_guang.pipelines.convnext_kd._loader",
                   side_effect=lambda *a, **kw: [batch_train if kw.get("training") else batch_valid]), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step")):
            output = run_kd_preflight(path, self.root, "preflight", 216)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual((report["status"], report["model_forward_sample_instances"]), ("completed", 216))
        self.assertEqual((report["teacher_forward_sample_instances"], report["student_backward_samples"]), (168, 48))
        self.assertFalse((self.root / "experiments" / self.config["experiment"]["name"]).exists())
        teachers.verify_frozen()

    def test_training_failure_records_partial_progress_and_does_not_retry(self):
        build_offline_cache(self.config, self.root, workers=0)
        path = self.root / "failure.yaml"
        write_yaml(path, self.config)
        def fail_epoch(*args, **kwargs):
            kwargs["progress"]["consumed_train_samples"] = 2
            raise FloatingPointError("synthetic gradient failure")
        with patch("eye_qing_guang.pipelines.convnext_kd.train_epoch", side_effect=fail_epoch) as train:
            with self.assertRaisesRegex(FloatingPointError, "synthetic"):
                run_kd_experiment(path, self.root)
            train.assert_called_once()
        output = self.root / "experiments" / self.config["experiment"]["name"]
        report = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual((report["status"], report["stages"]["training"]["status"]), ("failed", "failed"))
        self.assertEqual(report["training_progress"]["consumed_train_samples"], 2)
        self.assertFalse((output / "submission/submission.csv").exists())


def resolve_fixture(path, root):
    return (root / path).resolve()


if __name__ == "__main__":
    unittest.main()

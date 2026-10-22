"""Precision regression tests use tiny synthetic tensors, never competition data."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.contracts.precision import training_precision
from eye_qing_guang.pipelines.b0_development import new_progress
from eye_qing_guang.pipelines.preflight import ReplayLoader, clipped_gradients, compare_precision, cpu_state, fp32_reference
from eye_qing_guang.trainers.binary import (
    check_runtime_precision, finish_update, make_training_scaler, train_accumulated_epoch, training_autocast,
)
from eye_qing_guang.trainers.schedules import SuccessfulUpdateSchedule
from config_fixtures import load_config

ROOT = Path(__file__).resolve().parents[1]


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.linear = nn.Linear(4, 1)
        self.output_dtypes = []

    def forward(self, image, age=None):
        result = self.linear(self.bn(self.conv(image)).relu().mean(dim=(2, 3)))
        self.output_dtypes.append(result.dtype)
        return result


class PrecisionTest(unittest.TestCase):
    def test_explicit_and_legacy_precision_contract(self):
        legacy = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml")
        self.assertEqual(training_precision(legacy["training"]), "fp16")
        for precision in ("fp32", "bf16"):
            config = load_config(ROOT / f"configs/b0_resnet50_512_dev_ddp4_ppu_{precision}_v2.yaml")
            self.assertEqual(training_precision(config["training"]), precision)
            self.assertFalse(make_training_scaler(config["training"]).is_enabled())
            invalid = copy.deepcopy(config)
            invalid["training"]["amp"] = not invalid["training"]["amp"]
            with self.assertRaises(ValueError):
                validate_config(invalid)
        for invalid in ("auto", "float16", None, []):
            with self.assertRaises(ValueError):
                training_precision({"precision": invalid, "amp": True})
        # Disabled BF16 scaling must not disable BF16 autocast.
        with patch("torch.autocast") as autocast:
            training_autocast({"precision": "bf16", "amp": True}, torch.device("cuda", 0))
        self.assertEqual(autocast.call_args.kwargs, {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True})

    def test_nonfinite_unscaled_gradient_fails_before_step(self):
        for precision in ("fp32", "bf16"):
            model = nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            before = cpu_state(model)
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, float("inf"))
            scaler = make_training_scaler({"precision": precision, "amp": precision != "fp32"})
            with self.assertRaises(FloatingPointError):
                finish_update(model, optimizer, scaler, 5)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(before[name], value))

    def run_cuda_update(self, precision):
        torch.manual_seed(42)
        device = torch.device("cuda", 0)
        config = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml")
        settings = {**config["training"], "batch_size": 2, "accumulation_steps": 2,
                    "precision": precision, "amp": precision != "fp32"}
        check_runtime_precision(settings, device)
        model = ConvModel().to(device)
        initial = cpu_state(model)
        data = [{"image": torch.linspace(-1, 1, 192).reshape(3, 8, 8) + index / 10,
                 "age": torch.zeros(1), "label": torch.tensor([float(index % 2)])} for index in range(4)]
        replay = ReplayLoader(list(DataLoader(data, batch_size=2)), 2)
        criterion = nn.BCEWithLogitsLoss()
        reference = fp32_reference(model, replay, criterion, device, settings, new_progress())
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 2, settings)
        gradients = {}

        def record(row):
            self.assertTrue(row["success"])
            gradients.update(clipped_gradients(model))

        progress = new_progress()
        metrics = train_accumulated_epoch(model, replay, optimizer, criterion, device, settings,
                                          make_training_scaler(settings), schedule, progress, record)
        self.assertEqual(progress["successful_updates"], 1)
        self.assertEqual(progress["consumed_samples"], 4)
        self.assertTrue(any(not torch.equal(initial[name], parameter.detach().cpu()) for name, parameter in model.named_parameters()))
        self.assertEqual(model.output_dtypes[-1], torch.bfloat16 if precision == "bf16" else torch.float32)
        comparison = compare_precision(metrics["train_loss"], gradients, reference)
        self.assertEqual(comparison["status"], "passed", comparison)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA synthetic FP32 check")
    def test_cuda_fp32_update(self):
        self.run_cuda_update("fp32")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA synthetic BF16 check")
    def test_cuda_bf16_update_against_fp32(self):
        self.run_cuda_update("bf16")


if __name__ == "__main__":
    unittest.main()

"""Bounded resume/diagnostic checks; no competition training is launched."""

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from eye_qing_guang.contracts.checkpoints import save_checkpoint
from eye_qing_guang.contracts.config import validate_config
from eye_qing_guang.contracts.resume import load_resume_checkpoint, restore_training_state
from eye_qing_guang.datasets.loaders import make_loader
from eye_qing_guang.trainers.binary import make_training_scaler
from eye_qing_guang.trainers.schedules import DevelopmentStopping, SuccessfulUpdateSchedule
from eye_qing_guang.utils.distributed import DistributedContext
from eye_qing_guang.utils.experiment_io import load_config, new_experiment

ROOT = Path(__file__).resolve().parents[1]


class ResumeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="eqg_resume_synthetic_")
        self.root = Path(self.temporary.name)
        self.config = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_fp32_v2.yaml")
        self.config["training"].update(batch_size=2, accumulation_steps=2, epochs=3, min_epochs=1)
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.schedule = SuccessfulUpdateSchedule(self.optimizer, 1, self.config["training"])
        self.schedule.prepare()
        self.model(torch.ones(2, 1)).square().mean().backward()
        self.optimizer.step()
        self.schedule.record(True)
        stopping = DevelopmentStopping(1, 15, 1e-4)
        stopping.update(1, 0.7)
        state = {"scheduler": self.schedule.state_dict(), "scaler": {}, "early_stopping": stopping.state_dict(),
                 "progress": {"attempted_groups": 1, "successful_updates": 1, "skipped_groups": 0,
                              "consumed_samples": 16, "successful_samples": 16, "consecutive_skips": 0},
                 "distributed": {"world_size": 4}, "resume_supported": False}
        directory = new_experiment(self.root, "source")
        self.path = directory / "checkpoints/epoch_0001.pt"
        save_checkpoint(self.path, self.model, self.optimizer, 1, self.config, None, 1.0, {"roc_auc": 0.7}, state)

    def tearDown(self):
        self.temporary.cleanup()

    def test_restore_optimizer_schedule_and_per_rank_progress(self):
        target = copy.deepcopy(self.config)
        target["training"].update(batch_size=1, accumulation_steps=4)
        payload, info = load_resume_checkpoint(self.path, target)
        self.assertEqual(info["start_epoch"], 2)
        self.assertFalse(info["exact_replay"])
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 1, target["training"])
        progress, stopping = restore_training_state(payload, model, optimizer, schedule, make_training_scaler(target["training"]), 16, 4)
        self.assertEqual(progress["consumed_samples"], 4)
        self.assertEqual(progress["successful_updates"], 1)
        self.assertEqual(stopping.best_epoch, 1)
        self.assertEqual(schedule.prepare(), self.schedule.value(2))
        for actual, expected in zip(model.parameters(), self.model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state[actual]["exp_avg"], self.optimizer.state[expected]["exp_avg"])

    def test_incompatible_recipe_and_epoch_boundary_are_rejected(self):
        for section, key, value in (("training", "learning_rate", 0.002),
                                    ("data", "raw_root", "other_data"), ("development", "associations_csv", "other.csv")):
            target = copy.deepcopy(self.config)
            target[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_resume_checkpoint(self.path, target)
        payload, _ = load_resume_checkpoint(self.path, self.config)
        payload["training_state"]["progress"]["consumed_samples"] -= 1
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(ValueError, "epoch boundary"):
            restore_training_state(payload, model, optimizer, SuccessfulUpdateSchedule(optimizer, 1, self.config["training"]),
                                   make_training_scaler(self.config["training"]), 16, 4)

    def test_batch_change_preserves_next_lr_moments_and_repeated_resume(self):
        target = copy.deepcopy(self.config)
        target["training"].update(batch_size=1, accumulation_steps=1)
        payload, info = load_resume_checkpoint(self.path, target)
        self.assertEqual(info["batch_transition"]["target_global_batch"], 4)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 4, target["training"])
        progress, stopping = restore_training_state(payload, model, optimizer, schedule,
                                                    make_training_scaler(target["training"]), 16, 4)
        self.assertEqual(schedule.successful_updates, 1)
        self.assertEqual(schedule.total, 1 + 79 * 4)
        self.assertEqual(schedule.prepare(), self.schedule.value(2))
        self.assertAlmostEqual(schedule.value(schedule.total), target["training"]["min_lr"])
        before = schedule.state_dict()
        schedule.record(False)
        self.assertEqual(schedule.state_dict(), before)
        for actual, expected in zip(model.parameters(), self.model.parameters()):
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizer.state[actual][key], self.optimizer.state[expected][key])
        # Complete the next synthetic epoch with four successful updates.
        for _ in range(4):
            schedule.prepare()
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 1)).square().mean().backward()
            optimizer.step()
            schedule.record(True)
        progress.update(attempted_groups=5, successful_updates=5, consumed_samples=32, successful_samples=32)
        path = self.root / "changed_epoch_0002.pt"
        save_checkpoint(path, model, optimizer, 2, target, None, 1.0, {"roc_auc": 0.7}, {
            "scheduler": schedule.state_dict(), "scaler": {}, "early_stopping": stopping.state_dict(),
            "progress": progress, "distributed": {"world_size": 4}})
        for batch in (1, 2):
            next_config = copy.deepcopy(target)
            next_config["training"]["batch_size"] = batch
            loaded, _ = load_resume_checkpoint(path, next_config)
            next_model = nn.Linear(1, 1)
            next_optimizer = torch.optim.AdamW(next_model.parameters(), lr=0.001)
            next_schedule = SuccessfulUpdateSchedule(next_optimizer, 4 // batch, next_config["training"])
            counters, _ = restore_training_state(loaded, next_model, next_optimizer, next_schedule,
                                                make_training_scaler(next_config["training"]), 16, 4)
            self.assertEqual(counters["attempted_groups"], 5)
            self.assertEqual(next_schedule.successful_updates, 5)
            self.assertEqual(next_schedule.prepare(), schedule.value(6))
            self.assertEqual(next_schedule.total, schedule.total if batch == 1 else 5 + 78 * 2)
            self.assertAlmostEqual(next_schedule.value(next_schedule.total), 1e-6)

    def test_epoch32_batch104_schedule_and_corrupt_accounting(self):
        source = copy.deepcopy(self.config)
        source["training"].update(batch_size=128, accumulation_steps=1, epochs=80)
        payload = torch.load(self.path, weights_only=True)
        payload.update(config=source, epoch=32)
        state = payload["training_state"]
        source_schedule = SuccessfulUpdateSchedule(self.optimizer, 81, source["training"])
        source_schedule.successful_updates = 2592
        state["scheduler"] = source_schedule.state_dict()
        state["early_stopping"].update(best_epoch=32)
        state["progress"].update(attempted_groups=2592, successful_updates=2592,
                                 consumed_samples=32 * 41116, successful_samples=32 * 41116)
        for values in payload["optimizer_state"]["state"].values():
            values["step"].fill_(2592)
        payload["optimizer_state"]["param_groups"][0]["lr"] = source_schedule.value(2592)
        source_path = self.root / "synthetic_epoch_0032.pt"
        torch.save(payload, source_path)
        target = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume32_b104_v4.yaml")
        self.assertEqual((target["training"]["batch_size"], target["training"]["eval_batch_size"],
                          target["training"]["accumulation_steps"], target["data"]["num_workers"]), (104, 104, 1, 8))
        # Use the synthetic source recipe; only batch/eval batch differ.
        target = {**source, "training": {**source["training"], "batch_size": 104, "eval_batch_size": 104}}
        loaded, _ = load_resume_checkpoint(source_path, target)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 99, target["training"])
        counters, _ = restore_training_state(loaded, model, optimizer, schedule,
                                            make_training_scaler(target["training"]), 41116, 4)
        self.assertEqual(counters["consumed_samples"], 32 * 10279)
        self.assertEqual(schedule.successful_updates, 2592)
        self.assertEqual(schedule.total, 7344)
        self.assertEqual(schedule.prepare(), source_schedule.value(2593))
        rates = [schedule.value(index) for index in range(2593, schedule.total + 1)]
        self.assertTrue(all(math.isfinite(x) for x in rates))
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[-1], 1e-6)
        bad = copy.deepcopy(loaded)
        bad["training_state"]["progress"]["attempted_groups"] += 1
        with self.assertRaisesRegex(ValueError, "update count"):
            restore_training_state(bad, model, optimizer, SuccessfulUpdateSchedule(optimizer, 99, target["training"]),
                                   make_training_scaler(target["training"]), 41116, 4)

    def test_skips_are_not_remapped_into_successful_updates(self):
        payload = torch.load(self.path, weights_only=True)
        payload["epoch"] = 2
        payload["training_state"]["progress"].update(attempted_groups=2, skipped_groups=1, consumed_samples=32)
        target = copy.deepcopy(self.config)
        target["training"].update(batch_size=1, accumulation_steps=1, epochs=4)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 4, target["training"])
        counters, stopping = restore_training_state(payload, model, optimizer, schedule,
                                                    make_training_scaler(target["training"]), 16, 4)
        self.assertEqual(schedule.total, 1 + 78 * 4)
        self.assertEqual(schedule.successful_updates, 1)
        # Another completed epoch has four attempts but only three updates.
        schedule.record(False)
        for _ in range(3):
            schedule.prepare()
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(1, 1)).square().mean().backward()
            optimizer.step()
            schedule.record(True)
        counters.update(attempted_groups=6, successful_updates=4, skipped_groups=2,
                        consumed_samples=48, successful_samples=28)
        path = self.root / "skipped_epoch_0003.pt"
        save_checkpoint(path, model, optimizer, 3, target, None, 1.0, {"roc_auc": 0.7}, {
            "scheduler": schedule.state_dict(), "scaler": {}, "early_stopping": stopping.state_dict(),
            "progress": counters, "distributed": {"world_size": 4}})
        loaded, _ = load_resume_checkpoint(path, target)
        resumed = SuccessfulUpdateSchedule(optimizer, 4, target["training"])
        restored, _ = restore_training_state(loaded, model, optimizer, resumed,
                                            make_training_scaler(target["training"]), 16, 4)
        self.assertEqual(restored["skipped_groups"], 2)
        self.assertEqual(resumed.total, schedule.total)
        self.assertEqual(resumed.prepare(), schedule.value(5))

    def test_nonfinite_checkpoint_is_rejected(self):
        payload = torch.load(self.path, weights_only=True)
        payload["model_state"]["weight"].fill_(float("nan"))
        path = self.root / "corrupt.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_resume_checkpoint(path, self.config)

    @unittest.skipUnless(torch.cuda.is_available(), "Synthetic CUDA optimizer-device resume check")
    def test_cuda_optimizer_restore_and_next_update(self):
        target = copy.deepcopy(self.config)
        target["training"].update(batch_size=1, accumulation_steps=1)
        payload, _ = load_resume_checkpoint(self.path, target)
        model = nn.Linear(1, 1).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        schedule = SuccessfulUpdateSchedule(optimizer, 4, target["training"])
        restore_training_state(payload, model, optimizer, schedule, make_training_scaler(target["training"]), 16, 4)
        self.assertEqual(schedule.total, 1 + 79 * 4)
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].device.type, "cuda")
            self.assertEqual(state["exp_avg_sq"].device.type, "cuda")
        schedule.prepare()
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 1, device="cuda")).square().mean().backward()
        optimizer.step()
        schedule.record(True)
        self.assertEqual(schedule.successful_updates, 2)
        self.assertTrue(all(int(state["step"].item()) == 2 for state in optimizer.state.values()))

    def test_diagnostic_timeouts_and_zero_workers(self):
        config = load_config(ROOT / "configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume17_diag_v3.yaml")
        self.assertEqual((config["training"]["batch_size"], config["training"]["accumulation_steps"],
                          config["training"]["eval_batch_size"], config["data"]["num_workers"]), (128, 1, 128, 8))
        context = DistributedContext(0, 4, torch.device("cpu"))
        loader = make_loader([{}] * 4, config, ROOT, training=True, device=context.device, distributed=context)
        self.assertEqual(loader.timeout, 90)
        config["data"]["num_workers"] = 0
        self.assertEqual(make_loader([{}] * 4, config, ROOT, device=context.device).timeout, 0)
        config["diagnostics"]["data_timeout_seconds"] = 180
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_stalled_phase_produces_automatic_stack(self):
        code = '''
from pathlib import Path
from types import SimpleNamespace
import time, torch
from eye_qing_guang.utils.training_diagnostics import TrainingDiagnostics
p=Path(__import__('sys').argv[1]); (p/'reports').mkdir()
context=SimpleNamespace(rank=0,primary=True,device=torch.device('cpu'))
d=TrainingDiagnostics(p,context,{'stack_timeout_seconds':0.1,'device_poll_timeout_seconds':1,'device_poll_seconds':1})
d.mark('synthetic_blocked_phase',epoch=18,group=1442)
time.sleep(0.25)
d.close()
'''
        subprocess.run([sys.executable, "-c", code, str(self.root)], check=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
        stages = [json.loads(line) for line in (self.root / "reports/diagnostics_rank_0000.jsonl").read_text().splitlines()]
        self.assertEqual(stages[-1]["phase"], "synthetic_blocked_phase")
        self.assertEqual(stages[-1]["group"], 1442)
        self.assertIn("Timeout", (self.root / "reports/stacks_rank_0000.log").read_text())


if __name__ == "__main__":
    unittest.main()

"""Per-rank phase evidence and bounded, read-only device snapshots for DDP hangs."""

import faulthandler
import json
import shutil
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone

import torch
from torch.utils.data.distributed import DistributedSampler


class TrainingDiagnostics:
    def __init__(self, directory, context, options):
        self.context, self.options = context, options
        self.details, self.expected, self.records = {}, None, None
        self.stop = threading.Event()
        self.events = (directory / "reports" / f"diagnostics_rank_{context.rank:04d}.jsonl").open("x", encoding="utf-8")
        self.stacks = (directory / "reports" / f"stacks_rank_{context.rank:04d}.log").open("x", encoding="utf-8")
        self.snapshots, self.thread = None, None
        if context.primary and context.device.type == "cuda":
            self.snapshots = (directory / "reports" / "device_snapshots.jsonl").open("x", encoding="utf-8")
            self.thread = threading.Thread(target=self._poll_devices, daemon=True)
            self.thread.start()
        self.mark("diagnostics_started")

    def mark(self, phase, memory=False, **details):
        self.details.update({key: value for key, value in details.items() if not key.endswith("image_ids")})
        row = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": time.monotonic(),
               "rank": self.context.rank, "device": str(self.context.device), "phase": phase, **self.details, **details}
        if memory and self.context.device.type == "cuda":
            # Allocator counters only: no device synchronization or driver memory query in the hot path.
            row["allocator"] = {"allocated_bytes": torch.cuda.memory_allocated(self.context.device),
                                "reserved_bytes": torch.cuda.memory_reserved(self.context.device),
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.context.device),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.context.device)}
        self.events.write(json.dumps(row, allow_nan=False) + "\n")
        self.events.flush()
        faulthandler.dump_traceback_later(self.options["stack_timeout_seconds"], repeat=True, file=self.stacks)

    def bind_loader(self, loader):
        self.records = getattr(loader.dataset, "records", None)
        self.expected = list(loader.sampler) if self.records is not None and isinstance(loader.sampler, DistributedSampler) else None

    def waiting_for_batch(self, offset, batch_size, **details):
        ids = ([self.records[index]["image_id"] for index in self.expected[offset:offset+batch_size]]
               if self.expected is not None else [])
        self.mark("data_wait", expected_image_ids=ids, **details)

    def capture_failure(self):
        traceback.print_exc(file=self.stacks)
        self.stacks.flush()
        faulthandler.dump_traceback(file=self.stacks, all_threads=True)

    def _poll_devices(self):
        executable = shutil.which("nvidia-smi") or shutil.which("ppu-smi")
        while not self.stop.is_set():
            row = {"utc": datetime.now(timezone.utc).isoformat(), "command": executable}
            try:
                if executable is None:
                    raise FileNotFoundError("nvidia-smi/ppu-smi not available; allocator counters remain in rank logs")
                result = subprocess.run([executable], capture_output=True, text=True,
                                        timeout=self.options["device_poll_timeout_seconds"])
                row.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            except (OSError, subprocess.TimeoutExpired) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            self.snapshots.write(json.dumps(row) + "\n")
            self.snapshots.flush()
            if executable is None or self.stop.wait(self.options["device_poll_seconds"]):
                break

    def close(self):
        faulthandler.cancel_dump_traceback_later()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.options["device_poll_timeout_seconds"] + 2)
            if not self.thread.is_alive():
                self.snapshots.close()
        self.events.close()
        self.stacks.close()

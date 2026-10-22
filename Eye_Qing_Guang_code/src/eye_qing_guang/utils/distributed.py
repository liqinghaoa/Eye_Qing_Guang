"""Single-host DDP coordination; all communication tensors use the bound device."""

import json
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self):
        return self.rank == 0

    def reduce(self, values, operation=dist.ReduceOp.SUM):
        tensor = torch.tensor(values, dtype=torch.float64 if self.device.type == "cpu" else torch.float32,
                              device=self.device)
        dist.all_reduce(tensor, op=operation)
        return tensor.cpu().tolist()

    def agree(self, values, description):
        """Check small numerical state BEFORE an optimizer can diverge."""
        tensor = torch.tensor(values, dtype=torch.float64 if self.device.type == "cpu" else torch.float32,
                              device=self.device)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        if any(not torch.equal(gathered[0], other) for other in gathered[1:]):
            raise RuntimeError(f"DDP ranks disagree: {description}")

    def all_true(self, condition):
        return bool(self.reduce([int(condition)], dist.ReduceOp.MIN)[0])

    def gather_json(self, value):
        """Primitive metadata/results via UTF-8 bytes, without CPU/NCCL or pickle."""
        encoded = json.dumps(value, allow_nan=False).encode("utf-8")
        size = torch.tensor([len(encoded)], dtype=torch.int64, device=self.device)
        sizes = [torch.empty_like(size) for _ in range(self.world_size)]
        dist.all_gather(sizes, size)
        lengths = [int(item.item()) for item in sizes]
        payload = torch.zeros(max(lengths), dtype=torch.uint8, device=self.device)
        payload[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=self.device)
        buffers = [torch.empty_like(payload) for _ in range(self.world_size)]
        dist.all_gather(buffers, payload)
        if not self.primary:
            return None
        return [json.loads(bytes(buffer[:length].cpu().tolist()).decode("utf-8"))
                for buffer, length in zip(buffers, lengths)]

    def broadcast_json(self, value=None):
        encoded = json.dumps(value, allow_nan=False).encode("utf-8") if self.primary else b""
        size = torch.tensor([len(encoded)], dtype=torch.int64, device=self.device)
        dist.broadcast(size, src=0)
        payload = torch.empty(int(size.item()), dtype=torch.uint8, device=self.device)
        if self.primary:
            payload.copy_(torch.tensor(list(encoded), dtype=torch.uint8, device=self.device))
        dist.broadcast(payload, src=0)
        return json.loads(bytes(payload.cpu().tolist()).decode("utf-8"))

    def on_primary(self, function):
        """Root-only I/O/metrics with a collective success or failure decision."""
        result, error = None, None
        if self.primary:
            try:
                result = {"ok": True, "value": function()}
                json.dumps(result, allow_nan=False)
            except BaseException as exc:
                error = exc
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = self.broadcast_json(result)
        if not result["ok"]:
            if error is not None:
                raise error
            raise RuntimeError(f"Primary rank failed: {result['error']}")
        return result["value"]

    def synchronize_buffers(self, model):
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        return {"rank": self.rank,
                "allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                "reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0}


def wrap_model(model, context):
    """CPU branch supports synthetic tests only; production initialization requires PPU."""
    options = ({"device_ids": [context.device.index], "output_device": context.device.index}
               if context.device.type == "cuda" else {})
    return torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=True, **options)


def initialize_distributed(config):
    """Production entry: four permitted PPU devices only; CPU/Gloo is test-only."""
    from eye_qing_guang.contracts.config import validate_config

    validate_config(config)
    return _initialize_ppu_ddp(config)


def initialize_timm_distributed(config):
    """Initialize the same four-rank PPU/NCCL runtime for the independent timm recipe."""
    from eye_qing_guang.contracts.timm_config import validate_timm_config

    validate_timm_config(config)
    return _initialize_ppu_ddp(config)


def _initialize_ppu_ddp(config):
    if config["experiment"]["protocol"] != "b0_dev_ddp_v1":
        if config["experiment"]["protocol"] != "timm_official_trainvalid_v1":
            raise ValueError("An explicitly supported four-rank DDP configuration is required.")
    options = config["distributed"]
    try:
        rank, local_rank, world_size = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Start DDP with torchrun --standalone --nnodes=1 --nproc_per_node=4.") from exc
    if (world_size != 4 or not 0 <= rank < 4 or rank != local_rank
            or int(os.environ.get("LOCAL_WORLD_SIZE", "4")) != 4):
        raise RuntimeError("DDP v1 requires one host and exactly four local ranks.")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") != ["4", "5", "6", "7"]:
        raise RuntimeError("Preserve the container CUDA_VISIBLE_DEVICES=4,5,6,7 assignment.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Exactly four assigned CUDA-compatible PPU devices must be available.")
    if not dist.is_nccl_available():
        raise RuntimeError("The installed PPU build has no nccl-compatible backend; do not replace its stack.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=options["backend"], rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=options["timeout_seconds"]))
    return DistributedContext(rank, world_size, device)


def close_distributed():
    # No final barrier: a failed peer must not trap the remaining ranks here.
    if dist.is_initialized():
        dist.destroy_process_group()


def communication_check(config):
    context = initialize_distributed(config)
    try:
        actual = context.reduce([context.rank + 1])[0]
        if actual != 10:
            raise RuntimeError(f"Four-rank all_reduce returned {actual}, expected 10.")
        metadata = context.gather_json({"rank": context.rank, "device": str(context.device)})
        context.agree([1.0, 65536.0], "communication smoke check")
        context.on_primary(lambda: print(f"DDP communication PASS: {metadata}; no training data consumed", flush=True))
    finally:
        close_distributed()

"""Local single-device execution and the existing four-rank native DDP adapter."""

import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

from eye_qing_guang.utils.distributed import (
    close_distributed, initialize_timm_distributed, wrap_model,
)


def launch_coordinates():
    try:
        coordinates = tuple(int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Start the four-rank experiment with external torchrun.") from exc
    rank, local_rank, world = coordinates
    if world != 4 or rank != local_rank or not 0 <= rank < 4:
        raise RuntimeError("This experiment requires one node and exactly four local ranks.")
    return coordinates


class SingleDeviceRuntime:
    """Local execution only: no process group, DDP wrapper or collectives."""

    rank = local_rank = 0
    world_size = 1

    def __init__(self, config, device):
        self.config, self.device = config, device
        self.closed = False

    def checked(self, phase, action):
        return action()

    def primary(self, phase, action):
        return action()

    def broadcast(self, value):
        return value

    def sum_numbers(self, numbers):
        return numbers

    def gather_predictions(self, rows):
        return rows

    def initialize_model(self, model):
        return model

    def unwrap_model(self, model):
        return model

    def verify_parameters(self, model):
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Non-finite model parameters.")

    def average_gradients(self, model, local_samples):
        return None

    def accumulation_context(self, model, synchronize):
        return nullcontext()

    def evaluation_state(self, model):
        return nullcontext()

    def close(self):
        self.closed = True


def _single_device_runtime(config):
    # Reject a multi-process launcher before any device or output is touched.
    try:
        coordinates = {key: int(os.environ.get(key, default)) for key, default in (
            ("WORLD_SIZE", "1"), ("LOCAL_WORLD_SIZE", "1"),
            ("RANK", "0"), ("LOCAL_RANK", "0"))}
    except ValueError as exc:
        raise RuntimeError("Invalid launcher environment for single-process training.") from exc
    if coordinates != {"WORLD_SIZE": 1, "LOCAL_WORLD_SIZE": 1, "RANK": 0, "LOCAL_RANK": 0}:
        raise RuntimeError("Single-card configuration requires one Python process; do not launch multiple ranks.")
    if dist.is_initialized():
        raise RuntimeError("Single-card training requires no active distributed process group.")
    device = torch.device(config["training"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Single-card training requires an available CUDA device.")
    device = torch.device("cuda", 0 if device.index is None else device.index)
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Selected device {device} is outside CUDA_VISIBLE_DEVICES.")
    torch.cuda.set_device(device)
    return SingleDeviceRuntime(config, device)


class B0DDPRuntime:
    """Keeps B0's NCCL/DDP process group; DDP reduces each accumulation group once."""

    def __init__(self, config, context=None):
        self.config = config
        self.context = context if context is not None else initialize_timm_distributed(config)
        self.rank, self.local_rank = self.context.rank, self.context.rank
        self.world_size, self.device = self.context.world_size, self.context.device
        self.closed = False

    def checked(self, phase, action):
        value, error = None, None
        try:
            value = action()
        except BaseException as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        success = self.context.all_true(error is None)
        if not success:
            gathered = self.context.gather_json({"error": error})
            errors = [item["error"] for item in gathered] if self.rank == 0 else None
            errors = self.broadcast(errors)
            raise RuntimeError(f"{phase} failed: " + "; ".join(item for item in errors if item))
        return value

    def primary(self, phase, action):
        result = None
        if self.rank == 0:
            try:
                result = {"ok": True, "value": action()}
            except BaseException as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result = self.broadcast(result)
        if not result["ok"]:
            raise RuntimeError(f"{phase} failed on rank 0: {result['error']}")
        return result["value"]

    def broadcast(self, value):
        return self.context.broadcast_json(value)

    def sum_numbers(self, numbers):
        return self.context.reduce(numbers)

    def gather_predictions(self, rows):
        parts = self.context.gather_json(rows)
        return [row for part in parts for row in part] if self.rank == 0 else None

    def initialize_model(self, model):
        return wrap_model(model, self.context)

    def unwrap_model(self, model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    def verify_parameters(self, model):
        bare = self.unwrap_model(model)
        for parameter in bare.parameters():
            value = parameter.detach().contiguous()
            reference = value.clone()
            dist.broadcast(reference, src=0)
            if not torch.isfinite(value).all() or not torch.allclose(
                    value, reference, rtol=1e-6, atol=1e-6):
                raise FloatingPointError("Updated DDP parameters are non-finite or differ across ranks.")

    def average_gradients(self, model, local_samples):
        # DDP performed its averaged all-reduce on the final backward of each group.
        return None

    def accumulation_context(self, model, synchronize):
        if not synchronize and isinstance(model, torch.nn.parallel.DistributedDataParallel):
            return model.no_sync()
        return nullcontext()

    @contextmanager
    def evaluation_state(self, model):
        bare = self.unwrap_model(model)
        local_buffers = {name: value.detach().clone() for name, value in bare.named_buffers()}
        self.context.synchronize_buffers(bare)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in bare.named_buffers():
                    value.copy_(local_buffers[name])

    def close(self):
        if not self.closed:
            self.closed = True
            close_distributed()


def create_source_runtime(config, root=None):
    if (config["experiment"]["protocol"] in {"timm_official_trainvalid_v1", "timm_convnext_official_trainvalid_v1", "timm_convnext_final_fit_v1"}
            and config["distributed"]["enabled"] is False):
        if config["experiment"]["protocol"] == "timm_convnext_final_fit_v1":
            from eye_qing_guang.contracts.convnext_final_config import validate_convnext_final_config
            validate_convnext_final_config(config)
        elif config["experiment"]["protocol"] == "timm_convnext_official_trainvalid_v1":
            from eye_qing_guang.contracts.convnext_config import validate_convnext_config
            validate_convnext_config(config)
        else:
            from eye_qing_guang.contracts.timm_config import validate_timm_config
            validate_timm_config(config)
        runtime = _single_device_runtime(config)
        if config["training"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            runtime.close()
            raise RuntimeError("BF16 is unavailable on the selected device; choose explicit FP32 configuration.")
        return runtime, runtime.device
    rank, local_rank, world = launch_coordinates()
    runtime = B0DDPRuntime(config)
    if (runtime.rank, runtime.local_rank, runtime.world_size) != (rank, local_rank, world):
        runtime.close()
        raise RuntimeError("torchrun rank topology changed during initialization.")
    if config["training"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        runtime.close()
        raise RuntimeError("BF16 is unavailable on the selected PPU build; no automatic precision downgrade.")
    return runtime, runtime.device

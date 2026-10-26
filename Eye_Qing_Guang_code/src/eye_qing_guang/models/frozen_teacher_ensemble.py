"""Three independent frozen FP32 classifiers, sharing one augmented RGB view."""

import torch

from eye_qing_guang.contracts.convnext_checkpoints import load_convnext_checkpoint
from eye_qing_guang.contracts.timm_checkpoints import load_timm_checkpoint
from eye_qing_guang.contracts.convnext_kd_config import (
    IMAGENET_MEAN, IMAGENET_STD, teacher_identity, validate_teacher_provenance,
)
from eye_qing_guang.utils.paths import resolve_path


def finite_state(model):
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite teacher state: {name}")


def teacher_input(student_images, index):
    if student_images.dtype != torch.float32 or student_images.ndim != 4 or student_images.shape[1] != 3:
        raise ValueError("Shared teacher view must be the original FP32 [B,3,H,W] student input.")
    if not torch.isfinite(student_images).all():
        raise FloatingPointError("Non-finite shared augmented view.")
    if index < 2:
        return student_images
    if index != 2:
        raise ValueError("Invalid teacher index.")
    mean = student_images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = student_images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    converted = (student_images * std + mean - .5) / .5
    if not torch.isfinite(converted).all():
        raise FloatingPointError("Non-finite teacher normalization conversion.")
    return converted


class FrozenTeacherEnsemble:
    """Not a student submodule: train(), optimizer and student state_dict exclude teachers."""

    def __init__(self, models, provenance, *, chunk=8, epsilon=1e-6):
        if len(models) != 3 or type(chunk) is not int or chunk < 1:
            raise ValueError("Three teachers and a positive forward chunk are required.")
        self.models = list(models)
        self.provenance = validate_teacher_provenance(provenance)
        self.chunk, self.epsilon = chunk, epsilon
        self.forward_sample_instances = 0
        self._initial_state = []
        for model in self.models:
            finite_state(model)
            model.float().eval().requires_grad_(False)
            for parameter in model.parameters():
                parameter.grad = None
            self._initial_state.append({name: value.detach().cpu().clone()
                                        for name, value in model.state_dict().items()})
        self.verify_frozen()

    @classmethod
    def load(cls, config, root, device):
        models, provenance = [], []
        for index, spec in enumerate(config["distillation"]["teachers"]):
            path = resolve_path(spec["checkpoint"], root)
            if not path.is_file() or path.name != "best.pt":
                raise FileNotFoundError(f"Required teacher best checkpoint is missing: {path}")
            loader = load_convnext_checkpoint if spec["loader"] == "convnext_format3" else load_timm_checkpoint
            model, payload = loader(path, device, root)
            expected, frozen = teacher_identity(index), payload["config"]
            actual = {
                "epoch": payload.get("epoch"), "model_name": payload.get("model_name"),
                "num_features": payload.get("num_features"), "timm_version": payload.get("timm_version"),
                "image_size": frozen["data"]["image_size"], "pos_weight": payload.get("pos_weight"),
                "protocol": frozen["experiment"]["protocol"], "mean": frozen["data"]["mean"],
                "std": frozen["data"]["std"],
            }
            if (payload.get("format_version") != 3 or payload.get("checkpoint_kind") != "inference_weight"
                    or any(actual[key] != expected[key] for key in actual)
                    or frozen["training"]["pos_weight"] != 5.0 or frozen["model"]["use_age"] is not False
                    or frozen["model"]["name"] != spec["model_name"]):
                raise ValueError(f"Teacher {spec['id']} differs from the frozen model/epoch/512/pw5 identity.")
            source = payload.get("model_metadata", {}).get("source")
            item = {**expected, "size_bytes": path.stat().st_size, "pretrained_source": source}
            models.append(model)
            provenance.append(item)
        return cls(models, provenance, chunk=config["distillation"]["teacher_forward_chunk"],
                   epsilon=config["distillation"]["epsilon"])

    def verify_frozen(self):
        for model, reference in zip(self.models, self._initial_state):
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
                raise ValueError("Teachers must stay in eval mode without gradients.")
            finite_state(model)
            state = model.state_dict()
            if set(state) != set(reference):
                raise ValueError("Teacher state structure changed.")
            for name, value in state.items():
                if value.is_floating_point() and value.dtype != torch.float32:
                    raise ValueError("Teacher floating state must stay FP32.")
                if not torch.equal(value.detach().cpu(), reference[name]):
                    raise ValueError(f"Frozen teacher state changed: {name}")

    def assert_student_optimizer(self, student, optimizer):
        expected = {id(p) for p in student.parameters() if p.requires_grad}
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        teacher_ids = {id(p) for model in self.models for p in model.parameters()}
        if (not expected or len(actual) != len(expected) or {id(p) for p in actual} != expected
                or expected & teacher_ids):
            raise ValueError("Optimizer must contain exactly the trainable student parameters.")

    @torch.no_grad()
    def individual_predictions(self, images):
        if not len(images):
            raise ValueError("Empty teacher batch.")
        outputs = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            for index, model in enumerate(self.models):
                if model.training or any(p.requires_grad for p in model.parameters()):
                    raise ValueError("Teacher mode or gradient flags changed.")
                pieces = []
                for chunk in images.split(self.chunk):
                    view = teacher_input(chunk, index)
                    self.forward_sample_instances += len(chunk)
                    logits = model(view, None)
                    if logits.dtype != torch.float32 or logits.shape != (len(chunk), 1):
                        raise ValueError("Teacher must return FP32 [B,1] logits.")
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite teacher logits.")
                    pieces.append(logits.detach())
                logits = torch.cat(pieces)
                probability = torch.sigmoid(logits)
                if not torch.isfinite(probability).all():
                    raise FloatingPointError("Non-finite teacher probability.")
                outputs.append((logits, probability))
        return outputs

    @torch.no_grad()
    def probabilities(self, images):
        outputs = self.individual_predictions(images)
        q = torch.stack([probability for _, probability in outputs]).mean(dim=0)
        if q.shape != (len(images), 1) or not torch.isfinite(q).all():
            raise FloatingPointError("Invalid fused teacher probability.")
        return q.clamp(self.epsilon, 1 - self.epsilon).detach()

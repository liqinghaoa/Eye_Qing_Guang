"""B0 successful-update learning-rate indexing and independent stopping state."""

import copy
import math
from dataclasses import asdict, dataclass


class SuccessfulUpdateSchedule:
    def __init__(self, optimizer, groups_per_epoch, settings):
        self.optimizer = optimizer
        self.total = settings["schedule_total_epochs"] * groups_per_epoch
        self.warmup = settings["warmup_epochs"] * groups_per_epoch
        self.start, self.peak, self.minimum = (
            settings["warmup_start_lr"], settings["learning_rate"], settings["min_lr"])
        self.successful_updates = 0
        if not 1 < self.warmup < self.total:
            raise ValueError("Invalid successful-update schedule length.")

    def value(self, index):
        if not 1 <= index <= self.total:
            raise ValueError("Successful-update index exceeds frozen schedule.")
        return self._value(self.state_dict(), index)

    @classmethod
    def _value(cls, state, index):
        continuation = state.get("continuation")
        if continuation is not None:
            anchor, previous = continuation["anchor"], continuation["previous"]
            if index <= anchor:
                return cls._value(previous, index)
            # Preserve the old NEXT learning rate, stretching only the unexecuted
            # tail. Successful updates (not fetched batches/skips) advance it.
            fraction = (index - anchor - 1) / (state["total"] - anchor - 1)
            reference = anchor + 1 + fraction * (previous["total"] - anchor - 1)
            return cls._value(previous, reference)
        if index <= state["warmup"]:
            return state["start"] + (state["peak"] - state["start"]) * (index - 1) / (state["warmup"] - 1)
        fraction = (index - state["warmup"]) / (state["total"] - state["warmup"])
        return state["minimum"] + 0.5 * (state["peak"] - state["minimum"]) * (1 + math.cos(math.pi * fraction))

    def load_state_dict(self, state):
        self._validate_state(state)
        for key in ("start", "peak", "minimum"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume schedule mismatch: {key}")
        self.__dict__.pop("continuation", None)
        self.__dict__.update(copy.deepcopy(state))

    @classmethod
    def _validate_state(cls, state):
        keys = {"total", "warmup", "start", "peak", "minimum", "successful_updates"}
        if not isinstance(state, dict) or not keys <= state.keys() or state.keys() - keys - {"continuation"}:
            raise ValueError("Invalid saved scheduler fields.")
        if (type(state["total"]) is not int or type(state["warmup"]) is not int or state["warmup"] <= 1
                or type(state["successful_updates"]) is not int
                or not 0 <= state["successful_updates"] <= state["total"]
                or any(not isinstance(state[k], (int, float)) or not math.isfinite(state[k])
                       or state[k] <= 0 for k in ("start", "peak", "minimum"))):
            raise ValueError("Invalid saved scheduler values.")
        continuation = state.get("continuation")
        if continuation is None:
            if not state["warmup"] < state["total"]:
                raise ValueError("Invalid saved warmup length.")
            return
        if not isinstance(continuation, dict) or set(continuation) != {"anchor", "previous"}:
            raise ValueError("Invalid batch-change scheduler continuation.")
        previous, anchor = continuation["previous"], continuation["anchor"]
        cls._validate_state(previous)
        if (type(anchor) is not int or anchor < 1 or anchor != previous["successful_updates"]
                or not anchor <= state["successful_updates"] <= state["total"]
                or min(state["total"], previous["total"]) < anchor + 2
                or any(state[k] != previous[k] for k in ("warmup", "start", "peak", "minimum"))):
            raise ValueError("Invalid batch-change scheduler anchor.")

    def retime_remaining(self, remaining_updates):
        """Keep history/AdamW step indices; map the new tail to the saved curve."""
        if remaining_updates < 2 or self.total - self.successful_updates < 2:
            raise ValueError("Batch-change resume needs at least two remaining scheduled updates.")
        previous = copy.deepcopy(self.state_dict())
        self.continuation = {"anchor": self.successful_updates, "previous": previous}
        self.total = self.successful_updates + remaining_updates

    def prepare(self):
        rate = self.value(self.successful_updates + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def record(self, success):
        self.successful_updates += int(success)

    def state_dict(self):
        return {key: value for key, value in vars(self).items() if key != "optimizer"}


@dataclass
class DevelopmentStopping:
    min_epochs: int
    patience: int
    min_delta: float
    best_auc: float = -float("inf")
    best_epoch: int | None = None
    reference_auc: float | None = None
    stale: int = 0

    def update(self, epoch, auc):
        if auc is None or not math.isfinite(auc):
            raise ValueError("Internal development AUROC must be finite.")
        improved = auc > self.best_auc
        if improved:
            self.best_auc, self.best_epoch = auc, epoch
        if self.reference_auc is None or auc > self.reference_auc + self.min_delta:
            self.reference_auc, self.stale = auc, 0
        else:
            self.stale += 1
        return improved, epoch >= self.min_epochs and self.stale >= self.patience

    def state_dict(self):
        return asdict(self)


def fit_length(best_epochs):
    """Protocol arithmetic only; full-fit training is deliberately not dispatched."""
    import statistics

    if not best_epochs or any(type(epoch) is not int or epoch <= 0 for epoch in best_epochs):
        raise ValueError("All planned development runs must supply positive completed best epochs.")
    raw = math.ceil(statistics.median(best_epochs))
    return {"raw": raw, "fit_epochs": min(80, max(30, raw))}

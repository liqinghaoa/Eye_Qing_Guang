"""B0 successful-update learning-rate indexing and independent stopping state."""

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
        if index <= self.warmup:
            return self.start + (self.peak - self.start) * (index - 1) / (self.warmup - 1)
        fraction = (index - self.warmup) / (self.total - self.warmup)
        return self.minimum + 0.5 * (self.peak - self.minimum) * (1 + math.cos(math.pi * fraction))

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

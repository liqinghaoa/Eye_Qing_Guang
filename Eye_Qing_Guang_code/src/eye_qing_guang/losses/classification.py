"""BCE or focal BCE with optional training-set positive-class weighting."""

import torch
from torch import nn
from torch.nn import functional as F


class BinaryLoss(nn.Module):
    def __init__(self, mode: str, pos_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, labels):
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight, reduction="none",
        )
        if self.mode == "focal":
            probability = torch.sigmoid(logits)
            pt = probability * labels + (1 - probability) * (1 - labels)
            loss = (1 - pt).pow(self.gamma) * loss
        return loss.mean()


def build_loss(config: dict, pos_weight: float = 1.0):
    return BinaryLoss(config["loss"], pos_weight, float(config.get("focal_gamma", 2)))

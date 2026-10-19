"""Concatenate image features with a learned normalized-age embedding."""

import torch
from torch import nn


class AgeFusionClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, features: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.age_encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features + 16, 1))

    def forward(self, image, age=None):
        if age is None:
            raise ValueError("Age fusion requires normalized age inputs.")
        features = torch.cat((self.encoder(image), self.age_encoder(age)), dim=1)
        return self.classifier(features)

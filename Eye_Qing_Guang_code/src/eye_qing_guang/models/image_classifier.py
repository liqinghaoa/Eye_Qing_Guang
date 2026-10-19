"""Image encoder and single-logit classification head."""

from torch import nn
from torchvision import models


def build_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone in {"resnet18", "resnet50"}:
        weights_class = models.ResNet18_Weights if backbone == "resnet18" else models.ResNet50_Weights
        network = getattr(models, backbone)(weights=weights_class.DEFAULT if pretrained else None)
        features = network.fc.in_features
        network.fc = nn.Identity()
        return network, features
    if backbone == "efficientnet_b0":
        network = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
        )
        features = network.classifier[1].in_features
        network.classifier = nn.Identity()
        return network, features
    if backbone == "tiny_cnn" and not pretrained:
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        ), 16
    raise ValueError(f"Unsupported backbone/pretraining combination: {backbone}")


class ImageClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, features: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 1))

    def forward(self, image, age=None):
        return self.classifier(self.encoder(image))

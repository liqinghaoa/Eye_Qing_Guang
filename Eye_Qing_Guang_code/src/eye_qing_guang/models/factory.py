"""Construct models from explicit configuration."""

from eye_qing_guang.models.age_fusion import AgeFusionClassifier
from eye_qing_guang.models.image_classifier import ImageClassifier, build_encoder


def build_model(config: dict, initialize_pretrained: bool | None = None):
    pretrained = config["pretrained"] if initialize_pretrained is None else initialize_pretrained
    encoder, features = build_encoder(config["backbone"], pretrained)
    classifier = AgeFusionClassifier if config["use_age"] else ImageClassifier
    return classifier(encoder, features, float(config.get("dropout", 0)))

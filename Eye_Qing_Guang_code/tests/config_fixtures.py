"""Read-only historical config fixtures for tests of retired B0 recipes."""

from pathlib import Path

from eye_qing_guang.utils.experiment_io import load_config as _load_config


def load_config(path):
    path = Path(path)
    if not path.exists() and path.parent.name == "configs":
        fixture = Path(__file__).resolve().parent / "fixtures" / "legacy_configs" / path.name
        if fixture.is_file():
            path = fixture
    return _load_config(path)

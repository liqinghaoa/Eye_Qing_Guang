"""Portable project-relative paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def image_index(directory: Path) -> dict[str, Path]:
    """Index filenames case-insensitively, rejecting ambiguous matches."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            key = path.relative_to(directory).as_posix().casefold()
            if key in result:
                raise ValueError(f"Ambiguous case-insensitive image path: {key}")
            result[key] = path
    return result


def find_image(name: str, directory: Path, index: dict[str, Path]) -> Path:
    relative = Path(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError(f"Image_name must be a safe relative path: {name}")
    key = relative.as_posix().casefold()
    if key not in index:
        raise FileNotFoundError(directory / relative)
    return index[key]

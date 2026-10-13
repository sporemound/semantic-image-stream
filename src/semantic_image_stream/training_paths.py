"""Local training files live separately from saved books and disposable images."""
from __future__ import annotations

import os
from pathlib import Path


def default_training_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    return Path(base or Path.home() / ".local" / "share") / "SemanticImageStream" / "training"


def resolve_training_dir(settings) -> Path:
    value = getattr(settings, "training_dir", "")
    return Path(os.path.expandvars(value)).expanduser().resolve() if value else default_training_dir()


def training_file(settings, name: str) -> Path:
    """An explicit training folder never silently falls back to bundled files."""
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Training file must name a single file")
    local = resolve_training_dir(settings) / name
    if local.exists() or getattr(settings, "training_dir", ""):
        return local
    bundled = Path(__file__).resolve().parent / "data" / "training" / name
    return bundled if bundled.is_file() else local

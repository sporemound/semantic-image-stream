"""Narration profiles applied to a copy of the listener's current settings."""
from copy import deepcopy
from dataclasses import dataclass

from .configuration import Settings


@dataclass(frozen=True)
class AudiobookPreset:
    name: str
    description: str
    transcript_window_seconds: float
    semantic_interval: float
    search_interval: float
    slideshow_interval: float
    semantic_sensitivity: float
    relevance_threshold: float
    providers: tuple[str, ...]


AUDIOBOOK_PRESETS = {
    "fiction": AudiobookPreset(
        "Fiction", "Understands the full story locally; images change every 10 seconds when available.",
        90, 8, 15, 10, 0.25, 0.30, ("wikimedia", "openverse"),
    ),
    "history": AudiobookPreset(
        "History & Biography", "Uses full-session context and Commons/archival sources; 12-second slides.",
        120, 8, 15, 12, 0.30, 0.35, ("wikimedia", "archive"),
    ),
    "science": AudiobookPreset(
        "Science & Technical", "Uses full-session explanations with a 6-second context check and 10-second slides.",
        90, 6, 12, 10, 0.35, 0.30, ("wikimedia", "openverse"),
    ),
}


def apply_audiobook_preset(settings: Settings, key: str) -> Settings:
    """Preserve capture/model/cache choices until the listener explicitly changes them."""
    try:
        preset = AUDIOBOOK_PRESETS[key]
    except KeyError:
        raise ValueError("Choose a supported audiobook preset") from None
    updated = deepcopy(settings)
    for field in ("transcript_window_seconds", "semantic_interval", "search_interval", "slideshow_interval",
                  "semantic_sensitivity", "relevance_threshold"):
        setattr(updated, field, getattr(preset, field))
    updated.providers = list(preset.providers)
    updated.prefetch_count = 8
    if updated.ranking != "learned_context":
        updated.ranking = "context_clip"
    updated.semantic_provider = "ollama"
    return updated.validate()

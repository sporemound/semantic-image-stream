"""Validated TOML configuration, with paths expanded only at the boundary."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import math
import tomllib


def default_data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return str(Path(base or Path.home() / ".cache") / "SemanticImageStream")


@dataclass
class Settings:
    audio_mode: str = "loopback"
    audio_device: int | None = None
    audio_file: str = ""
    chunk_seconds: float = 6.0
    silence_seconds: float = 0.8
    silence_rms: float = 0.008
    audio_queue_size: int = 8
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    language: str = ""
    transcript_window_seconds: float = 45.0
    semantic_interval: float = 4.0
    search_interval: float = 12.0
    slideshow_interval: float = 9.0
    semantic_sensitivity: float = 0.45
    relevance_threshold: float = 0.30
    time_filter_enabled: bool = False
    time_filter_target: str = "depicted"
    time_filter_start: float = 1000.0
    time_filter_end: float = 2026.0
    time_filter_scale: str = "calendar"
    time_filter_include_undated: bool = False
    exclude_ai_generated: bool = True
    semantic_provider: str = "contextual"
    context_model: str = "qwen2.5:7b"
    book_context: str = ""
    retain_context: bool = True
    session_dir: str = ""
    training_dir: str = ""
    use_story_examples: bool = True
    context_cpu_threads: int = 6
    context_timeout: float = 180.0
    providers: list[str] = field(default_factory=lambda: ["wikimedia"])
    results_per_provider: int = 10
    prefetch_count: int = 6
    min_width: int = 480
    min_height: int = 300
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 3.0
    max_download_bytes: int = 15_000_000
    max_image_pixels: int = 40_000_000
    download_timeout: float = 12.0
    ranking: str = "metadata"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"
    transition: str = "crossfade"
    transition_seconds: float = 0.8
    attribution_visible: bool = True
    display_index: int = 0
    cache_dir: str = field(default_factory=default_data_dir)
    cache_ttl_hours: float = 24.0
    cache_max_mb: int = 512
    history_enabled: bool = True
    history_limit: int = 500

    def validate(self) -> "Settings":
        if not isinstance(self.cache_dir, str) or not self.cache_dir.strip():
            raise ValueError("cache_dir must be a nonempty path")
        self.cache_dir = str(Path(os.path.expandvars(self.cache_dir)).expanduser().resolve())
        enums = {"audio_mode": {"loopback", "microphone", "file"},
                 "whisper_device": {"auto", "cuda", "cpu"},
                 "ranking": {"metadata", "clip", "context_clip", "learned_context"},
                 "semantic_provider": {"contextual", "spacy", "ollama"},
                 "transition": {"crossfade", "dissolve", "zoom", "cut"},
                 "clip_device": {"cpu", "cuda"}}
        enums.update(time_filter_target={"depicted", "created"}, time_filter_scale={"calendar", "geological"})
        for key, choices in enums.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"{key} must be one of {sorted(choices)}")
        for key in ("chunk_seconds", "silence_seconds", "transcript_window_seconds",
                    "semantic_interval", "search_interval", "slideshow_interval",
                    "download_timeout", "cache_ttl_hours", "context_timeout"):
            if (not isinstance(getattr(self, key), (float, int))
                    or not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0):
                raise ValueError(f"{key} must be positive")
        for key in ("audio_queue_size", "results_per_provider", "prefetch_count", "min_width",
                    "min_height", "max_download_bytes", "max_image_pixels", "cache_max_mb", "history_limit", "context_cpu_threads"):
            if not isinstance(getattr(self, key), int) or isinstance(getattr(self, key), bool) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.chunk_seconds > 30 or self.audio_queue_size > 64:
            raise ValueError("Audio buffering must stay within 30-second chunks and 64 queued chunks")
        if self.results_per_provider > 50 or self.prefetch_count > 30:
            raise ValueError("Search batches exceed the supported bound")
        for key in ("semantic_sensitivity", "relevance_threshold", "silence_rms"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if not 0 < self.min_aspect_ratio <= self.max_aspect_ratio:
            raise ValueError("Invalid aspect-ratio range")
        if not 0 <= self.transition_seconds <= 5:
            raise ValueError("transition_seconds must be between 0 and 5")
        if (not isinstance(self.providers, list) or not self.providers
                or not all(isinstance(p, str) for p in self.providers)
                or set(self.providers) - {"wikimedia", "duckduckgo", "openverse", "archive"}):
            raise ValueError("Select at least one supported image provider")
        if not isinstance(self.display_index, int) or self.display_index < 0:
            raise ValueError("display_index must be a nonnegative integer")
        if not isinstance(self.whisper_model, str) or not self.whisper_model.strip():
            raise ValueError("whisper_model must be a nonempty model name or path")
        if (not isinstance(self.context_model, str) or not self.context_model.strip()
                or len(self.context_model) > 120 or any(c.isspace() for c in self.context_model)
                or any(token in self.context_model.lower() for token in (":cloud", "://", "\\", ".."))):
            raise ValueError("context_model must name a local installed model")
        if not isinstance(self.book_context, str) or len(self.book_context) > 2000:
            raise ValueError("book_context must contain at most 2000 characters")
        if not isinstance(self.retain_context, bool):
            raise ValueError("retain_context must be true or false")
        if (not isinstance(self.session_dir, str) or len(self.session_dir) > 4096
                or any(character in self.session_dir for character in "\x00\r\n")):
            raise ValueError("session_dir must name a local folder")
        if (not isinstance(self.training_dir, str) or len(self.training_dir) > 4096
                or any(character in self.training_dir for character in "\x00\r\n")):
            raise ValueError("training_dir must name a local folder")
        if not isinstance(self.use_story_examples, bool):
            raise ValueError("use_story_examples must be true or false")
        for key in ("time_filter_enabled", "time_filter_include_undated", "exclude_ai_generated"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be true or false")
        for key in ("time_filter_start", "time_filter_end"):
            value = getattr(self, key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                    or not 1950 - 14_000_000_000 <= value <= 100_000):
                raise ValueError(f"{key} must be a finite supported calendar or geological year")
        if self.time_filter_start > self.time_filter_end:
            raise ValueError("The time filter's start must be older than or equal to its end")
        if self.context_cpu_threads > 32:
            raise ValueError("context_cpu_threads must be at most 32")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**data).validate()

    def save(self, path: str | Path) -> None:
        import json
        lines = []
        for key, value in asdict(self).items():
            if value is not None:
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

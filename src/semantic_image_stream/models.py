from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioChunk:
    samples: Any  # mono float32 at 16 kHz
    start: float
    sample_rate: int = 16000


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str = ""


@dataclass(frozen=True)
class Topic:
    query: str
    context: str
    concepts: tuple[str, ...] = ()
    kind: str = "general"
    summary: str = ""
    scene: str = ""
    visual_subjects: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()
    transcript: tuple[str, ...] = ()
    context_used_segments: int = 0
    visual_update: str = "replace"
    image_guidance: str | None = None
    guidance_strength: float | None = None
    guidance_only: bool = False


@dataclass
class ImageCandidate:
    image_url: str
    source_url: str
    title: str
    provider: str
    creator: str = "Unknown"
    license: str = "Unknown — check source before reuse"
    license_url: str = ""
    description: str = ""
    width: int = 0
    height: int = 0
    local_path: str = ""
    phash: str = ""
    score: float = 0.0
    authenticity: str = "Unverified source imagery"
    ai_generated: bool | None = None
    temporal_evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageCandidate":
        from .temporal_filter import validated_evidence

        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "temporal_evidence" in fields:
            fields["temporal_evidence"] = validated_evidence(fields["temporal_evidence"])
        if "ai_generated" in fields and type(fields["ai_generated"]) is not bool:
            fields["ai_generated"] = None
        return cls(**fields)

    @property
    def path(self) -> Path:
        return Path(self.local_path)

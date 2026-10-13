from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageCandidate":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def path(self) -> Path:
        return Path(self.local_path)

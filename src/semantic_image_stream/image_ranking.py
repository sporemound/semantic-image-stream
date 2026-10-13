"""Local metadata baseline and optional genuine CLIP image/text embeddings."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import math
import threading
from urllib.parse import urlsplit

from .models import ImageCandidate, Topic
from .semantic_analysis import tokens


def is_duplicate(phash: str, previous: list[str], max_distance: int = 6) -> bool:
    if not phash:
        return False
    return any((int(phash, 16) ^ int(p, 16)).bit_count() <= max_distance for p in previous if p)


def source_weight(candidate: ImageCandidate) -> float:
    host = (urlsplit(candidate.source_url).hostname or "").casefold()
    if host.endswith((".gov", ".edu")) or candidate.provider == "wikimedia":
        return 1.0
    # Archive is an upload platform, so its host alone is not a factual
    # endorsement. Openverse licenses likewise do not establish accuracy.
    return {"archive": 0.25, "openverse": 0.35}.get(candidate.provider, 0.15)


class MetadataRanker:
    """Scores lie in 0..1, but are heuristic scores, not calibrated probability."""
    name = "metadata"

    def rank(self, topic: Topic, candidates: list[ImageCandidate], recent: list[str]) -> list[ImageCandidate]:
        query = tokens(topic.query)
        context = tokens(topic.context)
        ranked = []
        for c in candidates:
            if is_duplicate(c.phash, recent):
                continue
            text = tokens(c.title + " " + c.description)
            # Broad discovery tags can mention every scientific domain on an
            # unrelated stock photo. Require a subject in the title too for
            # sources without Commons' structured descriptive metadata.
            if c.provider != "wikimedia" and not (query & tokens(c.title)):
                continue
            overlap = len(query & text) / max(1, len(query))
            contextual = len(context & text) / max(1, min(len(context), 20))
            credible = source_weight(c)
            specific = topic.kind in {"science", "history"} and credible >= 1.0
            quality = min(1, max(0, math.log2(max(1, c.width * c.height / 100_000))) / 5)
            c.score = min(1.0, 0.72 * overlap + 0.06 * contextual + 0.08 * credible + 0.05 * specific + 0.09 * quality)
            # A credible/high-res unrelated picture must not pass on quality alone.
            if overlap > 0:
                ranked.append(c)
        return sorted(ranked, key=lambda c: c.score, reverse=True)


class ClipRanker(MetadataRanker):
    name = "clip"

    def __init__(self, settings, gpu_lock: threading.Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock or threading.Lock()
        self.model = None
        self.processor = None
        self.device = settings.clip_device

    def _load(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Local CLIP ranking requires installing semantic-image-stream[clip].") from exc
        self.device = self.settings.clip_device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        # No remote custom code is allowed. Model weights are fetched once.
        model_cache = str(Path(self.settings.cache_dir) / "models" / "clip")
        self.processor = CLIPProcessor.from_pretrained(self.settings.clip_model, cache_dir=model_cache,
                                                      trust_remote_code=False)
        self.model = CLIPModel.from_pretrained(self.settings.clip_model, cache_dir=model_cache,
                                              trust_remote_code=False).to(self.device).eval()

    def rank(self, topic, candidates, recent):
        import torch
        from PIL import Image
        candidates = [c for c in candidates if not is_duplicate(c.phash, recent)]
        if not candidates:
            return []
        # CPU model loading/inference must never pause CUDA speech recognition.
        # In particular, a stopped CPU model download can outlive its session.
        access = self.gpu_lock if self.settings.clip_device == "cuda" else nullcontext()
        with access:
            if self.model is None:
                self._load()
            similarities = []
            # Small batches keep RAM/VRAM bounded. CPU is the default for CLIP.
            for offset in range(0, len(candidates), 4):
                batch = candidates[offset:offset + 4]
                images = []
                for c in batch:
                    with Image.open(Path(c.local_path)) as im:
                        images.append(im.convert("RGB"))
                inputs = self.processor(text=[topic.query], images=images, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    # logits include a learned scale; divide it out to get cosine.
                    out = self.model(**inputs)
                    scores = out.logits_per_image[:, 0] / self.model.logit_scale.exp()
                similarities.extend(scores.detach().cpu().tolist())
                for im in images:
                    im.close()
        for c, similarity in zip(candidates, similarities):
            # Threshold keeps its meaning as approximate cosine relevance.
            quality = min(1, (c.width * c.height) / 2_000_000)
            specificity = topic.kind in {"science", "history"} and source_weight(c) >= 1.0
            c.score = max(0, min(1, 0.9 * similarity + 0.04 * quality + 0.06 * specificity))
        return sorted(candidates, key=lambda c: c.score, reverse=True)


def build_ranker(settings, gpu_lock=None):
    return ClipRanker(settings, gpu_lock) if settings.ranking == "clip" else MetadataRanker()

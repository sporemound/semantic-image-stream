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
        context = tokens(topic.scene + " " + topic.summary) if topic.scene or topic.summary else tokens(topic.context)
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
            if topic.scene or topic.summary:
                # A one-word object match alone cannot establish narrative fit.
                c.score = min(1.0, 0.30 * overlap + 0.48 * contextual + 0.08 * credible + 0.05 * specific + 0.09 * quality)
            else:
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
        # Resolve an already complete snapshot before consulting the network.
        # A newer partial snapshot must not hide the installed usable weights.
        snapshot_root = Path(model_cache) / ("models--" + self.settings.clip_model.replace("/", "--")) / "snapshots"
        snapshots = sorted(snapshot_root.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
        complete = next((path for path in snapshots if all((path / name).is_file() and (path / name).stat().st_size
                         for name in ("config.json", "tokenizer.json", "preprocessor_config.json"))
                         and any((path / name).is_file() and (path / name).stat().st_size
                                 for name in ("model.safetensors", "pytorch_model.bin"))), None)
        source = str(complete) if complete else self.settings.clip_model
        options = dict(cache_dir=model_cache, trust_remote_code=False, local_files_only=complete is not None)
        self.processor = CLIPProcessor.from_pretrained(source, **options)
        self.model = CLIPModel.from_pretrained(source, **options).to(self.device).eval()

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


class ContextClipRanker(ClipRanker):
    """Rank image pixels against the scene, story summary, and every recorded token.

    CLIP accepts only a short text window. Full-session text is therefore split
    at token boundaries and averaged by token count, rather than truncated.
    Incremental sufficient statistics keep embedding storage bounded; raw text
    remains in the session so a reset or edit can rebuild it correctly.
    """
    name = "context_clip"

    def __init__(self, settings, gpu_lock=None):
        super().__init__(settings, gpu_lock)
        self._history = ()
        self._history_sum = None
        self._history_weight = 0
        self._rank_lock = threading.Lock()

    def _windows(self, text):
        tokenizer = self.processor.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        limit = max(1, min(75, int(tokenizer.model_max_length) - 2))
        return [(ids[index:index + limit], len(ids[index:index + limit]))
                for index in range(0, len(ids), limit)]

    def _text_sum(self, texts):
        import torch
        windows = [window for text in texts for window in self._windows(text)]
        total, weight = None, 0
        for offset in range(0, len(windows), 16):
            batch = windows[offset:offset + 16]
            # Build from exact token IDs: decoding and re-tokenizing can expand
            # windows past CLIP's limit and silently discard trailing content.
            prepared = [self.processor.tokenizer.prepare_for_model(ids, add_special_tokens=True,
                        return_attention_mask=True, truncation=False) for ids, _ in batch]
            inputs = self.processor.tokenizer.pad(prepared, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                vectors = self.model.get_text_features(**inputs).float()
                vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = torch.tensor([length for _, length in batch], device=self.device, dtype=vectors.dtype)
            subtotal = (vectors * weights[:, None]).sum(dim=0)
            total = subtotal if total is None else total + subtotal
            weight += sum(length for _, length in batch)
        return total, weight

    def _text_vector(self, texts):
        total, _ = self._text_sum(texts)
        return None if total is None else total / total.norm().clamp_min(1e-8)

    def _history_vector(self, transcript):
        transcript = tuple(transcript)
        if transcript[:len(self._history)] != self._history:
            self._history, self._history_sum, self._history_weight = (), None, 0
        addition, weight = self._text_sum(transcript[len(self._history):])
        if addition is not None:
            self._history_sum = addition if self._history_sum is None else self._history_sum + addition
            self._history_weight += weight
        self._history = transcript
        return (None if self._history_sum is None else
                self._history_sum / self._history_sum.norm().clamp_min(1e-8))

    def rank(self, topic, candidates, recent):
        import torch
        from PIL import Image
        candidates = [candidate for candidate in candidates if not is_duplicate(candidate.phash, recent)]
        if not candidates:
            return []
        access = self.gpu_lock if self.settings.clip_device == "cuda" else nullcontext()
        with self._rank_lock, access:
            if self.model is None:
                self._load()
            history = self._history_vector(topic.transcript or (topic.context,))
            background = getattr(self.settings, "book_context", "")
            summary = self._text_vector((background, topic.summary)) if background or topic.summary else history
            scene = self._text_vector((topic.scene, *topic.visual_subjects)) if topic.scene else summary
            contexts = [(vector, weight) for vector, weight in ((scene, .50), (summary, .30), (history, .20))
                        if vector is not None]
            if not contexts:
                return []
            denominator = sum(weight for _, weight in contexts)
            for offset in range(0, len(candidates), 4):
                batch = candidates[offset:offset + 4]
                images = []
                try:
                    for candidate in batch:
                        with Image.open(Path(candidate.local_path)) as original:
                            images.append(original.convert("RGB"))
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        vectors = self.model.get_image_features(**inputs).float()
                        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        scores = sum((vectors @ vector) * weight for vector, weight in contexts) / denominator
                    for candidate, similarity in zip(batch, scores.detach().cpu().tolist()):
                        # Rescale the narrow CLIP cosine range for the 0..1 UI.
                        # This is a relevance score, never a probability. No
                        # title, resolution, or source bonus can rescue pixels
                        # which do not match the recorded scene and story.
                        candidate.score = max(0.0, min(1.0, (similarity - .14) / .30))
                finally:
                    for image in images:
                        image.close()
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def build_ranker(settings, gpu_lock=None):
    if settings.ranking == "context_clip":
        return ContextClipRanker(settings, gpu_lock)
    return ClipRanker(settings, gpu_lock) if settings.ranking == "clip" else MetadataRanker()

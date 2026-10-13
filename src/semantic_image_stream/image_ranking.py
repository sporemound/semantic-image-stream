"""Local metadata baseline and optional genuine CLIP image/text embeddings."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import math
import re
import threading
from urllib.parse import urlsplit

from .models import ImageCandidate, Topic
from .semantic_analysis import tokens


def _cached_encoder_snapshot(settings):
    model_cache = Path(settings.cache_dir) / "models" / "clip"
    root = model_cache / ("models--" + settings.clip_model.replace("/", "--")) / "snapshots"
    snapshots = sorted(root.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return next((path for path in snapshots if all((path / name).is_file() and (path / name).stat().st_size
                 for name in ("config.json", "tokenizer.json", "preprocessor_config.json"))
                 and any((path / name).is_file() and (path / name).stat().st_size
                         for name in ("model.safetensors", "pytorch_model.bin"))), None)


def _fingerprint_snapshot(snapshot):
    if snapshot is None:
        raise RuntimeError("The frozen CLIP encoder is not cached locally. Install its weights before local relevance training.")
    weight = "model.safetensors" if (snapshot / "model.safetensors").is_file() else "pytorch_model.bin"
    files = (weight, "config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.json", "merges.txt")
    hashes = {}
    try:
        for name in files:
            path = snapshot / name
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1_048_576), b""):
                    digest.update(chunk)
            hashes[name] = digest.hexdigest()
    except OSError:
        raise RuntimeError("Cached CLIP encoder files could not be checked locally.") from None
    data = {"files": hashes, "use_fast": False}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def encoder_fingerprint(settings):
    """Identify exact cached weights/preprocessing using local reads only."""
    return _fingerprint_snapshot(_cached_encoder_snapshot(settings))


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

    def _load(self, local_only=False, snapshot=None):
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
        complete = snapshot if snapshot is not None else _cached_encoder_snapshot(self.settings)
        if local_only and complete is None:
            raise RuntimeError("The frozen CLIP encoder is not cached locally. Install its weights before local relevance training.")
        source = str(complete) if complete else self.settings.clip_model
        options = dict(cache_dir=model_cache, trust_remote_code=False, local_files_only=complete is not None)
        self.processor = CLIPProcessor.from_pretrained(source, use_fast=False, **options)
        model_options = {"use_safetensors": (complete / "model.safetensors").is_file()} if complete else {}
        self.model = CLIPModel.from_pretrained(source, **model_options, **options).to(self.device).eval()

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

    def _score_vectors(self, vectors, contexts):
        denominator = sum(weight for _, weight in contexts)
        scores = sum((vectors @ vector) * weight for vector, weight in contexts) / denominator
        # Rescale the narrow cosine range for the existing UI score. It is
        # never a probability; no source/title/resolution bonus changes it.
        return [max(0.0, min(1.0, (similarity - .14) / .30))
                for similarity in scores.detach().cpu().tolist()]

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
                        scores = self._score_vectors(vectors, contexts)
                    for candidate, score in zip(batch, scores):
                        candidate.score = score
                finally:
                    for image in images:
                        image.close()
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


class LearnedContextRanker(ContextClipRanker):
    """Checked local pair head, sharing complete story/scene embeddings."""
    name = "learned_context"

    def __init__(self, settings, gpu_lock=None):
        super().__init__(settings, gpu_lock)
        self._adapter = None

    def _load_adapter(self):
        from .learned_relevance import load_adapter
        from .training_paths import training_file
        self._adapter = load_adapter(training_file(self.settings, "relevance-model.json"), self.settings.clip_model)

    def _load(self):
        from .learned_relevance import AdapterError

        self._load_adapter()
        expected = self._adapter.artifact.get("metadata", {}).get("encoder_fingerprint")
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise AdapterError("The trained relevance model has no exact encoder fingerprint. Train it again with the cached encoder.")
        snapshot = _cached_encoder_snapshot(self.settings)
        if expected != _fingerprint_snapshot(snapshot):
            raise AdapterError("The trained relevance model uses different CLIP weights or preprocessing. Train it again with this cached encoder.")
        super()._load(local_only=True, snapshot=snapshot)

    def _score_vectors(self, vectors, contexts):
        from .learned_relevance import pair_features

        if self._adapter is None:
            self._load_adapter()
        images = vectors.detach().cpu().numpy()
        denominator = sum(weight for _, weight in contexts)
        if self._adapter.artifact["report"]["selection"]["chosen_head"] == "cosine_only":
            # Averaging nonlinear per-context probabilities can change the
            # original scene/story ordering. Calibrate its weighted raw cosine
            # once so the selected conservative head keeps that ordering.
            raw = sum(pair_features(images, vector.detach().cpu().numpy())[:, -1] * weight
                      for vector, weight in contexts) / denominator
            return self._adapter.predict_cosines(raw).tolist()
        scores = sum(self._adapter.predict(images, vector.detach().cpu().numpy()) * weight
                     for vector, weight in contexts) / denominator
        return scores.tolist()


def embed_pairs(settings, image_paths, texts, gpu_lock=None):
    """Extract training vectors from the already cached encoder, without downloads."""
    import numpy as np
    import torch
    from PIL import Image
    from .learned_relevance import AdapterError, EMBEDDING_DIM, MAX_SAMPLES

    paths, captions = list(image_paths), list(texts)
    if (not paths or len(paths) != len(captions) or len(paths) > MAX_SAMPLES
            or any(not isinstance(text, str) or not text.strip() for text in captions)):
        raise AdapterError("Provide a bounded set of paired training images and nonempty captions.")
    ranker = ContextClipRanker(settings, gpu_lock)
    image_vectors, text_vectors = [], []
    access = ranker.gpu_lock if settings.clip_device == "cuda" else nullcontext()
    with access:
        ranker._load(local_only=True)
        for offset in range(0, len(paths), 4):
            images = []
            try:
                for path in paths[offset:offset + 4]:
                    with Image.open(path) as source:
                        images.append(source.convert("RGB"))
                inputs = ranker.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(ranker.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    vectors = ranker.model.get_image_features(**inputs).float()
                    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                image_vectors.extend(vectors.detach().cpu().numpy())
            finally:
                for image in images:
                    image.close()
        for text in captions:
            vector = ranker._text_vector((text,))
            if vector is None:
                raise AdapterError("A training caption produced no CLIP text features.")
            text_vectors.append(vector.detach().cpu().numpy())
    result = np.asarray(image_vectors, dtype=np.float32), np.asarray(text_vectors, dtype=np.float32)
    if any(matrix.shape != (len(paths), EMBEDDING_DIM) for matrix in result):
        raise AdapterError("The selected cached encoder has incompatible training dimensions.")
    return result


def build_ranker(settings, gpu_lock=None):
    if settings.ranking == "learned_context":
        return LearnedContextRanker(settings, gpu_lock)
    if settings.ranking == "context_clip":
        return ContextClipRanker(settings, gpu_lock)
    return ClipRanker(settings, gpu_lock) if settings.ranking == "clip" else MetadataRanker()

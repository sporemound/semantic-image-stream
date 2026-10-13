"""Local metadata baseline and optional genuine CLIP image/text embeddings."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import logging
import math
import re
import threading
from urllib.parse import urlsplit

from .models import ImageCandidate, Topic
from .semantic_analysis import tokens

CLIP_INITIALIZATION_LOCK = threading.Lock()
log = logging.getLogger(__name__)
LEXICAL_FEEDBACK_FINGERPRINT = hashlib.sha256(b"semantic-image-stream:feedback-lexical-v1:512").hexdigest()


def _book_components(topic, settings):
    from .book_guidance import guidance_for
    text, strength = guidance_for(topic, settings)
    book = .35 * strength if text else 0.0
    # Scene stays the largest component even at maximum book guidance; every
    # recorded token retains at least .15625 of the complete context score.
    return text, book, (.50 - .375 * book, .30 - .50 * book, .20 - .125 * book)


def _context_fingerprint(encoder, method, topic, settings):
    from .book_guidance import GUIDANCE_VERSION
    _, book, weights = _book_components(topic, settings)
    if not book:
        return encoder  # Existing unguided votes keep their compatible vectors.
    identity = {"encoder": encoder, "method": method, "guidance": GUIDANCE_VERSION,
                "blend": "separate-book-v1", "book": book, "scene_summary_history": weights}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _lexical_feedback_vector(texts):
    """Deterministic local features for the metadata baseline, separate from CLIP."""
    import numpy as np
    vector = np.zeros(512, dtype=np.float64)
    for text in texts:
        for word in tokens(text):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1 if digest[4] & 1 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


class ClipLoadError(RuntimeError):
    """Controlled, path/transcript-free diagnostics for local CLIP loading."""


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
    feedback = None
    settings = None  # Compatible subclasses may provide their own constructor.

    def __init__(self, settings=None):
        self.settings = settings

    def _has_feedback(self):
        if self.feedback is None:
            return False
        count = getattr(self.feedback, "count", None)
        if count is None:
            return True
        try:
            return count() > 0
        except Exception:
            return False

    def _lexical_context_vector(self, topic):
        import numpy as np
        scene = _lexical_feedback_vector((topic.scene, *topic.visual_subjects, topic.query))
        summary = _lexical_feedback_vector((topic.summary,))
        history = _lexical_feedback_vector(topic.transcript or (topic.context,))
        guide, book, weights = _book_components(topic, self.settings)
        vector = scene * weights[0] + summary * weights[1] + history * weights[2]
        if book:
            vector += _lexical_feedback_vector((guide,)) * book
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-8 else vector

    def feedback_vectors(self, topic, candidate):
        return {"image_vector": _lexical_feedback_vector((candidate.title, candidate.description)).tolist(),
                "context_vector": self._lexical_context_vector(topic).tolist(),
                "encoder_fingerprint": _context_fingerprint(LEXICAL_FEEDBACK_FINGERPRINT, self.name, topic, self.settings)}

    def _personalize(self, scores, images, context, fingerprint):
        if self.feedback is None:
            return scores
        try:
            return self.feedback.adjust_scores(scores, images, context, fingerprint)
        except Exception as error:
            # An optional private preference file must never stop the slideshow.
            if not getattr(self, "_feedback_warning", False):
                log.warning("Local image feedback unavailable (%s)", type(error).__name__)
                self._feedback_warning = True
            return scores

    def rank(self, topic: Topic, candidates: list[ImageCandidate], recent: list[str]) -> list[ImageCandidate]:
        query = tokens(topic.query)
        context = tokens(topic.scene + " " + topic.summary) if topic.scene or topic.summary else tokens(topic.context)
        guide, book, _ = _book_components(topic, self.settings)
        guide_vector = _lexical_feedback_vector((guide,)) if book else None
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
                if book:
                    fit = max(0.0, float(_lexical_feedback_vector((c.title, c.description)) @ guide_vector))
                    c.score = (1 - book) * c.score + book * fit
                ranked.append(c)
        if ranked and self._has_feedback():
            vectors = [_lexical_feedback_vector((candidate.title, candidate.description)) for candidate in ranked]
            scores = self._personalize([candidate.score for candidate in ranked], vectors,
                                      self._lexical_context_vector(topic),
                                      _context_fingerprint(LEXICAL_FEEDBACK_FINGERPRINT, self.name, topic, self.settings))
            for candidate, score in zip(ranked, scores):
                candidate.score = score
        return sorted(ranked, key=lambda c: c.score, reverse=True)


class ClipRanker(MetadataRanker):
    name = "clip"

    def __init__(self, settings, gpu_lock: threading.Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock or threading.Lock()
        self.model = None
        self.processor = None
        self.device = settings.clip_device
        self._rank_lock = threading.Lock()
        self._encoder_snapshot = None
        self._feedback_encoder_fingerprint = None
        self._manual_text = None
        self._manual_embedding = None

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
            # Exact IDs preserve all supplied tokens without re-tokenization.
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

    def _manual_vector(self, text):
        # One latest full-brief vector is enough; advancing images must not
        # repeatedly encode a stable manual description under model load.
        if text != self._manual_text:
            self._manual_embedding = self._text_vector((text,))
            self._manual_text = text
        return self._manual_embedding

    def _feedback_fingerprint(self):
        if self._feedback_encoder_fingerprint is None:
            self._feedback_encoder_fingerprint = _fingerprint_snapshot(self._encoder_snapshot)
        return self._feedback_encoder_fingerprint

    def _personalize_pixels(self, scores, vectors, context, topic=None):
        if not self._has_feedback():
            return scores
        try:
            fingerprint = self._feedback_fingerprint()
        except RuntimeError:
            return scores
        if topic is not None:
            fingerprint = _context_fingerprint(fingerprint, self.name, topic, self.settings)
        return self._personalize(scores, vectors.detach().cpu().numpy(),
                                 context.detach().cpu().numpy(), fingerprint)

    def feedback_vectors(self, topic, candidate):
        import torch
        from PIL import Image
        access = self.gpu_lock if self.settings.clip_device == "cuda" else nullcontext()
        with self._rank_lock, access:
            if self.model is None:
                self._load(local_only=True)
            with Image.open(candidate.path) as source:
                with source.convert("RGB") as raster:
                    inputs = self.processor(text=[topic.query], images=[raster], return_tensors="pt",
                                            padding=True, truncation=True)
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        output = self.model(**inputs)
            guide, book, _ = _book_components(topic, self.settings)
            context = output.text_embeds[0]
            if book:
                manual = self._manual_vector(guide)
                if manual is not None:
                    context = context * (1 - book) + manual * book
                    context = context / context.norm().clamp_min(1e-8)
            return {"image_vector": output.image_embeds[0].detach().cpu().tolist(),
                    "context_vector": context.detach().cpu().tolist(),
                    "encoder_fingerprint": _context_fingerprint(self._feedback_fingerprint(), self.name, topic, self.settings)}

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
        model_options = {"use_safetensors": (complete / "model.safetensors").is_file()} if complete else {}
        # Transformers' empty-weight initialization temporarily changes global
        # torch parameter constructors. Different stopped/current CPU workers
        # must never overlap those constructors or leave loaded weights on meta.
        with CLIP_INITIALIZATION_LOCK:
            if self.model is not None:
                return
            self.processor = None
            try:
                processor = CLIPProcessor.from_pretrained(source, use_fast=False, **options)
                model, loading = CLIPModel.from_pretrained(source, output_loading_info=True,
                                                           **model_options, **options)
                if not isinstance(loading, dict):
                    raise ClipLoadError("CLIP could not verify its cached weights. Restart monitoring to retry safely.")
                # These deterministic position indices are nonpersistent buffers,
                # not trained weights, in the official CLIP implementation.
                generated = {"text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"}
                missing = [key for key in loading.get("missing_keys", ()) if key not in generated]
                if missing or loading.get("mismatched_keys") or loading.get("error_msgs"):
                    raise ClipLoadError("CLIP cached weights are incomplete or incompatible. A complete cached model is required.")
                if any(getattr(value, "is_meta", False) or getattr(getattr(value, "device", None), "type", "") == "meta"
                       for value in (*model.parameters(), *model.buffers())):
                    raise ClipLoadError("CLIP weights were not fully materialized. Restart the app to retry safely.")
                model = model.to(self.device).eval()
            except NotImplementedError:
                raise ClipLoadError("CLIP could not finish local model initialization. Restart the app to retry safely.") from None
            # Publish only a fully loaded model. Failed attempts leave model=None
            # so the next attempt can safely reload the original cached weights.
            self.processor, self.model = processor, model
            self._encoder_snapshot = complete or _cached_encoder_snapshot(self.settings)

    def rank(self, topic, candidates, recent):
        import torch
        from PIL import Image
        candidates = [c for c in candidates if not is_duplicate(c.phash, recent)]
        if not candidates:
            return []
        # CPU model loading/inference must never pause CUDA speech recognition.
        # In particular, a stopped CPU model download can outlive its session.
        access = self.gpu_lock if self.settings.clip_device == "cuda" else nullcontext()
        with self._rank_lock, access:
            if self.model is None:
                self._load()
            similarities = []
            pixel_vectors, text_vector = [], None
            personalizing = self._has_feedback()
            guide, book, _ = _book_components(topic, self.settings)
            manual = self._manual_vector(guide) if book else None
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
                    if manual is not None:
                        scores = scores * (1 - book) + (out.image_embeds @ manual) * book
                similarities.extend(scores.detach().cpu().tolist())
                if personalizing:
                    pixel_vectors.extend(out.image_embeds.detach().cpu().tolist())
                    context = out.text_embeds[0]
                    if manual is not None:
                        context = context * (1 - book) + manual * book
                        context = context / context.norm().clamp_min(1e-8)
                    text_vector = context.detach().cpu().tolist()
                for im in images:
                    im.close()
        for c, similarity in zip(candidates, similarities):
            # Threshold keeps its meaning as approximate cosine relevance.
            quality = min(1, (c.width * c.height) / 2_000_000)
            specificity = topic.kind in {"science", "history"} and source_weight(c) >= 1.0
            c.score = max(0, min(1, 0.9 * similarity + 0.04 * quality + 0.06 * specificity))
        if pixel_vectors and text_vector is not None:
            try:
                personalized = self._personalize([c.score for c in candidates], pixel_vectors,
                                                  text_vector, _context_fingerprint(self._feedback_fingerprint(), self.name,
                                                                                    topic, self.settings))
            except RuntimeError:
                personalized = [c.score for c in candidates]
            for candidate, score in zip(candidates, personalized):
                candidate.score = score
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

    def _context_vectors(self, topic):
        history = self._history_vector(topic.transcript or (topic.context,))
        guide, book, weights = _book_components(topic, self.settings)
        summary = self._text_vector((topic.summary,)) if topic.summary else history
        scene = self._text_vector((topic.scene, *topic.visual_subjects)) if topic.scene else summary
        contexts = [(vector, weight) for vector, weight in ((scene, weights[0]), (summary, weights[1]), (history, weights[2]))
                    if vector is not None]
        if book:
            manual = self._manual_vector(guide)
            if manual is not None:
                contexts.append((manual, book))
        return contexts

    @staticmethod
    def _combined_context(contexts):
        vector = sum(value * weight for value, weight in contexts)
        return vector / vector.norm().clamp_min(1e-8)

    def feedback_vectors(self, topic, candidate):
        import torch
        from PIL import Image
        access = self.gpu_lock if self.settings.clip_device == "cuda" else nullcontext()
        with self._rank_lock, access:
            if self.model is None:
                if _cached_encoder_snapshot(self.settings) is None:
                    raise RuntimeError("Feedback learning needs the locally cached image model.")
                self._load()
            contexts = self._context_vectors(topic)
            if not contexts:
                raise RuntimeError("Feedback learning needs an established spoken context.")
            with Image.open(candidate.path) as source:
                with source.convert("RGB") as raster:
                    inputs = self.processor(images=[raster], return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        image = self.model.get_image_features(**inputs).float()[0]
                        image = image / image.norm().clamp_min(1e-8)
            return {"image_vector": image.detach().cpu().tolist(),
                    "context_vector": self._combined_context(contexts).detach().cpu().tolist(),
                    "encoder_fingerprint": _context_fingerprint(self._feedback_fingerprint(), self.name, topic, self.settings)}

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
            contexts = self._context_vectors(topic)
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
                        scores = self._personalize_pixels(scores, vectors, self._combined_context(contexts), topic)
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
    return ClipRanker(settings, gpu_lock) if settings.ranking == "clip" else MetadataRanker(settings)

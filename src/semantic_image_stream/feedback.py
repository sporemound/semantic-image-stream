"""Private, bounded image-fit feedback for frozen local relevance encoders.

Labels and feature vectors stay beside saved stories, independently of the image
cache. This is a similarity-weighted preference memory, not model fine-tuning.
Both the context and image must resemble a labeled pair before its vote matters.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time

import numpy as np

from .session_store import default_session_dir

VERSION = 1
MAX_LABELS = 1000
MAX_VECTOR_DIM = 2048
MAX_FILE_BYTES = 64 * 1024 * 1024
SIMILARITY_MINIMUM = 0.75
MAX_CORRECTION = 0.12
REGULARIZATION = 2.0
REPLACE_RETRY_DELAYS = (0.01, 0.025, 0.05)
_DIGEST = re.compile(r"[a-f0-9]{64}")
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class FeedbackError(RuntimeError):
    """Controlled diagnostics never contain private context, URLs or vectors."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _context_key(topic):
    # The vote evaluates the exact immutable context that produced this image.
    # Only its digest is stored; the raw recording never enters the feedback file.
    fields = ("query", "scene", "summary", "kind", "queries", "visual_subjects", "concepts", "context")
    return _digest({field: getattr(topic, field, "") for field in fields})


def _image_key(candidate):
    # URL plus decoded-image fingerprint distinguishes a replacement at the same
    # URL without persisting either value or relying on a temporary cache path.
    return _digest([candidate.image_url, candidate.phash.casefold()])


def _vector(value):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf" or array.ndim != 1 or not 2 <= array.size <= MAX_VECTOR_DIM:
            return None
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            return None
        norm = float(np.linalg.norm(array))
        if not math.isfinite(norm) or norm <= 1e-12:
            return None
        return array / norm
    except (TypeError, ValueError, OverflowError):
        return None


def _features(vectors):
    if not isinstance(vectors, dict) or set(vectors) != {"image_vector", "context_vector", "encoder_fingerprint"}:
        return None
    fingerprint = vectors["encoder_fingerprint"]
    if not isinstance(fingerprint, str) or not _DIGEST.fullmatch(fingerprint):
        return None
    image, context = _vector(vectors["image_vector"]), _vector(vectors["context_vector"])
    if image is None or context is None or image.size != context.size:
        return None
    return {"encoder_fingerprint": fingerprint,
            "image_vector": np.round(image, 8).tolist(), "context_vector": np.round(context, 8).tolist()}


def _replace_file(source, destination):
    # Windows scanners/readers can briefly deny replacement of a just-written
    # file. Retry that specific native failure, never a schema or general IO
    # error, and leave the existing destination intact if it remains locked.
    for attempt in range(len(REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if (getattr(error, "winerror", None) not in {5, 32, 33}
                    or attempt == len(REPLACE_RETRY_DELAYS)):
                raise
            time.sleep(REPLACE_RETRY_DELAYS[attempt])


class FeedbackStore:
    def __init__(self, settings):
        root = getattr(settings, "session_dir", "") or default_session_dir()
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "feedback.json"
        with _PATH_LOCKS_GUARD:
            self._lock = _PATH_LOCKS.setdefault(str(self.path), threading.RLock())
        self._records = {}
        self._stamp = None
        with self._lock:
            self._refresh()

    def _refresh(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._records, self._stamp = {}, None
            return
        except OSError:
            raise FeedbackError("Saved image feedback could not be read.") from None
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._stamp:
            return
        if stat.st_size > MAX_FILE_BYTES:
            raise FeedbackError("Saved image feedback exceeds the local storage limit.")
        try:
            with self.path.open("rb") as saved:
                # A small vote file must not reserve the entire 64MiB ceiling
                # while the native speech/image models occupy memory.
                raw = saved.read(min(stat.st_size + 1, MAX_FILE_BYTES + 1))
            if len(raw) > stat.st_size or len(raw) > MAX_FILE_BYTES:
                raise ValueError("limit")
            data = json.loads(raw)
            if (not isinstance(data, dict) or set(data) != {"version", "records"}
                    or type(data["version"]) is not int or data["version"] != VERSION
                    or not isinstance(data["records"], list) or len(data["records"]) > MAX_LABELS):
                raise ValueError("schema")
            records = {}
            for row in data["records"]:
                if (not isinstance(row, dict)
                        or set(row) - {"context_key", "image_key", "vote", "updated", "features"}
                        or any(not isinstance(row.get(field), str) or not _DIGEST.fullmatch(row[field])
                               for field in ("context_key", "image_key"))
                        or type(row.get("vote")) is not bool or type(row.get("updated")) is not int
                        or not 0 <= row["updated"] <= 10**22):
                    raise ValueError("label")
                clean = {field: row[field] for field in ("context_key", "image_key", "vote", "updated")}
                features = _features(row.get("features"))
                if features:
                    clean["features"] = features
                key = (clean["context_key"], clean["image_key"])
                if key not in records or clean["updated"] >= records[key]["updated"]:
                    records[key] = clean
        except (OSError, ValueError, TypeError, KeyError, RecursionError, MemoryError):
            raise FeedbackError("Saved image feedback is invalid; its existing file was preserved.") from None
        self._records, self._stamp = records, stamp

    def _write(self, records):
        temporary = None
        try:
            encoded = json.dumps({"version": VERSION, "records": list(records.values())},
                                 ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(encoded) > MAX_FILE_BYTES:
                raise ValueError("limit")
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".feedback-", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            _replace_file(temporary, self.path)
            temporary = None
            stat = self.path.stat()
            self._records = records
            self._stamp = (stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, TypeError):
            raise FeedbackError("Image feedback could not be saved locally.") from None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def count(self):
        with self._lock:
            self._refresh()
            return len(self._records)

    def get_vote(self, topic, candidate) -> bool | None:
        key = (_context_key(topic), _image_key(candidate))
        with self._lock:
            self._refresh()
            row = self._records.get(key)
            return row["vote"] if row else None

    def record(self, topic, candidate, positive, vectors=None):
        if type(positive) is not bool:
            raise FeedbackError("Choose a good-fit or poor-fit image vote.")
        key = (_context_key(topic), _image_key(candidate))
        features = _features(vectors)
        with self._lock:
            self._refresh()
            row = dict(self._records.get(key, {}))
            row.update(context_key=key[0], image_key=key[1], vote=positive, updated=time.time_ns())
            if features:
                row["features"] = features
            records = dict(self._records)
            records[key] = row
            if len(records) > MAX_LABELS:
                newest = sorted(records.values(), key=lambda item: item["updated"], reverse=True)[:MAX_LABELS]
                records = {(item["context_key"], item["image_key"]): item for item in newest}
            self._write(records)
            return {"context_key": key[0], "vote": positive, "count": len(records),
                    "learned": bool(row.get("features"))}

    def adjust_scores(self, scores, image_vectors, context_vector, encoder_fingerprint) -> list[float]:
        try:
            baseline = [float(score) for score in scores]
        except (TypeError, ValueError, OverflowError):
            raise FeedbackError("Image feedback score data is invalid.") from None
        if any(not math.isfinite(score) for score in baseline):
            raise FeedbackError("Image feedback score data is invalid.")
        if not isinstance(encoder_fingerprint, str) or not _DIGEST.fullmatch(encoder_fingerprint):
            return baseline
        context = _vector(context_vector)
        try:
            images = list(image_vectors)
        except TypeError:
            return baseline
        if context is None or len(images) != len(baseline):
            return baseline
        with self._lock:
            self._refresh()
            labeled = [row for row in self._records.values()
                       if row.get("features", {}).get("encoder_fingerprint") == encoder_fingerprint]
        # Normalize each stored pair once, then compare only context-compatible
        # images. Cost stays bounded when a saved collection reaches 1000 votes.
        matched_images, matched_labels, context_weights = [], [], []
        for row in labeled:
            features = row["features"]
            previous_context, previous_image = _vector(features["context_vector"]), _vector(features["image_vector"])
            if previous_context is None or previous_image is None or previous_context.size != context.size:
                continue
            similarity = float(context @ previous_context)
            if similarity <= SIMILARITY_MINIMUM:
                continue
            matched_images.append(previous_image)
            matched_labels.append(1.0 if row["vote"] else -1.0)
            context_weights.append(min(1.0, (similarity - SIMILARITY_MINIMUM) / (1 - SIMILARITY_MINIMUM)))
        if not matched_images:
            return baseline
        previous_images = np.stack(matched_images)
        labels = np.asarray(matched_labels)
        context_weights = np.asarray(context_weights)
        corrections = []
        for image in images:
            image = _vector(image)
            if image is None or image.size != context.size:
                corrections.append(0.0)
                continue
            # No marginal/global theme preference: both members of a pair
            # independently gate its influence, with shrinkage for few votes.
            weights = np.clip((previous_images @ image - SIMILARITY_MINIMUM)
                              / (1 - SIMILARITY_MINIMUM), 0.0, 1.0) * context_weights
            total, strength = float(weights @ labels), float(weights.sum())
            corrections.append(float(np.clip(MAX_CORRECTION * total / (REGULARIZATION + strength),
                                             -MAX_CORRECTION, MAX_CORRECTION)))
        return [min(1.0, max(0.0, score + correction)) for score, correction in zip(baseline, corrections)]

"""Atomic local story snapshots, independent of the disposable image cache."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time

from .models import Topic, TranscriptSegment

MAX_SESSION_BYTES = 256 * 1024 * 1024
SCHEMA_VERSION = 1


class SessionStoreError(RuntimeError):
    """A saved story could not be read or written; in-memory speech stays usable."""


def default_session_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    return Path(base or Path.home() / ".local" / "state") / "SemanticImageStream" / "sessions"


def _integer(value, minimum=0, maximum=1_000_000_000):
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError("Invalid saved integer")
    return value


def _text(value, maximum):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Invalid saved text")
    return value


def _texts(value, count=64, maximum=1000):
    if not isinstance(value, list) or len(value) > count:
        raise ValueError("Invalid saved text list")
    return tuple(_text(item, maximum) for item in value)


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _topic_fields(topic):
    if topic is None:
        return None
    fields = {key: getattr(topic, key) for key in ("query", "concepts", "kind", "summary", "scene",
                                                  "visual_subjects", "queries", "context_used_segments", "visual_update",
                                                  "image_guidance", "guidance_strength", "guidance_only")}
    return {key: list(value) if isinstance(value, tuple) else value for key, value in fields.items()}


def snapshot_session(state) -> dict:
    # Never acquire the analysis/model lock here: its cached checkpoint can
    # lag behind the complete raw transcript while local inference is active.
    with state.lock:
        return {"version": SCHEMA_VERSION, "session_id": state.session_id, "revision": state.revision,
                "saved_at": time.time(), "book_context": state.book_context,
                "book_guidance_strength": state.book_guidance_strength, "guidance_revision": state.guidance_revision,
                "guidance_draft_marker": state.guidance_draft_marker,
                "guidance_draft_revision": state.guidance_draft_revision,
                "provider_key": list(state.provider_key), "segments": [asdict(segment) for segment in state.segments],
                "topic": _topic_fields(state.topic), "analyzer_checkpoint": deepcopy(state.analyzer_checkpoint)}


def restore_session(data):
    from .engine import SessionState

    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported saved session")
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("Invalid saved session identity")
    revision = _integer(data.get("revision"))
    book = _text(data.get("book_context", ""), 2000)
    strength = data.get("book_guidance_strength", .65)
    if not _finite_number(strength) or not 0 <= strength <= 1:
        raise ValueError("Invalid saved guidance strength")
    guidance_revision = _integer(data.get("guidance_revision", 0))
    draft_marker = data.get("guidance_draft_marker", "")
    if not isinstance(draft_marker, str) or (draft_marker and not re.fullmatch(r"[a-f0-9]{64}", draft_marker)):
        raise ValueError("Invalid saved guidance draft marker")
    draft_revision = _integer(data.get("guidance_draft_revision", 0))
    key = _texts(data.get("provider_key", []), 4, 2000)
    if key and (len(key) not in {3, 4} or key[0] not in {"contextual", "spacy", "ollama"}
                or (len(key) == 4 and (key[0] != "ollama" or not re.fullmatch(r"[a-f0-9]{64}", key[3])))):
        raise ValueError("Invalid saved interpreter")
    rows = data.get("segments")
    if not isinstance(rows, list) or len(rows) > 1_000_000:
        raise ValueError("Invalid saved transcript")
    segments = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid saved passage")
        start, end = row.get("start"), row.get("end")
        if not _finite_number(start) or not _finite_number(end) or not 0 <= start <= end:
            raise ValueError("Invalid saved timestamp")
        segments.append(TranscriptSegment(start, end, _text(row.get("text"), 1_000_000),
                                          _text(row.get("language", ""), 100)))
    state = SessionState(session_id=session_id, revision=revision, book_context=book,
                         segments=segments, provider_key=key, book_guidance_strength=strength,
                         guidance_revision=guidance_revision, guidance_draft_marker=draft_marker,
                         guidance_draft_revision=draft_revision)
    topic = data.get("topic")
    # A damaged derived summary must not discard an otherwise intact recording.
    try:
        if topic is not None:
            if not isinstance(topic, dict):
                raise ValueError("Invalid saved topic")
            transcript = tuple(segment.text for segment in segments)
            state.topic = Topic(query=_text(topic.get("query", ""), 180), context=" ".join(transcript),
                                concepts=_texts(topic.get("concepts", [])), kind=_text(topic.get("kind", "general"), 100),
                                summary=_text(topic.get("summary", ""), 400), scene=_text(topic.get("scene", ""), 280),
                                visual_subjects=_texts(topic.get("visual_subjects", []), 16),
                                queries=_texts(topic.get("queries", []), 8, 180), transcript=transcript,
                                context_used_segments=_integer(topic.get("context_used_segments", 0), 0, len(segments)),
                                visual_update=_text(topic.get("visual_update", "replace"), 20),
                                image_guidance=(_text(topic["image_guidance"], 2000)
                                                if topic.get("image_guidance") is not None else None),
                                guidance_strength=topic.get("guidance_strength"),
                                guidance_only=topic.get("guidance_only", False))
            if not isinstance(state.topic.guidance_only, bool):
                raise ValueError("Invalid saved topic guidance type")
            if (state.topic.guidance_strength is not None and
                    (not _finite_number(state.topic.guidance_strength) or not 0 <= state.topic.guidance_strength <= 1)):
                raise ValueError("Invalid saved topic guidance")
            if state.topic.visual_update not in {"replace", "retain", "clear"}:
                raise ValueError("Invalid saved visual update")
    except (ValueError, TypeError):
        state.topic = None
    checkpoint = data.get("analyzer_checkpoint")
    if isinstance(checkpoint, dict):
        try:
            if len(json.dumps(checkpoint, allow_nan=False).encode("utf-8")) <= 16 * 1024:
                state.analyzer_checkpoint = checkpoint
        except (ValueError, TypeError):
            pass
    return state


class SessionStore:
    def __init__(self, settings, on_status=None):
        root = settings.session_dir or default_session_dir()
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "last-session.json"
        self.on_status = on_status or (lambda message: None)
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._active = None
        self._epoch = 0
        self._ticket = 0
        self._latest_ticket = 0
        self._saved_ticket = 0
        self._pending = None
        self._last_error = None
        self._blocked = False
        self._closed = False
        self._worker = None

    def _status(self, message, epoch=None):
        with self._condition:
            if epoch is not None and (epoch != self._epoch or self._closed):
                return
            try:
                self.on_status(message)
            except Exception:
                pass

    def load(self):
        try:
            if not self.path.exists():
                return None
            with self.path.open("rb") as source:
                size = os.fstat(source.fileno()).st_size
                if size > MAX_SESSION_BYTES:
                    raise ValueError("Saved context exceeds the supported size")
                raw = source.read(size + 1)
                if len(raw) > size or source.read(1):
                    raise ValueError("Saved context changed while reading")
            if len(raw) > MAX_SESSION_BYTES:
                raise ValueError("Saved context exceeds the supported size")
            state = restore_session(json.loads(raw))
        except (OSError, ValueError, TypeError, RecursionError, MemoryError):
            self._blocked = True
            raise SessionStoreError("Saved context could not be read and was left untouched. "
                                    "New book can clear it; recording can continue in this window.") from None
        self._status(f"Saved context restored locally · {len(state.segments)} recorded segments")
        return state

    def bind(self, state):
        with self._condition:
            if self._closed:
                raise SessionStoreError("Local context saving is closed")
            if self._active is state:
                return
            self._detach_locked()
            self._active = state
            state.on_change = self.save

    def _detach_locked(self):
        if self._active is not None and self._active.on_change == self.save:
            self._active.on_change = None
        self._active = None
        self._epoch += 1
        self._pending = None
        self._last_error = None
        self._condition.notify_all()

    def detach(self):
        with self._condition:
            self._detach_locked()

    def save(self, state):
        with self._condition:
            if self._closed or self._active is not state:
                return None
            if self._blocked:
                raise SessionStoreError("The unreadable saved context was left untouched. "
                                        "Use New book before saving a fresh session.")
            self._ticket += 1
            ticket, epoch = self._ticket, self._epoch
        snapshot = snapshot_session(state)
        with self._condition:
            if self._closed or self._active is not state or epoch != self._epoch:
                return None
            if ticket > self._latest_ticket:
                self._latest_ticket = ticket
                self._pending = (epoch, ticket, state, snapshot)
                self._last_error = None
            if self._worker is None:
                self._worker = threading.Thread(target=self._write_pending, daemon=True, name="sis-session-save")
                self._worker.start()
            self._condition.notify_all()
        return ticket

    def _atomic_write(self, raw):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.root, prefix=".session-", suffix=".tmp",
                                             delete=False) as target:
                temporary = Path(target.name)
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    # Windows readers/antivirus may briefly hold the old file.
                    # Keep that complete snapshot while retrying the same
                    # atomic replacement, with a bounded background wait.
                    if attempt == 4:
                        raise
                    time.sleep(.02 * (attempt + 1))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _write_pending(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                epoch, ticket, state, snapshot = self._pending
                self._pending = None
            try:
                raw = json.dumps(snapshot, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8")
                if len(raw) > MAX_SESSION_BYTES:
                    raise ValueError("The recorded session exceeds the supported save size")
                with self._write_lock:
                    with self._condition:
                        if (epoch != self._epoch or self._active is not state or self._closed
                                or ticket < self._latest_ticket):
                            continue
                    self._atomic_write(raw)
                    with self._condition:
                        if epoch == self._epoch:
                            self._saved_ticket = max(self._saved_ticket, ticket)
                            self._last_error = None
                            self._condition.notify_all()
                self._status(f"Context saved locally · {len(snapshot['segments'])} recorded segments", epoch)
            except (OSError, ValueError, TypeError, RecursionError):
                current_error = False
                with self._condition:
                    if epoch == self._epoch and ticket == self._latest_ticket:
                        self._last_error = (ticket, "Could not save context locally. "
                                            "The current transcript remains in this window.")
                        current_error = True
                        self._condition.notify_all()
                if current_error:
                    self._status("Could not save context locally. The current transcript remains in this window.", epoch)

    def flush(self, state, timeout=5.0):
        ticket = self.save(state)
        if ticket is None:
            return
        with self._condition:
            epoch = self._epoch
            done = self._condition.wait_for(lambda: self._saved_ticket >= ticket
                                           or self._last_error is not None and self._last_error[0] >= ticket
                                           or self._epoch != epoch or self._closed, timeout=timeout)
            if self._epoch != epoch or self._closed:
                return
            if self._last_error is not None and self._last_error[0] >= ticket:
                raise SessionStoreError(self._last_error[1])
            if not done:
                raise SessionStoreError("Context is still being saved. Keep this window open and try closing again.")

    def clear(self):
        self.detach()
        try:
            with self._write_lock:
                self.path.unlink(missing_ok=True)
        except OSError:
            raise SessionStoreError("Saved context could not be deleted; it was left on this computer.") from None
        self._blocked = False
        self._status("Saved context cleared locally")

    def close(self):
        with self._condition:
            self._detach_locked()
            self._closed = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=1)

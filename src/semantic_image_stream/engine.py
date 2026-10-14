"""Independent, bounded capture/transcription/semantic/search clocks."""
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
import asyncio
from email.utils import parsedate_to_datetime
import json
import logging
import queue
import re
import shutil
import threading
import time
from typing import Callable
from uuid import uuid4

from .cache import Cache
from .image_ranking import build_ranker, is_duplicate
from .models import ImageCandidate, Topic
from .semantic_analysis import TopicGate, build_analyzer, fallback_queries, sanitize_query, tokens

log = logging.getLogger(__name__)
GPU_LOCK = threading.Lock()
_NATIVE_RETIREMENT_LOCK = threading.Lock()
_NATIVE_RETIREMENT_EVENTS = []
_SOURCE_PAUSED = object()
IMAGE_FILTER_FIELDS = ("time_filter_enabled", "time_filter_target", "time_filter_start", "time_filter_end",
                       "time_filter_scale", "time_filter_include_undated", "exclude_ai_generated", "allow_nsfw")


def _interpreter_key(settings):
    key = (settings.semantic_provider, getattr(settings, "context_model", ""),
           getattr(settings, "book_context", ""))
    if settings.semantic_provider == "ollama":
        from .story_references import story_references_for
        identity = story_references_for(settings).identity
        if identity:
            key += (identity,)
    return key


@dataclass
class SessionState:
    """Story data shared by monitoring runs, with optional local-save callbacks."""
    segments: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    analyzer: object | None = None
    provider_key: tuple = ()
    topic: object | None = None
    playback_scope: tuple | None = None
    playback_generation: int = 0
    session_id: str = field(default_factory=lambda: uuid4().hex)
    revision: int = 0
    book_context: str = ""
    book_guidance_strength: float = 0.65
    guidance_revision: int = 0
    guidance_draft_marker: str = ""
    guidance_draft_revision: int = 0
    analyzer_checkpoint: dict | None = None
    image_matches: dict[str, dict] = field(default_factory=dict)
    on_change: Callable[["SessionState"], None] | None = None

    def notify_changed(self):
        with self.lock:
            self.revision += 1
            callback = self.on_change
        if callback:
            try:
                callback(self)
            except Exception as exc:
                # Persistence is optional; disk/callback failure cannot stop
                # capture or expose recorded text through exception messages.
                log.warning("Local session update failed (%s)", type(exc).__name__)


PipelineSessionState = SessionState


async def _background(function, *args):
    """Daemon workers let closing the UI return during uncancellable native loads."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def finish(value=None, error=None):
        if not future.done():
            if error:
                future.set_exception(error)
            else:
                future.set_result(value)

    def work():
        try:
            result = function(*args)
        except Exception as exc:
            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(finish, None, exc)
                except RuntimeError:
                    pass
        else:
            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(finish, result, None)
                except RuntimeError:
                    pass

    threading.Thread(target=work, daemon=True, name="sis-model-work").start()
    return await future


def export_image(candidate: ImageCandidate, destination: str | Path) -> tuple[Path, Path]:
    """Copy the retrieved image alongside its complete source metadata."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(candidate.local_path)
    if not source.is_file():
        raise FileNotFoundError("The selected image is no longer in cache. Retrieve it again before saving.")
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    metadata = target.with_suffix(target.suffix + ".json")
    data = candidate.to_dict()
    data.pop("local_path", None)
    data["saved_filename"] = target.name
    metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target, metadata


class Pipeline:
    def __init__(self, settings, on_event, session_state=None):
        # Input/model settings are snapshots; image filters change on the worker loop.
        self.settings = replace(settings, providers=list(settings.providers)).validate()
        self.on_event = on_event
        self.stop_event = threading.Event()
        self._thread = None
        self._capture = None
        self._transcriber = None
        self.session_state = session_state if session_state is not None else SessionState()
        self._segments = self.session_state.segments
        self._segments_lock = self.session_state.lock
        self._recent_hashes = set()
        self._displayed_urls = set()
        self._rejected_image_urls = set()
        self._query_pages = {}
        self._refresh_queries = set()
        self._fresh_query_cycles = set()
        self._fallback_positions = {}
        self._last_query_failed = False
        self._last_query_incomplete = False
        self._last_query_all_paused = False
        self._last_query_succeeded = False
        self._search_retry_delay = 30.0
        self._source_retry_at = {}
        self._protected_paths = deque(maxlen=120)
        self._history_lock = threading.Lock()
        self._generation = 0
        self._last_search = 0.0
        from .book_guidance import snapshot_guidance
        self._latest_topic = snapshot_guidance(self.session_state.topic, self.settings) if self.session_state.topic else None
        self._approved_images = {}
        self._approved_scope = None
        self._approved_generation = None
        self._last_fill_signature = None
        self._ranker = None
        self._feedback_store = None
        self._feedback_store_lock = threading.Lock()
        self._feedback_request_lock = threading.Lock()
        self._feedback_requests = deque()
        self._feedback_worker_running = False
        self._loop = None
        self._image_filter_revision = 0
        self._native_workers = {}
        self._native_workers_lock = threading.Lock()

    def update_book_guidance(self, settings):
        """Apply a visual brief immediately; update interpretation independently."""
        snapshot = replace(settings, providers=list(settings.providers)).validate()
        text, strength = snapshot.book_context.strip(), snapshot.book_guidance_strength

        def apply():
            if (self.settings.book_context, self.settings.book_guidance_strength) == (text, strength):
                self._emit_context("ready", "The current book guidance is already applied.", guidance_ack=True)
                return
            self.settings.book_context, self.settings.book_guidance_strength = text, strength
            from .book_guidance import guidance_seed, refresh_guidance
            with self._segments_lock:
                state = self.session_state
                state.book_context, state.book_guidance_strength = text, strength
                state.guidance_revision += 1
                state.guidance_draft_marker = ""
                state.provider_key = _interpreter_key(self.settings)
                state.analyzer_checkpoint = None
                if self._latest_topic:
                    self._latest_topic = refresh_guidance(self._latest_topic, self.settings)
                    state.topic = self._latest_topic
                else:
                    self._latest_topic = state.topic = guidance_seed(self.settings, (segment.text for segment in state.segments))
            self._restart_image_discovery()
            self.session_state.notify_changed()
            self._emit_context("ready", "Book guidance applied; spoken context and audio are retained.",
                               playback_changed=True, guidance_ack=True)
            # Waiting for a current local-model call must not delay new searches.
            threading.Thread(target=self._rebind_book_guidance, args=(text, strength, state.guidance_revision),
                             daemon=True, name="sis-book-guidance").start()

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(apply)
        else:
            apply()

    def _rebind_book_guidance(self, text, strength, revision):
        with self.session_state.analysis_lock:
            with self._segments_lock:
                state = self.session_state
                if state.guidance_revision != revision:
                    return
                analyzer = state.analyzer
            update = getattr(analyzer, "update_book_guidance", None)
            if not callable(update):
                return
            try:
                update(text, strength)
                checkpoint = analyzer.export_checkpoint()
            except Exception as exc:
                log.warning("Book guidance checkpoint update deferred (%s)", type(exc).__name__)
                return
            with self._segments_lock:
                if state.analyzer is analyzer and state.guidance_revision == revision:
                    state.analyzer_checkpoint = checkpoint
            state.notify_changed()

    def _restart_image_discovery(self):
        self._image_filter_revision += 1
        self._generation += 1
        with self._segments_lock:
            self.session_state.playback_generation += 1
        self._approved_images.clear()
        self._query_pages.clear()
        self._refresh_queries.clear()
        self._fresh_query_cycles.clear()
        self._fallback_positions.clear()
        pending = getattr(self, "topic_queue", None)
        if pending is not None:
            while not pending.empty():
                pending.get_nowait()
            if self._latest_topic and self._latest_topic.query:
                pending.put_nowait((self._generation, self._latest_topic))

    def update_image_filters(self, settings):
        """Apply validated filter controls without restarting capture or narration."""
        snapshot = replace(settings, providers=list(settings.providers)).validate()
        values = {name: getattr(snapshot, name) for name in IMAGE_FILTER_FIELDS}

        def apply():
            from .temporal_filter import filter_identity
            previous = (filter_identity(self.settings), self.settings.exclude_ai_generated, self.settings.allow_nsfw)
            for name, value in values.items():
                setattr(self.settings, name, value)
            if previous == (filter_identity(self.settings), self.settings.exclude_ai_generated, self.settings.allow_nsfw):
                return
            self._restart_image_discovery()
            self._emit_context("ready", "Image filters updated; the spoken story is retained.", playback_changed=True)

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(apply)
        else:
            apply()

    def refresh_images(self):
        """Refresh the current scene without resetting speech or source cooldowns."""
        if not self._latest_topic or not self._latest_topic.query:
            return False

        def apply():
            topic = self._latest_topic
            if not topic or not topic.query or self.stop_event.is_set():
                return
            from .book_guidance import guided_queries
            queries = list(guided_queries(topic, self.settings))
            for alternative in fallback_queries(topic):
                queries.extend(guided_queries(replace(topic, query=alternative), self.settings))
            queries.append(sanitize_query(topic.query))
            for query in queries:
                if query:
                    key = query.casefold()
                    self._query_pages[key] = 0
                    self._refresh_queries.add(key)
            # Fence obsolete requests and ranking before fetching fresh head
            # pages. Durable displayed fingerprints and story state are retained.
            self._image_filter_revision += 1
            self._generation += 1
            self._approved_images.clear()
            self._last_search = 0.0
            pending = getattr(self, "topic_queue", None)
            if pending is not None:
                while not pending.empty():
                    pending.get_nowait()
                pending.put_nowait((self._generation, topic))
            self.emit("status", "Refreshing images for the current scene…")

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(apply)
        else:
            apply()
        return True

    def _image_filter_match(self, candidate):
        from .search_providers import is_known_generated
        from .temporal_filter import filter_match
        if not self.settings.allow_nsfw and getattr(candidate, "mature_content", None) is True:
            return False, "Marked as mature content"
        if (self.settings.exclude_ai_generated
                and (getattr(candidate, "ai_generated", None) is True
                     or is_known_generated(candidate.title, candidate.description))):
            return False, "Marked as AI-generated"
        result = filter_match(self.settings, candidate)
        return result.matched, result.reason

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def image_request_revision(self):
        return self._image_filter_revision

    def emit(self, name, value=None):
        try:
            self.on_event(name, value)
        except Exception:
            log.exception("Event consumer failed for %s", name)

    def start(self):
        if self.running:
            return
        with self._history_lock:
            self._rejected_image_urls.clear()
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sis-pipeline")
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        # Closing an audio stream can block in a driver; never do that on the UI.
        if self._capture:
            threading.Thread(target=self._capture.stop, daemon=True, name="sis-audio-stop").start()
        if self._transcriber:
            self._transcriber.stop()
        self._retire_native_workers()

    def _retire_native_workers(self):
        with self._native_workers_lock:
            workers, self._native_workers = self._native_workers, {}
        if not workers:
            return
        releases = {"speech": getattr(self._transcriber, "close", None),
                    "ranking": getattr(self._ranker, "close", None)}
        # Capture bound old-resource methods, never mutable Pipeline fields.
        events = [worker.retire(releases.get(name)) for name, worker in workers.items()]
        self._transcriber = self._ranker = None
        with _NATIVE_RETIREMENT_LOCK:
            _NATIVE_RETIREMENT_EVENTS[:] = [event for event in _NATIVE_RETIREMENT_EVENTS if not event.is_set()]
            _NATIVE_RETIREMENT_EVENTS.extend(events)

    async def _wait_for_native_cleanup(self):
        reported = False
        while not self.stop_event.is_set():
            with _NATIVE_RETIREMENT_LOCK:
                _NATIVE_RETIREMENT_EVENTS[:] = [event for event in _NATIVE_RETIREMENT_EVENTS if not event.is_set()]
                waiting = bool(_NATIVE_RETIREMENT_EVENTS)
            if not waiting:
                return True
            if not reported:
                self.emit("status", "Waiting for the previous local model work to release memory before resuming.")
                reported = True
            await asyncio.sleep(.1)
        return False

    async def _native_background(self, name, function, *args):
        if self.stop_event.is_set():
            from .native_workers import NativeWorkerClosed
            raise NativeWorkerClosed("Local model work stopped")
        with self._native_workers_lock:
            if self.stop_event.is_set():
                from .native_workers import NativeWorkerClosed
                raise NativeWorkerClosed("Local model work stopped")
            worker = self._native_workers.get(name)
        if worker is None:
            # Small direct/isolated operations also use this class without a
            # monitoring lifecycle. Production creates persistent workers.
            return await _background(function, *args)
        return await worker.run(function, *args)

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as exc:
            # Full transcripts and queries are deliberately absent from logs.
            log.error("Pipeline failed (%s)", type(exc).__name__)
            self.emit("error", str(exc))
        finally:
            self.stop_event.set()
            self.emit("stopped", None)

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        if not await self._wait_for_native_cleanup():
            return
        from .native_workers import NativeWorker
        from .audio_capture import create_capture
        from .transcription import FasterWhisperTranscriber
        from .search_providers import build_providers
        from .image_download import ImageDownloader
        self.audio_queue = queue.Queue(maxsize=self.settings.audio_queue_size)
        self.topic_queue = asyncio.Queue(maxsize=1)
        self._capture = create_capture(self.settings)
        self._transcriber = FasterWhisperTranscriber(self.settings, GPU_LOCK, self.stop_event,
                                                    lambda message: self.emit("status", message))
        try:
            analyzer = self._session_analyzer()
        except Exception as exc:
            analyzer = None
            self._report_context_failure(exc)
        ranker = build_ranker(self.settings, GPU_LOCK)
        self._ranker = ranker
        try:
            ranker.feedback = self._get_feedback_store()
        except Exception as exc:
            log.warning("Local image feedback unavailable (%s)", type(exc).__name__)
            self.emit("status", "Local image feedback is unavailable; image monitoring continues.")
        providers = build_providers(self.settings)
        cache = Cache(self.settings)
        downloader = ImageDownloader(self.settings)
        self._cache = cache
        with self._native_workers_lock:
            self._native_workers = {name: NativeWorker(name=f"sis-{name}")
                                    for name in ("speech", "ranking", "context")}
        with self._history_lock:
            self._recent_hashes.update(cache.seen_phashes())
            self._displayed_urls.update(cache.seen_urls())
        self.capture_done = False
        with self._segments_lock:
            capture_offset = max((segment.end for segment in self._segments), default=0.0)
            previous_segments = list(self._segments)
        if previous_segments:
            self.emit("transcript", previous_segments)
        if not self._latest_topic:
            from .book_guidance import guidance_seed
            self._latest_topic = self.session_state.topic = guidance_seed(self.settings,
                (segment.text for segment in previous_segments))
        if self._latest_topic:
            self._last_fill_signature = self._fill_signature(self._latest_topic)
            self._emit_context("ready", "Retained story context is ready." if not self._latest_topic.guidance_only else
                               "Images follow your book guidance while local spoken context develops.")
            self.emit("topic", self._latest_topic)
            if self._latest_topic.query:
                self.topic_queue.put_nowait((self._generation, self._latest_topic))

        def on_chunk(chunk):
            if self.stop_event.is_set():
                return
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                self.emit("status", "Transcription is behind playback; dropped the oldest audio chunk.")
            try:
                # A restarted device begins at zero, while the retained story
                # transcript must keep a stable, increasing session timeline.
                self.audio_queue.put_nowait(replace(chunk, start=chunk.start + capture_offset))
            except queue.Full:
                pass

        def capture():
            try:
                self._capture.run(on_chunk, lambda message: self.emit("status", message), self.stop_event)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.emit("error", f"Audio capture stopped: {exc}")
                    self.stop_event.set()
            finally:
                self.capture_done = True

        threading.Thread(target=capture, daemon=True, name="sis-capture").start()
        tasks = [asyncio.create_task(self._transcribe()),
                 asyncio.create_task(self._interpret(analyzer)),
                 asyncio.create_task(self._supervise_image_search(providers, downloader, ranker, cache))]
        try:
            self.emit("status", "Monitoring. Audio stays local; only compact search queries go to selected providers.")
            while not self.stop_event.is_set():
                for task in tasks:
                    if task.done() and not task.cancelled():
                        error = task.exception()
                        if error:
                            raise error
                await asyncio.sleep(0.1)
        finally:
            self.stop_event.set()
            self._retire_native_workers()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await downloader.close()
            for provider in providers:
                close = getattr(provider, "close", None)
                if close:
                    await close()
            cache.close()
            self._cache = None

    async def _transcribe(self):
        retry_chunk = None
        while not self.stop_event.is_set():
            if retry_chunk is None:
                try:
                    chunk = self.audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
            else:
                chunk = retry_chunk
            self.emit("status", "Recognizing captured audio locally…")
            try:
                segments = await self._native_background("speech", self._transcriber.transcribe, chunk)
            except Exception as exc:
                from .memory_pressure import MemoryPressureError, is_memory_allocation_error
                if self.stop_event.is_set():
                    break
                if not isinstance(exc, MemoryPressureError) and not is_memory_allocation_error(exc):
                    raise
                log.warning("Local transcription paused for memory pressure (%s)", type(exc).__name__)
                self.emit("status", "Windows memory is low. Close unused apps; transcription will retry. "
                                    "Recorded story context is retained.")
                retry_chunk = chunk
                deadline = time.monotonic() + 10
                while not self.stop_event.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(.1)
                continue
            retry_chunk = None
            if self.stop_event.is_set():
                break
            if segments:
                with self._segments_lock:
                    # Stop/close can win after inference's outer check. A final
                    # flush takes this same lock, so no later append escapes it.
                    if self.stop_event.is_set():
                        break
                    self._segments.extend(segments)
                    window = list(self._segments)
                self.session_state.notify_changed()
                self.emit("transcript", window)
                self.emit("status", "Spoken context updated · listening")
            else:
                self.emit("status", "Audio received, but no speech was recognized. Listening for the next segment.")

    async def _interpret(self, analyzer):
        gate = TopicGate(self.settings.semantic_sensitivity)
        if self._latest_topic:
            gate.accept(self._latest_topic)
        last_count = -1
        last_guidance_revision = -1
        failed_count = -1
        retry_at = 0.0
        while not self.stop_event.is_set():
            with self._segments_lock:
                window = list(self._segments)
            count = len(window)
            guidance_revision = self.session_state.guidance_revision
            if (window and (count != last_count or guidance_revision != last_guidance_revision
                            or getattr(analyzer, "has_pending_segments", False))
                    and (count != failed_count or time.monotonic() >= retry_at)):
                self._emit_context("updating", "Updating local story understanding from all recorded text.")
                topic, changed, refill = None, False, False
                try:
                    if analyzer is None:
                        analyzer = self._session_analyzer()
                    topic = await self._native_background("context", self._analyze_session, analyzer, window)
                except Exception as exc:
                    if self.stop_event.is_set():
                        break
                    # Recognition and capture remain independent of the local
                    # context model. No transcript/query is included in logs.
                    self._report_context_failure(exc)
                    failed_count = count
                    retry_at = time.monotonic() + max(5.0, self.settings.semantic_interval)
                else:
                    if self.stop_event.is_set():
                        break
                    if guidance_revision != self.session_state.guidance_revision:
                        # Reuse the already processed window after the brief is
                        # revalidated; an obsolete model result cannot publish.
                        last_count = last_guidance_revision = -1
                        continue
                    last_count = count
                    last_guidance_revision = guidance_revision
                    failed_count = -1
                    if topic:
                        from .book_guidance import snapshot_guidance
                        topic = snapshot_guidance(replace(topic, transcript=tuple(segment.text for segment in window)), self.settings)
                        with self._segments_lock:
                            if self.stop_event.is_set():
                                break
                            previous = self._latest_topic
                            retain_requested = topic.visual_update == "retain"
                            retaining = (retain_requested and previous is not None and bool(previous.query)
                                         and (not topic.query or all(getattr(previous, name) == getattr(topic, name)
                                              for name in ("query", "concepts", "kind", "scene", "visual_subjects", "queries"))))
                            if retaining:
                                # Abstract narration can update the complete
                                # book understanding without erasing its last
                                # grounded visual scene or queued images.
                                topic = replace(previous, summary=topic.summary, context=topic.context,
                                                transcript=topic.transcript,
                                                context_used_segments=topic.context_used_segments,
                                                image_guidance=topic.image_guidance, guidance_strength=topic.guidance_strength,
                                                visual_update="retain")
                                playback_changed, changed = False, False
                            else:
                                # An earlier batch may have advanced the
                                # analyzer before a later batch failed. Its
                                # retained scene can therefore be newer than
                                # the last scene published by this pipeline.
                                if retain_requested and previous is not None and topic.query:
                                    topic = replace(topic, visual_update="replace")
                                playback_changed = self._update_playback_scope(topic)
                                changed = gate.changed(topic) or playback_changed
                            # Every successful context update affects in-flight
                            # ranking, even if its compact search query is stable.
                            self._latest_topic = topic
                            self.session_state.topic = topic
                        self.session_state.notify_changed()
                        self._sync_image_pool(topic)
                        signature = self._fill_signature(topic)
                        refill = (bool(topic.query) and len(self._approved_images) < self.settings.prefetch_count
                                  and signature != self._last_fill_signature)
                        message = ("Keeping the current scene while recording more of the story." if retaining else
                                   "Local story understanding updated." if topic.query else
                                   "No grounded visual subject in this passage yet.")
                        self._emit_context("retaining" if retaining else "ready", message,
                                           playback_changed=playback_changed, retaining=retaining)
                    else:
                        changed = False
                if topic and failed_count == -1 and (changed or refill):
                    self._latest_topic = topic
                    self._generation += 1
                    gate.accept(topic)
                    self._last_fill_signature = self._fill_signature(topic)
                    if changed:
                        self.emit("topic", topic)
                    while not self.topic_queue.empty():
                        self.topic_queue.get_nowait()
                    if topic.query:
                        self.topic_queue.put_nowait((self._generation, topic))
            await asyncio.sleep(self.settings.semantic_interval)

    def _session_analyzer(self):
        key = _interpreter_key(self.settings)
        changed, reread = False, False
        try:
            with self._segments_lock:
                state = self.session_state
                exact_key = state.provider_key == key
                previous_key = state.provider_key
                guidance_change = (len(previous_key) >= 3 and previous_key[:2] == key[:2]
                                   and previous_key[3:] == key[3:] and previous_key[2] != key[2])
                # Adding non-factual illustrations need not replay hours of a
                # valid saved book. The new analyzer must still verify the old
                # checkpoint digest and every accepted fact against raw text.
                legacy_pack_addition = len(key) == 4 and state.provider_key == key[:3]
                matching = exact_key or legacy_pack_addition or guidance_change
                if state.book_context != self.settings.book_context:
                    state.book_context = self.settings.book_context
                    changed = True
                if state.book_guidance_strength != self.settings.book_guidance_strength:
                    state.book_guidance_strength = self.settings.book_guidance_strength
                    changed = True
                if state.analyzer is not None and exact_key:
                    from .book_guidance import snapshot_guidance
                    if state.topic:
                        updated = snapshot_guidance(state.topic, self.settings)
                        changed = changed or updated is not state.topic
                        state.topic = self._latest_topic = updated
                    return state.analyzer
                if guidance_change and callable(getattr(state.analyzer, "update_book_guidance", None)):
                    from .book_guidance import refresh_guidance
                    try:
                        state.analyzer.update_book_guidance(self.settings.book_context, self.settings.book_guidance_strength)
                        state.analyzer_checkpoint = state.analyzer.export_checkpoint()
                    except Exception as exc:
                        state.analyzer_checkpoint = None
                        log.warning("Restored book guidance waits for a new passage (%s)", type(exc).__name__)
                    state.provider_key = key
                    if state.topic:
                        state.topic = refresh_guidance(state.topic, self.settings)
                    self._latest_topic = state.topic
                    changed = True
                    return state.analyzer
                checkpoint = state.analyzer_checkpoint if matching else None
                if not matching:
                    reread = bool(state.provider_key and state.segments)
                    if state.provider_key:
                        state.playback_generation += 1
                    state.topic = None
                    self._latest_topic = None
                    state.playback_scope = None
                    state.analyzer_checkpoint = None
                state.provider_key = key
                state.analyzer = None
                changed = True
                initial_settings = replace(self.settings, providers=list(self.settings.providers))
                if guidance_change and self.settings.semantic_provider == "ollama":
                    initial_settings.book_context = previous_key[2]
                    previous_strength = getattr(state.topic, "guidance_strength", None)
                    initial_settings.book_guidance_strength = .65 if previous_strength is None else previous_strength
                analyzer = build_analyzer(
                    initial_settings, stop_event=self.stop_event,
                    on_status=lambda message: self._emit_context("updating", message))
                if matching and self.settings.semantic_provider == "ollama":
                    restore = getattr(analyzer, "restore_checkpoint", None)
                    restored = False
                    if checkpoint is not None and restore:
                        try:
                            restored = restore(checkpoint, list(state.segments)) is True
                        except Exception as exc:
                            log.warning("Local story checkpoint restore failed (%s)", type(exc).__name__)
                    if guidance_change and restored:
                        from .book_guidance import refresh_guidance
                        try:
                            analyzer.update_book_guidance(self.settings.book_context, self.settings.book_guidance_strength)
                            state.analyzer_checkpoint = analyzer.export_checkpoint()
                        except Exception as exc:
                            state.analyzer_checkpoint = None
                            log.warning("Saved book guidance waits for a new passage (%s)", type(exc).__name__)
                        if state.topic:
                            state.topic = refresh_guidance(state.topic, self.settings)
                    if not restored:
                        # A rejected checkpoint is derived data only: the full
                        # recorded transcript remains available for re-reading.
                        reread = bool(state.segments)
                        if state.topic is not None or state.playback_scope is not None:
                            state.playback_generation += 1
                        state.topic = None
                        state.playback_scope = None
                        state.analyzer_checkpoint = None
                state.analyzer = analyzer
                from .book_guidance import guidance_seed, snapshot_guidance
                self._latest_topic = snapshot_guidance(state.topic, self.settings) if state.topic else None
                if self._latest_topic is not state.topic:
                    state.topic = self._latest_topic
                    changed = True
                if not self._latest_topic and self._loop is not None:
                    self._latest_topic = state.topic = guidance_seed(self.settings,
                        (segment.text for segment in state.segments))
                return analyzer
        finally:
            if changed:
                self.session_state.notify_changed()
            if reread:
                self.emit("status", "Re-reading the complete retained transcript with the selected local interpreter.")

    def _analyze_session(self, analyzer, segments):
        # Native/local HTTP work can briefly outlive a cancelled asyncio task.
        # Serialize reuse and only rebind callbacks after that work has ended.
        while not self.stop_event.is_set():
            if self.session_state.analysis_lock.acquire(timeout=0.1):
                break
        else:
            return None
        checkpoint_changed = False
        key = _interpreter_key(self.settings)
        try:
            if self.stop_event.is_set():
                return None
            update = getattr(analyzer, "update_book_guidance", None)
            if callable(update) and (analyzer.settings.book_context, analyzer.settings.book_guidance_strength) != (
                    self.settings.book_context, self.settings.book_guidance_strength):
                update(self.settings.book_context, self.settings.book_guidance_strength)
            bind = getattr(analyzer, "bind", None)
            if bind:
                bind(self.stop_event, lambda message: self._emit_context("updating", message))
            if self.settings.semantic_provider == "ollama":
                incremental = getattr(analyzer, "analyze_incremental", None)
                return (incremental(segments, self.stop_event) if callable(incremental)
                        else analyzer.analyze(segments, self.stop_event))
            return analyzer.analyze(segments)
        finally:
            try:
                export = getattr(analyzer, "export_checkpoint", None)
                if export:
                    try:
                        checkpoint = export()
                    except Exception as exc:
                        log.warning("Local story checkpoint export failed (%s)", type(exc).__name__)
                    else:
                        with self._segments_lock:
                            if self.session_state.analyzer is analyzer and self.session_state.provider_key == key:
                                self.session_state.analyzer_checkpoint = checkpoint
                                checkpoint_changed = True
            finally:
                self.session_state.analysis_lock.release()
            if checkpoint_changed:
                self.session_state.notify_changed()

    def _report_context_failure(self, error):
        from .narrative_context import NarrativeContextError
        log.warning("Story understanding failed (%s)", type(error).__name__)
        detail = str(error) if isinstance(error, NarrativeContextError) else f"Local story understanding failed ({type(error).__name__})."
        self._emit_context("error", f"{detail} Speech recording continues; context will retry.")

    @staticmethod
    def _visual_scope(topic):
        subjects = frozenset(tokens(" ".join(topic.visual_subjects or (topic.query,))))
        scene = frozenset(tokens(topic.scene))
        # Content-token distance drops function words, including negation.
        # Preserve basic visual polarity without treating paraphrase as a scene.
        visual = " ".join((topic.scene, *topic.visual_subjects)).casefold()
        visual = re.sub(r"\b(?:not|never)\s+(?:open|opens|opened|opening)\b", "closed", visual)
        visual = re.sub(r"\b(?:not|never)\s+(?:closed|closes|closing|shut)\b", "open", visual)
        polarity = frozenset(label for label, pattern in (
            ("negated", r"\b(?:no|not|without|never)\b|n't\b"),
            ("open", r"\b(?:open|opens|opened|opening)\b"),
            ("closed", r"\b(?:closed|closes|closing|shut)\b"),
        ) if re.search(pattern, visual))
        return bool(topic.query), subjects, scene, polarity

    @staticmethod
    def _scope_changed(previous, current):
        def distance(first, second):
            return 1 - len(first & second) / max(1, len(first | second))

        return (previous is None or previous[0] != current[0]
                or (current[0] and (distance(previous[1], current[1]) >= 0.5
                                   or (current[2] and previous[2] and distance(previous[2], current[2]) >= 0.6)
                                   or (previous[3] if len(previous) > 3 else frozenset()) != current[3])))

    def _update_playback_scope(self, topic):
        scope = self._visual_scope(topic)
        changed = self._scope_changed(self.session_state.playback_scope, scope)
        if changed:
            self.session_state.playback_scope = scope
            self.session_state.playback_generation += 1
        return changed

    @staticmethod
    def _fill_signature(topic):
        # Counts, raw-text appends and summary paraphrases alone do not justify
        # another source search. Grounded alternatives and visual content do.
        queries = tuple(dict.fromkeys(sanitize_query(query).casefold()
                                      for query in (topic.query, *topic.queries) if sanitize_query(query)))
        return queries, frozenset(tokens(" ".join(topic.visual_subjects))), frozenset(tokens(topic.scene))

    def _sync_image_pool(self, topic):
        scope = self._visual_scope(topic)
        generation = self.session_state.playback_generation
        if self._approved_generation != generation or self._scope_changed(self._approved_scope, scope):
            self._approved_images.clear()
            self._approved_scope, self._approved_generation = scope, generation
        with self._history_lock:
            seen_urls, seen_hashes = set(self._displayed_urls), list(self._recent_hashes)
            rejected_urls = set(self._rejected_image_urls)
        self._approved_images = {url: image for url, image in self._approved_images.items()
                                 if image.path.is_file() and url not in seen_urls and url not in rejected_urls
                                 and self._image_filter_match(image)[0]
                                 and not is_duplicate(image.phash, seen_hashes)}

    def _emit_context(self, phase, message, playback_changed=False, retaining=False, guidance_ack=False):
        with self._segments_lock:
            recorded = len(self._segments)
        topic = self._latest_topic
        analyzer = self.session_state.analyzer
        used = getattr(analyzer, "context_used_segments", getattr(topic, "context_used_segments", 0))
        self.emit("context", {"phase": phase, "summary": getattr(topic, "summary", ""),
                              "scene": getattr(topic, "scene", ""),
                              "visual_subjects": list(getattr(topic, "visual_subjects", ())),
                              "segments_used": used,
                              "segments_recorded": recorded,
                              "playback_key": f"scene:{self.session_state.playback_generation}",
                              "playback_changed": playback_changed,
                              "retaining": retaining,
                              "topic_snapshot": topic,
                              "model": getattr(self.settings, "context_model", "")
                              if self.settings.semantic_provider == "ollama" else "",
                              "method": self.settings.semantic_provider,
                              "summary_method": getattr(analyzer, "summary_method", ""), "message": message,
                              **({"book_context": self.settings.book_context,
                                  "book_guidance_strength": self.settings.book_guidance_strength,
                                  "guidance_revision": self.session_state.guidance_revision} if guidance_ack else {})})

    async def _supervise_image_search(self, providers, downloader, ranker, cache):
        """Recover an exited image worker without restarting speech or story data."""
        while not self.stop_event.is_set():
            try:
                await self._search(providers, downloader, ranker, cache)
            except asyncio.CancelledError:
                if self.stop_event.is_set():
                    raise
                reason = "CancelledError"
            except Exception as exc:
                reason = type(exc).__name__
            else:
                if self.stop_event.is_set():
                    return
                reason = "UnexpectedExit"
            if self.stop_event.is_set():
                return
            log.warning("Image discovery worker restarting (%s)", reason)
            self.emit("retrieval", {"phase": "error", "accepted": len(self._approved_images),
                                    "found": 0, "downloaded": 0, "rejected": 0,
                                    "threshold": self.settings.relevance_threshold, "best_score": None,
                                    "failures": {"Image discovery interrupted": 1},
                                    "message": "Image discovery was interrupted; restarting it automatically. "
                                               "Speech and story context are retained."})
            # A failed worker must not spin on a topic already queued before
            # its exit. An actual new generation may wake recovery sooner.
            generation = self._generation
            delay = max(30.0, self.settings.search_interval)
            deadline = time.monotonic() + delay
            while (not self.stop_event.is_set() and self._generation == generation
                   and time.monotonic() < deadline):
                await asyncio.sleep(.1)

    def _next_discovery_query(self, primary, alternatives, tried):
        # Each poll still has two fallback requests. Rotate their starting
        # point so later grounded subjects are not starved by exhausted ones.
        options = {}
        for query in alternatives:
            clean = sanitize_query(query)
            if clean and clean.casefold() != primary.casefold():
                options.setdefault(clean.casefold(), clean)
        if not options:
            return ""
        values = tuple(options.values())
        key = (primary.casefold(), *options)
        start = self._fallback_positions.get(key, 0)
        for offset in range(len(values)):
            position = (start + offset) % len(values)
            query = values[position]
            if query.casefold() not in tried:
                self._fallback_positions[key] = (position + 1) % len(values)
                if len(self._fallback_positions) > 200:
                    self._fallback_positions.pop(next(iter(self._fallback_positions)))
                return query
        return ""

    async def _search(self, providers, downloader, ranker, cache):
        while not self.stop_event.is_set():
            try:
                generation, topic = await asyncio.wait_for(self.topic_queue.get(), timeout=.25)
            except asyncio.TimeoutError:
                topic = self._latest_topic
                if not topic or not topic.query:
                    continue
                self._sync_image_pool(topic)
                if len(self._approved_images) >= self.settings.prefetch_count:
                    continue
                generation = self._generation
            last_search = self._last_search
            delay = self.settings.search_interval - (time.monotonic() - last_search)
            while delay > 0 and not self.stop_event.is_set():
                # Stop and an explicit refresh should wake a provider cooldown.
                pause = min(.1, delay)
                await asyncio.sleep(pause)
                if self._last_search != last_search:
                    break
                delay = min(delay - pause, self.settings.search_interval - (time.monotonic() - last_search))
            if self.stop_event.is_set():
                return
            # Collapse transitions queued during the provider cooldown.
            while not self.topic_queue.empty():
                generation, topic = self.topic_queue.get_nowait()
            # A queued retry may carry an older summary or an empty passage.
            # Search from the latest accepted scene, never restore that input.
            topic = self._latest_topic or topic
            generation = self._generation
            if not topic.query:
                continue
            self._sync_image_pool(topic)
            if len(self._approved_images) >= self.settings.prefetch_count:
                continue
            self._last_search = time.monotonic()
            from .book_guidance import guided_queries
            guided = guided_queries(topic, self.settings)
            query = guided[0] if guided else sanitize_query(topic.query)
            primary_query = query
            query_basis = topic
            is_fallback = False
            tried = {query.casefold()}
            remaining = 2
            rejected_urls = {}
            attempt_failures = Counter()
            attempt_stats = {"found": 0, "downloaded": 0, "best_score": None}
            published = False
            ranking_failed = False
            all_queries_paused = True
            healthy_sources = False
            download_retry_needed = False
            self._search_retry_delay = max(1.0, self.settings.search_interval)
            while not self.stop_event.is_set():
                page_key = query.casefold()
                page = self._query_pages.setdefault(page_key, 0)
                revision = self._image_filter_revision
                evaluated = False
                transient_download = False
                evaluated_urls = set()

                async def evaluate_batch(candidates):
                    nonlocal evaluated, published, ranking_failed, transient_download, download_retry_needed
                    if (revision != self._image_filter_revision or generation != self._generation
                            or self.stop_event.is_set()):
                        return True
                    candidates = [candidate for candidate in candidates if candidate.image_url not in evaluated_urls]
                    evaluated_urls.update(candidate.image_url for candidate in candidates)
                    evaluated = True
                    previous_failures = attempt_failures.copy()
                    ready, failed = await self._evaluate_candidates(
                        topic, candidates, query_basis if is_fallback else None,
                        query if is_fallback else None, downloader, ranker, cache,
                        rejected_urls, attempt_failures, attempt_stats)
                    published = published or ready
                    ranking_failed = ranking_failed or failed
                    transient_download = transient_download or any(
                        attempt_failures[reason] > previous_failures[reason]
                        for reason in ("Source temporarily rate-limited", "Source unavailable", "Network error",
                                       "Image download timed out", "Low system memory"))
                    download_retry_needed = download_retry_needed or transient_download
                    return (ranking_failed or len(self._approved_images) >= self.settings.prefetch_count
                            or revision != self._image_filter_revision or generation != self._generation
                            or self.stop_event.is_set())

                await self._query_candidates(query, providers, cache, page=page,
                                             refresh=(page_key in self._refresh_queries
                                                      or page_key in self._fresh_query_cycles),
                                             on_candidates=evaluate_batch)
                # A refresh during a slow request owns its new force flag and
                # page cursor. An obsolete request must not consume either.
                if revision != self._image_filter_revision or generation != self._generation:
                    break
                if not evaluated and not self.stop_event.is_set():
                    await evaluate_batch([])
                failed_query = self._last_query_failed
                incomplete_query = self._last_query_incomplete
                succeeded_query = self._last_query_succeeded
                healthy_sources = healthy_sources or succeeded_query
                all_queries_paused = all_queries_paused and self._last_query_all_paused
                self._refresh_queries.discard(page_key)
                if incomplete_query:
                    # Partial results must not hide a recovering source behind
                    # a combined query cache. A healthy source may still page.
                    self._refresh_queries.add(page_key)
                if failed_query:
                    attempt_failures["Source unavailable"] += 1
                if generation != self._generation:
                    # A slow old batch yielded to a newer scene. Return to the
                    # current primary query instead of paging its predecessor.
                    break
                # An actual healthy response exhausts this page even if a
                # different source is cooling down. Failed downloads retain
                # their page so its already-cached URLs can be retried safely.
                if succeeded_query and not transient_download and not ranking_failed:
                    self._query_pages[page_key] = (page + 1) % 10
                    if page == 9:
                        # Every later page needs a fresh request after the
                        # first exhausted cycle, including non-head pages.
                        self._fresh_query_cycles.add(page_key)
                if len(self._query_pages) > 200:
                    oldest = next(iter(self._query_pages))
                    self._query_pages.pop(oldest)
                    self._refresh_queries.discard(oldest)
                    self._fresh_query_cycles.discard(oldest)
                # Raw source hits are not usable matches. Try the remaining
                # subjects only after validation/ranking exhausts this batch.
                # A failed local ranker instead retains its explicit diagnostic.
                if ranking_failed or len(self._approved_images) >= self.settings.prefetch_count or not remaining:
                    break
                basis = self._latest_topic or topic
                if not basis.query:
                    break
                alternatives = guided_queries(basis, self.settings)[1:]
                for alternative in fallback_queries(basis):
                    alternatives.extend(guided_queries(replace(basis, query=alternative), self.settings))
                query = self._next_discovery_query(primary_query, alternatives, tried)
                if not query:
                    break
                query_basis = basis
                is_fallback = True
                tried.add(query.casefold())
                remaining -= 1
                self.emit("retrieval", {"phase": "searching", "found": 0, "downloaded": 0,
                                        "rejected": 0, "accepted": len(self._approved_images),
                                        "threshold": self.settings.relevance_threshold,
                                        "best_score": None, "failures": {},
                                        "message": (f"{len(self._approved_images)}/{self.settings.prefetch_count} relevant images ready; "
                                                    "looking for more current-scene images." if self._approved_images else
                                                    "No usable matches yet; trying a shorter query from the current spoken context.")})
            if generation != self._generation:
                continue
            transient = any(attempt_failures[reason] for reason in ("Source temporarily rate-limited", "Source unavailable",
                                                                  "Network error", "Image download timed out", "Low system memory"))
            if self._approved_images and not ranking_failed:
                count = len(self._approved_images)
                message = (f"{count} relevant images ready for this scene." if count >= self.settings.prefetch_count else
                           f"{count}/{self.settings.prefetch_count} relevant images ready. No further images passed the current checks; "
                           "continuing to look for fresh matches.")
                if transient and count < self.settings.prefetch_count:
                    message = (f"{count}/{self.settings.prefetch_count} relevant images ready. Some image sources are temporarily unavailable; "
                               "keeping ready images while they retry.")
                if is_fallback and (self._latest_topic or topic) is query_basis:
                    message += " · using a shorter query from the current spoken context"
                self.emit("retrieval", {"phase": "ready", "accepted": count,
                                        **attempt_stats,
                                        "rejected": sum(attempt_failures.values()), "failures": dict(attempt_failures),
                                        "threshold": self.settings.relevance_threshold, "message": message})
            if (ranking_failed or not self._approved_images
                    or ((transient or all_queries_paused) and len(self._approved_images) < self.settings.prefetch_count)):
                if ranking_failed or download_retry_needed or (transient and not healthy_sources):
                    self._search_retry_delay = max(30.0, self.settings.search_interval)
                if all_queries_paused:
                    deadlines = [self._source_retry_at.get(provider.name, 0) for provider in providers]
                    remaining_delays = [deadline - time.monotonic() for deadline in deadlines
                                        if deadline > time.monotonic()]
                    if remaining_delays:
                        self._search_retry_delay = max(self._search_retry_delay, min(remaining_delays))
                if not ranking_failed:
                    message = ("Keeping the current scene while image sources cool down; images will retry." if all_queries_paused else
                               "Keeping ready images while unavailable sources retry." if self._approved_images else
                               "No new images met the relevance and quality filters. See image status for details.")
                    self.emit("status", message)
                # Exhausted healthy pages follow the user's search cadence;
                # model/network failures and source cooldowns retain backoff.
                await self._wait_for_search_retry(generation, topic)
            cache.prune(set(self._protected_paths))

    async def _evaluate_candidates(self, topic, candidates, fallback_topic, fallback_query,
                                   downloader, ranker, cache, rejected_urls, attempt_failures, attempt_stats):
        revision = self._image_filter_revision
        def ranking_topic():
            current = self._latest_topic or topic
            # Retain full context, concepts and kind, but score matches to
            # the shorter subject that actually found them. Once speech
            # changes, obsolete fallback wording must not bless old images.
            effective = replace(current, query=fallback_query) if current is fallback_topic else current
            return current, effective

        self._sync_image_pool(self._latest_topic or topic)
        by_url = {}
        for candidate in candidates:
            by_url.setdefault(candidate.image_url, candidate)
        with self._history_lock:
            seen_urls = self._displayed_urls | self._rejected_image_urls
        unique = [candidate for url, candidate in by_url.items()
                  if url not in self._approved_images and url not in seen_urls]
        attempt_stats["found"] += len(unique)
        filtered = []
        for candidate in unique:
            matched, reason = self._image_filter_match(candidate)
            if matched:
                filtered.append(candidate)
            else:
                attempt_failures[reason] += 1
        unique = filtered
        # Rank metadata before prefetch to start with promising source
        # images; retain other candidates for actual multimodal ranking.
        if ranker.name == "metadata" and unique:
            try:
                current = ranking_topic()[1]
                preferred = ranker.rank(current, unique, []) if current.query else []
            except Exception as exc:
                self._report_ranking_failure(ranker, exc, len(unique), 0)
                return False, True
            preferred_urls = {c.image_url for c in preferred}
            unique = preferred + [c for c in unique if c.image_url not in preferred_urls]
        semaphore = asyncio.Semaphore(3)
        downloaded = 0
        failures = Counter()
        published = set()
        last_selection = ()

        def report(phase, accepted=0, best_score=None):
            if phase == "ready":
                message = f"{accepted} relevant images ready · {downloaded} downloaded from {len(unique)} results"
            elif phase == "empty":
                if not unique:
                    filter_rejections = {reason: count for reason, count in attempt_failures.items()
                                         if reason.startswith(("The recorded period", "No supported period", "Marked as AI",
                                                               "Marked as mature content"))}
                    filter_label = "image" if "Marked as mature content" in filter_rejections else "time/AI"
                    message = (f"No results passed the selected {filter_label} filters. " +
                               "; ".join(f"{count} {reason.lower()}" for reason, count in filter_rejections.items())
                               if filter_rejections else "No images found for this query.")
                elif not downloaded:
                    details = "; ".join(f"{count} {reason.lower()}" for reason, count in failures.items())
                    message = f"Found {len(unique)} results, but no images could be downloaded. {details}"
                elif best_score is not None and best_score < self.settings.relevance_threshold:
                    message = (f"Downloaded {downloaded} images. Best relevance {best_score:.2f} is below "
                               f"the {self.settings.relevance_threshold:.2f} minimum.")
                else:
                    message = f"Downloaded {downloaded} images, but none are new matches for the current discussion."
            elif phase == "loading":
                message = ("Loading the cached local trained relevance model. " if ranker.name == "learned_context" else
                           "Loading the local image relevance model; first use may download weights. ")
                message += "Speech monitoring continues."
            else:
                message = f"Checking images · {downloaded} downloaded from {len(unique)} results"
            if (self._latest_topic or topic) is fallback_topic:
                message += " · using a shorter query from the current spoken context"
            self.emit("retrieval", {"phase": phase, "found": len(unique), "downloaded": downloaded,
                                    "rejected": sum(failures.values()), "accepted": accepted,
                                    "threshold": self.settings.relevance_threshold,
                                    "best_score": best_score, "failures": dict(failures), "message": message})

        report("downloading")

        async def prepare(candidate):
            async with semaphore:
                cached = cache.get_image(candidate.image_url)
                if cached:
                    # Reapply current geometry filters to cached images.
                    ratio = cached.width / max(1, cached.height)
                    if (cached.width >= self.settings.min_width and cached.height >= self.settings.min_height
                            and cached.width * cached.height <= self.settings.max_image_pixels
                            and cached.path.stat().st_size <= self.settings.max_download_bytes
                            and self.settings.min_aspect_ratio <= ratio <= self.settings.max_aspect_ratio):
                        # Keep cached validated pixels while using current source
                        # date/attribution metadata. Known content marks remain
                        # recorded even if another source omits them later.
                        updated = replace(candidate, local_path=cached.local_path, width=cached.width,
                                          height=cached.height, phash=cached.phash,
                                          ai_generated=(True if cached.ai_generated is True else candidate.ai_generated),
                                          mature_content=(True if cached.mature_content is True else candidate.mature_content),
                                          temporal_evidence=(candidate.temporal_evidence or cached.temporal_evidence))
                        cache.put_image(updated)
                        return updated, None
                if candidate.image_url in rejected_urls:
                    return None, rejected_urls[candidate.image_url]
                try:
                    image = await downloader.download(candidate)
                    cache.put_image(image)
                    return image, None
                except Exception as exc:
                    log.debug("Rejected image (%s)", type(exc).__name__)
                    reason = getattr(exc, "reason", "Download or validation failed")
                    rejected_urls[candidate.image_url] = reason
                    return None, reason

        prepared = []
        ranked_by_url = {}
        ranked_scope = None
        ranked_generation = None
        best_score = None
        ranking_failed = False
        scene_corrections = 0
        tasks = [asyncio.create_task(prepare(c)) for c in unique]
        try:
            # Publish useful validated images as they arrive. Waiting for
            # every slow/broken URL made an otherwise working preview blank.
            for completed in asyncio.as_completed(tasks):
                image, reason = await completed
                if revision != self._image_filter_revision or self.stop_event.is_set():
                    return False, False
                if image is None:
                    failures[reason] += 1
                    attempt_failures[reason] += 1
                    report("downloading", len(published), best_score)
                    continue
                downloaded += 1
                attempt_stats["downloaded"] += 1
                prepared.append(image)
                with self._history_lock:
                    recent = list(self._recent_hashes)
                # Each new image uses the latest complete context snapshot.
                # Rechecking the growing batch after every arrival was quadratic
                # model work. Only a changed visual scene invalidates its scores.
                while True:
                    basis, current = ranking_topic()
                    scope = self._visual_scope(basis)
                    generation = self.session_state.playback_generation
                    new_scene = (ranked_generation != generation or self._scope_changed(ranked_scope, scope))
                    to_rank = prepared if new_scene else [image]
                    if ranker.name in {"clip", "context_clip", "learned_context"} and getattr(ranker, "model", None) is None:
                        report("loading", len(published), best_score)
                    try:
                        batch = (await self._native_background("ranking", ranker.rank, current, to_rank, recent)) if current.query else []
                    except Exception as exc:
                        ranking_failed = True
                        self._report_ranking_failure(ranker, exc, len(unique), downloaded, len(published))
                        break
                    if self.stop_event.is_set() or revision != self._image_filter_revision:
                        return False, False
                    latest = self._latest_topic or topic
                    if (generation != self.session_state.playback_generation
                            or self._scope_changed(scope, self._visual_scope(latest))):
                        # An obsolete scene cannot approve images for its
                        # successor. Summary/transcript additions within the
                        # same scene must not starve a completed slow ranking.
                        scene_corrections += 1
                        if scene_corrections > 1:
                            # Repeated scene changes can otherwise rerank the
                            # same old-query pixels forever. Release remaining
                            # downloads and let discovery fetch the current scene.
                            return False, False
                        continue
                    if new_scene:
                        ranked_by_url.clear()
                        if ranked_scope is not None:
                            published.clear()
                            last_selection = ()
                    self._sync_image_pool(latest)
                    for candidate in to_rank:
                        ranked_by_url.pop(candidate.image_url, None)
                    ranked_by_url.update((candidate.image_url, candidate) for candidate in batch)
                    ranked_scope, ranked_generation = scope, generation
                    ranked = sorted(ranked_by_url.values(), key=lambda candidate: candidate.score, reverse=True)
                    break
                if ranking_failed:
                    break
                best_score = max((c.score for c in ranked), default=None)
                if best_score is not None:
                    attempt_stats["best_score"] = max(best_score, attempt_stats["best_score"] or 0.0)
                chosen, hashes = [], list(recent)
                with self._history_lock:
                    excluded_urls = self._displayed_urls | self._rejected_image_urls
                    hashes = list(self._recent_hashes)
                for candidate in ranked:
                    matched, reason = self._image_filter_match(candidate)
                    if not matched:
                        failures[reason] += 1
                        continue
                    other_hashes = [approved.phash for url, approved in self._approved_images.items()
                                    if url != candidate.image_url]
                    if (candidate.image_url not in excluded_urls and candidate.image_url not in self._approved_images
                            and candidate.score >= self.settings.relevance_threshold
                            and not is_duplicate(candidate.phash, hashes)
                            and not is_duplicate(candidate.phash, other_hashes)):
                        chosen.append(candidate)
                        hashes.append(candidate.phash)
                        if len(chosen) + len(self._approved_images) >= self.settings.prefetch_count:
                            break
                selection = (current.query, *(c.image_url for c in chosen))
                if chosen and selection != last_selection:
                    last_selection = selection
                    self._protected_paths.extend(c.local_path for c in chosen)
                    published.update(c.image_url for c in chosen)
                    self._approved_images.update((candidate.image_url, candidate) for candidate in chosen)
                    from .session_image_export import record_session_images
                    record_session_images(self.session_state, chosen, current,
                                          self.settings.relevance_threshold, ranker.name)
                    self.emit("images", sorted(self._approved_images.values(), key=lambda candidate: candidate.score, reverse=True))
                    self.emit("status", f"Ready: {len(chosen)} relevant images · {ranker.name} ranking")
                report("ready" if self._approved_images else "downloading", len(self._approved_images), best_score)
                if len(self._approved_images) >= self.settings.prefetch_count:
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if not ranking_failed:
            report("ready" if self._approved_images else "empty", len(self._approved_images), best_score)
        return bool(published), ranking_failed

    async def _query_candidates(self, query, providers, cache, page=0, refresh=False, on_candidates=None):
        from .temporal_filter import filter_identity, search_hints
        from .book_guidance import guidance_identity
        revision = self._image_filter_revision
        self._last_query_failed = False
        self._last_query_incomplete = False
        self._last_query_all_paused = False
        self._last_query_succeeded = False
        identity = [query.casefold(), self.settings.providers, self.settings.results_per_provider]
        # Version both modes so legacy query rows without mature metadata are
        # refreshed once, and permissive results never fill restricted searches.
        identity.append({"content_policy": "mature-v1", "allow_nsfw": self.settings.allow_nsfw})
        guidance = guidance_identity(self._latest_topic or Topic(query, ""), self.settings)
        if guidance:
            identity.append({"book_guidance": guidance})
        if self.settings.time_filter_enabled or not self.settings.exclude_ai_generated:
            identity.append({"time_filter": filter_identity(self.settings),
                             "exclude_ai_generated": self.settings.exclude_ai_generated})
        if "duckduckgo" in self.settings.providers:
            # Older combined cache entries can contain only the other sources'
            # partial results after DDGS's malformed HTTP/2 header failed.
            # Refresh those queries once while retaining image/history caches.
            identity.append("duckduckgo-http2-v1")
        if page:
            identity.append({"page": page})
        key = json.dumps(identity)
        cached = cache.get_query(key)
        with self._history_lock:
            consumed_urls = self._displayed_urls | self._rejected_image_urls
        # A cached page whose images have all been consumed cannot refill the
        # slideshow. Fetch it now instead of spending one cooldown per stale
        # cache page before the ten-page cycle finally refreshes the source.
        consumed = bool(cached) and all(candidate.image_url in consumed_urls for candidate in cached)
        if cached is not None and not refresh and not consumed:
            self._last_query_succeeded = True
            if on_candidates:
                await on_candidates(cached)
            return cached
        self.emit("status", "Retrieving existing images from selected sources…")
        hints = search_hints(self.settings) if (self.settings.time_filter_enabled
                 and self.settings.time_filter_target == "depicted"
                 and self.settings.time_filter_start < -100_000) else ()
        # Use one geological unit at a time. Calendar bounds are metadata
        # filters, not literal words every source document must contain.
        hint = hints[page % len(hints)] if hints else ""
        source_query = (query if not hint or hint.casefold() in query.casefold()
                        else sanitize_query(f"{hint} {query}"))[:180].strip()
        hinted_page = page // len(hints) if hints and source_query != query else page
        async def search(provider):
            if time.monotonic() < self._source_retry_at.get(provider.name, 0):
                return _SOURCE_PAUSED
            async def request(text):
                selected_page = hinted_page if text == source_query else page
                if not selected_page:
                    return await provider.search(text, self.settings.results_per_provider)
                paged = getattr(provider, "search_page", None)
                return await paged(text, self.settings.results_per_provider, selected_page) if paged else []
            found = await request(source_query)
            if (source_query != query and revision == self._image_filter_revision
                    and not self.stop_event.is_set()
                    and not any(self._image_filter_match(item)[0] for item in found)):
                # A source may not index period words. Try the original story
                # query too; its results still require source-backed dates.
                found.extend(await request(query))
            return found
        async def search_result(index, provider):
            try:
                return index, await search(provider)
            except Exception as exc:
                return index, exc

        tasks = [asyncio.create_task(search_result(index, provider)) for index, provider in enumerate(providers)]
        results = {}
        try:
            # A healthy source's hits can reach validation/ranking immediately.
            # Waiting for all providers delayed useful images by a slow source's
            # entire timeout. Cancel outstanding work once the buffer is full.
            for completed in asyncio.as_completed(tasks):
                index, result = await completed
                results[index] = result
                if revision != self._image_filter_revision or self.stop_event.is_set():
                    break
                if on_candidates and isinstance(result, list) and result:
                    if await on_candidates(result):
                        break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # A source may have already answered (including Retry-After) while
            # the fast batch was being ranked. Preserve every completed answer
            # before closing the batch, even if it no longer needs evaluation.
            for task in tasks:
                if not task.cancelled():
                    index, result = task.result()
                    results.setdefault(index, result)
        if revision != self._image_filter_revision:
            # An old request cannot fill a new range or create a false empty cache.
            return []
        candidates = []
        succeeded = len(results) == len(providers)
        self._last_query_incomplete = not succeeded
        self._last_query_all_paused = bool(results) and all(result is _SOURCE_PAUSED for result in results.values())
        for index, provider in enumerate(providers):
            if index not in results:
                continue
            result = results[index]
            if result is _SOURCE_PAUSED:
                succeeded = False
                self._last_query_incomplete = True
            elif isinstance(result, Exception):
                succeeded = False
                self._last_query_failed = True
                self._last_query_incomplete = True
                response = getattr(result, "response", None) or getattr(getattr(result, "__cause__", None), "response", None)
                delay = max(30.0, self.settings.search_interval)
                if response is not None and response.status_code == 429:
                    value = response.headers.get("Retry-After", "")
                    try:
                        delay = max(delay, float(value)) if value.isdigit() else max(delay, parsedate_to_datetime(value).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                # Back off the failed source alone. Other healthy sources keep
                # supplying images at the configured search cadence.
                self._source_retry_at[provider.name] = time.monotonic() + delay
                self.emit("status", f"{provider.name} unavailable ({type(result).__name__}); trying other sources.")
            else:
                self._last_query_succeeded = True
                candidates.extend(result)
        # A genuine zero-result query is cached too, preventing unchanged
        # narration from repeatedly hitting the same sources. Availability
        # errors are never treated as evidence that a subject has no images.
        if candidates or (succeeded and self._last_query_succeeded):
            cache.put_query(key, candidates)
        return candidates

    def _report_ranking_failure(self, ranker, error, found, downloaded, accepted=0):
        from .memory_pressure import is_memory_allocation_error
        # Exception text can include file paths, URLs or private model inputs.
        # The class name and selected ranking method give a safe diagnostic.
        name = {"clip": "CLIP", "context_clip": "context CLIP", "learned_context": "trained relevance",
                "metadata": "metadata"}.get(ranker.name, "selected")
        reason = "Local image ranking failed"
        log.warning("Image ranking failed (%s)", type(error).__name__)
        message = (f"Local {name} image ranking failed ({type(error).__name__}). "
                   "Speech monitoring continues; image ranking will retry.")
        if is_memory_allocation_error(error):
            reason = "Low system memory"
            message = "Windows memory is low. Close unused apps; image ranking will retry automatically."
        from .image_ranking import ClipLoadError
        if isinstance(error, ClipLoadError) and reason != "Low system memory":
            message += " " + str(error)
        self.emit("retrieval", {"phase": "error", "found": found, "downloaded": downloaded,
                                "rejected": 1, "accepted": accepted,
                                "threshold": self.settings.relevance_threshold, "best_score": None,
                                "failures": {reason: 1}, "message": message})

    async def _wait_for_search_retry(self, generation, topic):
        # New subjects wake retry immediately. Healthy empty pages keep the
        # configured cadence; unavailable models/sources use a longer backoff.
        end = time.monotonic() + self._search_retry_delay
        while self.topic_queue.empty() and time.monotonic() < end and not self.stop_event.is_set():
            await asyncio.sleep(0.1)
        if not self.stop_event.is_set() and generation == self._generation and self.topic_queue.empty():
            self.topic_queue.put_nowait((generation, topic))

    def _get_feedback_store(self):
        with self._feedback_store_lock:
            if self._feedback_store is None:
                from .feedback import FeedbackStore
                self._feedback_store = FeedbackStore(self.settings)
            return self._feedback_store

    def get_feedback_vote(self, topic_snapshot, candidate):
        try:
            return self._get_feedback_store().get_vote(topic_snapshot, candidate)
        except Exception as exc:
            log.warning("Local image feedback lookup failed (%s)", type(exc).__name__)
            return None

    def record_feedback(self, candidate, positive, topic_snapshot):
        """Queue a local vote using the immutable context shown at click time."""
        request_id = uuid4().hex
        if (not isinstance(candidate, ImageCandidate) or not isinstance(topic_snapshot, Topic)
                or not topic_snapshot.query or not isinstance(positive, bool)):
            self._emit_feedback_async(request_id, candidate, positive,
                                      "A visible image and accepted story context are needed to save feedback.")
            return request_id
        start_worker = False
        with self._feedback_request_lock:
            full = len(self._feedback_requests) >= 32
            if not full:
                self._feedback_requests.append((request_id, replace(candidate), positive, topic_snapshot, self._ranker))
                if not self._feedback_worker_running:
                    self._feedback_worker_running = True
                    start_worker = True
        if full:
            self._emit_feedback_async(request_id, candidate, positive,
                                      "Local feedback is busy. Please try saving this vote again shortly.")
        elif start_worker:
            # Drain votes even after Stop. Exit when empty so retired workers
            # do not retain old native models or complete book sessions.
            threading.Thread(target=self._save_feedback_requests, daemon=True, name="sis-image-feedback").start()
        return request_id

    def _emit_feedback_async(self, request_id, candidate, positive, message):
        # A direct GUI-thread callback could arrive before the caller stores
        # its request ID. Worker-thread delivery queues that acknowledgment.
        threading.Thread(target=self._emit_feedback, args=(request_id, candidate, positive),
                         kwargs={"success": False, "message": message},
                         daemon=True, name="sis-feedback-ack").start()

    def _save_feedback_requests(self):
        while True:
            with self._feedback_request_lock:
                if not self._feedback_requests:
                    self._feedback_worker_running = False
                    return
                request_id, candidate, positive, topic, ranker = self._feedback_requests.popleft()
            try:
                store = self._get_feedback_store()
                # Save the label before optional native work. This upsert is
                # durable even if visual features cannot be prepared.
                result = store.record(topic, candidate, positive)
            except Exception as exc:
                log.warning("Local image feedback save failed (%s)", type(exc).__name__)
                self._emit_feedback(request_id, candidate, positive, success=False,
                                    message="Feedback could not be saved locally. Please try again.")
                continue
            features_failed = False
            try:
                extract = getattr(ranker, "feedback_vectors", None)
                vectors = None
                if extract and not self.stop_event.is_set():
                    with self._native_workers_lock:
                        worker = self._native_workers.get("ranking")
                    loop = self._loop
                    if worker is not None and loop is not None and loop.is_running():
                        pending = asyncio.run_coroutine_threadsafe(worker.run(extract, topic, candidate), loop)
                        vectors = pending.result()
                    else:
                        vectors = extract(topic, candidate)
                if vectors is not None:
                    result = store.record(topic, candidate, positive, vectors=vectors)
            except Exception as exc:
                features_failed = True
                log.warning("Local image feedback features unavailable (%s)", type(exc).__name__)
            message = ("Saved locally with visual features for relevance learning." if result.get("learned") else
                       "Saved locally; visual features were unavailable for this vote.")
            if features_failed:
                message = "Feedback saved locally. Visual learning could not be updated for this vote."
            self._emit_feedback(request_id, candidate, positive, success=True, result=result,
                                message=message, features_failed=features_failed)

    def _emit_feedback(self, request_id, candidate, positive, *, success, message, result=None, features_failed=False):
        result = result or {}
        self.emit("feedback", {"request_id": request_id, "context_key": result.get("context_key", ""),
                               "image_url": getattr(candidate, "image_url", ""),
                               "vote": positive if isinstance(positive, bool) else None,
                               "count": result.get("count", 0),
                               "learned": bool(result.get("learned")) and not features_failed,
                               "success": success, "message": message})

    def reject_image(self, image):
        # Qt may reject a raster that passed the download decoder. The GUI
        # only records its URL; the search worker frees the slot and refills.
        # It was never displayed, so persistent viewing history stays intact.
        with self._history_lock:
            self._rejected_image_urls.add(image.image_url)

    def record_displayed(self, image):
        with self._history_lock:
            if re.fullmatch(r"[a-f0-9]{16}", image.phash or ""):
                self._recent_hashes.add(image.phash)
            self._displayed_urls.add(image.image_url)
        cache = getattr(self, "_cache", None)
        if cache:
            try:
                cache.record(image)
            except Exception:
                pass  # may race shutdown; never disrupt presentation
        else:
            with_cache = Cache(self.settings)
            try:
                with_cache.record(image)
            finally:
                with_cache.close()

    def clear_cache(self):
        if self.running:
            raise RuntimeError("Stop monitoring before clearing the cache.")
        with_cache = Cache(self.settings)
        try:
            with_cache.clear()
        finally:
            with_cache.close()

    def clear_history(self):
        if self.running:
            raise RuntimeError("Stop monitoring before clearing session history.")
        with_cache = Cache(self.settings)
        try:
            with_cache.clear_history()
        finally:
            with_cache.close()
        with self._history_lock:
            self._recent_hashes.clear()
            self._displayed_urls.clear()

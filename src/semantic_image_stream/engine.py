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
from .models import ImageCandidate
from .semantic_analysis import TopicGate, build_analyzer, fallback_queries, sanitize_query, tokens

log = logging.getLogger(__name__)
GPU_LOCK = threading.Lock()


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
    analyzer_checkpoint: dict | None = None
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
        # Runtime receives a snapshot, so changing controls never races workers.
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
        self._query_pages = {}
        self._refresh_queries = set()
        self._last_query_failed = False
        self._source_retry_at = {}
        self._protected_paths = deque(maxlen=120)
        self._history_lock = threading.Lock()
        self._generation = 0
        self._last_search = 0.0
        self._latest_topic = self.session_state.topic
        self._approved_images = {}
        self._approved_scope = None
        self._approved_generation = None
        self._last_fill_signature = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def emit(self, name, value=None):
        try:
            self.on_event(name, value)
        except Exception:
            log.exception("Event consumer failed for %s", name)

    def start(self):
        if self.running:
            return
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
        providers = build_providers(self.settings)
        cache = Cache(self.settings)
        downloader = ImageDownloader(self.settings)
        self._cache = cache
        with self._history_lock:
            self._recent_hashes.update(cache.seen_phashes())
            self._displayed_urls.update(cache.seen_urls())
        self.capture_done = False
        with self._segments_lock:
            capture_offset = max((segment.end for segment in self._segments), default=0.0)
            previous_segments = list(self._segments)
        if previous_segments:
            self.emit("transcript", previous_segments)
        if self._latest_topic:
            self._last_fill_signature = self._fill_signature(self._latest_topic)
            self._emit_context("ready", "Retained story context is ready.")
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
                 asyncio.create_task(self._search(providers, downloader, ranker, cache))]
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
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            self.emit("status", "Recognizing captured audio locally…")
            segments = await _background(self._transcriber.transcribe, chunk)
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
        failed_count = -1
        retry_at = 0.0
        while not self.stop_event.is_set():
            with self._segments_lock:
                window = list(self._segments)
            count = len(window)
            if (window and count != last_count
                    and (count != failed_count or time.monotonic() >= retry_at)):
                self._emit_context("updating", "Updating local story understanding from all recorded text.")
                topic, changed, refill = None, False, False
                try:
                    if analyzer is None:
                        analyzer = self._session_analyzer()
                    topic = await _background(self._analyze_session, analyzer, window)
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
                    last_count = count
                    failed_count = -1
                    if topic:
                        topic = replace(topic, transcript=tuple(segment.text for segment in window))
                        with self._segments_lock:
                            if self.stop_event.is_set():
                                break
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
                        message = ("Local story understanding updated." if topic.query else
                                   "No grounded visual subject in this passage yet.")
                        self._emit_context("ready", message, playback_changed=playback_changed)
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
                # Adding non-factual illustrations need not replay hours of a
                # valid saved book. The new analyzer must still verify the old
                # checkpoint digest and every accepted fact against raw text.
                legacy_pack_addition = len(key) == 4 and state.provider_key == key[:3]
                matching = exact_key or legacy_pack_addition
                if state.book_context != self.settings.book_context:
                    state.book_context = self.settings.book_context
                    changed = True
                if state.analyzer is not None and exact_key:
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
                analyzer = build_analyzer(
                    self.settings, stop_event=self.stop_event,
                    on_status=lambda message: self._emit_context("updating", message))
                if matching and self.settings.semantic_provider == "ollama":
                    restore = getattr(analyzer, "restore_checkpoint", None)
                    restored = False
                    if checkpoint is not None and restore:
                        try:
                            restored = restore(checkpoint, list(state.segments)) is True
                        except Exception as exc:
                            log.warning("Local story checkpoint restore failed (%s)", type(exc).__name__)
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
                self._latest_topic = state.topic
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
            bind = getattr(analyzer, "bind", None)
            if bind:
                bind(self.stop_event, lambda message: self._emit_context("updating", message))
            if self.settings.semantic_provider == "ollama":
                return analyzer.analyze(segments, self.stop_event)
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
        self._approved_images = {url: image for url, image in self._approved_images.items()
                                 if image.path.is_file() and url not in seen_urls
                                 and not is_duplicate(image.phash, seen_hashes)}

    def _emit_context(self, phase, message, playback_changed=False):
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
                              "model": getattr(self.settings, "context_model", "")
                              if self.settings.semantic_provider == "ollama" else "",
                              "method": self.settings.semantic_provider,
                              "summary_method": getattr(analyzer, "summary_method", ""), "message": message})

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
            delay = self.settings.search_interval - (time.monotonic() - self._last_search)
            if delay > 0:
                await asyncio.sleep(delay)
            # Collapse transitions queued during the provider cooldown.
            while not self.topic_queue.empty():
                generation, topic = self.topic_queue.get_nowait()
            self._last_search = time.monotonic()
            self._sync_image_pool(self._latest_topic or topic)
            if len(self._approved_images) >= self.settings.prefetch_count:
                continue
            query = sanitize_query(topic.query)
            query_basis = topic
            is_fallback = False
            tried = {query.casefold()}
            remaining = 2
            rejected_urls = {}
            attempt_failures = Counter()
            attempt_stats = {"found": 0, "downloaded": 0, "best_score": None}
            published = False
            ranking_failed = False
            while not self.stop_event.is_set():
                page_key = query.casefold()
                page = self._query_pages.get(page_key, 0)
                previous_failures = attempt_failures.copy()
                candidates = await self._query_candidates(query, providers, cache, page=page,
                                                          refresh=page_key in self._refresh_queries)
                failed_query = self._last_query_failed
                self._refresh_queries.discard(page_key)
                if failed_query:
                    # A blocked page is not an exhausted page. Retry it after
                    # source backoff, retaining successful partial results.
                    self._refresh_queries.add(page_key)
                    attempt_failures["Source unavailable"] += 1
                published, ranking_failed = await self._evaluate_candidates(
                    topic, candidates, query_basis if is_fallback else None,
                    query if is_fallback else None, downloader, ranker, cache, rejected_urls, attempt_failures, attempt_stats)
                transient_page = failed_query or any(attempt_failures[reason] > previous_failures[reason]
                    for reason in ("Source temporarily rate-limited", "Source unavailable", "Network error", "Image download timed out"))
                if not transient_page and not ranking_failed:
                    self._query_pages[page_key] = (page + 1) % 10
                    if page == 9:
                        self._refresh_queries.add(page_key)
                if len(self._query_pages) > 200:
                    oldest = next(iter(self._query_pages))
                    self._query_pages.pop(oldest)
                    self._refresh_queries.discard(oldest)
                # Raw source hits are not usable matches. Try the remaining
                # subjects only after validation/ranking exhausts this batch.
                # A failed local ranker instead retains its explicit diagnostic.
                if ranking_failed or len(self._approved_images) >= self.settings.prefetch_count or not remaining:
                    break
                basis = self._latest_topic or topic
                if not basis.query:
                    break
                alternatives = [sanitize_query(q) for q in fallback_queries(basis)]
                alternatives = [q for q in alternatives if q and q.casefold() not in tried]
                if not alternatives:
                    break
                query = alternatives[0]
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
            transient = any(attempt_failures[reason] for reason in ("Source temporarily rate-limited", "Source unavailable",
                                                                  "Network error", "Image download timed out"))
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
            if ranking_failed or not self._approved_images or (transient and len(self._approved_images) < self.settings.prefetch_count):
                if not ranking_failed:
                    message = ("Keeping ready images while unavailable sources retry." if self._approved_images else
                               "No new images met the relevance and quality filters. See image status for details.")
                    self.emit("status", message)
                # New speech topics wake retry immediately rather than waiting
                # behind the old topic's fixed thirty-second failure delay.
                await self._wait_for_search_retry(generation, topic)
            cache.prune(set(self._protected_paths))

    async def _evaluate_candidates(self, topic, candidates, fallback_topic, fallback_query,
                                   downloader, ranker, cache, rejected_urls, attempt_failures, attempt_stats):
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
            seen_urls = set(self._displayed_urls)
        unique = [candidate for url, candidate in by_url.items()
                  if url not in self._approved_images and url not in seen_urls]
        attempt_stats["found"] += len(unique)
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
                    message = "No images found for this query."
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
                        return cached, None
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
        tasks = [asyncio.create_task(prepare(c)) for c in unique]
        try:
            # Publish useful validated images as they arrive. Waiting for
            # every slow/broken URL made an otherwise working preview blank.
            for completed in asyncio.as_completed(tasks):
                image, reason = await completed
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
                        batch = (await _background(ranker.rank, current, to_rank, recent)) if current.query else []
                    except Exception as exc:
                        ranking_failed = True
                        self._report_ranking_failure(ranker, exc, len(unique), downloaded, len(published))
                        break
                    latest = self._latest_topic or topic
                    if (generation != self.session_state.playback_generation
                            or self._scope_changed(scope, self._visual_scope(latest))):
                        # An obsolete scene cannot approve images for its
                        # successor. Summary/transcript additions within the
                        # same scene must not starve a completed slow ranking.
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
                    displayed_urls = set(self._displayed_urls)
                    hashes = list(self._recent_hashes)
                for candidate in ranked:
                    other_hashes = [approved.phash for url, approved in self._approved_images.items()
                                    if url != candidate.image_url]
                    if (candidate.image_url not in displayed_urls and candidate.image_url not in self._approved_images
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

    async def _query_candidates(self, query, providers, cache, page=0, refresh=False):
        self._last_query_failed = False
        identity = [query.casefold(), self.settings.providers, self.settings.results_per_provider]
        if "duckduckgo" in self.settings.providers:
            # Older combined cache entries can contain only the other sources'
            # partial results after DDGS's malformed HTTP/2 header failed.
            # Refresh those queries once while retaining image/history caches.
            identity.append("duckduckgo-http2-v1")
        if page:
            identity.append({"page": page})
        key = json.dumps(identity)
        cached = cache.get_query(key)
        if cached is not None and not refresh:
            return cached
        self.emit("status", "Retrieving existing images from selected sources…")
        async def search(provider):
            if time.monotonic() < self._source_retry_at.get(provider.name, 0):
                return []
            if not page:
                return await provider.search(query, self.settings.results_per_provider)
            paged = getattr(provider, "search_page", None)
            return await paged(query, self.settings.results_per_provider, page) if paged else []
        results = await asyncio.gather(*(search(p) for p in providers),
                                       return_exceptions=True)
        candidates = []
        succeeded = True
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                succeeded = False
                self._last_query_failed = True
                response = getattr(result, "response", None) or getattr(getattr(result, "__cause__", None), "response", None)
                if response is not None and response.status_code == 429:
                    delay = 30.0
                    value = response.headers.get("Retry-After", "")
                    try:
                        delay = max(delay, float(value)) if value.isdigit() else max(delay, parsedate_to_datetime(value).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                    self._source_retry_at[provider.name] = time.monotonic() + delay
                self.emit("status", f"{provider.name} unavailable ({type(result).__name__}); trying other sources.")
            else:
                candidates.extend(result)
        # A genuine zero-result query is cached too, preventing unchanged
        # narration from repeatedly hitting the same sources. Availability
        # errors are never treated as evidence that a subject has no images.
        if candidates or succeeded:
            cache.put_query(key, candidates)
        return candidates

    def _report_ranking_failure(self, ranker, error, found, downloaded, accepted=0):
        # Exception text can include file paths, URLs or private model inputs.
        # The class name and selected ranking method give a safe diagnostic.
        name = {"clip": "CLIP", "context_clip": "context CLIP", "learned_context": "trained relevance",
                "metadata": "metadata"}.get(ranker.name, "selected")
        reason = "Local image ranking failed"
        log.warning("Image ranking failed (%s)", type(error).__name__)
        message = (f"Local {name} image ranking failed ({type(error).__name__}). "
                   "Speech monitoring continues; image ranking will retry.")
        from .image_ranking import ClipLoadError
        if isinstance(error, ClipLoadError):
            message += " " + str(error)
        self.emit("retrieval", {"phase": "error", "found": found, "downloaded": downloaded,
                                "rejected": 1, "accepted": accepted,
                                "threshold": self.settings.relevance_threshold, "best_score": None,
                                "failures": {reason: 1}, "message": message})

    async def _wait_for_search_retry(self, generation, topic):
        # New subjects wake a failed visual batch; unchanged subjects retain
        # the existing bounded retry delay so a broken local model cannot spin.
        end = time.monotonic() + max(30, self.settings.search_interval)
        while self.topic_queue.empty() and time.monotonic() < end and not self.stop_event.is_set():
            await asyncio.sleep(0.1)
        if not self.stop_event.is_set() and generation == self._generation and self.topic_queue.empty():
            self.topic_queue.put_nowait((generation, topic))

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

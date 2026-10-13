"""Independent, bounded capture/transcription/semantic/search clocks."""
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
import asyncio
import json
import logging
import queue
import re
import shutil
import threading
import time

from .cache import Cache
from .image_ranking import build_ranker, is_duplicate
from .models import ImageCandidate
from .semantic_analysis import TopicGate, build_analyzer, fallback_queries, sanitize_query, tokens

log = logging.getLogger(__name__)
GPU_LOCK = threading.Lock()


@dataclass
class SessionState:
    """In-memory story context shared by successive monitoring runs."""
    segments: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    analyzer: object | None = None
    provider_key: tuple = ()
    topic: object | None = None
    playback_scope: tuple | None = None
    playback_generation: int = 0


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
        self._recent_hashes = deque(maxlen=100)
        self._protected_paths = deque(maxlen=120)
        self._history_lock = threading.Lock()
        self._generation = 0
        self._last_search = 0.0
        self._latest_topic = self.session_state.topic

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
        self.capture_done = False
        with self._segments_lock:
            capture_offset = max((segment.end for segment in self._segments), default=0.0)
            previous_segments = list(self._segments)
        if previous_segments:
            self.emit("transcript", previous_segments)
        if self._latest_topic:
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
                    self._segments.extend(segments)
                    window = list(self._segments)
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
                topic, changed = None, False
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
                        playback_changed = self._update_playback_scope(topic)
                        changed = gate.changed(topic) or playback_changed
                        # Every successful context update affects in-flight
                        # ranking, even if its compact search query is stable.
                        self._latest_topic = topic
                        self.session_state.topic = topic
                        message = ("Local story understanding updated." if topic.query else
                                   "No grounded visual subject in this passage yet.")
                        self._emit_context("ready", message, playback_changed=playback_changed)
                    else:
                        changed = False
                if topic and failed_count == -1 and changed:
                    self._latest_topic = topic
                    self._generation += 1
                    gate.accept(topic)
                    self.emit("topic", topic)
                    while not self.topic_queue.empty():
                        self.topic_queue.get_nowait()
                    if topic.query:
                        self.topic_queue.put_nowait((self._generation, topic))
            await asyncio.sleep(self.settings.semantic_interval)

    def _session_analyzer(self):
        key = (self.settings.semantic_provider, getattr(self.settings, "context_model", ""),
               getattr(self.settings, "book_context", ""))
        with self._segments_lock:
            if self.session_state.analyzer is None or self.session_state.provider_key != key:
                if self.session_state.provider_key and self.session_state.provider_key != key:
                    self.session_state.playback_generation += 1
                self.session_state.topic = None
                self._latest_topic = None
                self.session_state.playback_scope = None
                self.session_state.provider_key = key
                self.session_state.analyzer = None
                self.session_state.analyzer = build_analyzer(
                    self.settings, stop_event=self.stop_event,
                    on_status=lambda message: self._emit_context("updating", message))
            return self.session_state.analyzer

    def _analyze_session(self, analyzer, segments):
        # Native/local HTTP work can briefly outlive a cancelled asyncio task.
        # Serialize reuse and only rebind callbacks after that work has ended.
        while not self.stop_event.is_set():
            if self.session_state.analysis_lock.acquire(timeout=0.1):
                break
        else:
            return None
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
            self.session_state.analysis_lock.release()

    def _report_context_failure(self, error):
        from .narrative_context import NarrativeContextError
        log.warning("Story understanding failed (%s)", type(error).__name__)
        detail = str(error) if isinstance(error, NarrativeContextError) else f"Local story understanding failed ({type(error).__name__})."
        self._emit_context("error", f"{detail} Speech recording continues; context will retry.")

    def _update_playback_scope(self, topic):
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
        scope = (bool(topic.query), subjects, scene, polarity)
        previous = self.session_state.playback_scope

        def distance(first, second):
            return 1 - len(first & second) / max(1, len(first | second))

        changed = (previous is None or previous[0] != scope[0]
                   or (scope[0] and (distance(previous[1], subjects) >= 0.5
                                     or (scene and previous[2] and distance(previous[2], scene) >= 0.6)
                                     or (previous[3] if len(previous) > 3 else frozenset()) != polarity)))
        if changed:
            self.session_state.playback_scope = scope
            self.session_state.playback_generation += 1
        return changed

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
            generation, topic = await self.topic_queue.get()
            delay = self.settings.search_interval - (time.monotonic() - self._last_search)
            if delay > 0:
                await asyncio.sleep(delay)
            # Collapse transitions queued during the provider cooldown.
            while not self.topic_queue.empty():
                generation, topic = self.topic_queue.get_nowait()
            self._last_search = time.monotonic()
            query = sanitize_query(topic.query)
            query_basis = topic
            is_fallback = False
            tried = {query.casefold()}
            remaining = 2
            rejected_urls = {}
            published = False
            ranking_failed = False
            while not self.stop_event.is_set():
                candidates = await self._query_candidates(query, providers, cache)
                published, ranking_failed = await self._evaluate_candidates(
                    topic, candidates, query_basis if is_fallback else None,
                    query if is_fallback else None, downloader, ranker, cache, rejected_urls)
                # Raw source hits are not usable matches. Try the remaining
                # subjects only after validation/ranking exhausts this batch.
                # A failed local ranker instead retains its explicit diagnostic.
                if published or ranking_failed or not remaining:
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
                                        "rejected": 0, "accepted": 0,
                                        "threshold": self.settings.relevance_threshold,
                                        "best_score": None, "failures": {},
                                        "message": "No usable matches yet; trying a shorter query from the current spoken context."})
            if not published:
                if not ranking_failed:
                    self.emit("status", "No new images met the relevance and quality filters. See image status for details.")
                # New speech topics wake retry immediately rather than waiting
                # behind the old topic's fixed thirty-second failure delay.
                await self._wait_for_search_retry(generation, topic)
            cache.prune(set(self._protected_paths))

    async def _evaluate_candidates(self, topic, candidates, fallback_topic, fallback_query,
                                   downloader, ranker, cache, rejected_urls):
        def ranking_topic():
            current = self._latest_topic or topic
            # Retain full context, concepts and kind, but score matches to
            # the shorter subject that actually found them. Once speech
            # changes, obsolete fallback wording must not bless old images.
            effective = replace(current, query=fallback_query) if current is fallback_topic else current
            return current, effective

        by_url = {}
        for candidate in candidates:
            by_url.setdefault(candidate.image_url, candidate)
        unique = list(by_url.values())
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
                message = ("Loading the local image relevance model; first use may download weights. "
                           "Speech monitoring continues.")
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
                    report("downloading", len(published), best_score)
                    continue
                downloaded += 1
                prepared.append(image)
                with self._history_lock:
                    recent = list(self._recent_hashes)
                # Topic changes should not discard an entire useful batch:
                # re-evaluate it against the CURRENT context before display.
                # Repeat if context changes while a local model is ranking.
                while True:
                    basis, current = ranking_topic()
                    if ranker.name in {"clip", "context_clip"} and getattr(ranker, "model", None) is None:
                        report("loading", len(published), best_score)
                    try:
                        ranked = (await _background(ranker.rank, current, prepared, recent)) if current.query else []
                    except Exception as exc:
                        ranking_failed = True
                        self._report_ranking_failure(ranker, exc, len(unique), downloaded, len(published))
                        break
                    if basis is (self._latest_topic or topic):
                        break
                if ranking_failed:
                    break
                best_score = max((c.score for c in ranked), default=None)
                chosen, hashes = [], list(recent)
                for candidate in ranked:
                    if (candidate.score >= self.settings.relevance_threshold
                            and not is_duplicate(candidate.phash, hashes)):
                        chosen.append(candidate)
                        hashes.append(candidate.phash)
                        if len(chosen) >= self.settings.prefetch_count:
                            break
                selection = (current.query, *(c.image_url for c in chosen))
                if chosen and selection != last_selection:
                    last_selection = selection
                    self._protected_paths.extend(c.local_path for c in chosen)
                    published.update(c.image_url for c in chosen)
                    self.emit("images", chosen)
                    self.emit("status", f"Ready: {len(chosen)} relevant images · {ranker.name} ranking")
                report("ready" if published else "downloading", len(published), best_score)
                if len(published) >= self.settings.prefetch_count:
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if not ranking_failed:
            report("ready" if published else "empty", len(published), best_score)
        return bool(published), ranking_failed

    async def _query_candidates(self, query, providers, cache):
        key = json.dumps([query.casefold(), self.settings.providers, self.settings.results_per_provider])
        cached = cache.get_query(key)
        if cached is not None:
            return cached
        self.emit("status", "Retrieving existing images from selected sources…")
        results = await asyncio.gather(*(p.search(query, self.settings.results_per_provider) for p in providers),
                                       return_exceptions=True)
        candidates = []
        succeeded = True
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                succeeded = False
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
        name = {"clip": "CLIP", "context_clip": "context CLIP", "metadata": "metadata"}.get(ranker.name, "selected")
        reason = "Local image ranking failed"
        log.warning("Image ranking failed (%s)", type(error).__name__)
        message = (f"Local {name} image ranking failed ({type(error).__name__}). "
                   "Speech monitoring continues; image ranking will retry.")
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
            self._recent_hashes.append(image.phash)
        cache = getattr(self, "_cache", None)
        if cache:
            try:
                cache.record(image)
            except Exception:
                pass  # may race shutdown; never disrupt presentation

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

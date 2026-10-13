"""Independent, bounded capture/transcription/semantic/search clocks."""
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
import asyncio
import json
import logging
import queue
import shutil
import threading
import time

from .cache import Cache
from .image_ranking import build_ranker, is_duplicate
from .models import ImageCandidate
from .semantic_analysis import TopicGate, build_analyzer, sanitize_query

log = logging.getLogger(__name__)
GPU_LOCK = threading.Lock()


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
    def __init__(self, settings, on_event):
        # Runtime receives a snapshot, so changing controls never races workers.
        self.settings = replace(settings, providers=list(settings.providers)).validate()
        self.on_event = on_event
        self.stop_event = threading.Event()
        self._thread = None
        self._capture = None
        self._transcriber = None
        self._segments = deque(maxlen=300)
        self._segments_lock = threading.Lock()
        self._recent_hashes = deque(maxlen=100)
        self._protected_paths = deque(maxlen=120)
        self._history_lock = threading.Lock()
        self._generation = 0
        self._last_search = 0.0
        self._latest_topic = None

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
        self._segments.clear()
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
        analyzer = build_analyzer(self.settings)
        ranker = build_ranker(self.settings, GPU_LOCK)
        providers = build_providers(self.settings)
        cache = Cache(self.settings)
        downloader = ImageDownloader(self.settings)
        self._cache = cache
        self.capture_done = False

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
                self.audio_queue.put_nowait(chunk)
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
            segments = await _background(self._transcriber.transcribe, chunk)
            if self.stop_event.is_set():
                break
            if segments:
                with self._segments_lock:
                    self._segments.extend(segments)
                    cutoff = segments[-1].end - self.settings.transcript_window_seconds
                    while self._segments and self._segments[0].end < cutoff:
                        self._segments.popleft()
                    window = list(self._segments)
                self.emit("transcript", window)

    async def _interpret(self, analyzer):
        gate = TopicGate(self.settings.semantic_sensitivity)
        last_end = -1
        while not self.stop_event.is_set():
            with self._segments_lock:
                window = list(self._segments)
            if window and window[-1].end != last_end:
                last_end = window[-1].end
                topic = await _background(analyzer.analyze, window)
                if topic and gate.changed(topic):
                    self._latest_topic = topic
                    self._generation += 1
                    gate.accept(topic)
                    self.emit("topic", topic)
                    if self.topic_queue.full():
                        self.topic_queue.get_nowait()
                    self.topic_queue.put_nowait((self._generation, topic))
            await asyncio.sleep(self.settings.semantic_interval)

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
            key = json.dumps([query.casefold(), self.settings.providers, self.settings.results_per_provider])
            candidates = cache.get_query(key)
            if candidates is None:
                self.emit("status", "Retrieving existing images from selected sources…")
                results = await asyncio.gather(*(p.search(query, self.settings.results_per_provider) for p in providers),
                                               return_exceptions=True)
                candidates = []
                for provider, result in zip(providers, results):
                    if isinstance(result, Exception):
                        self.emit("status", f"{provider.name} unavailable ({type(result).__name__}); trying other sources.")
                    else:
                        candidates.extend(result)
                if candidates:
                    cache.put_query(key, candidates)
            by_url = {}
            for candidate in candidates:
                by_url.setdefault(candidate.image_url, candidate)
            unique = list(by_url.values())
            # Rank metadata before prefetch to start with promising source
            # images; retain other candidates for actual multimodal ranking.
            if ranker.name == "metadata":
                preferred = ranker.rank(self._latest_topic or topic, unique, [])
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
                        message = "No images found for this topic. Waiting for the next subject or retry."
                    elif not downloaded:
                        details = "; ".join(f"{count} {reason.lower()}" for reason, count in failures.items())
                        message = f"Found {len(unique)} results, but no images could be downloaded. {details}"
                    elif best_score is not None and best_score < self.settings.relevance_threshold:
                        message = (f"Downloaded {downloaded} images. Best relevance {best_score:.2f} is below "
                                   f"the {self.settings.relevance_threshold:.2f} minimum.")
                    else:
                        message = f"Downloaded {downloaded} images, but none are new matches for the current discussion."
                else:
                    message = f"Checking images · {downloaded} downloaded from {len(unique)} results"
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
                    try:
                        image = await downloader.download(candidate)
                        cache.put_image(image)
                        return image, None
                    except Exception as exc:
                        log.debug("Rejected image (%s)", type(exc).__name__)
                        return None, getattr(exc, "reason", "Download or validation failed")

            prepared = []
            best_score = None
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
                        current = self._latest_topic or topic
                        ranked = await _background(ranker.rank, current, prepared, recent)
                        if current is (self._latest_topic or topic):
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
            if published:
                report("ready", len(published), best_score)
            else:
                report("empty", best_score=best_score)
                self.emit("status", "No new images met the relevance and quality filters. See image status for details.")
                # New speech topics wake retry immediately rather than waiting
                # behind the old topic's fixed thirty-second failure delay.
                end = time.monotonic() + max(30, self.settings.search_interval)
                while self.topic_queue.empty() and time.monotonic() < end and not self.stop_event.is_set():
                    await asyncio.sleep(0.1)
                if generation == self._generation and self.topic_queue.empty():
                    self.topic_queue.put_nowait((generation, topic))
            cache.prune(set(self._protected_paths))

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

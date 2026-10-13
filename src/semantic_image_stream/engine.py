"""Independent, bounded capture/transcription/semantic/search clocks."""
from collections import deque
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
            if generation != self._generation:
                continue
            by_url = {}
            for candidate in candidates:
                by_url.setdefault(candidate.image_url, candidate)
            unique = list(by_url.values())
            semaphore = asyncio.Semaphore(3)

            async def prepare(candidate):
                async with semaphore:
                    if generation != self._generation:
                        return None
                    cached = cache.get_image(candidate.image_url)
                    if cached:
                        # Reapply current geometry filters to cached images.
                        ratio = cached.width / max(1, cached.height)
                        if (cached.width >= self.settings.min_width and cached.height >= self.settings.min_height
                                and cached.width * cached.height <= self.settings.max_image_pixels
                                and cached.path.stat().st_size <= self.settings.max_download_bytes
                                and self.settings.min_aspect_ratio <= ratio <= self.settings.max_aspect_ratio):
                            return cached
                    try:
                        image = await downloader.download(candidate)
                        cache.put_image(image)
                        return image
                    except Exception as exc:
                        log.debug("Rejected image (%s)", type(exc).__name__)
                        return None

            prepared = [c for c in await asyncio.gather(*(prepare(c) for c in unique)) if c is not None]
            with self._history_lock:
                recent = list(self._recent_hashes)
            ranked = await _background(ranker.rank, topic, prepared, recent)
            if generation != self._generation:
                continue
            chosen, hashes = [], list(recent)
            for image in ranked:
                if image.score >= self.settings.relevance_threshold and not is_duplicate(image.phash, hashes):
                    chosen.append(image)
                    hashes.append(image.phash)
                    if len(chosen) >= self.settings.prefetch_count:
                        break
            if chosen:
                self._protected_paths.extend(c.local_path for c in chosen)
                self.emit("images", chosen)
                self.emit("status", f"Ready: {len(chosen)} relevant images · {ranker.name} ranking")
            else:
                self.emit("status", "No new images met the relevance and quality filters. Keeping the current slide.")
                # Bounded retries allow temporary failures without getting stuck
                # indefinitely on an accepted topic. New topics take precedence.
                await asyncio.sleep(max(30, self.settings.search_interval))
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

"""Offline integration of real file decoding and engine scheduling.

Only inference and external network boundaries are substituted here. Live
recording/model/provider validation is a separate opt-in script.
"""
import asyncio
from dataclasses import replace
import json
import queue
import threading
import time
import wave

import numpy as np
import pytest

from semantic_image_stream.audio_capture import FileCapture
from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import AudioChunk, ImageCandidate, Topic, TranscriptSegment


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [["wikimedia", "duckduckgo"], ["wikimedia"]])
async def test_duckduckgo_header_fix_refreshes_old_partial_searches_once_without_clearing_history(tmp_path, selected):
    settings = Settings(cache_dir=str(tmp_path), providers=selected)
    pipeline = Pipeline(settings, lambda *args: None)
    cache = Cache(settings)
    old = ImageCandidate(image_url="https://photos.example.org/old.jpg", source_url="https://museum.example.org/old",
                         title="Previous partial result", provider="wikimedia")
    legacy_key = json.dumps(["earth orbit", selected, settings.results_per_provider])
    cache.put_query(legacy_key, [old])
    cache.record(old)
    calls = []

    class Provider:
        def __init__(self, name):
            self.name = name

        async def search(self, query, limit):
            calls.append(self.name)
            return [ImageCandidate(image_url=f"https://photos.example.org/{self.name}.jpg",
                                   source_url=f"https://museum.example.org/{self.name}",
                                   title="Current result", provider=self.name)]

    providers = [Provider(name) for name in selected]
    try:
        result = await pipeline._query_candidates("Earth orbit", providers, cache)
        if "duckduckgo" in selected:
            assert calls == selected
            assert {candidate.provider for candidate in result} == set(selected)
            assert old.image_url not in {candidate.image_url for candidate in result}
        else:
            assert not calls
            assert [candidate.image_url for candidate in result] == [old.image_url]
        before = list(calls)
        repeated = await pipeline._query_candidates("Earth orbit", providers, cache)
        assert calls == before
        assert repeated == result
        assert cache.get_query(legacy_key) == [old]
        assert cache.history() == [old]
    finally:
        cache.close()


def test_pipeline_file_to_ranked_images_with_provider_failure(tmp_path, monkeypatch):
    recording = tmp_path / "input.wav"
    samples = (np.sin(np.arange(16000 * 2) * 2 * np.pi * 240 / 16000) * 12000).astype("<i2")
    with wave.open(str(recording), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(samples.tobytes())
    settings = Settings(audio_mode="file", audio_file=str(recording), cache_dir=str(tmp_path / "cache"),
                        chunk_seconds=2, semantic_interval=0.03, search_interval=0.03,
                        providers=["wikimedia", "openverse"])
    monkeypatch.setattr("semantic_image_stream.audio_capture.create_capture", lambda s: FileCapture(s, realtime=False))

    class Transcriber:
        def __init__(self, *args):
            pass

        def transcribe(self, chunk):
            assert chunk.sample_rate == 16000 and len(chunk.samples) <= 32000
            return [TranscriptSegment(chunk.start, chunk.start + 2,
                                      "Black holes curve spacetime and their event horizon prevents light from escaping.")]

        def stop(self):
            pass

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            assert len(query.split()) <= 8
            return [ImageCandidate("https://img.example/a.jpg", "https://source.example/a", "Black hole event horizon", "wikimedia", license="CC BY 4.0")]

        async def close(self):
            pass

    class BrokenProvider(Provider):
        name = "openverse"

        async def search(self, query, limit):
            raise RuntimeError("rate limited")

    class Downloader:
        def __init__(self, settings):
            self.root = tmp_path / "cache" / "images"
            self.root.mkdir(parents=True, exist_ok=True)

        async def download(self, image):
            from PIL import Image
            path = self.root / "verified.png"
            Image.new("RGB", (800, 600), "red").save(path)
            image.local_path, image.width, image.height, image.phash = str(path), 800, 600, "aabbccdd11223344"
            return image

        async def close(self):
            pass

    monkeypatch.setattr("semantic_image_stream.transcription.FasterWhisperTranscriber", Transcriber)
    monkeypatch.setattr("semantic_image_stream.search_providers.build_providers", lambda s: [BrokenProvider(), Provider()])
    monkeypatch.setattr("semantic_image_stream.image_download.ImageDownloader", Downloader)
    events = []
    ready = threading.Event()

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "images":
            ready.set()

    pipeline = Pipeline(settings, receive)
    pipeline.start()
    try:
        assert ready.wait(8), events
        image = next(data[0] for kind, data in events if kind == "images")
        assert image.path.exists()
        assert image.score >= settings.relevance_threshold
        assert image.license == "CC BY 4.0"
        assert any(kind == "transcript" for kind, _ in events)
        assert any(kind == "topic" for kind, _ in events)
        assert any(kind == "status" and "openverse unavailable" in text for kind, text in events)
        assert not any(kind == "error" for kind, _ in events)
        pipeline.record_displayed(image)
    finally:
        pipeline.stop()
        end = time.monotonic() + 3
        while pipeline.running and time.monotonic() < end:
            time.sleep(0.01)
    assert not pipeline.running
    assert events[-1][0] == "stopped"


def test_clip_ranking_failure_keeps_transcripts_and_topics_running(tmp_path, monkeypatch):
    """A failed visual model must not terminate the independent speech worker."""
    failed = threading.Event()
    second_failure = threading.Event()
    transcript_after_failure = threading.Event()
    topic_after_failure = threading.Event()
    slow_cancelled = threading.Event()
    events = []
    settings = Settings(audio_mode="microphone", audio_device=1, whisper_model="turbo", whisper_device="auto",
                        ranking="clip", clip_device="cpu", relevance_threshold=0.61,
                        cache_dir=str(tmp_path / "cache"), semantic_interval=0.02, search_interval=0.02)

    class Capture:
        def run(self, on_chunk, on_status, stopped):
            samples = np.zeros(16000, dtype=np.float32)
            on_chunk(AudioChunk(samples, 0))
            if failed.wait(3) and not stopped.is_set():
                on_chunk(AudioChunk(samples, 20))
            stopped.wait(6)

        def stop(self):
            pass

    class Transcriber:
        def __init__(self, *args):
            pass

        def transcribe(self, chunk):
            text = ("Black holes curve spacetime and their event horizon prevents light from escaping."
                    if chunk.start == 0 else
                    "Orchid pollination depends on insects carrying pollen between flowers and enabling plant reproduction.")
            return [TranscriptSegment(chunk.start, chunk.start + 1, text)]

        def stop(self):
            pass

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return [ImageCandidate(f"https://image.example/{kind}", "https://source.example/plate",
                                   "Scientific source plate", "wikimedia") for kind in ["fast", "slow"]]

        async def close(self):
            pass

    class Downloader:
        def __init__(self, settings):
            self.root = tmp_path / "cache" / "images"
            self.root.mkdir(parents=True, exist_ok=True)

        async def download(self, image):
            if image.image_url.endswith("slow"):
                try:
                    await asyncio.Event().wait()
                finally:
                    slow_cancelled.set()
            from PIL import Image
            path = self.root / "verified.png"
            Image.new("RGB", (800, 600), "red").save(path)
            image.local_path, image.width, image.height, image.phash = str(path), 800, 600, "aabbccdd11223344"
            return image

        async def close(self):
            pass

    class BrokenClip:
        name = "clip"
        model = None
        calls = 0

        def rank(self, *args):
            self.calls += 1
            raise OSError("private-ranking-detail https://private.invalid/?query=private")

    ranker = BrokenClip()
    monkeypatch.setattr("semantic_image_stream.audio_capture.create_capture", lambda settings: Capture())
    monkeypatch.setattr("semantic_image_stream.transcription.FasterWhisperTranscriber", Transcriber)
    monkeypatch.setattr("semantic_image_stream.search_providers.build_providers", lambda settings: [Provider()])
    monkeypatch.setattr("semantic_image_stream.image_download.ImageDownloader", Downloader)
    monkeypatch.setattr("semantic_image_stream.engine.build_ranker", lambda *args: ranker)

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "retrieval" and payload["phase"] == "error":
            if failed.is_set():
                second_failure.set()
            failed.set()
        elif failed.is_set() and kind == "transcript":
            transcript_after_failure.set()
        elif failed.is_set() and kind == "topic":
            topic_after_failure.set()

    pipeline = Pipeline(settings, receive)
    pipeline.start()
    try:
        assert failed.wait(3), events
        assert transcript_after_failure.wait(3), events
        assert topic_after_failure.wait(3), events
        assert second_failure.wait(3), "The new topic must wake the visual retry without the thirty-second delay"
        assert slow_cancelled.wait(1), "Remaining downloads in the failed batch must be cancelled"
        assert pipeline.running and not pipeline.stop_event.is_set()
        assert not any(kind in {"error", "stopped", "images"} for kind, _ in events)
        assert not any(kind == "retrieval" and payload["phase"] == "empty" for kind, payload in events)
        assert any(kind == "retrieval" and payload["phase"] == "loading" for kind, payload in events)
        errors = [payload for kind, payload in events if kind == "retrieval" and payload["phase"] == "error"]
        assert all("OSError" in payload["message"] and "Speech monitoring continues" in payload["message"] for payload in errors)
        assert all("private-ranking-detail" not in str(payload) for _, payload in events)
        assert pipeline.settings.ranking == "clip" and pipeline.settings.relevance_threshold == 0.61
        assert ranker.calls == 2
    finally:
        pipeline.stop()
        pipeline._thread.join(timeout=3)
    assert not pipeline.running
    assert events[-1][0] == "stopped"


@pytest.mark.asyncio
async def test_empty_recognition_reports_audio_received_and_continues(tmp_path):
    events = []
    empty = asyncio.Event()
    spoken = asyncio.Event()

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "status" and "no speech was recognized" in payload:
            empty.set()
        if kind == "status" and payload == "Spoken context updated · listening":
            spoken.set()

    class Transcriber:
        calls = 0

        def transcribe(self, chunk):
            self.calls += 1
            return [] if self.calls == 1 else [TranscriptSegment(1, 2, "Public fixture speech")]

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), receive)
    pipeline.audio_queue = queue.Queue()
    pipeline._transcriber = Transcriber()
    pipeline.audio_queue.put(AudioChunk(np.zeros(16000, dtype=np.float32), 0))
    task = asyncio.create_task(pipeline._transcribe())
    try:
        await asyncio.wait_for(empty.wait(), 2)
        assert not task.done() and not pipeline.stop_event.is_set()
        assert not any(kind == "transcript" for kind, _ in events)
        pipeline.audio_queue.put(AudioChunk(np.zeros(16000, dtype=np.float32), 1))
        await asyncio.wait_for(spoken.wait(), 2)
        assert any(kind == "transcript" for kind, _ in events)
        assert sum(kind == "status" and payload == "Recognizing captured audio locally…" for kind, payload in events) == 2
    finally:
        pipeline.stop_event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_full_session_keeps_more_than_three_hundred_segments_and_interprets_every_text(tmp_path):
    events = []
    pipeline = None

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "transcript" and len(payload) == 330:
            pipeline.stop_event.set()
        if kind == "context" and payload["phase"] == "ready":
            pipeline.stop_event.set()

    class Transcriber:
        calls = 0

        def transcribe(self, chunk):
            first, count = (0, 320) if self.calls == 0 else (320, 10)
            self.calls += 1
            return [TranscriptSegment(index * 6, index * 6 + 6, f"Recorded passage {index}")
                    for index in range(first, first + count)]

    class Analyzer:
        seen = None

        def analyze(self, segments):
            self.seen = list(segments)
            return Topic("Arctic expedition", "Full story", summary="An expedition crosses Arctic ice.",
                         scene="People on Arctic ice", context_used_segments=len(segments))

    settings = Settings(cache_dir=str(tmp_path), transcript_window_seconds=1, semantic_interval=0.01)
    pipeline = Pipeline(settings, receive)
    pipeline.audio_queue = queue.Queue()
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline._transcriber = Transcriber()
    for start in (0, 1920):
        pipeline.audio_queue.put(AudioChunk(np.zeros(16000, dtype=np.float32), start))
    await asyncio.wait_for(pipeline._transcribe(), 2)
    assert len(pipeline.session_state.segments) == 330
    assert pipeline.session_state.segments[0].text == "Recorded passage 0"
    assert pipeline.session_state.segments[-1].end > 120
    pipeline.stop_event.clear()
    analyzer = Analyzer()
    await asyncio.wait_for(pipeline._interpret(analyzer), 2)
    assert analyzer.seen == pipeline.session_state.segments
    assert pipeline._latest_topic.transcript == tuple(segment.text for segment in analyzer.seen)
    ready = [payload for kind, payload in events if kind == "context" and payload["phase"] == "ready"]
    assert ready[-1]["segments_recorded"] == ready[-1]["segments_used"] == 330


@pytest.mark.asyncio
async def test_same_query_updates_full_context_without_repeating_network_generation(tmp_path):
    events = []
    pipeline = None

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] == "ready":
            if payload["segments_recorded"] == 1:
                with pipeline._segments_lock:
                    pipeline._segments.append(TranscriptSegment(6, 12, "They search for shelter."))
            else:
                pipeline.stop_event.set()

    class Analyzer:
        def analyze(self, segments):
            return Topic("Arctic expedition", " ".join(segment.text for segment in segments),
                         summary=f"Story with {len(segments)} passages", scene="The expedition in Arctic snow",
                         visual_subjects=("Arctic snow",), context_used_segments=len(segments))

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), semantic_interval=0.01), receive)
    pipeline._segments.append(TranscriptSegment(0, 6, "The expedition reaches the Arctic."))
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    await asyncio.wait_for(pipeline._interpret(Analyzer()), 2)
    assert pipeline._generation == 1
    assert sum(kind == "topic" for kind, _ in events) == 1
    assert pipeline._latest_topic.summary == "Story with 2 passages"
    assert pipeline._latest_topic.transcript == ("The expedition reaches the Arctic.", "They search for shelter.")
    assert pipeline.session_state.topic is pipeline._latest_topic
    assert sum(kind == "context" and payload["phase"] == "ready" for kind, payload in events) == 2


@pytest.mark.asyncio
async def test_underfilled_set_refreshes_new_grounded_alternatives_with_stable_primary_and_scene(tmp_path):
    from dataclasses import replace
    from PIL import Image

    pipeline = None
    primary = "Arctic camp"
    initial = Topic(primary, "Full story", scene="Tents on Arctic ice", visual_subjects=("tents", "Arctic ice"),
                    queries=(primary, "old grounded alternative"))
    events = []

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] == "ready":
            count = payload["segments_recorded"]
            if count < 3:
                pipeline._segments.append(TranscriptSegment(count * 6, (count + 1) * 6, f"New passage {count}"))
            else:
                pipeline.stop_event.set()

    class Analyzer:
        def analyze(self, segments):
            queries = initial.queries if len(segments) == 1 else (primary, "new grounded alternative")
            return replace(initial, queries=queries, summary=f"Summary after {len(segments)} recorded passages.",
                           context_used_segments=len(segments))

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), semantic_interval=0.01, prefetch_count=4,
                                 relevance_threshold=0.40), receive)
    pipeline._latest_topic = initial
    pipeline._update_playback_scope(initial)
    pipeline._sync_image_pool(initial)
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (800, 600), "red").save(image_path)
    pipeline._approved_images = {str(index): ImageCandidate(str(index), "https://source.example/file", "Approved", "wikimedia",
                                                          local_path=str(image_path)) for index in range(2)}
    pipeline._segments.append(TranscriptSegment(0, 6, "Initial story passage"))
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    before = pipeline.session_state.playback_generation
    await asyncio.wait_for(pipeline._interpret(Analyzer()), 2)
    assert pipeline._generation == 2, "Only the initial underfilled request and new query alternative should schedule a search"
    generation, queued = pipeline.topic_queue.get_nowait()
    assert generation == 2 and queued.query == primary
    assert queued.queries == (primary, "new grounded alternative")
    assert pipeline.session_state.playback_generation == before
    assert len(pipeline._approved_images) == 2
    assert not any(kind == "topic" for kind, _ in events), "A refill must not invalidate the approved scene playlist"
    assert pipeline.settings.relevance_threshold == 0.40


def test_semantic_failure_keeps_capture_and_transcription_running(tmp_path, monkeypatch):
    failed, recovered = threading.Event(), threading.Event()
    events = []
    settings = Settings(cache_dir=str(tmp_path), semantic_interval=0.01, search_interval=0.01)

    class Capture:
        def run(self, on_chunk, on_status, stopped):
            samples = np.zeros(16000, dtype=np.float32)
            on_chunk(AudioChunk(samples, 0))
            if failed.wait(3) and not stopped.is_set():
                on_chunk(AudioChunk(samples, 1))
            stopped.wait(6)

        def stop(self):
            pass

    class Transcriber:
        def __init__(self, *args):
            pass

        def transcribe(self, chunk):
            return [TranscriptSegment(chunk.start, chunk.start + 1, "The expedition walks across Arctic snow.")]

        def stop(self):
            pass

    class Analyzer:
        calls = 0

        def analyze(self, segments):
            self.calls += 1
            if len(segments) == 1:
                raise RuntimeError("private-transcript-detail")
            return Topic("Arctic snow", "The expedition walks across Arctic snow.",
                         context_used_segments=len(segments))

    analyzer = Analyzer()
    monkeypatch.setattr("semantic_image_stream.audio_capture.create_capture", lambda _: Capture())
    monkeypatch.setattr("semantic_image_stream.transcription.FasterWhisperTranscriber", Transcriber)
    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", lambda settings, **kwargs: analyzer)
    monkeypatch.setattr("semantic_image_stream.search_providers.build_providers", lambda _: [])

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] == "error":
            failed.set()
        if kind == "context" and payload["phase"] == "ready":
            recovered.set()

    pipeline = Pipeline(settings, receive)
    pipeline.start()
    try:
        assert failed.wait(3), events
        assert recovered.wait(3), events
        assert pipeline.running and not pipeline.stop_event.is_set()
        assert len(pipeline.session_state.segments) == 2 and analyzer.calls == 2
        assert sum(kind == "transcript" for kind, _ in events) == 2
        assert not any(kind in {"error", "stopped"} for kind, _ in events)
        assert not any("private-transcript-detail" in str(payload) for _, payload in events)
    finally:
        pipeline.stop()
        pipeline._thread.join(timeout=3)
    assert not pipeline.running


def test_stop_start_retains_session_analyzer_and_increasing_capture_timestamps(tmp_path, monkeypatch):
    state = SessionState(segments=[TranscriptSegment(0, 400, "An expedition began earlier.")])
    starts, built = [], []

    class Capture:
        def run(self, on_chunk, on_status, stopped):
            on_chunk(AudioChunk(np.zeros(16000, dtype=np.float32), 0))
            stopped.wait(6)

        def stop(self):
            pass

    class Transcriber:
        def __init__(self, *args):
            pass

        def transcribe(self, chunk):
            starts.append(chunk.start)
            return [TranscriptSegment(chunk.start, chunk.start + 1, "The expedition continues.")]

        def stop(self):
            pass

    class Analyzer:
        def analyze(self, segments):
            return Topic("Arctic expedition", "Story context", context_used_segments=len(segments))

    def create_analyzer(settings, **kwargs):
        analyzer = Analyzer()
        built.append(analyzer)
        return analyzer

    monkeypatch.setattr("semantic_image_stream.audio_capture.create_capture", lambda _: Capture())
    monkeypatch.setattr("semantic_image_stream.transcription.FasterWhisperTranscriber", Transcriber)
    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", create_analyzer)
    monkeypatch.setattr("semantic_image_stream.search_providers.build_providers", lambda _: [])
    settings = Settings(cache_dir=str(tmp_path), semantic_interval=0.01, search_interval=0.01)
    for expected_count in (2, 3):
        recognized = threading.Event()

        def receive(kind, payload):
            if kind == "transcript" and len(payload) == expected_count:
                recognized.set()

        pipeline = Pipeline(settings, receive, session_state=state)
        pipeline.start()
        try:
            assert recognized.wait(3)
            assert pipeline._segments is state.segments
        finally:
            pipeline.stop()
            pipeline._thread.join(timeout=3)
        assert not pipeline.running
    assert starts == [400, 401]
    assert [segment.end for segment in state.segments] == [400, 401, 402]
    assert len(built) == 1 and state.analyzer is built[0]


def test_resume_emits_retained_context_and_topic_without_changing_playback_generation(tmp_path, monkeypatch):
    text = "The explorers camp in the Arctic."
    retained = Topic("Arctic camp", text, transcript=(text,), context_used_segments=1)
    settings = Settings(cache_dir=str(tmp_path), semantic_interval=0.01, search_interval=0.01)

    class Analyzer:
        def analyze(self, segments):
            return retained

    class Capture:
        def run(self, on_chunk, on_status, stopped):
            stopped.wait(3)

        def stop(self):
            pass

    class Transcriber:
        def __init__(self, *args):
            pass

        def stop(self):
            pass

    state = SessionState(segments=[TranscriptSegment(0, 6, text)], analyzer=Analyzer(), topic=retained,
                         provider_key=(settings.semantic_provider, settings.context_model, settings.book_context),
                         playback_scope=(True, frozenset({"arctic", "camp"}), frozenset()), playback_generation=7)
    monkeypatch.setattr("semantic_image_stream.audio_capture.create_capture", lambda _: Capture())
    monkeypatch.setattr("semantic_image_stream.transcription.FasterWhisperTranscriber", Transcriber)
    monkeypatch.setattr("semantic_image_stream.search_providers.build_providers", lambda _: [])
    restored = threading.Event()
    events = []

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "topic":
            restored.set()

    pipeline = Pipeline(settings, receive, session_state=state)
    pipeline.start()
    try:
        assert restored.wait(3), events
        topics = [payload for kind, payload in events if kind == "topic"]
        assert topics == [retained]
        first_topic = next(index for index, (kind, _) in enumerate(events) if kind == "topic")
        context_index = next(index for index, (kind, payload) in enumerate(events)
                             if kind == "context" and payload["phase"] == "ready")
        assert context_index < first_topic
        assert events[context_index][1]["playback_key"] == "scene:7"
        assert not events[context_index][1]["playback_changed"]
        assert pipeline._generation == 0 and state.playback_generation == 7
    finally:
        pipeline.stop()
        pipeline._thread.join(timeout=3)
    assert not pipeline.running


def test_changed_story_model_rebuilds_analyzer_without_erasing_recorded_text(tmp_path, monkeypatch):
    state = SessionState(segments=[TranscriptSegment(0, 400, "Earlier story context")])
    created = []

    def build(settings, **kwargs):
        analyzer = object()
        created.append(analyzer)
        return analyzer

    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", build)
    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama", context_model="first-local-model")
    first = Pipeline(settings, lambda *args: None, session_state=state)
    assert first._session_analyzer() is created[0]
    resumed = Pipeline(settings, lambda *args: None, session_state=state)
    assert resumed._session_analyzer() is created[0]
    settings.context_model = "other-local-model"
    changed = Pipeline(settings, lambda *args: None, session_state=state)
    assert changed._session_analyzer() is created[1]
    assert len(state.segments) == 1 and state.segments[0].text == "Earlier story context"


def test_session_change_callback_runs_outside_lock_and_never_logs_recorded_text(caplog):
    state = SessionState()
    acquired = threading.Event()

    def save(snapshot):
        assert snapshot is state and snapshot.revision == 1

        def read():
            with snapshot.lock:
                acquired.set()

        reader = threading.Thread(target=read)
        reader.start()
        assert acquired.wait(1), "The save callback must not retain the session lock"
        reader.join(timeout=1)
        raise OSError("private recorded book text")

    state.on_change = save
    state.notify_changed()
    assert state.revision == 1
    assert "OSError" in caplog.text and "private recorded book text" not in caplog.text
    assert len(state.session_id) == 32
    assert state.session_id != SessionState().session_id


@pytest.mark.parametrize("provider", ["ollama", "contextual"])
def test_rebuild_loaded_analyzer_preserves_matching_context_and_scope(tmp_path, monkeypatch, provider):
    settings = Settings(cache_dir=str(tmp_path), semantic_provider=provider, book_context="Arctic expedition")
    segments = [TranscriptSegment(index * 6, (index + 1) * 6, f"Recorded passage {index}")
                for index in range(330)]
    retained = Topic("Arctic camp", "Full text", context_used_segments=330)
    checkpoint = {"cursor": 330}
    state = SessionState(segments=segments, topic=retained, playback_generation=9,
                         playback_scope=(True, frozenset({"arctic", "camp"}), frozenset()),
                         provider_key=(provider, settings.context_model, settings.book_context),
                         book_context=settings.book_context,
                         analyzer_checkpoint=checkpoint if provider == "ollama" else None)
    restored, saved = [], []

    class Analyzer:
        def restore_checkpoint(self, value, recording):
            restored.append((value, recording))
            return True

    fresh = Analyzer()
    state.on_change = lambda snapshot: saved.append(snapshot.revision)
    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", lambda *args, **kwargs: fresh)
    pipeline = Pipeline(settings, lambda *args: None, session_state=state)
    assert pipeline._session_analyzer() is fresh
    assert state.analyzer is fresh
    assert state.topic is pipeline._latest_topic
    assert replace(state.topic, image_guidance=None, guidance_strength=None) == retained
    assert state.topic.image_guidance == settings.book_context
    assert state.playback_generation == 9 and state.playback_scope is not None
    assert state.segments is segments and len(state.segments) == 330
    assert saved == [1]
    if provider == "ollama":
        assert restored == [(checkpoint, segments)]
        assert state.analyzer_checkpoint is checkpoint
    else:
        assert restored == []


@pytest.mark.parametrize("reason", ["corrupt", "model_changed", "missing"])
def test_invalid_saved_model_context_rereads_all_raw_without_loss(tmp_path, monkeypatch, reason):
    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama", context_model="current-local", use_story_examples=False)
    segments = [TranscriptSegment(0, 400, "An earlier private recorded passage.")]
    old_key = ("ollama", "other-local" if reason == "model_changed" else settings.context_model, "")
    state = SessionState(segments=segments, provider_key=old_key, topic=Topic("Arctic camp", "Earlier context"),
                         playback_scope=(True, frozenset({"arctic", "camp"}), frozenset()), playback_generation=7,
                         analyzer_checkpoint=None if reason == "missing" else {"invalid": "private content"})
    restored, events = [], []

    class Analyzer:
        def restore_checkpoint(self, value, recording):
            restored.append(recording)
            return False

    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", lambda *args, **kwargs: Analyzer())
    pipeline = Pipeline(settings, lambda kind, payload: events.append((kind, payload)), session_state=state)
    pipeline._session_analyzer()
    assert state.segments is segments and state.segments[0].end == 400
    assert state.topic is pipeline._latest_topic is None
    assert state.playback_scope is state.analyzer_checkpoint is None
    assert state.playback_generation == 8
    assert state.provider_key == ("ollama", "current-local", "")
    assert len(restored) == (1 if reason == "corrupt" else 0)
    assert any(kind == "status" and "complete retained transcript" in payload for kind, payload in events)
    assert "private" not in str(events)


@pytest.mark.parametrize("fails", [False, True])
def test_analysis_worker_checkpoints_partial_progress_even_when_model_fails(tmp_path, fails):
    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama", use_story_examples=False)
    checkpoint = {"cursor": 1, "accepted_state": {"summary": "Grounded story context"}}

    class Analyzer:
        def analyze(self, segments, stopped):
            if fails:
                raise RuntimeError("controlled inference failure")
            return Topic("Arctic camp", "Story context")

        def export_checkpoint(self):
            return checkpoint

    state = SessionState(analyzer=Analyzer(), provider_key=("ollama", settings.context_model, ""))
    saved = []

    def save(snapshot):
        assert snapshot.analysis_lock.acquire(blocking=False), "Saving must not wait on model work"
        snapshot.analysis_lock.release()
        saved.append(snapshot.analyzer_checkpoint)

    state.on_change = save
    pipeline = Pipeline(settings, lambda *args: None, session_state=state)
    if fails:
        with pytest.raises(RuntimeError, match="controlled inference failure"):
            pipeline._analyze_session(state.analyzer, [])
    else:
        assert pipeline._analyze_session(state.analyzer, []).query == "Arctic camp"
    assert state.analyzer_checkpoint is checkpoint and saved == [checkpoint]
    assert state.revision == 1


@pytest.mark.parametrize("changed", ["analyzer", "key"])
def test_old_analysis_worker_cannot_replace_new_interpreter_checkpoint(tmp_path, changed):
    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama")
    retained = {"cursor": 0}
    state = SessionState(provider_key=("ollama", settings.context_model, ""), analyzer_checkpoint=retained)

    class OldAnalyzer:
        def analyze(self, segments, stopped):
            if changed == "analyzer":
                state.analyzer = object()
            else:
                state.provider_key = ("ollama", "different-local", "")
            return None

        def export_checkpoint(self):
            return {"cursor": 99}

    old = state.analyzer = OldAnalyzer()
    pipeline = Pipeline(settings, lambda *args: None, session_state=state)
    pipeline._analyze_session(old, [])
    assert state.analyzer_checkpoint is retained and state.revision == 0


@pytest.mark.asyncio
async def test_raw_transcript_saves_while_model_pending_and_callback_failure_is_nonfatal(tmp_path, caplog):
    pending, release = threading.Event(), threading.Event()
    settings = Settings(cache_dir=str(tmp_path))
    state = SessionState()

    class Analyzer:
        def analyze(self, segments):
            pending.set()
            release.wait(timeout=3)
            return None

    class Transcriber:
        def transcribe(self, chunk):
            return [TranscriptSegment(0, 6, "The explorers camp beside Arctic sea ice.")]

    pipeline = Pipeline(settings, lambda kind, value: pipeline.stop_event.set()
                        if kind == "transcript" else None, session_state=state)
    state.analyzer = Analyzer()
    state.provider_key = (settings.semantic_provider, settings.context_model, "")
    saved = []

    def save(snapshot):
        assert pending.is_set() and not release.is_set()
        saved.append(list(snapshot.segments))
        raise OSError("private transcript in disk error")

    state.on_change = save
    model = threading.Thread(target=pipeline._analyze_session, args=(state.analyzer, []))
    model.start()
    try:
        assert pending.wait(1)
        pipeline._transcriber = Transcriber()
        pipeline.audio_queue = queue.Queue()
        pipeline.audio_queue.put(AudioChunk(np.zeros(16000, dtype=np.float32), 0))
        await asyncio.wait_for(pipeline._transcribe(), 1)
        assert saved == [state.segments] and len(state.segments) == 1
        assert state.revision == 1
        assert "private transcript" not in caplog.text
        assert model.is_alive(), "Accepted raw text must save before the model returns"
    finally:
        release.set()
        model.join(timeout=1)
    assert not model.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["transcript", "topic"])
async def test_stop_and_final_flush_reject_result_waiting_to_accept_session_lock(tmp_path, result):
    from semantic_image_stream.session_store import SessionStore

    settings = Settings(cache_dir=str(tmp_path / "cache"), session_dir=str(tmp_path / "sessions"),
                        semantic_interval=0.01)
    recorded = TranscriptSegment(0, 6, "The explorers set up an Arctic camp.")
    retained = Topic("Arctic camp", recorded.text, summary=recorded.text,
                     scene="Arctic camp", visual_subjects=("Arctic camp",), context_used_segments=1)
    state = SessionState(segments=[recorded], topic=retained, playback_generation=3)
    store = SessionStore(settings)
    store.bind(state)
    store.flush(state)
    events = []
    pipeline = Pipeline(settings, lambda kind, value: events.append((kind, value)), session_state=state)

    class CloseBeforeAcceptance:
        armed = result == "transcript"

        def __enter__(self):
            if self.armed:
                self.armed = False
                # Deterministically place close after the outer stop check,
                # before acceptance can acquire the real session-state lock.
                pipeline.stop_event.set()
                store.flush(state)
                store.close()
            return state.lock.__enter__()

        def __exit__(self, *args):
            return state.lock.__exit__(*args)

    admission = CloseBeforeAcceptance()
    pipeline._segments_lock = admission

    class Transcriber:
        def transcribe(self, chunk):
            return [TranscriptSegment(6, 12, "A new unaccepted passage.")]

    class Analyzer:
        def analyze(self, segments):
            admission.armed = True
            return Topic("Arctic door", "New model result", scene="A wooden door on Arctic ice",
                         visual_subjects=("wooden door", "Arctic ice"), context_used_segments=1)

    try:
        if result == "transcript":
            pipeline._transcriber = Transcriber()
            pipeline.audio_queue = queue.Queue()
            pipeline.audio_queue.put(AudioChunk(np.zeros(16000, dtype=np.float32), 6))
            await asyncio.wait_for(pipeline._transcribe(), 2)
        else:
            pipeline.topic_queue = asyncio.Queue(maxsize=1)
            await asyncio.wait_for(pipeline._interpret(Analyzer()), 2)
        assert pipeline.stop_event.is_set()
        assert state.segments == [recorded]
        assert state.topic is pipeline._latest_topic is retained
        assert state.playback_generation == 3 and state.revision == 0
        assert not any(kind == "transcript" or kind == "topic" or
                       kind == "context" and value.get("phase") == "ready" for kind, value in events)
        fresh = SessionStore(settings)
        try:
            saved = fresh.load()
            assert saved.segments == state.segments
            assert saved.topic.query == state.topic.query and saved.topic.scene == state.topic.scene
        finally:
            fresh.close()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_unchanged_failed_context_waits_for_cooldown_instead_of_spinning(tmp_path):
    failed = asyncio.Event()

    class Analyzer:
        calls = 0

        def analyze(self, segments):
            self.calls += 1
            raise OSError("private-model-input")

    def receive(kind, payload):
        if kind == "context" and payload["phase"] == "error":
            failed.set()

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), semantic_interval=0.01), receive)
    pipeline._segments.append(TranscriptSegment(0, 6, "Arctic expedition"))
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    analyzer = Analyzer()
    task = asyncio.create_task(pipeline._interpret(analyzer))
    try:
        await asyncio.wait_for(failed.wait(), 2)
        await asyncio.sleep(0.12)
        assert analyzer.calls == 1 and not task.done()
    finally:
        pipeline.stop_event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_waiting_session_analysis_worker_can_stop_without_rebinding_old_model(tmp_path):
    state = SessionState()
    state.analysis_lock.acquire()
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None, session_state=state)
    called = threading.Event()

    class Analyzer:
        def analyze(self, segments):
            called.set()

    worker = threading.Thread(target=pipeline._analyze_session, args=(Analyzer(), []))
    worker.start()
    pipeline.stop_event.set()
    try:
        worker.join(timeout=1)
        assert not worker.is_alive() and not called.is_set()
    finally:
        state.analysis_lock.release()
        worker.join(timeout=1)


def test_playback_scope_changes_only_for_meaningful_scene_or_grounded_subject_changes(tmp_path):
    from dataclasses import replace

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    camp = Topic("Arctic expedition", "All story text", summary="The explorers set up camp.",
                 scene="Camp tents beside Arctic sea ice", visual_subjects=("Arctic sea ice", "camp tents"))
    assert pipeline._update_playback_scope(camp)
    first = pipeline.session_state.playback_generation
    updated = replace(camp, summary="An expedition camps among ice.", transcript=("Earlier", "New narration"),
                      context_used_segments=2)
    assert not pipeline._update_playback_scope(updated)
    assert pipeline.session_state.playback_generation == first
    door = replace(updated, scene="A mysterious wooden door stands alone in the ice",
                   visual_subjects=("wooden door", "Arctic sea ice"))
    assert pipeline._update_playback_scope(door)
    assert pipeline.session_state.playback_generation == first + 1
    assert pipeline._update_playback_scope(replace(door, query="", visual_subjects=()))
    assert pipeline.session_state.playback_generation == first + 2


@pytest.mark.parametrize("first_scene,next_scene,changes", [
    ("A wooden door stands in the Arctic camp with snow.", "A wooden door stands in the Arctic camp without snow.", True),
    ("A wooden door stands in the Arctic camp without snow.", "A wooden door stands in the Arctic camp with no snow.", False),
    ("An open wooden door stands in the Arctic camp.", "An opened wooden door stands in the Arctic camp.", False),
    ("An open wooden door stands in the Arctic camp.", "A closed wooden door stands in the Arctic camp.", True),
    ("The wooden door is open in the Arctic camp.", "The wooden door is not open in the Arctic camp.", True),
    ("The wooden door opens in the Arctic camp.", "The wooden door never opens in the Arctic camp.", True),
    ("The wooden door is closed in the Arctic camp.", "The wooden door is not open in the Arctic camp.", False),
    ("The wooden door is open in the Arctic camp.", "The wooden door is not closed in the Arctic camp.", False),
])
def test_scene_polarity_invalidates_playback_without_clearing_for_equivalent_wording(tmp_path, first_scene, next_scene, changes):
    from dataclasses import replace

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    topic = Topic("Arctic camp door", "Full session", scene=first_scene,
                  visual_subjects=("wooden door", "Arctic camp"))
    assert pipeline._update_playback_scope(topic)
    before = pipeline.session_state.playback_generation
    changed = pipeline._update_playback_scope(replace(topic, scene=next_scene,
                                                     summary="A paraphrased summary", context_used_segments=9))
    assert changed is changes
    assert pipeline.session_state.playback_generation == before + int(changes)


@pytest.mark.asyncio
async def test_no_grounded_visual_topic_updates_context_and_clears_pending_search(tmp_path):
    events = []
    pipeline = None

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] == "ready":
            if payload["segments_recorded"] == 1:
                pipeline._segments.append(TranscriptSegment(6, 12, "There is no grounded visible scene yet."))
            else:
                pipeline.stop_event.set()

    class Analyzer:
        def analyze(self, segments):
            return Topic("Arctic expedition" if len(segments) == 1 else "", "Full story",
                         summary="The passage discusses an abstract thought.",
                         scene="", context_used_segments=len(segments))

    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), semantic_interval=0.01), receive)
    pipeline._segments.append(TranscriptSegment(0, 6, "The Arctic expedition camps."))
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    await asyncio.wait_for(pipeline._interpret(Analyzer()), 2)
    assert pipeline._latest_topic.query == "" and pipeline.topic_queue.empty()
    ready = [payload for kind, payload in events if kind == "context" and payload["phase"] == "ready"]
    assert ready[-1]["segments_used"] == ready[-1]["segments_recorded"] == 2
    assert ready[-1]["message"] == "No grounded visual subject in this passage yet."
    assert ready[-1]["playback_key"] != ready[0]["playback_key"]
    assert ready[-1]["playback_changed"]

"""Offline integration of real file decoding and engine scheduling.

Only inference and external network boundaries are substituted here. Live
recording/model/provider validation is a separate opt-in script.
"""
import asyncio
import queue
import threading
import time
import wave

import numpy as np
import pytest

from semantic_image_stream.audio_capture import FileCapture
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.models import AudioChunk, ImageCandidate, TranscriptSegment


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

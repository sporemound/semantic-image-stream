"""Offline integration of real file decoding and engine scheduling.

Only inference and external network boundaries are substituted here. Live
recording/model/provider validation is a separate opt-in script.
"""
import threading
import time
import wave

import numpy as np

from semantic_image_stream.audio_capture import FileCapture
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.models import ImageCandidate, TranscriptSegment


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

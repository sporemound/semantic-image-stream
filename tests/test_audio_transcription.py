"""Injected models validate control flow, not Whisper recognition accuracy."""
import sys
import threading
import types

import numpy as np
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.models import AudioChunk
from semantic_image_stream.transcription import FasterWhisperTranscriber, TranscriptionCancelled


def _chunk(start=12):
    return AudioChunk(np.full(16000, 0.1, dtype=np.float32), start)


def _segment(text="The rings of Saturn", start=0.1, end=0.8):
    return types.SimpleNamespace(text=text, start=start, end=end)


def _install_model(monkeypatch, constructor, gpu_count=0):
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=constructor))
    monkeypatch.setitem(sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: gpu_count))


def test_model_is_lazy_and_incremental_chunks_have_absolute_timestamps(monkeypatch):
    calls, received = [], []

    class Model:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

        def transcribe(self, samples, **kwargs):
            received.append((samples.copy(), kwargs))
            return iter([_segment()]), types.SimpleNamespace(language="en")

    _install_model(monkeypatch, Model)
    transcriber = FasterWhisperTranscriber(Settings(whisper_device="cpu"))
    assert calls == []
    assert transcriber.transcribe(AudioChunk(np.zeros(16000, dtype=np.float32), 0)) == []
    assert calls == []
    first = transcriber.transcribe(_chunk(12))
    second = transcriber.transcribe(_chunk(25))
    assert len(calls) == 1
    assert len(received) == 2 and all(len(values) == 16000 for values, _ in received)
    assert first[0].start == pytest.approx(12.1)
    assert second[0].end == pytest.approx(25.8)
    assert received[0][1]["vad_filter"] is True
    assert received[0][1]["condition_on_previous_text"] is False
    assert received[0][1]["language"] is None


def test_cuda_constructor_failure_falls_back_to_cpu(monkeypatch):
    devices = []

    class Model:
        def __init__(self, model, **kwargs):
            devices.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("cublas64_12.dll not found")

        def transcribe(self, samples, **kwargs):
            return iter([_segment()]), types.SimpleNamespace(language="en")

    _install_model(monkeypatch, Model, gpu_count=1)
    transcriber = FasterWhisperTranscriber(Settings())
    assert len(transcriber.transcribe(_chunk())) == 1
    assert devices == ["cuda", "cpu"]
    assert transcriber.device == "cpu"


def test_lazy_generator_cuda_error_retries_once_without_duplicate_segments(monkeypatch):
    lock = threading.Lock()
    devices = []

    class Model:
        def __init__(self, model, **kwargs):
            self.device = kwargs["device"]
            devices.append(self.device)
            if self.device == "cuda":
                assert lock.locked()

        def transcribe(self, samples, **kwargs):
            def segments():
                if self.device == "cuda":
                    assert lock.locked()
                yield _segment()
                if self.device == "cuda":
                    raise RuntimeError("CUDA error: out of memory during lazy decoder")
                yield _segment("Then Jupiter", 0.8, 1)
            return segments(), types.SimpleNamespace(language="en")

    _install_model(monkeypatch, Model, gpu_count=1)
    transcriber = FasterWhisperTranscriber(Settings(), gpu_lock=lock)
    result = transcriber.transcribe(_chunk())
    assert [segment.text for segment in result] == ["The rings of Saturn", "Then Jupiter"]
    assert devices == ["cuda", "cpu"]
    assert not lock.locked()


def test_non_cuda_failures_remain_visible(monkeypatch):
    devices = []

    class Model:
        def __init__(self, model, **kwargs):
            devices.append(kwargs["device"])

        def transcribe(self, samples, **kwargs):
            def segments():
                raise ValueError("Unsupported language choice")
                yield
            return segments(), types.SimpleNamespace(language="")

    _install_model(monkeypatch, Model, gpu_count=1)
    transcriber = FasterWhisperTranscriber(Settings())
    with pytest.raises(ValueError, match="Unsupported language"):
        transcriber.transcribe(_chunk())
    assert devices == ["cuda"]


def test_cancel_during_model_load_discards_loaded_model(monkeypatch):
    stopped = threading.Event()

    def constructor(*args, **kwargs):
        stopped.set()
        return object()

    _install_model(monkeypatch, constructor)
    transcriber = FasterWhisperTranscriber(Settings(), stop_event=stopped)
    with pytest.raises(TranscriptionCancelled):
        transcriber.transcribe(_chunk())
    assert transcriber._model is None


def test_cancel_while_waiting_for_gpu_does_not_load_model(monkeypatch):
    lock = threading.Lock()
    lock.acquire()
    loaded = []
    _install_model(monkeypatch, lambda *args, **kwargs: loaded.append(True), gpu_count=1)
    transcriber = FasterWhisperTranscriber(Settings(), gpu_lock=lock)
    errors = []

    def work():
        try:
            transcriber.transcribe(_chunk())
        except TranscriptionCancelled:
            errors.append(True)

    worker = threading.Thread(target=work)
    worker.start()
    transcriber.stop()
    worker.join(timeout=1)
    lock.release()
    assert not worker.is_alive()
    assert loaded == [] and errors == [True]


def test_invalid_chunk_never_enters_native_model(monkeypatch):
    _install_model(monkeypatch, lambda *args, **kwargs: pytest.fail("Invalid chunk reached native model"))
    transcriber = FasterWhisperTranscriber(Settings())
    with pytest.raises(ValueError, match="16000"):
        transcriber.transcribe(AudioChunk(np.ones(10), 0, sample_rate=48000))
    with pytest.raises(ValueError, match="at most 30"):
        transcriber.transcribe(AudioChunk(np.ones(31 * 16000), 0))
    with pytest.raises(ValueError, match="non-finite"):
        transcriber.transcribe(AudioChunk(np.array([np.nan], dtype=np.float32), 0))

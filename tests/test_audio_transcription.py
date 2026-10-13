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


def test_cached_model_avoids_remote_metadata_request(monkeypatch):
    options = []
    class Model:
        def __init__(self, model, **kwargs):
            options.append(kwargs)
        def transcribe(self, samples, **kwargs):
            return iter([_segment()]), types.SimpleNamespace(language="en")
    _install_model(monkeypatch, Model)
    assert FasterWhisperTranscriber(Settings(whisper_device="cpu")).transcribe(_chunk())
    assert len(options) == 1
    assert options[0]["local_files_only"] is True


def test_only_cache_miss_enables_initial_download(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    options, statuses = [], []
    class Model:
        def __init__(self, model, **kwargs):
            options.append(kwargs["local_files_only"])
            if kwargs["local_files_only"]:
                raise LocalEntryNotFoundError("Weights are not cached")
        def transcribe(self, samples, **kwargs):
            return iter([_segment()]), types.SimpleNamespace(language="en")
    _install_model(monkeypatch, Model)
    assert FasterWhisperTranscriber(Settings(whisper_device="cpu"), on_status=statuses.append).transcribe(_chunk())
    assert options == [True, False]
    assert any("Downloading Whisper" in status for status in statuses)


def decode_segments(decoded):
    transcriber = FasterWhisperTranscriber(Settings(whisper_device="cpu"))
    transcriber._model = types.SimpleNamespace(
        transcribe=lambda samples, **kwargs: (iter(decoded), types.SimpleNamespace(language="en")))
    transcriber.device = "cpu"
    return transcriber.transcribe(_chunk())


def test_hallucinated_out_of_chunk_repeats_are_not_clamped_to_chunk_end():
    decoded = [_segment("The door opened.", 0.1, 0.7)]
    decoded.extend(_segment("The door.", 1 + index * 0.2, 1.2 + index * 0.2) for index in range(25))
    result = decode_segments(decoded)
    assert [segment.text for segment in result] == ["The door opened."]
    assert all(segment.end > segment.start for segment in result)


@pytest.mark.parametrize("start,end", [
    (0.2, 0.2), (0.8, 0.1), (-0.8, -0.1), (1, 2),
    (float("nan"), 0.5), (0.1, float("inf")), (float("-inf"), 0.5), (None, 0.5),
])
def test_invalid_or_nonpositive_decoded_intervals_are_ignored(start, end):
    result = decode_segments([_segment("Invalid decoder output", start, end), _segment("Real speech", 0.2, 0.8)])
    assert [segment.text for segment in result] == ["Real speech"]


def test_exact_overlapping_repeats_are_removed_but_distinct_utterances_remain():
    result = decode_segments([
        _segment("The door.", 0.1, 0.3),
        _segment("The door.", 0.1, 0.3),
        _segment("The door.", 0.2, 0.35),
        _segment("The door.", 0.4, 0.6),
        _segment("The door.", 0.6, 0.8),
    ])
    assert len(result) == 3
    assert [(segment.start, segment.end) for segment in result] == [(12.1, 12.3), (12.4, 12.6), (12.6, 12.8)]


def test_partially_overlapping_audio_boundaries_preserve_real_speech():
    result = decode_segments([_segment("Opening words", -0.1, 0.2), _segment("Closing words", 0.8, 1.2)])
    assert [(segment.start, segment.end) for segment in result] == [(12, 12.2), (12.8, 13)]

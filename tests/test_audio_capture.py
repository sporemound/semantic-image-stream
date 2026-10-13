"""Signal/recorded-file integration tests; these do not validate audio hardware."""
import sys
import threading
import time
import types
import wave

import numpy as np
import pytest

from semantic_image_stream import audio_capture as audio
from semantic_image_stream.configuration import Settings


def test_silence_is_not_enqueued_and_buffer_is_bounded():
    segmenter = audio.ChunkSegmenter(Settings(chunk_seconds=1, silence_seconds=0.2))
    for index in range(1000):
        assert segmenter.feed(np.zeros(1600, dtype=np.float32), index * 0.1) == []
        assert segmenter.buffered_samples <= 3520
    assert segmenter.flush() == []


def test_long_utterance_is_incremental_without_retranscribing_samples():
    segmenter = audio.ChunkSegmenter(Settings(chunk_seconds=1, silence_seconds=0.2))
    signal = np.full(10 * 16000, 0.1, dtype=np.float32)
    chunks = []
    # Uneven input sizes exercise frame carry and accurate timestamps.
    for index in range(0, len(signal), 997):
        chunks.extend(segmenter.feed(signal[index:index + 997], index / 16000))
    chunks.extend(segmenter.flush())
    assert len(chunks) == 10
    assert all(len(c.samples) == 16000 for c in chunks)
    assert [c.start for c in chunks] == pytest.approx(list(range(10)), abs=0.0001)
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), signal)


def test_silence_ends_utterance_and_leading_gap_stays_in_timestamps():
    segmenter = audio.ChunkSegmenter(Settings(chunk_seconds=3, silence_seconds=0.4))
    assert not segmenter.feed(np.zeros(16000, dtype=np.float32))
    assert not segmenter.feed(np.full(16000, 0.1, dtype=np.float32))
    chunks = segmenter.feed(np.zeros(16000, dtype=np.float32))
    assert len(chunks) == 1
    assert chunks[0].start == pytest.approx(0.8)
    assert len(chunks[0].samples) / 16000 == pytest.approx(1.6)
    assert not segmenter.flush()


def test_discontinuity_does_not_join_unrelated_audio():
    segmenter = audio.ChunkSegmenter(Settings(chunk_seconds=3, silence_seconds=0.4))
    assert not segmenter.feed(np.full(4800, 0.1, dtype=np.float32), 0)
    chunks = segmenter.feed(np.full(4800, 0.2, dtype=np.float32), 100)
    chunks.extend(segmenter.flush())
    assert len(chunks) == 2
    assert chunks[0].start == pytest.approx(0)
    assert chunks[1].start == pytest.approx(100)
    assert len(chunks[0].samples) == len(chunks[1].samples) == 4800


@pytest.mark.parametrize("sample_rate", [8000, 44100, 48000])
def test_streaming_resampler_downmixes_and_keeps_rate_across_blocks(sample_rate):
    pytest.importorskip("av")
    values = np.sin(2 * np.pi * 440 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    stereo = np.column_stack((values * 0.2, values * 0.4))
    resampler = audio.MonoResampler(sample_rate)
    output = []
    for index in range(0, len(stereo), 1003):
        output.extend(resampler.feed(stereo[index:index + 1003]))
    output.extend(resampler.flush())
    signal = np.concatenate(output)
    assert abs(len(signal) - 16000) <= 1
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) == pytest.approx(0.3, abs=0.01)
    peak_hz = np.argmax(np.abs(np.fft.rfft(signal))) * 16000 / len(signal)
    assert peak_hz == pytest.approx(440, abs=1)


def _write_wave(path):
    # Generated recording fixture: silence + two tones. This tests real media
    # decoding/resampling, not speech-recognition quality.
    rate = 48000
    silence = np.zeros(rate // 5, dtype=np.float32)
    t = np.arange(rate // 2) / rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    signal = np.concatenate((silence, tone, silence, tone, silence))
    stereo = np.column_stack((signal, signal * 0.5))
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(2)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes((stereo * 32767).astype("<i2").tobytes())
    return len(signal) / rate


def test_real_wave_replay_has_bounded_mono_chunks_and_timestamped_gaps(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "recorded_fixture.wav"
    duration = _write_wave(path)
    settings = Settings(audio_mode="file", audio_file=str(path), chunk_seconds=0.4,
                        silence_seconds=0.16, silence_rms=0.005)
    source = audio.FileCapture(settings, realtime=False)
    chunks, statuses = [], []
    source.run(chunks.append, statuses.append, threading.Event())
    assert len(chunks) >= 3
    assert all(c.samples.ndim == 1 and c.sample_rate == 16000 and len(c.samples) <= 6400 for c in chunks)
    assert all(np.isfinite(c.samples).all() for c in chunks)
    assert chunks[-1].start + len(chunks[-1].samples) / 16000 <= duration + 0.001
    assert all(a.start + len(a.samples) / 16000 <= b.start + 0.001 for a, b in zip(chunks, chunks[1:]))
    assert statuses[-1] == "Audio file finished"


def test_file_replay_stop_prevents_further_decoding(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "fixture.wav"
    _write_wave(path)
    source = audio.FileCapture(Settings(audio_mode="file", audio_file=str(path), chunk_seconds=0.3), realtime=False)
    chunks = []

    def receive(chunk):
        chunks.append(chunk)
        source.stop()

    source.run(receive, lambda message: None, threading.Event())
    assert len(chunks) == 1


def test_device_disconnect_retry_is_bounded_and_cancel_aware(monkeypatch):
    source = audio.DeviceCapture(Settings())
    attempts = []

    def disconnect(*args):
        attempts.append(True)
        raise OSError("disconnected fixture device")

    monkeypatch.setattr(source, "_run_device", disconnect)
    monkeypatch.setattr(source, "_wait", lambda seconds, event: False)
    with pytest.raises(RuntimeError, match="after 5 attempts"):
        source.run(lambda chunk: None, lambda status: None, threading.Event())
    assert len(attempts) == 5
    attempts.clear()
    monkeypatch.setattr(source, "_wait", lambda seconds, event: (source.stop() or True))
    source.run(lambda chunk: None, lambda status: None, threading.Event())
    assert len(attempts) == 1


def test_windows_loopback_enumeration_and_cleanup(monkeypatch):
    info = {"index": 7, "name": "Fixture speakers [Loopback]", "defaultSampleRate": 48000,
            "maxInputChannels": 2, "isLoopbackDevice": True}
    terminated = []
    backend = types.SimpleNamespace(get_loopback_device_info_generator=lambda: iter([info]),
                                    get_default_wasapi_loopback=lambda: info,
                                    terminate=lambda: terminated.append(True))
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", types.SimpleNamespace(PyAudio=lambda: backend))
    monkeypatch.setattr(audio.sys, "platform", "win32")
    devices = audio.list_devices()
    assert devices == [audio.AudioDevice(7, info["name"], 48000, 2, True, True)]
    assert terminated == [True]


def test_windows_callback_backpressure_drops_old_audio_and_closes_stream(monkeypatch):
    pytest.importorskip("av")
    info = {"index": 7, "name": "Fixture loopback", "defaultSampleRate": 48000,
            "maxInputChannels": 2, "isLoopbackDevice": True}
    closed, terminated = [], []
    source = audio.DeviceCapture(Settings(chunk_seconds=0.4))

    class Stream:
        def __init__(self, callback):
            self.callback = callback

        def start_stream(self):
            values = np.full((4800, 2), 6553, dtype="<i2").tobytes()
            for _ in range(60):
                self.callback(values, 4800, {}, 0)

        def is_active(self):
            return True

        def stop_stream(self):
            pass

        def close(self):
            closed.append(True)

    backend = types.SimpleNamespace(
        get_loopback_device_info_generator=lambda: iter([info]), get_default_wasapi_loopback=lambda: info,
        open=lambda **kwargs: Stream(kwargs["stream_callback"]), terminate=lambda: terminated.append(True))
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", types.SimpleNamespace(
        PyAudio=lambda: backend, paInt16=8, paContinue=0, paComplete=1, paAbort=2))
    monkeypatch.setattr(audio.sys, "platform", "win32")
    chunks, statuses = [], []

    def receive(chunk):
        chunks.append(chunk)
        source.stop()

    source.run(receive, statuses.append, threading.Event())
    assert len(chunks) == 1
    assert chunks[0].start >= 2.8
    assert any("dropped" in status for status in statuses)
    assert closed == terminated == [True]


def test_parallel_windows_discovery_serializes_native_lifecycle(monkeypatch):
    """Regression for process-global PortAudio initialization/termination races."""
    info = {"index": 7, "name": "Fixture speakers", "defaultSampleRate": 48000,
            "maxInputChannels": 2, "isLoopbackDevice": True}
    active_managers = [0]
    collision = []
    guard = threading.Lock()
    start = threading.Event()
    results = []

    class Backend:
        def __init__(self):
            with guard:
                active_managers[0] += 1
                if active_managers[0] != 1:
                    collision.append(True)
            # Yield while native initialization/enumeration owns global state.
            time.sleep(0.01)

        def get_loopback_device_info_generator(self):
            return iter([info])

        def get_default_wasapi_loopback(self):
            return info

        def terminate(self):
            time.sleep(0.01)
            with guard:
                active_managers[0] -= 1

    monkeypatch.setitem(sys.modules, "pyaudiowpatch", types.SimpleNamespace(PyAudio=Backend))
    monkeypatch.setattr(audio.sys, "platform", "win32")

    def discover():
        start.wait()
        results.append(audio.list_devices())

    workers = [threading.Thread(target=discover) for _ in range(4)]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=2)
    assert all(not worker.is_alive() for worker in workers)
    assert len(results) == 4
    assert all(devices[0].index == 7 for devices in results)
    assert collision == [] and active_managers == [0]

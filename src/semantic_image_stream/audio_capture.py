"""Real input sources and bounded, silence-aware 16 kHz audio segmentation.

Windows uses PyAudioWPatch's WASAPI loopback *input* endpoints: no output
stream is opened, so normal playback is unaffected. PortAudio callbacks only
copy into a bounded queue; resampling and user callbacks run on the worker.
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Callable

import numpy as np

from .configuration import Settings
from .models import AudioChunk

SAMPLE_RATE = 16000
log = logging.getLogger(__name__)
# PortAudio keeps process-global host API state. Parallel Pa_Initialize or
# Pa_Terminate from UI discovery and capture can abort the process on Windows.
# Serialize lifecycle operations, including enumeration and stream startup.
_WINDOWS_BACKEND_LOCK = threading.RLock()


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    sample_rate: int
    channels: int
    is_loopback: bool = False
    is_default: bool = False


def _windows_devices(p, mode: str) -> list[AudioDevice]:
    if mode == "loopback":
        infos = list(p.get_loopback_device_info_generator())
        try:
            default_index = int(p.get_default_wasapi_loopback()["index"])
        except (OSError, ValueError):
            default_index = -1
    else:
        infos = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
        infos = [d for d in infos if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")]
        try:
            default_index = int(p.get_default_input_device_info()["index"])
        except (OSError, ValueError):
            default_index = -1
    return [AudioDevice(int(d["index"]), d["name"], int(d["defaultSampleRate"]),
                        int(d["maxInputChannels"]), bool(d.get("isLoopbackDevice")),
                        int(d["index"]) == default_index)
            for d in infos if d["maxInputChannels"] > 0]


def list_devices(mode: str = "loopback") -> list[AudioDevice]:
    """Enumerate input devices; failures are visible to callers, never simulated."""
    if mode == "file":
        return []
    if mode not in {"loopback", "microphone"}:
        raise ValueError("Audio mode must be loopback, microphone, or file")
    if sys.platform == "win32":
        import pyaudiowpatch as pa
        with _WINDOWS_BACKEND_LOCK:
            p = pa.PyAudio()
            try:
                return _windows_devices(p, mode)
            finally:
                p.terminate()
    if mode == "loopback":
        raise RuntimeError("WASAPI loopback requires Windows. On Linux select a PulseAudio/PipeWire monitor input in microphone mode.")
    import sounddevice as sd
    default_index = int(sd.default.device[0])
    return [AudioDevice(i, d["name"], int(d["default_samplerate"]), int(d["max_input_channels"]),
                        False, i == default_index)
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


class ChunkSegmenter:
    """RMS pre-gate, 200 ms pre-roll, and hard bounded utterance chunks.

    This inexpensive energy check does not replace Whisper's speech VAD. It
    avoids sending pure silence to the model. Timestamps include silent gaps.
    No sample is retranscribed in a later chunk.
    """

    def __init__(self, settings: Settings):
        self.max_samples = max(1, int(settings.chunk_seconds * SAMPLE_RATE))
        self.silence_samples = max(1, int(settings.silence_seconds * SAMPLE_RATE))
        self.threshold = settings.silence_rms
        self.frame_samples = min(320, self.max_samples)  # 20 ms
        self._pre: deque[tuple[np.ndarray, float]] = deque(maxlen=max(1, 3200 // self.frame_samples))
        self._parts: list[np.ndarray] = []
        self._size = self._voiced = self._silent = 0
        self._start = self._cursor = 0.0
        self._pending = np.empty(0, dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        return self._size + len(self._pending) + sum(len(x) for x, _ in self._pre)

    def _emit(self) -> list[AudioChunk]:
        result = []
        if self._parts and self._voiced:
            result.append(AudioChunk(np.concatenate(self._parts).astype(np.float32, copy=False), self._start))
        self._parts = []
        self._size = self._voiced = self._silent = 0
        return result

    def _frame(self, frame: np.ndarray, start: float) -> list[AudioChunk]:
        result = []
        voiced = bool(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) > self.threshold)
        if not self._parts:
            if not voiced:
                self._pre.append((frame.copy(), start))
                return result
            if self._pre:
                self._start = self._pre[0][1]
                self._parts = [a for a, _ in self._pre]
                self._size = sum(len(a) for a in self._parts)
                self._pre.clear()
            else:
                self._start = start
        # Small configured chunk sizes can be shorter than pre-roll.
        while self._size >= self.max_samples:
            joined = np.concatenate(self._parts)
            extra = joined[self.max_samples:].copy()
            self._parts = [joined[:self.max_samples]]
            self._size = self.max_samples
            result.extend(self._emit())
            self._start += self.max_samples / SAMPLE_RATE
            if len(extra):
                self._parts = [extra]
                self._size = len(extra)
        offset = 0
        while offset < len(frame):
            count = min(len(frame) - offset, self.max_samples - self._size)
            if not self._parts:
                self._start = start + offset / SAMPLE_RATE
            self._parts.append(frame[offset:offset + count].copy())
            self._size += count
            self._voiced += count if voiced else 0
            self._silent = 0 if voiced else self._silent + count
            offset += count
            if self._size >= self.max_samples or self._silent >= self.silence_samples:
                result.extend(self._emit())
        return result

    def feed(self, samples: np.ndarray, start: float | None = None) -> list[AudioChunk]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        result: list[AudioChunk] = []
        if start is not None and abs(start - (self._cursor + len(self._pending) / SAMPLE_RATE)) > 0.025:
            result.extend(self.flush())
            self._pre.clear()
            self._cursor = float(start)
        if self._pending.size:
            samples = np.concatenate((self._pending, samples))
            self._pending = np.empty(0, dtype=np.float32)
        for offset in range(0, len(samples), self.frame_samples):
            frame = samples[offset:offset + self.frame_samples]
            if len(frame) < self.frame_samples:
                self._pending = frame.copy()
                break
            result.extend(self._frame(frame, self._cursor))
            self._cursor += len(frame) / SAMPLE_RATE
        return result

    def flush(self) -> list[AudioChunk]:
        result = []
        if self._pending.size:
            result.extend(self._frame(self._pending, self._cursor))
            self._cursor += len(self._pending) / SAMPLE_RATE
            self._pending = np.empty(0, dtype=np.float32)
        result.extend(self._emit())
        self._pre.clear()
        return result


class MonoResampler:
    """Stateful FFmpeg resampling, retaining filter state across input blocks."""
    def __init__(self, sample_rate: int):
        import av
        self._av = av
        self.sample_rate = sample_rate
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        mono = np.asarray(samples, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1, dtype=np.float32)
        frame = self._av.AudioFrame.from_ndarray(np.ascontiguousarray(mono.reshape(1, -1)), format="flt", layout="mono")
        frame.sample_rate = self.sample_rate
        return [f.to_ndarray().reshape(-1).astype(np.float32, copy=False) for f in self._resampler.resample(frame)]

    def flush(self) -> list[np.ndarray]:
        return [f.to_ndarray().reshape(-1).astype(np.float32, copy=False) for f in self._resampler.resample(None)]


class _CaptureBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._stop = threading.Event()

    def stop(self) -> None:
        # Wake the worker without calling PortAudio from the UI thread.
        self._stop.set()

    def _stopped(self, event: threading.Event) -> bool:
        return self._stop.is_set() or event.is_set()

    def _wait(self, seconds: float, event: threading.Event) -> bool:
        end = time.monotonic() + seconds
        while not self._stopped(event):
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            self._stop.wait(min(remaining, 0.05))
        return True


class FileCapture(_CaptureBase):
    """Replay a local media file at real-time speed with bounded decoding memory."""
    def __init__(self, settings: Settings, *, realtime: bool = True):
        super().__init__(settings)
        self.realtime = realtime

    def run(self, on_chunk: Callable[[AudioChunk], None], on_status: Callable[[str], None],
            stop_event: threading.Event) -> None:
        import av
        path = Path(self.settings.audio_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Choose a readable audio or video file: {path}")
        segmenter = ChunkSegmenter(self.settings)
        started = time.monotonic()
        offset = 0
        on_status(f"Replaying {path.name}")
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("This media file has no audio stream")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

            def consume(frame) -> bool:
                nonlocal offset
                values = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                # Decoder frames can be larger than our processing window.
                for index in range(0, len(values), 1600):
                    if self._stopped(stop_event):
                        return False
                    block = values[index:index + 1600]
                    for chunk in segmenter.feed(block, offset / SAMPLE_RATE):
                        on_chunk(chunk)
                    offset += len(block)
                    if self.realtime and self._wait(started + offset / SAMPLE_RATE - time.monotonic(), stop_event):
                        return False
                return True

            for frame in container.decode(audio=0):
                if self._stopped(stop_event):
                    return
                # Reset media timestamps: chunks use a stable replay timeline.
                frame.pts = None
                for converted in resampler.resample(frame):
                    if not consume(converted):
                        return
            for frame in resampler.resample(None):
                if not consume(frame):
                    return
        if not self._stopped(stop_event):
            for chunk in segmenter.flush():
                on_chunk(chunk)
            on_status("Audio file finished")


class DeviceCapture(_CaptureBase):
    MAX_RECONNECT_ATTEMPTS = 4
    NATIVE_QUEUE_BLOCKS = 32  # ~3.2 seconds, independent of session length.

    def run(self, on_chunk: Callable[[AudioChunk], None], on_status: Callable[[str], None],
            stop_event: threading.Event) -> None:
        failures = 0
        started = time.monotonic()
        while not self._stopped(stop_event):
            attempt_started = time.monotonic()
            try:
                self._run_device(on_chunk, on_status, stop_event, started)
                return
            except (OSError, RuntimeError) as error:
                if self._stopped(stop_event):
                    return
                if time.monotonic() - attempt_started > 10:
                    failures = 0
                failures += 1
                if failures > self.MAX_RECONNECT_ATTEMPTS:
                    raise RuntimeError(f"Audio device unavailable after {failures} attempts: {error}") from error
                log.warning("Audio input interrupted: %s", error)
                on_status(f"Audio interrupted; reconnecting ({failures}/{self.MAX_RECONNECT_ATTEMPTS}): {error}")
                if self._wait(min(2 ** (failures - 1), 4), stop_event):
                    return

    def _run_device(self, on_chunk, on_status, stop_event, session_start):
        blocks: queue.Queue = queue.Queue(self.NATIVE_QUEUE_BLOCKS)
        dropped = [0]
        callback_error = [None]
        native_offset = [time.monotonic() - session_start]

        def enqueue(values, sample_rate, status=None):
            start = native_offset[0]
            native_offset[0] += len(values) / sample_rate
            if status:
                dropped[0] += 1
            try:
                blocks.put_nowait((values.copy(), start))
            except queue.Full:
                try:
                    blocks.get_nowait()
                except queue.Empty:
                    pass
                try:
                    blocks.put_nowait((values.copy(), start))
                except queue.Full:
                    pass
                dropped[0] += 1

        p = stream = None
        try:
            if sys.platform == "win32":
                import pyaudiowpatch as pa
                with _WINDOWS_BACKEND_LOCK:
                    p = pa.PyAudio()
                    devices = _windows_devices(p, self.settings.audio_mode)
                    device = self._select(devices)

                    def callback(data, frames, info, status):
                        if self._stopped(stop_event):
                            return None, pa.paComplete
                        try:
                            values = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                            enqueue(values.reshape(-1, device.channels), device.sample_rate, status)
                        except Exception as error:
                            callback_error[0] = error
                            return None, pa.paAbort
                        return None, pa.paContinue

                    stream = p.open(format=pa.paInt16, channels=device.channels, rate=device.sample_rate,
                                    input=True, input_device_index=device.index,
                                    frames_per_buffer=max(1, device.sample_rate // 10),
                                    stream_callback=callback, start=False)
                    stream.start_stream()
                    active = stream.is_active
            else:
                if self.settings.audio_mode == "loopback":
                    raise RuntimeError("Select a Linux monitor input using microphone mode")
                import sounddevice as sd
                device = self._select(list_devices("microphone"))

                def callback(data, frames, info, status):
                    if self._stopped(stop_event):
                        raise sd.CallbackStop
                    enqueue(data, device.sample_rate, status)

                stream = sd.InputStream(device=device.index, samplerate=device.sample_rate,
                                        channels=device.channels, dtype="float32",
                                        blocksize=max(1, device.sample_rate // 10), callback=callback)
                stream.start()
                def active():
                    return stream.active
            on_status(f"Listening: {device.name}")
            segmenter = ChunkSegmenter(self.settings)
            resampler = MonoResampler(device.sample_rate)
            output_offset = native_offset[0]
            native_expected = None
            last_report = 0
            while not self._stopped(stop_event):
                try:
                    values, start = blocks.get(timeout=0.1)
                except queue.Empty:
                    if callback_error[0] is not None:
                        raise RuntimeError(f"Audio callback failed: {callback_error[0]}") from callback_error[0]
                    if not active():
                        raise OSError("The audio input stopped or was disconnected")
                    continue
                if native_expected is None:
                    output_offset = start
                elif abs(start - native_expected) > 0.025:
                    # Do not join unrelated samples after callback backpressure.
                    for chunk in segmenter.flush():
                        on_chunk(chunk)
                    segmenter = ChunkSegmenter(self.settings)
                    resampler = MonoResampler(device.sample_rate)
                    output_offset = start
                native_expected = start + len(values) / device.sample_rate
                for samples in resampler.feed(values):
                    for chunk in segmenter.feed(samples, output_offset):
                        on_chunk(chunk)
                    output_offset += len(samples) / SAMPLE_RATE
                if dropped[0] > last_report:
                    on_status("Audio processing fell behind; older buffered audio was dropped")
                    last_report = dropped[0]
        finally:
            with _WINDOWS_BACKEND_LOCK if p is not None else nullcontext():
                if stream is not None:
                    try:
                        if sys.platform == "win32":
                            stream.stop_stream()
                        else:
                            stream.abort()
                    except (OSError, RuntimeError):
                        pass
                    try:
                        stream.close()
                    except (OSError, RuntimeError):
                        pass
                if p is not None:
                    p.terminate()

    def _select(self, devices: list[AudioDevice]) -> AudioDevice:
        if self.settings.audio_device is not None:
            for device in devices:
                if device.index == self.settings.audio_device:
                    return device
            raise OSError("The selected input device is unavailable; refresh the device list")
        if not devices:
            raise OSError("No compatible audio input devices were found")
        return next((d for d in devices if d.is_default), devices[0])


def create_capture(settings: Settings) -> FileCapture | DeviceCapture:
    settings.validate()
    return FileCapture(settings) if settings.audio_mode == "file" else DeviceCapture(settings)

"""Local, incremental faster-whisper with lazy loading and CUDA recovery."""
from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
from pathlib import Path
import threading

import numpy as np

from .configuration import Settings
from .models import AudioChunk, TranscriptSegment

log = logging.getLogger(__name__)


class TranscriptionCancelled(RuntimeError):
    pass


class FasterWhisperTranscriber:
    """Transcribe only the incoming chunk, not accumulated audio history.

    Model construction and a native decoder step cannot be forcibly interrupted
    upstream. Cancellation is checked before/after loading, while waiting for
    the shared GPU lock, and between generated segments. Call on a worker.
    """
    def __init__(self, settings: Settings, gpu_lock=None, stop_event=None, on_status=None):
        self.settings = settings
        self.gpu_lock = gpu_lock or threading.Lock()
        self.stop_event = stop_event or threading.Event()
        self._stop = threading.Event()
        self.on_status = on_status or (lambda message: log.info(message))
        self._model = None
        self.device = ""
        self._load_lock = threading.Lock()

    def stop(self):
        self._stop.set()

    def _check_cancel(self):
        if self._stop.is_set() or self.stop_event.is_set():
            raise TranscriptionCancelled("Transcription stopped")

    @contextmanager
    def _gpu_access(self):
        while not self.gpu_lock.acquire(timeout=0.1):
            self._check_cancel()
        try:
            self._check_cancel()
            yield
        finally:
            self.gpu_lock.release()

    @staticmethod
    def _gpu_failure(error):
        message = str(error).lower()
        return any(token in message for token in ("cuda", "cudnn", "cublas", "nvidia", "out of memory", "driver"))

    def _choose_device(self):
        if self.settings.whisper_device != "auto":
            return self.settings.whisper_device
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except (ImportError, RuntimeError):
            return "cpu"

    def _construct(self, device):
        self._check_cancel()
        from faster_whisper import WhisperModel
        compute = self.settings.whisper_compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        elif device == "cpu" and compute in {"float16", "int8_float16"}:
            compute = "int8"
        self.on_status(f"Loading local Whisper {self.settings.whisper_model} on {device}; first use downloads model weights")
        model = WhisperModel(self.settings.whisper_model, device=device, compute_type=compute,
                             download_root=str(Path(self.settings.cache_dir) / "models"),
                             num_workers=1, cpu_threads=4)
        self._check_cancel()
        self._model = model
        self.device = device
        self.on_status(f"Local transcription ready ({device}, {compute})")

    def _ensure_model(self):
        self._check_cancel()
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            device = self._choose_device()
            if device == "cuda":
                try:
                    with self._gpu_access():
                        self._construct("cuda")
                except Exception as error:
                    if isinstance(error, TranscriptionCancelled) or not self._gpu_failure(error):
                        raise
                    self.on_status(f"CUDA unavailable; using CPU: {error}")
                    self._construct("cpu")
            else:
                self._construct("cpu")

    def _decode(self, chunk, samples):
        self._check_cancel()
        # Iteration belongs inside the try/GPU-lock scope: this API is lazy.
        segments, info = self._model.transcribe(
            samples, language=self.settings.language or None, beam_size=3,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False, word_timestamps=False,
            temperature=0.0,
        )
        result = []
        duration = len(samples) / 16000
        for segment in segments:
            self._check_cancel()
            text = " ".join(segment.text.split())
            if text:
                start = max(0.0, min(duration, float(segment.start)))
                end = max(start, min(duration, float(segment.end)))
                result.append(TranscriptSegment(chunk.start + start, chunk.start + end, text,
                                                getattr(info, "language", "") or ""))
        self._check_cancel()
        return result

    def transcribe(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        self._check_cancel()
        if chunk.sample_rate != 16000:
            raise ValueError("Transcription requires mono float32 audio at 16000 Hz")
        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim != 1 or len(samples) > 30 * 16000:
            raise ValueError("Transcription chunks must be mono and at most 30 seconds")
        if not np.isfinite(samples).all():
            raise ValueError("Audio contains non-finite samples")
        if not samples.size or np.sqrt(np.mean(np.square(samples, dtype=np.float64))) <= self.settings.silence_rms:
            return []
        samples = np.ascontiguousarray(samples)
        self._ensure_model()
        if self.device != "cuda":
            return self._decode(chunk, samples)
        try:
            with self._gpu_access():
                return self._decode(chunk, samples)
        except Exception as error:
            if isinstance(error, TranscriptionCancelled) or not self._gpu_failure(error):
                raise
            self.on_status(f"CUDA transcription failed; retrying this chunk on CPU: {error}")
            # Release VRAM before loading CPU weights; partially decoded results
            # were local and cannot be emitted twice after the retry.
            self._model = None
            gc.collect()
            self._construct("cpu")
            return self._decode(chunk, samples)

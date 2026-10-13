"""Real prerecorded-media decoding; no speech model or network needed in CI."""
import json
from pathlib import Path
import threading
import wave

import numpy as np
import pytest

from semantic_image_stream.audio_capture import FileCapture
from semantic_image_stream.configuration import Settings

FIXTURES = Path(__file__).parent / "fixtures"
RECORDINGS = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))["recordings"]


@pytest.mark.parametrize("recording", RECORDINGS, ids=[recording["file"] for recording in RECORDINGS])
def test_committed_speech_fixture_replays_through_real_decoder(recording):
    pytest.importorskip("av")
    path = FIXTURES / recording["file"]
    with wave.open(str(path), "rb") as file:
        assert (file.getframerate(), file.getnchannels(), file.getsampwidth()) == (16000, 1, 2)
        duration = file.getnframes() / file.getframerate()
        assert 15 < duration < 60
    settings = Settings(audio_mode="file", audio_file=str(path), chunk_seconds=6)
    source = FileCapture(settings, realtime=False)
    chunks = []
    source.run(chunks.append, lambda status: None, threading.Event())
    assert len(chunks) >= 3
    assert all(c.sample_rate == 16000 and c.samples.dtype == np.float32 for c in chunks)
    assert all(c.samples.ndim == 1 and 0 < len(c.samples) <= 6 * 16000 for c in chunks)
    assert all(np.isfinite(c.samples).all() for c in chunks)
    assert chunks[-1].start + len(chunks[-1].samples) / 16000 <= duration + 0.001
    assert all(a.start + len(a.samples) / 16000 <= b.start + 0.001 for a, b in zip(chunks, chunks[1:]))

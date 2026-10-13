"""Local CPU ranking must not claim the transcription GPU's lock."""
import sys
import types

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_ranking import ClipRanker
from semantic_image_stream.models import ImageCandidate, Topic


class GPUAccess:
    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *args):
        pass


class FinishedInference(Exception):
    pass


def test_cpu_clip_loading_does_not_claim_gpu_lock(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    gpu = GPUAccess()
    ranker = ClipRanker(Settings(clip_device="cpu"), gpu)
    def load():
        assert gpu.entries == 0
        raise FinishedInference
    monkeypatch.setattr(ranker, "_load", load)
    import pytest
    with pytest.raises(FinishedInference):
        ranker.rank(Topic("black hole", ""), [ImageCandidate("https://example.com/a", "https://example.com/b", "Black hole", "wikimedia")], [])
    assert gpu.entries == 0


def test_cuda_clip_loading_coordinates_with_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    gpu = GPUAccess()
    ranker = ClipRanker(Settings(clip_device="cuda"), gpu)
    def load():
        assert gpu.entries == 1
        raise FinishedInference
    monkeypatch.setattr(ranker, "_load", load)
    import pytest
    with pytest.raises(FinishedInference):
        ranker.rank(Topic("black hole", ""), [ImageCandidate("https://example.com/a", "https://example.com/b", "Black hole", "wikimedia")], [])
    assert gpu.entries == 1

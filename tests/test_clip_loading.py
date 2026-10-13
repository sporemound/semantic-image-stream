"""CLIP initialization is serialized and incomplete native models never publish."""
from types import SimpleNamespace
import sys
import threading

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_ranking import ClipLoadError, ClipRanker


@pytest.fixture
def cached(tmp_path):
    root = tmp_path / "models" / "clip" / "models--openai--clip-vit-base-patch32" / "snapshots" / "complete"
    root.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors"):
        (root / name).write_text("cached fixture")
    return Settings(cache_dir=str(tmp_path), clip_device="cpu")


def fake_libraries(monkeypatch, factory):
    class Processor:
        @classmethod
        def from_pretrained(cls, source, **options):
            assert options["local_files_only"] and options["use_fast"] is False
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, source, **options):
            assert options["output_loading_info"] is True and options["local_files_only"]
            return factory()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor))


class NativeModel:
    def __init__(self, *, meta_parameter=False, meta_buffer=False, to_error=False):
        self.meta_parameter, self.meta_buffer, self.to_error = meta_parameter, meta_buffer, to_error
        self.moved = False

    def parameters(self):
        return iter([SimpleNamespace(is_meta=self.meta_parameter)])

    def buffers(self):
        return iter([SimpleNamespace(is_meta=self.meta_buffer)])

    def to(self, device):
        self.moved = True
        if self.to_error:
            raise NotImplementedError("private path or transcript must not be shown")
        assert not self.meta_parameter and not self.meta_buffer
        return self

    def eval(self):
        return self


@pytest.mark.parametrize("failure", ["meta_parameter", "meta_buffer", "missing", "mismatched", "loader", "device"])
def test_failed_load_never_moves_meta_weights_or_publishes_partial_model(cached, monkeypatch, failure):
    model = NativeModel(meta_parameter=failure == "meta_parameter", meta_buffer=failure == "meta_buffer",
                        to_error=failure == "device")

    def load():
        if failure == "loader":
            raise NotImplementedError("private path or transcript must not be shown")
        return model, {"missing_keys": ["private-weight-name"] if failure == "missing" else [],
                       "mismatched_keys": ["private-weight-name"] if failure == "mismatched" else [], "error_msgs": []}

    fake_libraries(monkeypatch, load)
    ranker = ClipRanker(cached)
    with pytest.raises(ClipLoadError) as error:
        ranker._load(local_only=True)
    assert "private" not in str(error.value)
    assert ranker.model is ranker.processor is None
    assert model.moved is (failure == "device")
    healthy = NativeModel()
    fake_libraries(monkeypatch, lambda: (healthy, {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}))
    ranker._load(local_only=True)
    assert ranker.model is healthy and ranker.processor is not None


def test_known_nonpersistent_position_buffers_do_not_reject_trained_model(cached, monkeypatch):
    model = NativeModel()
    fake_libraries(monkeypatch, lambda: (model, {"missing_keys": ["text_model.embeddings.position_ids",
                                                                "vision_model.embeddings.position_ids"]}))
    ranker = ClipRanker(cached)
    ranker._load(local_only=True)
    assert ranker.model is model and model.moved


def test_different_cpu_workers_never_overlap_global_empty_weight_initialization(cached, monkeypatch):
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    state_lock = threading.Lock()
    active, maximum, calls = 0, 0, 0
    errors = []

    def load():
        nonlocal active, maximum, calls
        with state_lock:
            active += 1
            calls += 1
            maximum = max(maximum, active)
            number = calls
        if number == 1:
            entered.set()
            assert release.wait(2)
        else:
            second_entered.set()
        with state_lock:
            active -= 1
        return NativeModel(), {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

    fake_libraries(monkeypatch, load)
    rankers = [ClipRanker(cached), ClipRanker(cached)]

    def work(ranker):
        try:
            ranker._load(local_only=True)
        except Exception as error:
            errors.append(error)

    workers = [threading.Thread(target=work, args=(ranker,)) for ranker in rankers]
    workers[0].start()
    try:
        assert entered.wait(1)
        workers[1].start()
        assert not second_entered.wait(0.05)
        assert all(ranker.model is ranker.processor is None for ranker in rankers)
    finally:
        release.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=2)
    assert not errors and maximum == 1 and calls == 2
    assert all(ranker.model is not None and ranker.model.moved for ranker in rankers)

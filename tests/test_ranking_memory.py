"""Bound CPU working sets without loading native models or large image assets."""
from contextlib import nullcontext
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream import image_ranking
from semantic_image_stream.image_ranking import ClipRanker, ContextClipRanker, LearnedContextRanker, embed_pairs
from semantic_image_stream.models import ImageCandidate, Topic


@pytest.fixture(autouse=True)
def mock_memory_headroom(monkeypatch):
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", lambda *args, **kwargs: None)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.dtype = self.values.dtype

    def __add__(self, other):
        return Tensor(self.values + (other.values if isinstance(other, Tensor) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return Tensor(self.values * (other.values if isinstance(other, Tensor) else other))

    def __truediv__(self, other):
        return Tensor(self.values / (other.values if isinstance(other, Tensor) else other))

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def norm(self, dim=None, keepdim=False):
        return Tensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim))

    def clamp_min(self, minimum):
        return Tensor(np.maximum(self.values, minimum))

    def sum(self, dim=None):
        return Tensor(self.values.sum(axis=dim))

    def exp(self):
        return Tensor(np.exp(self.values))

    def float(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *args):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


def images(tmp_path, count=5):
    result = []
    for index in range(count):
        path = tmp_path / f"cave-{index}.png"
        with Image.new("RGB", (24, 16), (index, 0, 0)) as image:
            image.save(path)
        result.append(ImageCandidate(f"https://photos.example/cave-{index}", "https://museum.example/cave",
                                     "Coastal cave", "wikimedia", local_path=str(path), width=24, height=16))
    return result


def fixture_ranker(monkeypatch, ranker_class, device, failure=None):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    ranker = ranker_class(Settings(clip_device=device, relevance_threshold=.40))
    calls, opened = [], []

    class Processor:
        def __call__(self, images, **kwargs):
            calls.append(len(images))
            opened.extend(images)
            if failure == "processor":
                raise MemoryError("controlled small-fixture failure")
            return {"pixel_values": Tensor([len(images)])}

    class Model:
        logit_scale = Tensor(0)

        def get_image_features(self, pixel_values):
            if failure == "inference":
                raise MemoryError("controlled small-fixture failure")
            return Tensor([[1, 0]] * int(pixel_values.values[0]))

        def __call__(self, pixel_values):
            if failure == "inference":
                raise MemoryError("controlled small-fixture failure")
            return SimpleNamespace(logits_per_image=Tensor([[.5]] * int(pixel_values.values[0])))

    ranker.processor, ranker.model = Processor(), Model()
    if isinstance(ranker, ContextClipRanker):
        monkeypatch.setattr(ranker, "_context_vectors", lambda topic: [(Tensor([1, 0]), 1)])
        monkeypatch.setattr(ranker, "_score_vectors", lambda values, contexts: [.5] * len(values.values))
    return ranker, calls, opened


def assert_closed(opened):
    assert opened
    for image in opened:
        with pytest.raises(ValueError, match="closed"):
            image.getpixel((0, 0))


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
@pytest.mark.parametrize("device,batches", [("cpu", [1, 1, 1, 1, 1]), ("cuda", [4, 1])])
def test_cpu_batches_one_image_and_cuda_keeps_four_without_cutoff_change(tmp_path, monkeypatch, ranker_class, device, batches):
    ranker, calls, opened = fixture_ranker(monkeypatch, ranker_class, device)
    result = ranker.rank(Topic("coastal cave", ""), images(tmp_path), [])
    assert len(result) == 5 and calls == batches
    assert ranker.settings.relevance_threshold == .40
    assert_closed(opened)


@pytest.mark.parametrize("device,expected", [("cpu", [24 * 16 * 12] * 5), ("cuda", [4 * 24 * 16 * 12, 24 * 16 * 12])])
def test_processor_preflight_estimates_actual_batch_pixels(tmp_path, monkeypatch, device, expected):
    ranker, _, _ = fixture_ranker(monkeypatch, ContextClipRanker, device)
    budgets = []
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory",
                        lambda estimated_bytes, reserve_bytes: budgets.append((estimated_bytes, reserve_bytes)))
    ranker.rank(Topic("coastal cave", ""), images(tmp_path), [])
    assert budgets == [(estimate, 134_217_728) for estimate in expected]


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_low_pixel_headroom_closes_raster_without_running_processor(tmp_path, monkeypatch, ranker_class):
    from semantic_image_stream.memory_pressure import MemoryPressureError, is_memory_allocation_error
    ranker, calls, _ = fixture_ranker(monkeypatch, ranker_class, "cpu")
    candidates = images(tmp_path, 1)
    converted, original_convert = [], Image.Image.convert

    def convert(source, *args, **kwargs):
        image = original_convert(source, *args, **kwargs)
        converted.append(image)
        return image

    def low_memory(*args, **kwargs):
        raise MemoryPressureError("controlled pressure")

    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", low_memory)
    with pytest.raises(image_ranking.ClipLoadError, match="low on available memory") as caught:
        ranker.rank(Topic("coastal cave", ""), candidates, [])
    assert not calls and is_memory_allocation_error(caught.value)
    assert_closed(converted)


def test_pixel_rank_waits_for_the_same_lock_as_image_decode(tmp_path, monkeypatch):
    ranker, calls, _ = fixture_ranker(monkeypatch, ContextClipRanker, "cpu")
    candidates = images(tmp_path, 1)
    begun, finished, failures = threading.Event(), threading.Event(), []

    def rank():
        begun.set()
        try:
            ranker.rank(Topic("coastal cave", ""), candidates, [])
        except Exception as error:
            failures.append(error)
        finally:
            finished.set()

    image_ranking.PIXEL_MEMORY_LOCK.acquire()
    worker = threading.Thread(target=rank)
    worker.start()
    try:
        assert begun.wait(1) and not finished.wait(.05)
        assert not calls
    finally:
        image_ranking.PIXEL_MEMORY_LOCK.release()
        worker.join(2)
    assert finished.is_set() and not failures and calls == [1]


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
@pytest.mark.parametrize("failure", ["processor", "inference"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rasters_close_when_processing_or_inference_fails(tmp_path, monkeypatch, ranker_class, failure, device):
    ranker, _, opened = fixture_ranker(monkeypatch, ranker_class, device, failure)
    with pytest.raises(MemoryError):
        ranker.rank(Topic("coastal cave", ""), images(tmp_path), [])
    assert_closed(opened)


@pytest.mark.parametrize("threads,device,expected", [(16, "cpu", [4]), (4, "cpu", []), (1, "cpu", []), (16, "cuda", [])])
def test_cpu_thread_cap_precedes_model_operations_and_never_increases_smaller_choice(monkeypatch, threads, device, expected):
    calls, limits = [], []
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(image_ranking, "_cached_encoder_snapshot", lambda settings: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: threads,
        set_num_threads=lambda value: limits.append(value), cuda=SimpleNamespace(is_available=lambda: True)))

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append("processor")
            assert limits == expected
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append("model")
            assert limits == expected
            return cls(), {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

        def parameters(self):
            return ()

        def buffers(self):
            return ()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPModel=Model, CLIPProcessor=Processor))
    ranker = ClipRanker(Settings(clip_device=device))
    ranker._load()
    assert calls == ["processor", "model"] and ranker.model is not None


def test_low_memory_blocks_cpu_load_before_any_native_import(monkeypatch):
    import builtins
    from semantic_image_stream.memory_pressure import MemoryPressureError
    original_import = builtins.__import__
    checks = []

    def low_memory(estimated_bytes, reserve_bytes):
        checks.append((estimated_bytes, reserve_bytes))
        raise MemoryPressureError("controlled pressure")

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            pytest.fail("Low-memory refusal must precede native imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", low_memory)
    monkeypatch.setattr(image_ranking, "_cached_encoder_snapshot", lambda settings: None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ranker = ClipRanker(Settings())
    with pytest.raises(image_ranking.ClipLoadError, match="low on available memory"):
        ranker._load()
    assert checks == [(1_073_741_824, 134_217_728)]
    assert ranker.model is ranker.processor is None


def test_loaded_model_skips_memory_check_and_native_reimports(monkeypatch):
    ranker = ClipRanker(Settings())
    ranker.model = object()
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory",
                        lambda *args, **kwargs: pytest.fail("A loaded model does not reload weights"))
    ranker._load()


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker])
@pytest.mark.parametrize("method", ["rank", "feedback_vectors"])
def test_pixel_entrypoints_check_pressure_before_importing_torch(tmp_path, monkeypatch, ranker_class, method):
    import builtins
    from semantic_image_stream.memory_pressure import MemoryPressureError
    original_import = builtins.__import__
    candidates = images(tmp_path, 1)

    def low_memory(*args, **kwargs):
        raise MemoryPressureError("controlled pressure")

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            pytest.fail("The public pixel entrypoint imported native code before checking pressure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", low_memory)
    monkeypatch.setattr(image_ranking, "_cached_encoder_snapshot", lambda settings: tmp_path / "cached")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    ranker = ranker_class(Settings())
    with pytest.raises(image_ranking.ClipLoadError, match="low on available memory"):
        if method == "rank":
            ranker.rank(Topic("coastal cave", ""), candidates, [])
        else:
            ranker.feedback_vectors(Topic("coastal cave", ""), candidates[0])


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_empty_pixel_batches_do_not_import_native_libraries(monkeypatch, ranker_class):
    import builtins
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            pytest.fail("Empty batches cannot allocate a native model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert ranker_class(Settings()).rank(Topic("coastal cave", ""), [], []) == []


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_closed_ranker_refuses_pixel_jobs_and_reload(tmp_path, monkeypatch, ranker_class):
    ranker = ranker_class(Settings())
    ranker.model = object()
    monkeypatch.setattr(image_ranking.gc, "collect", lambda: None)
    ranker.close()
    candidate = images(tmp_path, 1)[0]
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory",
                        lambda *args, **kwargs: pytest.fail("Retired models cannot reload"))
    for method in (lambda: ranker._load(), lambda: ranker.rank(Topic("coastal cave", ""), [candidate], []),
                   lambda: ranker.feedback_vectors(Topic("coastal cave", ""), candidate)):
        with pytest.raises(image_ranking.ClipLoadError, match="worker has stopped"):
            method()


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_close_waits_for_actual_rank_lock_then_releases_native_and_context_objects(monkeypatch, ranker_class):
    ranker = ranker_class(Settings())
    ranker.model = ranker.processor = ranker._manual_embedding = object()
    ranker._manual_text = "manual brief"
    ranker._feedback_encoder_fingerprint = "a" * 64
    ranker._encoder_snapshot = object()
    if isinstance(ranker, ContextClipRanker):
        ranker._history, ranker._history_sum, ranker._history_weight = ("old passage",), object(), 75
    if isinstance(ranker, LearnedContextRanker):
        ranker._adapter = object()
    collected, begun, finished = [], threading.Event(), threading.Event()
    monkeypatch.setattr(image_ranking.gc, "collect", lambda: collected.append(True))

    def close():
        begun.set()
        ranker.close()
        finished.set()

    ranker._rank_lock.acquire()
    worker = threading.Thread(target=close)
    worker.start()
    try:
        assert begun.wait(1) and not finished.wait(.05)
        assert ranker.model is not None and not collected
    finally:
        ranker._rank_lock.release()
        worker.join(2)
    assert finished.is_set()
    assert ranker.model is ranker.processor is ranker._manual_embedding is None
    assert ranker._manual_text is ranker._feedback_encoder_fingerprint is ranker._encoder_snapshot is None
    if isinstance(ranker, ContextClipRanker):
        assert ranker._history == () and ranker._history_sum is None and ranker._history_weight == 0
    if isinstance(ranker, LearnedContextRanker):
        assert ranker._adapter is None
    assert collected == [True]


def test_smaller_text_batches_encode_every_token_and_preserve_weighted_history(monkeypatch):
    prepared, batches = [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext,
        tensor=lambda values, **kwargs: Tensor(values)))

    class Tokenizer:
        model_max_length = 77

        def encode(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return list(range(int(text)))

        def prepare_for_model(self, ids, **kwargs):
            assert kwargs["truncation"] is False
            prepared.extend(ids)
            return ids

        def pad(self, rows, **kwargs):
            batches.append(len(rows))
            return {"rows": Tensor([len(rows)])}

    ranker = ContextClipRanker(Settings())
    ranker.processor = SimpleNamespace(tokenizer=Tokenizer())
    ranker.model = SimpleNamespace(get_text_features=lambda rows: Tensor([[1, 0]] * int(rows.values[0])))
    total, weight = ranker._text_sum(("317", "739"))
    assert max(batches) == 4
    assert prepared == [*range(317), *range(739)]
    assert weight == 1056 and total.tolist() == pytest.approx([1056, 0])


@pytest.mark.parametrize("device,batches", [("cpu", [1, 1, 1, 1, 1]), ("cuda", [4, 1])])
def test_training_embeddings_keep_dimensions_with_same_bounded_image_batches(tmp_path, monkeypatch, device, batches):
    calls, opened = [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    class Processor:
        def __call__(self, images, **kwargs):
            calls.append(len(images))
            opened.extend(images)
            return {"pixel_values": Tensor([len(images)])}

    def load(self, **kwargs):
        assert kwargs["local_only"] is True
        self.processor = Processor()
        self.model = SimpleNamespace(get_image_features=lambda pixel_values:
            Tensor([[1] + [0] * 511] * int(pixel_values.values[0])))

    monkeypatch.setattr(ContextClipRanker, "_load", load)
    monkeypatch.setattr(ContextClipRanker, "_text_vector", lambda self, texts: Tensor([1] + [0] * 511))
    candidates = images(tmp_path)
    pixels, captions = embed_pairs(Settings(clip_device=device), [image.local_path for image in candidates],
                                   ["Coastal cave"] * len(candidates))
    assert calls == batches and pixels.shape == captions.shape == (5, 512)
    assert np.array_equal(pixels, captions)
    assert_closed(opened)

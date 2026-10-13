from types import SimpleNamespace
import sys

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_ranking import ContextClipRanker, build_ranker


class Vector:
    def __init__(self, values):
        self.values = tuple(values)

    def __add__(self, other):
        return Vector(a + b for a, b in zip(self.values, other.values))

    def __truediv__(self, value):
        return Vector(item / value for item in self.values)

    def norm(self):
        return SimpleNamespace(clamp_min=lambda minimum: max(minimum, sum(x*x for x in self.values)**.5))


def test_entire_long_passage_is_encoded_without_token_truncation():
    class Tokenizer:
        model_max_length = 77

        def encode(self, text, **options):
            assert options == {"add_special_tokens": False, "truncation": False, "verbose": False}
            return list(range(int(text)))

    ranker = ContextClipRanker(Settings())
    ranker.processor = SimpleNamespace(tokenizer=Tokenizer())
    windows = ranker._windows("311")
    assert [token for ids, _ in windows for token in ids] == list(range(311))
    assert max(length for _, length in windows) == 75
    assert windows[-1][0][-1] == 310


def test_full_history_influences_embedding_after_recent_context_changes(monkeypatch):
    ranker = ContextClipRanker(Settings())
    calls = []

    def embed(texts):
        calls.append(tuple(texts))
        if not texts:
            return None, 0
        return Vector((sum(text.count("sea") for text in texts), sum(text.count("door") for text in texts))), len(texts)

    monkeypatch.setattr(ranker, "_text_sum", embed)
    ranker._history_vector(("sea sea sea",))
    result = ranker._history_vector(("sea sea sea", "door"))
    assert result.values == pytest.approx((3 / 10**.5, 1 / 10**.5))
    assert calls == [("sea sea sea",), ("door",)]
    ranker._history_vector(("sea sea sea", "door"))
    assert calls[-1] == ()
    assert ranker._history_weight == 2


def test_new_book_rebuilds_embedding_without_previous_story(monkeypatch):
    ranker = ContextClipRanker(Settings())
    calls = []

    def embed(texts):
        calls.append(tuple(texts))
        return (Vector((len(texts), 0)), len(texts)) if texts else (None, 0)

    monkeypatch.setattr(ranker, "_text_sum", embed)
    ranker._history_vector(("old story", "old conclusion"))
    ranker._history_vector(("new story",))
    assert calls == [("old story", "old conclusion"), ("new story",)]
    assert ranker._history == ("new story",)
    assert ranker._history_weight == 1


def test_full_context_mode_selects_actual_image_embedding_ranker():
    assert isinstance(build_ranker(Settings(ranking="context_clip")), ContextClipRanker)


def test_complete_cached_clip_snapshot_wins_over_newer_partial_download(tmp_path, monkeypatch):
    from semantic_image_stream.image_ranking import ClipRanker
    root = tmp_path / "models" / "clip" / "models--openai--clip-vit-base-patch32" / "snapshots"
    complete = root / "complete"
    partial = root / "partial"
    complete.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json", "pytorch_model.bin"):
        (complete / name).write_text("cached")
    partial.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (partial / name).write_text("partial")
    calls = []

    class Cached:
        @classmethod
        def from_pretrained(cls, source, **options):
            calls.append((source, options))
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(CLIPModel=Cached, CLIPProcessor=Cached))
    ClipRanker(Settings(cache_dir=str(tmp_path)))._load()
    assert len(calls) == 2
    assert all(source == str(complete) and options["local_files_only"] for source, options in calls)


@pytest.mark.parametrize("field,value", [("context_model", "qwen:cloud"), ("book_context", "x" * 2001),
                                          ("context_cpu_threads", 33), ("context_timeout", 0)])
def test_local_context_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value}).validate()

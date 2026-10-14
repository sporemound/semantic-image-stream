"""Separate book guidance is stable across modes; no native model is loaded."""
from contextlib import nullcontext
from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.feedback import FeedbackStore, _context_key
from semantic_image_stream.image_ranking import (
    ClipRanker, ContextClipRanker, LearnedContextRanker, MetadataRanker, _book_components, _context_fingerprint,
)
from semantic_image_stream.models import ImageCandidate, Topic


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __add__(self, other):
        return Tensor(self.values + (other.values if isinstance(other, Tensor) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return Tensor(self.values * other)

    def __truediv__(self, other):
        return Tensor(self.values / (other.values if isinstance(other, Tensor) else other))

    def __matmul__(self, other):
        return Tensor(self.values @ other.values)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def norm(self, dim=None, keepdim=False):
        return Tensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim))

    def clamp_min(self, minimum):
        return Tensor(np.maximum(self.values, minimum))

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


def passage():
    texts = tuple(f"The wooden door remains visible, recorded passage {index}." for index in range(321))
    return Topic("wooden door", " ".join(texts), summary="The group waits beside the wooden door.",
                 scene="A wooden door", visual_subjects=("wooden door",), transcript=texts)


def guided(settings, strength=.65):
    return replace(passage(), image_guidance=settings.book_context, guidance_strength=strength)


@pytest.mark.parametrize("strength", [0, .2, .65, 1])
def test_book_weights_keep_scene_largest_and_every_transcript_active(strength):
    settings = Settings(book_context="Arctic tundra, Inuit community, uncanny horror.", book_guidance_strength=strength)
    _, book, weights = _book_components(guided(settings, strength), settings)
    assert sum(weights) + book == pytest.approx(1)
    assert weights[0] > max(book, weights[1], weights[2])
    assert weights[2] >= .15
    assert settings.relevance_threshold == .30


def test_long_summary_never_dilutes_full_separate_book_embedding(monkeypatch):
    settings = Settings(book_context="Arctic tundra; Inuit community; eerie wooden door.")
    ranker = ContextClipRanker(settings)
    current = replace(guided(settings), summary="Long accepted summary " * 400)
    calls, histories = [], []
    scene, summary, history, manual = [Tensor(row) for row in np.eye(4)]

    def vector(texts):
        calls.append(tuple(texts))
        return manual if texts == (settings.book_context,) else summary if texts == (current.summary,) else scene

    monkeypatch.setattr(ranker, "_text_vector", vector)
    monkeypatch.setattr(ranker, "_history_vector", lambda texts: histories.append(tuple(texts)) or history)
    first = ranker._context_vectors(current)
    second = ranker._context_vectors(replace(current, summary=current.summary * 2))
    assert first[-1] == second[-1] == (manual, pytest.approx(.35 * .65))
    assert (settings.book_context,) in calls
    assert not any(settings.book_context in texts and len(texts) > 1 for texts in calls)
    assert calls.count((settings.book_context,)) == 1, "Stable guidance is encoded only once per ranker"
    assert histories == [current.transcript, current.transcript]
    assert len(current.transcript) == 321


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_all_native_modes_keep_every_manual_token_in_bounded_windows(ranker_class):
    ranker = ranker_class(Settings())
    ranker.processor = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=77,
        encode=lambda text, **kwargs: list(range(413))))
    windows = ranker._windows("the complete manual brief")
    assert [token for ids, _ in windows for token in ids] == list(range(413))
    assert max(length for _, length in windows) == 75


def pixels(tmp_path):
    candidates = []
    for name, color in (("arctic", "green"), ("urban", "red")):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (80, 60), color).save(path)
        candidates.append(ImageCandidate(f"https://images.example/{name}.png", "https://source.example/item",
                                         "Wooden door", "wikimedia", local_path=str(path), width=80, height=60))
    return candidates


def fixture_ranker(ranker_class, settings, monkeypatch):
    monkeypatch.setattr("semantic_image_stream.memory_pressure.require_memory", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    ranker = ranker_class(settings)
    neutral, arctic = np.eye(512)[:2]
    images = np.zeros((2, 512))
    images[:, 0] = .3
    images[:, 1] = [.3, -.3]
    images[:, 2] = np.sqrt(1 - .3**2 - .3**2)

    class Processor:
        def __call__(self, images, **kwargs):
            return {"pixel_values": Tensor([0 if image.getpixel((0, 0))[1] else 1 for image in images])}

    class Model:
        logit_scale = Tensor(0)

        def get_image_features(self, pixel_values):
            return Tensor(images[pixel_values.values.astype(int)])

        def __call__(self, pixel_values):
            indices = pixel_values.values.astype(int)
            return SimpleNamespace(logits_per_image=Tensor(np.full((len(indices), 1), .3)),
                                   image_embeds=Tensor(images[indices]), text_embeds=Tensor([neutral]))

    ranker.processor, ranker.model = Processor(), Model()
    ranker._feedback_encoder_fingerprint = "a" * 64
    monkeypatch.setattr(ranker, "_text_vector", lambda texts: Tensor(arctic if texts == (settings.book_context,) else neutral))
    if isinstance(ranker, ContextClipRanker):
        monkeypatch.setattr(ranker, "_history_vector", lambda texts: Tensor(neutral))
    if isinstance(ranker, LearnedContextRanker):
        ranker._adapter = SimpleNamespace(artifact={"report": {"selection": {"chosen_head": "cosine_only"}}},
                                         predict_cosines=lambda values: np.clip(values, 0, 1))
    return ranker


@pytest.mark.parametrize("ranker_class", [ClipRanker, ContextClipRanker, LearnedContextRanker])
def test_controlled_arctic_door_outranks_urban_door_without_native_models_or_cutoff_changes(
        tmp_path, monkeypatch, ranker_class):
    settings = Settings(book_context="Arctic tundra and an Inuit community, uncanny horror.", relevance_threshold=.40)
    ranker = fixture_ranker(ranker_class, settings, monkeypatch)
    images = pixels(tmp_path)
    baseline = ranker.rank(replace(passage(), image_guidance="", guidance_strength=.65), images, [])
    assert baseline[0].score == pytest.approx(baseline[1].score)
    result = ranker.rank(guided(settings), images, [])
    assert result[0].image_url.endswith("arctic.png") and result[0].score > result[1].score
    assert settings.relevance_threshold == .40
    vectors = ranker.feedback_vectors(guided(settings), result[0])
    assert vectors["encoder_fingerprint"] != "a" * 64
    assert vectors["context_vector"][1] > 0


def test_metadata_book_fit_prefers_arctic_door_and_keeps_query_guard():
    settings = Settings(book_context="Arctic Inuit tundra", book_guidance_strength=.65)
    ranker = MetadataRanker(settings)
    images = [ImageCandidate("https://images.example/arctic", "https://source.example", "Wooden door Arctic Inuit tundra", "wikimedia"),
              ImageCandidate("https://images.example/urban", "https://source.example", "Wooden door urban street traffic", "wikimedia"),
              ImageCandidate("https://images.example/background", "https://source.example", "Arctic Inuit tundra", "wikimedia")]
    selected = ranker.rank(guided(settings), images, [])
    assert [item.image_url for item in selected] == [images[0].image_url, images[1].image_url]
    assert selected[0].score > selected[1].score
    assert settings.relevance_threshold == .30


def test_guided_votes_are_identified_by_snapshot_and_weight_schema_and_legacy_votes_survive(tmp_path):
    settings = Settings(book_context="Arctic tundra", session_dir=str(tmp_path / "stories"))
    current = guided(settings)
    image = ImageCandidate("https://images.example/door", "https://source.example", "Wooden door", "wikimedia")
    ranker = MetadataRanker(settings)
    store = FeedbackStore(settings)
    first = ranker.feedback_vectors(current, image)
    store.record(current, image, True, first)
    changed = replace(current, guidance_strength=.9)
    second = ranker.feedback_vectors(changed, image)
    assert first["encoder_fingerprint"] != second["encoder_fingerprint"]
    assert store.get_vote(changed, image) is None
    assert store.adjust_scores([.5], [second["image_vector"]], second["context_vector"],
                               second["encoder_fingerprint"]) == [.5]
    settings.book_context = "A later different manual brief"
    settings.book_guidance_strength = .1
    assert ranker.feedback_vectors(current, image) == first, "Clicks use immutable frame guidance"
    baseline = passage()
    assert _context_key(baseline) == _context_key(replace(baseline, image_guidance="", guidance_strength=.65))
    assert _context_key(baseline) == _context_key(replace(baseline, image_guidance="Arctic", guidance_strength=0))
    assert _context_fingerprint("b" * 64, "context_clip", replace(baseline, image_guidance="", guidance_strength=1), settings) == "b" * 64

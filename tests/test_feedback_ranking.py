"""Votes affect real ranking paths without downloading models or changing cutoffs."""
from dataclasses import replace
from types import SimpleNamespace

from PIL import Image
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.feedback import FeedbackStore
from semantic_image_stream.image_ranking import ContextClipRanker, MetadataRanker
from semantic_image_stream.models import ImageCandidate, Topic


def test_saved_votes_personalize_metadata_and_survive_a_new_ranker(tmp_path):
    settings = Settings(session_dir=str(tmp_path / "story"), relevance_threshold=.30)
    topic = Topic("Arctic camp", "The party remains at an Arctic camp.",
                  scene="Arctic camp tents on ice", summary="Arctic camp tents on ice",
                  visual_subjects=("camp tents", "ice"))
    candidate = ImageCandidate("https://images.example/camp.jpg", "https://museum.example/item",
                               "Arctic camp tents", "wikimedia", width=1000, height=800)
    baseline = MetadataRanker().rank(topic, [replace(candidate)], [])[0].score
    ranker = MetadataRanker()
    ranker.feedback = FeedbackStore(settings)
    ranker.feedback.record(topic, candidate, True, ranker.feedback_vectors(topic, candidate))
    positive = ranker.rank(topic, [replace(candidate)], [])[0].score
    assert positive == pytest.approx(baseline + .04)
    fresh = MetadataRanker()
    fresh.feedback = FeedbackStore(settings)
    assert fresh.rank(topic, [replace(candidate)], [])[0].score == pytest.approx(positive)
    fresh.feedback.record(topic, candidate, False, fresh.feedback_vectors(topic, candidate))
    assert fresh.rank(topic, [replace(candidate)], [])[0].score == pytest.approx(baseline - .04)
    assert fresh.feedback.count() == 1
    assert settings.relevance_threshold == .30


def test_pixel_ranking_learns_pair_fit_from_the_full_context_and_respects_encoder_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    settings = Settings(session_dir=str(tmp_path / "story"), cache_dir=str(tmp_path / "cache"),
                        relevance_threshold=.30, clip_device="cpu")
    ranker = ContextClipRanker(settings)
    first = torch.zeros(512)
    second = torch.zeros(512)
    first[0], first[1] = .30, (1 - .30 ** 2) ** .5
    second[0], second[2] = .30, (1 - .30 ** 2) ** .5
    context = torch.zeros(512)
    context[0] = 1
    history = tuple(f"Earlier recorded camp passage {index}" for index in range(200)) + ("The party remains in camp.",)
    topic = Topic("camp", " ".join(history), scene="camp tents", summary="The party remains in camp.",
                  visual_subjects=("camp tents",), transcript=history, context_used_segments=len(history))
    captured = []
    monkeypatch.setattr(ranker, "_history_vector", lambda texts: captured.append(tuple(texts)) or context)
    monkeypatch.setattr(ranker, "_text_vector", lambda texts: context)
    ranker._feedback_encoder_fingerprint = "a" * 64

    class Processor:
        def __call__(self, images, **kwargs):
            green = [image.getpixel((0, 0))[1] > image.getpixel((0, 0))[0] for image in images]
            return {"pixel_values": torch.tensor(green)}

    ranker.processor = Processor()
    ranker.model = SimpleNamespace(get_image_features=lambda pixel_values:
                                   torch.stack([first if bool(green) else second for green in pixel_values]))
    candidates = []
    for name, color in (("other", "red"), ("preferred", "green")):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (80, 60), color).save(path)
        candidates.append(ImageCandidate(f"https://images.example/{name}.png", "https://museum.example/item",
                                         "Camp", "wikimedia", local_path=str(path), width=80, height=60))
    baseline = {candidate.image_url: candidate.score for candidate in ranker.rank(topic, candidates, [])}
    ranker.feedback = FeedbackStore(settings)
    features = ranker.feedback_vectors(topic, candidates[1])
    assert features["encoder_fingerprint"] == "a" * 64
    assert len(features["context_vector"]) == 512
    assert captured[-1] == history
    ranker.feedback.record(topic, candidates[1], True, features)
    ranked = ranker.rank(topic, candidates, [])
    assert ranked[0].image_url.endswith("preferred.png")
    assert ranked[0].score == pytest.approx(baseline[ranked[0].image_url] + .04)
    assert ranked[1].score == pytest.approx(baseline[ranked[1].image_url])
    ranker._feedback_encoder_fingerprint = "b" * 64
    for candidate in ranker.rank(topic, candidates, []):
        assert candidate.score == pytest.approx(baseline[candidate.image_url])
    assert settings.relevance_threshold == .30


def test_unavailable_feedback_cannot_stop_image_ranking_or_expose_private_error(caplog):
    ranker = MetadataRanker()
    ranker.feedback = SimpleNamespace(adjust_scores=lambda *args: (_ for _ in ()).throw(RuntimeError("private story")))
    topic = Topic("Arctic camp", "camp", scene="Arctic camp")
    candidate = ImageCandidate("https://images.example/camp.jpg", "https://source.example/camp",
                               "Arctic camp", "wikimedia")
    assert ranker.rank(topic, [candidate], [])
    assert "private story" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_empty_feedback_does_not_hash_or_encode_weights_during_initial_image_search(tmp_path, monkeypatch):
    settings = Settings(session_dir=str(tmp_path / "story"))
    ranker = ContextClipRanker(settings)
    ranker.feedback = FeedbackStore(settings)

    def unnecessary_work():
        raise AssertionError("An empty preference memory must not delay first-image startup")

    monkeypatch.setattr(ranker, "_feedback_fingerprint", unnecessary_work)
    assert ranker._personalize_pixels([.40], None, None) == [.40]
    assert not ranker.feedback.path.exists()

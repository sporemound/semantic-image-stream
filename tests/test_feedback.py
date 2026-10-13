from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_image_stream import feedback
from semantic_image_stream.feedback import FeedbackError, FeedbackStore
from semantic_image_stream.models import ImageCandidate, Topic


FINGERPRINT = hashlib.sha256(b"frozen-test-encoder").hexdigest()
OTHER_FINGERPRINT = hashlib.sha256(b"other-test-encoder").hexdigest()


def settings(tmp_path):
    return SimpleNamespace(session_dir=str(tmp_path / "story-memory"), cache_dir=str(tmp_path / "image-cache"))


def topic(text="Tents stood on the flat landscape."):
    return Topic(query="tents flat landscape", context=text, summary=text, scene=text,
                 visual_subjects=("tents", "flat landscape"), queries=("tents flat landscape",),
                 transcript=(text,), context_used_segments=1)


def candidate(number=1):
    return ImageCandidate(image_url=f"https://images.example.test/{number}.jpg", source_url="https://example.test/",
                          title="Landscape", provider="test", phash=f"{number:016x}")


def vectors(context=(1, 0, 0), image=(1, 0, 0), fingerprint=FINGERPRINT):
    return {"context_vector": context, "image_vector": image, "encoder_fingerprint": fingerprint}


def test_vote_and_actual_learning_survive_relaunch_without_private_transcript(tmp_path):
    config = settings(tmp_path)
    store = FeedbackStore(config)
    private = "Original private fictional dialogue and its context."
    current, image = topic(private), candidate()
    result = store.record(current, image, True, vectors())
    assert result == {"context_key": result["context_key"], "vote": True, "count": 1, "learned": True}
    assert len(result["context_key"]) == 64
    content = store.path.read_text(encoding="utf-8")
    assert private not in content
    assert current.query not in content
    assert image.image_url not in content
    assert image.phash not in content
    assert store.path.parent == store.root == (tmp_path / "story-memory").resolve()
    assert not (tmp_path / "image-cache").exists()
    reopened = FeedbackStore(config)
    assert reopened.get_vote(current, image) is True
    assert reopened.count() == 1
    adjusted = reopened.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT)
    assert adjusted == pytest.approx([0.54])
    assert not list(store.root.glob(".feedback-*.tmp"))


def test_changed_vote_upserts_and_feature_enrichment_never_duplicates(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    current, image = topic(), candidate()
    saved = store.record(current, image, False)
    assert saved["learned"] is False
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == [0.5]
    enriched = store.record(current, image, False, vectors())
    assert enriched["count"] == 1 and enriched["learned"] is True
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == pytest.approx([0.46])
    updated = store.record(current, image, True)
    assert updated["count"] == 1 and updated["learned"] is True
    assert store.get_vote(current, image) is True
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == pytest.approx([0.54])


def test_both_context_and_image_similarity_are_required(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(), False, vectors())
    assert store.adjust_scores([0.5], [[0, 1, 0]], [1, 0, 0], FINGERPRINT) == [0.5]
    assert store.adjust_scores([0.5], [[1, 0, 0]], [0, 1, 0], FINGERPRINT) == [0.5]
    assert store.adjust_scores([0.5], [[1, 0, 0]], [-1, 0, 0], FINGERPRINT) == [0.5]
    # Different pixel representation and context both transfer weakly, without
    # any global ban on a noun or source image outside similar story context.
    similar = [0.96, 0.28, 0]
    score = store.adjust_scores([0.5], [similar], similar, FINGERPRINT)[0]
    assert 0.46 < score < 0.5


def test_encoder_and_dimension_mismatch_have_no_influence(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(), False, vectors())
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], OTHER_FINGERPRINT) == [0.5]
    assert store.adjust_scores([0.5], [[1, 0]], [1, 0], FINGERPRINT) == [0.5]
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0], FINGERPRINT) == [0.5]
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], "not-a-fingerprint") == [0.5]


def test_exact_vote_identity_uses_context_and_decoded_image_fingerprint(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    current, image = topic(), candidate()
    store.record(current, image, True, vectors())
    assert store.get_vote(replace(current, context=current.context + " New spoken detail."), image) is None
    assert store.get_vote(replace(current, scene="Another location."), image) is None
    changed_pixels = candidate()
    changed_pixels.phash = "abcdef0123456789"
    assert store.get_vote(current, changed_pixels) is None
    # Moving an identical decoded image between cache paths does not change it.
    image.local_path = str(tmp_path / "another-cache.jpg")
    assert store.get_vote(current, image) is True


@pytest.mark.parametrize("bad", [
    None,
    vectors(image=[0, 0, 0]),
    vectors(context=[float("nan"), 0, 0]),
    vectors(image=[float("inf"), 0, 0]),
    vectors(context=[True, False, False]),
    vectors(image=[1, 0]),
    vectors(image=[[1, 0, 0]]),
    vectors(fingerprint="cloud:unverified"),
    vectors(fingerprint=FINGERPRINT.upper()),
    vectors(image=[1] * (feedback.MAX_VECTOR_DIM + 1)),
    {**vectors(), "unexpected": "private text"},
])
def test_bad_features_still_save_an_honest_label_without_learning(tmp_path, bad):
    store = FeedbackStore(settings(tmp_path))
    result = store.record(topic(), candidate(), True, bad)
    assert result["count"] == 1
    assert result["learned"] is False
    reopened = FeedbackStore(settings(tmp_path))
    assert reopened.get_vote(topic(), candidate()) is True
    assert reopened.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == [0.5]


def test_scores_are_regularized_bounded_and_never_change_configured_threshold(tmp_path):
    config = settings(tmp_path)
    config.relevance_threshold = 0.5
    store = FeedbackStore(config)
    for number in range(80):
        store.record(topic(), candidate(number), True, vectors())
    adjusted = store.adjust_scores([0.5, 0.99, 0.02], [[1, 0, 0], [1, 0, 0], [0, 1, 0]], [1, 0, 0], FINGERPRINT)
    assert 0.5 < adjusted[0] < 0.62
    assert adjusted[1] == 1
    assert adjusted[2] == 0.02
    assert config.relevance_threshold == 0.5


def test_opposing_labels_regularize_each_other(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(1), True, vectors())
    store.record(topic(), candidate(2), False, vectors())
    assert store.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == [0.5]


def test_storage_is_bounded_and_updating_a_vote_keeps_it_recent(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "MAX_LABELS", 3)
    store = FeedbackStore(settings(tmp_path))
    for number in range(3):
        store.record(topic(), candidate(number), True)
    store.record(topic(), candidate(0), False)
    store.record(topic(), candidate(3), True)
    assert store.count() == 3
    assert store.get_vote(topic(), candidate(0)) is False
    assert store.get_vote(topic(), candidate(1)) is None
    assert len(json.loads(store.path.read_text())["records"]) == 3


def test_separate_instances_refresh_and_merge_upserts_without_lost_labels(tmp_path):
    config = settings(tmp_path)
    old, new = FeedbackStore(config), FeedbackStore(config)
    assert old._lock is new._lock
    old.record(topic(), candidate(1), False, vectors())
    new.record(topic(), candidate(2), True, vectors())
    assert old.count() == new.count() == 2
    assert old.get_vote(topic(), candidate(2)) is True
    assert new.get_vote(topic(), candidate(1)) is False
    # Updating in another instance becomes visible to the scoring instance.
    new.record(topic(), candidate(1), True)
    assert old.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT)[0] > 0.5


def test_failed_atomic_write_preserves_prior_file_and_in_memory_labels(tmp_path, monkeypatch):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(1), True, vectors())
    before = store.path.read_bytes()

    def fail(*args):
        raise OSError("Private path must never appear in an error")

    monkeypatch.setattr(feedback.os, "replace", fail)
    with pytest.raises(FeedbackError, match="could not be saved locally") as error:
        store.record(topic(), candidate(2), False, vectors())
    assert "Private path" not in str(error.value)
    assert store.path.read_bytes() == before
    assert store.count() == 1
    assert store.get_vote(topic(), candidate(2)) is None
    assert not list(store.root.glob(".feedback-*.tmp"))


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_replace_failure_recovers_without_duplicate_vote(tmp_path, monkeypatch, code):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(1), True, vectors())
    native_replace = feedback.os.replace
    calls, delays = [], []

    def temporarily_locked(*args):
        calls.append(args)
        if len(calls) <= 2:
            error = PermissionError(13, "private path must not be echoed")
            error.winerror = code
            raise error
        return native_replace(*args)

    monkeypatch.setattr(feedback.os, "replace", temporarily_locked)
    monkeypatch.setattr(feedback.time, "sleep", delays.append)
    result = store.record(topic(), candidate(2), False, vectors())
    assert result["count"] == 2 and result["learned"] is True
    assert len(calls) == 3
    assert delays == [0.01, 0.025]
    assert FeedbackStore(settings(tmp_path)).get_vote(topic(), candidate(2)) is False
    assert not list(store.root.glob(".feedback-*.tmp"))


def test_permanent_windows_replace_failure_is_bounded_and_preserves_previous_file(tmp_path, monkeypatch):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(1), True)
    before = store.path.read_bytes()
    calls, delays = [], []

    def permanently_locked(*args):
        calls.append(args)
        error = PermissionError(13, "private data must not be echoed")
        error.winerror = 32
        raise error

    monkeypatch.setattr(feedback.os, "replace", permanently_locked)
    monkeypatch.setattr(feedback.time, "sleep", delays.append)
    with pytest.raises(FeedbackError, match="could not be saved locally") as error:
        store.record(topic(), candidate(2), False)
    assert "private data" not in str(error.value)
    assert len(calls) == 4 and delays == [0.01, 0.025, 0.05]
    assert store.path.read_bytes() == before
    assert store.count() == 1
    assert not list(store.root.glob(".feedback-*.tmp"))


def test_non_windows_permission_failure_is_not_retried(tmp_path, monkeypatch):
    store = FeedbackStore(settings(tmp_path))
    calls, delays = [], []

    def denied(*args):
        calls.append(args)
        raise PermissionError(13, "private path")

    monkeypatch.setattr(feedback.os, "replace", denied)
    monkeypatch.setattr(feedback.time, "sleep", delays.append)
    with pytest.raises(FeedbackError):
        store.record(topic(), candidate(), True)
    assert len(calls) == 1 and not delays


def test_corrupt_file_is_not_overwritten_or_echoed(tmp_path):
    config = settings(tmp_path)
    root = tmp_path / "story-memory"
    root.mkdir()
    path = root / "feedback.json"
    content = b'{"private": "untrusted story", "version": 999}'
    path.write_bytes(content)
    with pytest.raises(FeedbackError, match="existing file was preserved") as error:
        FeedbackStore(config)
    assert "untrusted story" not in str(error.value)
    assert path.read_bytes() == content


def test_tampered_or_mismatched_saved_vectors_do_not_influence_scores(tmp_path):
    config = settings(tmp_path)
    store = FeedbackStore(config)
    store.record(topic(), candidate(), False, vectors())
    data = json.loads(store.path.read_text())
    data["records"][0]["features"]["encoder_fingerprint"] = "not-valid"
    store.path.write_text(json.dumps(data))
    reopened = FeedbackStore(config)
    assert reopened.get_vote(topic(), candidate()) is False
    assert reopened.adjust_scores([0.5], [[1, 0, 0]], [1, 0, 0], FINGERPRINT) == [0.5]


def test_small_feedback_file_reader_allocates_its_actual_size(tmp_path, monkeypatch):
    config = settings(tmp_path)
    store = FeedbackStore(config)
    store.record(topic(), candidate(), True, vectors())
    actual_size = store.path.stat().st_size
    original_open = Path.open
    requested_sizes = []

    class Reader:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size):
            requested_sizes.append(size)
            return self.file.read(size)

    def tracked_open(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        return Reader(file) if path == store.path else file

    monkeypatch.setattr(Path, "open", tracked_open)
    reopened = FeedbackStore(config)
    assert reopened.get_vote(topic(), candidate()) is True
    assert requested_sizes == [actual_size + 1]
    assert requested_sizes[0] < 4096 < feedback.MAX_FILE_BYTES


def test_reader_memory_pressure_is_controlled_and_keeps_existing_file(tmp_path, monkeypatch):
    config = settings(tmp_path)
    store = FeedbackStore(config)
    store.record(topic(), candidate(), True)
    before = store.path.read_bytes()
    original_open = Path.open

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            raise MemoryError("Private story details must not be echoed")

    def pressure_open(path, *args, **kwargs):
        return Reader() if path == store.path else original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", pressure_open)
    with pytest.raises(FeedbackError, match="existing file was preserved") as error:
        FeedbackStore(config)
    assert "Private story" not in str(error.value)
    with original_open(store.path, "rb") as saved:
        assert saved.read() == before


def test_adjust_scores_accepts_native_numpy_features_and_ignores_bad_candidates(tmp_path):
    store = FeedbackStore(settings(tmp_path))
    store.record(topic(), candidate(), True, vectors(context=np.array([3, 0, 0]), image=np.array([7, 0, 0])))
    result = store.adjust_scores(np.array([0.5, 0.5]), [[1, 0, 0], [np.nan, 0, 0]],
                                 np.array([10, 0, 0]), FINGERPRINT)
    assert result == pytest.approx([0.54, 0.5])
    assert store.adjust_scores([0.5], [], [1, 0, 0], FINGERPRINT) == [0.5]

"""Local image-fit votes use the visible immutable context and survive Stop."""
from dataclasses import replace
import hashlib
import json
import threading

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.feedback import FeedbackStore
from semantic_image_stream.image_ranking import MetadataRanker
from semantic_image_stream.models import ImageCandidate, Topic


def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path / "cache"), session_dir=str(tmp_path / "stories"),
                    ranking="metadata", use_story_examples=False)


def topic():
    return Topic("Arctic camp", "Private recorded camp narration.", summary="The group waits at camp.",
                 scene="Tents on Arctic ice", visual_subjects=("camp tents", "Arctic ice"),
                 transcript=("Private recorded camp narration.",), context_used_segments=1)


def candidate():
    return ImageCandidate("https://images.example.org/camp.jpg", "https://source.example.org/camp",
                          "Arctic camp tents", "wikimedia", phash="0123456789abcdef")


def wait_for_count(events, count, condition):
    with condition:
        assert condition.wait_for(lambda: len(events) >= count, timeout=5)


def test_votes_save_before_features_and_changed_vote_survives_stop_fifo_and_reload(tmp_path):
    events, contexts = [], []
    condition = threading.Condition()
    entered, release = threading.Event(), threading.Event()
    configuration = settings(tmp_path)

    def receive(kind, payload):
        if kind == "feedback":
            with condition:
                events.append(payload)
                condition.notify_all()

    pipeline = Pipeline(configuration, receive)
    visible = topic()
    image = candidate()

    class Ranker:
        def feedback_vectors(self, context, image):
            contexts.append(context)
            if len(contexts) == 1:
                entered.set()
                assert release.wait(5)
            return {"image_vector": [1.0, 0.0, 0.0], "context_vector": [1.0, 0.0, 0.0],
                    "encoder_fingerprint": hashlib.sha256(b"frozen test encoder").hexdigest()}

    pipeline._ranker = Ranker()
    first = pipeline.record_feedback(image, True, visible)
    try:
        assert entered.wait(5), "The UI call returns while model features remain blocked"
        assert FeedbackStore(configuration).get_vote(visible, image) is True
        second = pipeline.record_feedback(image, False, visible)
        pipeline._latest_topic = Topic("New cave", "A later accepted passage must not alter either vote.")
        pipeline.stop()
        assert not events, "Feature preparation is still blocked, but the first label is already durable"
        release.set()
        wait_for_count(events, 2, condition)
        assert [event["request_id"] for event in events] == [first, second]
        assert [event["vote"] for event in events] == [True, False]
        assert all(event["success"] and event["learned"] and event["count"] == 1 for event in events)
        assert len(first) == len(second) == 32 and first != second
        assert contexts == [visible], "After Stop, keep the durable vote without loading another native model"
        resumed = Pipeline(configuration, lambda *args: None)
        assert resumed.get_feedback_vote(visible, image) is False
        assert resumed.get_feedback_vote(pipeline._latest_topic, image) is None
        assert FeedbackStore(configuration).count() == 1
        assert not pipeline._displayed_urls and not pipeline._recent_hashes
        assert not (tmp_path / "cache").exists()
        saved = (tmp_path / "stories" / "feedback.json").read_text(encoding="utf-8")
        assert visible.context not in saved and visible.transcript[0] not in saved and image.image_url not in saved
        assert pipeline.settings.relevance_threshold == configuration.relevance_threshold
    finally:
        release.set()


def test_real_metadata_features_personalize_matching_pairs_without_context_or_theme_leak(tmp_path):
    configuration = settings(tmp_path)
    events = []
    saved = threading.Event()

    def receive(kind, payload):
        if kind == "feedback":
            events.append(payload)
            saved.set()

    pipeline = Pipeline(configuration, receive)
    pipeline._ranker = MetadataRanker()
    visible, image = topic(), candidate()
    pipeline.record_feedback(image, True, visible)
    assert saved.wait(5)
    assert events[0]["success"] and events[0]["learned"] and events[0]["count"] == 1
    store = FeedbackStore(configuration)
    vectors = pipeline._ranker.feedback_vectors(visible, image)
    positive = store.adjust_scores([0.5], [vectors["image_vector"]], vectors["context_vector"], vectors["encoder_fingerprint"])
    assert positive[0] > 0.5
    saved.clear()
    pipeline.record_feedback(image, False, visible)
    assert saved.wait(5)
    negative = store.adjust_scores([0.5], [vectors["image_vector"]], vectors["context_vector"], vectors["encoder_fingerprint"])
    assert negative[0] < 0.5 and store.count() == 1
    unrelated = Topic("Kitchen cooking", "Kitchen cooking and dinner preparations.", scene="A bright kitchen")
    other_vectors = pipeline._ranker.feedback_vectors(unrelated, image)
    assert store.adjust_scores([0.5], [vectors["image_vector"]], other_vectors["context_vector"],
                               vectors["encoder_fingerprint"]) == [0.5]
    assert not pipeline._displayed_urls and not pipeline._recent_hashes


def test_feature_failure_keeps_durable_label_and_reports_saved_without_learning(tmp_path):
    events = []
    saved = threading.Event()

    def receive(kind, payload):
        if kind == "feedback":
            events.append(payload)
            saved.set()

    pipeline = Pipeline(settings(tmp_path), receive)

    class BrokenRanker:
        def feedback_vectors(self, context, image):
            raise RuntimeError("PRIVATE RAW NARRATION AND PATH MUST NOT APPEAR")

    pipeline._ranker = BrokenRanker()
    visible, image = topic(), candidate()
    request = pipeline.record_feedback(image, False, visible)
    assert saved.wait(5)
    event = events[0]
    assert event["request_id"] == request and event["success"] and not event["learned"]
    assert pipeline.get_feedback_vote(visible, image) is False
    assert event["count"] == 1 and "saved locally" in event["message"].casefold()
    assert "PRIVATE" not in json.dumps(event)


def test_store_save_and_lookup_errors_are_controlled_and_do_not_modify_history(tmp_path):
    events = []
    saved = threading.Event()

    def receive(kind, payload):
        if kind == "feedback":
            events.append(payload)
            saved.set()

    pipeline = Pipeline(settings(tmp_path), receive)

    class BrokenStore:
        def record(self, *args, **kwargs):
            raise OSError("PRIVATE FILE PATH")

        def get_vote(self, *args):
            raise OSError("PRIVATE RECORDED CONTEXT")

    pipeline._feedback_store = BrokenStore()
    request = pipeline.record_feedback(candidate(), True, topic())
    assert saved.wait(5)
    assert events[0]["request_id"] == request and not events[0]["success"]
    assert "PRIVATE" not in json.dumps(events)
    assert pipeline.get_feedback_vote(topic(), candidate()) is None
    assert not pipeline._displayed_urls and not pipeline._recent_hashes


def test_full_pending_feedback_queue_rejects_promptly_with_controlled_ack(tmp_path):
    events = []
    saved = threading.Event()

    def receive(kind, payload):
        events.append((kind, payload))
        saved.set()

    pipeline = Pipeline(settings(tmp_path), receive)
    pipeline._feedback_worker_running = True
    pipeline._feedback_requests.extend([None] * 32)
    request = pipeline.record_feedback(candidate(), True, topic())
    assert saved.wait(5)
    assert len(pipeline._feedback_requests) == 32
    assert events == [("feedback", {"request_id": request, "context_key": "", "image_url": candidate().image_url,
                                    "vote": True, "count": 0, "learned": False, "success": False,
                                    "message": "Local feedback is busy. Please try saving this vote again shortly."})]


@pytest.mark.parametrize("image,context,vote", [(None, topic(), True), (candidate(), None, True),
                                               (candidate(), replace(topic(), query=""), True),
                                               (candidate(), topic(), "private raw text")])
def test_invalid_feedback_inputs_fail_without_native_work_or_private_echo(tmp_path, image, context, vote):
    events = []
    saved = threading.Event()

    def receive(kind, payload):
        events.append((kind, payload))
        saved.set()

    pipeline = Pipeline(settings(tmp_path), receive)
    request = pipeline.record_feedback(image, vote, context)
    assert saved.wait(5)
    assert events[0][1]["request_id"] == request and not events[0][1]["success"]
    assert not pipeline._feedback_worker_running and not pipeline._feedback_requests
    assert "private raw text" not in json.dumps(events)


def test_context_event_exposes_exact_immutable_accepted_snapshot_including_retention(tmp_path):
    events = []
    pipeline = Pipeline(settings(tmp_path), lambda kind, payload: events.append((kind, payload)))
    accepted = replace(topic(), visual_update="retain", summary="The group is still waiting at camp.")
    pipeline._latest_topic = accepted
    pipeline._emit_context("retaining", "Keeping the current scene.", retaining=True)
    assert events[0][0] == "context" and events[0][1]["topic_snapshot"] is accepted
    assert events[0][1]["retaining"] and events[0][1]["scene"] == accepted.scene

"""Image votes use exact visible context, durable upserts and safe asynchronous acknowledgments."""
from dataclasses import asdict, replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate, Topic


class LocalFeedbackBoundary:
    def __init__(self):
        self.requests = []
        self.votes = {}
        self.displayed = []

    @staticmethod
    def identity(topic, candidate):
        return topic.query, topic.scene, candidate.image_url

    def record_feedback(self, candidate, positive, topic):
        request_id = str(len(self.requests) + 1)
        self.requests.append((request_id, candidate, positive, topic))
        return request_id

    def get_feedback_vote(self, topic, candidate):
        return self.votes.get(self.identity(topic, candidate))

    def complete(self, window, index=-1, success=True, learned=True):
        request_id, candidate, positive, topic = self.requests[index]
        key = self.identity(topic, candidate)
        if success:
            self.votes[key] = positive
        window._handle_event(window._session, "feedback", {
            "request_id": request_id, "image_url": candidate.image_url, "context_key": str(key),
            "vote": positive, "count": len(self.votes), "success": success, "learned": learned,
        })

    def record_displayed(self, candidate):
        self.displayed.append(candidate)

    def stop(self):
        pass


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    widget = MainWindow(Settings(cache_dir=str(tmp_path / "cache"), training_dir=str(tmp_path / "training"),
                                 session_dir=str(tmp_path / "sessions"), retain_context=False,
                                 providers=["wikimedia", "archive"], relevance_threshold=.1,
                                 book_context="The listener's own hint."))
    widget.pipeline = LocalFeedbackBoundary()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture
def topic():
    return Topic("frozen harbor", "Private spoken text.", summary="A journey across the frozen harbor.",
                 scene="A boat crosses the frozen harbor.", visual_subjects=("boat", "frozen harbor"),
                 transcript=("Private spoken text.",), context_used_segments=1)


@pytest.fixture
def images(tmp_path):
    result = []
    for name, color in (("boat", "blue"), ("harbor", "white"), ("ice", "cyan")):
        path = tmp_path / f"{name}.png"
        pixels = QImage(128, 96, QImage.Format.Format_RGB32)
        pixels.fill(QColor(color))
        assert pixels.save(str(path))
        result.append(ImageCandidate(f"https://images.example/{name}.png", f"https://source.example/{name}",
                                     name, "wikimedia", local_path=str(path)))
    return result


def present(window, topic, images):
    window._handle_event(window._session, "context", {"phase": "ready", "topic_snapshot": topic,
                        "summary": topic.summary, "scene": topic.scene, "visual_subjects": topic.visual_subjects,
                        "playback_key": "scene:1"})
    window._handle_event(window._session, "topic", topic)
    window._handle_event(window._session, "images", images)


def test_votes_require_usable_context_and_a_decoded_frame(window, topic, images, tmp_path):
    assert not window.good_fit_button.isEnabled() and not window.poor_fit_button.isEnabled()
    window._handle_event(0, "images", images[:1])
    assert not window.good_fit_button.isEnabled()
    window._handle_event(0, "context", {"phase": "ready", "topic_snapshot": topic})
    assert window.good_fit_button.isEnabled()
    window.slideshow.clear(clear_seen=False)
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"Not a raster")
    broken = replace(images[0], image_url="https://images.example/broken", local_path=str(path))
    window._handle_event(0, "images", [broken])
    assert not window.good_fit_button.isEnabled()
    assert not window.pipeline.requests


def test_good_vote_is_selected_after_local_save_and_repeated_click_is_not_duplicate(window, topic, images):
    present(window, topic, images[:1])
    before = asdict(window.settings)
    window.good_fit_button.click()
    assert not window.good_fit_button.isEnabled() and not window.poor_fit_button.isEnabled()
    assert "Saving vote locally" in window.feedback_status_label.text()
    assert window.pipeline.requests[0][3] is topic
    assert not window.pipeline.votes
    window.pipeline.complete(window)
    assert window.good_fit_button.isChecked() and not window.poor_fit_button.isChecked()
    assert "Saved locally" in window.feedback_status_label.text()
    window.good_fit_button.click()
    assert len(window.pipeline.requests) == 1 and len(window.pipeline.votes) == 1
    assert asdict(window.settings) == before


def test_vote_reversal_upserts_and_poor_vote_advances_only_to_unseen_pending(window, topic, images):
    present(window, topic, images[:2])
    window.good_fit_button.click()
    window.pipeline.complete(window)
    window.poor_fit_button.click()
    assert window.slideshow.current is images[0]
    window.pipeline.complete(window)
    assert len(window.pipeline.votes) == 1
    assert next(iter(window.pipeline.votes.values())) is False
    assert window.slideshow.current is images[1]
    assert not window.slideshow.pending
    window._select_history(window.image_history.item(1))
    assert window.slideshow.current is images[0]
    assert window.poor_fit_button.isChecked()
    assert window.good_fit_button.isEnabled()


def test_poor_vote_with_no_next_frame_keeps_the_image_and_supports_change(window, topic, images):
    present(window, topic, images[:1])
    window.poor_fit_button.click()
    window.pipeline.complete(window, learned=False)
    assert window.slideshow.current is images[0]
    assert window.poor_fit_button.isChecked() and window.good_fit_button.isEnabled()
    assert "image-model learning is not available" in window.feedback_status_label.text()
    window.good_fit_button.click()
    window.pipeline.complete(window)
    assert len(window.pipeline.votes) == 1 and window.good_fit_button.isChecked()


def test_old_frame_ack_cannot_advance_a_new_frame_or_another_history_view_of_same_url(window, topic, images):
    present(window, topic, images)
    window.poor_fit_button.click()
    window.next_image()
    assert window.slideshow.current is images[1]
    window._select_history(window.image_history.item(1))
    assert window.slideshow.current is images[0]
    checked = window.good_fit_button.isChecked(), window.poor_fit_button.isChecked()
    window.pipeline.complete(window)
    assert window.slideshow.current is images[0]
    assert list(window.slideshow.pending) == [images[2]]
    assert (window.good_fit_button.isChecked(), window.poor_fit_button.isChecked()) == checked


def test_snapshot_stays_exact_when_retained_context_advances_during_save(window, topic, images):
    present(window, topic, images[:1])
    window.good_fit_button.click()
    updated = replace(topic, summary="More of the private story has been recorded.", context_used_segments=2)
    window._handle_event(0, "context", {"phase": "retaining", "topic_snapshot": updated,
                                      "summary": updated.summary})
    assert window.pipeline.requests[0][3] is topic
    assert window._feedback_topic is updated
    assert not window.good_fit_button.isEnabled()
    window.pipeline.complete(window)
    assert window.good_fit_button.isEnabled()
    assert "previous context" in window.feedback_status_label.text()
    assert "Private spoken text" not in window.feedback_status_label.text()


def test_failed_save_leaves_vote_change_available_without_advancing(window, topic, images):
    present(window, topic, images[:2])
    window.poor_fit_button.click()
    window.pipeline.complete(window, success=False)
    assert window.slideshow.current is images[0]
    assert window.good_fit_button.isEnabled() and window.poor_fit_button.isEnabled()
    assert not window.poor_fit_button.isChecked()
    assert "Could not save" in window.feedback_status_label.text()


def test_votes_work_when_paused_or_stopped_and_reset_controls_on_new_book(window, topic, images):
    present(window, topic, images[:2])
    backend = window.pipeline
    window.toggle_pause()
    window._handle_event(0, "stopped", None)
    assert window.poor_fit_button.isEnabled()
    window.poor_fit_button.click()
    backend.complete(window)
    assert window.slideshow.current is images[1] and window.slideshow.paused
    assert not window.slideshow.timer.isActive()
    window.new_book()
    assert not window.good_fit_button.isEnabled() and not window.poor_fit_button.isEnabled()
    assert window._feedback_topic is None and not window._feedback_pending
    assert len(backend.votes) == 1


def test_feedback_controls_follow_fullscreen_chrome_and_fit_small_window(window, topic, images, app):
    present(window, topic, images[:1])
    window.resize(980, 690)
    app.processEvents()
    assert window.good_fit_button.isVisible() and window.poor_fit_button.isVisible()
    assert window.transport.geometry().width() <= window.stage.geometry().width()
    window.toggle_fullscreen()
    assert not window.good_fit_button.isVisible()
    window.leave_fullscreen()
    app.processEvents()
    assert window.good_fit_button.isVisible() and window.poor_fit_button.isVisible()

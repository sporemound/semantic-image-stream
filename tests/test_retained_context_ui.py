"""No-new-information model polls retain usable context and independent slide timing."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate, TranscriptSegment


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    widget = MainWindow(Settings(cache_dir=str(tmp_path / "cache"), session_dir=str(tmp_path / "sessions"),
                                 training_dir=str(tmp_path / "training"), retain_context=False,
                                 semantic_provider="ollama", relevance_threshold=.1))
    widget.show()
    app.processEvents()
    widget._handle_event(0, "transcript", [TranscriptSegment(0, 6, "The travelers crossed the frozen harbor.")])
    widget._handle_event(0, "context", {"phase": "ready", "summary": "Travelers visit a frozen harbor.",
                                      "scene": "A boat crosses the frozen harbor.", "visual_subjects": ["boat", "frozen harbor"],
                                      "segments_used": 1, "segments_recorded": 1,
                                      "method": "ollama", "playback_key": "scene:1", "message": "Scene ready."})
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


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


@pytest.mark.parametrize("payload", [
    {"phase": "updating"}, {"phase": "error", "message": "Local model timed out."},
    {"phase": "retaining"}, {"phase": "ready", "retaining": True},
])
def test_incomplete_or_retained_polls_keep_context_key_pending_and_timer(window, images, payload):
    window._handle_event(0, "images", images[:2])
    before_context = window.context_text.toPlainText()
    before_transcript = window.transcript.toPlainText()
    before_key = window._playback_key
    timer_id = window.slideshow.timer.timerId()
    window._handle_event(0, "context", {**payload, "summary": "", "scene": "", "visual_subjects": [],
                                      "playback_key": "scene:0", "segments_used": 1, "segments_recorded": 1})
    assert window.context_text.toPlainText() == before_context
    assert window.transcript.toPlainText() == before_transcript
    assert window._playback_key == before_key
    assert list(window.slideshow.pending) == [images[1]]
    assert window.slideshow.current is images[0]
    assert window.slideshow.timer.timerId() == timer_id
    assert window.slideshow.timer.isActive()


@pytest.mark.parametrize("payload", [{"phase": "retaining"}, {"phase": "ready", "retaining": True}])
def test_retaining_status_explains_fresh_searches_from_last_scene(window, payload):
    window._handle_event(0, "context", {**payload, "summary": "", "scene": "", "visual_subjects": []})
    assert "Retaining established scene" in window.context_status.text()
    assert "continuing fresh searches from the established scene" in window.image_status_label.text()
    assert "no visual subject" not in window.image_status_label.text()
    assert "frozen harbor" in window.context_text.toPlainText()
    assert window.threshold.value() == .1


def test_real_retrieval_diagnostic_remains_authoritative_during_retained_context(window):
    window._handle_event(0, "retrieval", {"phase": "searching", "message": "Searching the next source page.",
                                        "found": 12, "downloaded": 3, "accepted": 2})
    diagnostic = window.image_status_label.text()
    window._handle_event(0, "context", {"phase": "ready", "retaining": True,
                                      "summary": "", "scene": "", "visual_subjects": []})
    assert window.image_status_label.text() == diagnostic
    assert "Retaining established scene" in window.context_status.text()


def test_confirmed_ready_scene_change_still_invalidates_old_pending(window, images):
    window._handle_event(0, "images", images[:2])
    window._handle_event(0, "context", {"phase": "ready", "retaining": False,
                                      "summary": "The travelers arrive at a forest.",
                                      "scene": "A trail through the forest.", "visual_subjects": ["forest trail"],
                                      "playback_key": "scene:2"})
    assert window._playback_key == "context:scene:2"
    assert not window.slideshow.pending
    assert window.slideshow.current is images[0]
    assert "forest trail" in window.context_text.toPlainText()
    assert "frozen harbor" not in window.context_text.toPlainText()


def test_repeated_retained_model_polls_do_not_interrupt_chosen_slide_interval(window, images):
    window.interval.setValue(1)
    observed = []
    window.slideshow.current_changed.connect(observed.append)
    window._handle_event(0, "images", images)
    original_key = window._playback_key
    for _ in range(12):
        window._handle_event(0, "context", {"phase": "ready", "retaining": True,
                                          "summary": "", "scene": "", "visual_subjects": [],
                                          "playback_key": "scene:0", "segments_used": 1})
        QTest.qWait(200)
    assert observed == images
    assert window.slideshow.current is images[-1]
    assert window._playback_key == original_key
    assert not window.slideshow.pending
    assert "frozen harbor" in window.context_text.toPlainText()
    assert window.interval.value() == 1
    QTest.qWait(1100)
    assert observed == images  # Exhaustion holds the frame; it never repeats.

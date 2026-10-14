"""Narrative fullscreen follows accepted live context without changing playback."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QRectF, Qt
from PySide6.QtGui import QColor, QFontDatabase, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate, Topic
from semantic_image_stream.slideshow import ImageCanvas


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    # Windows' offscreen platform needs existing system fonts registered.
    if not QFontDatabase.families() and os.name == "nt":
        for name in ("segoeui.ttf", "georgia.ttf", "georgiai.ttf"):
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if font_path.is_file():
                assert QFontDatabase.addApplicationFont(str(font_path)) >= 0
    return application


@pytest.fixture
def image_factory(tmp_path):
    def make(name="one", title="Moonlight on the Frozen Harbor", color="blue"):
        path = tmp_path / f"{name}.png"
        pixels = QImage(160, 100, QImage.Format.Format_RGB32)
        pixels.fill(QColor(color))
        assert pixels.save(str(path))
        return ImageCandidate(f"https://images.example/{name}.png", f"https://source.example/{name}",
                              title, "wikimedia", local_path=str(path))
    return make


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    widget = MainWindow(Settings(cache_dir=str(tmp_path / "cache"), training_dir=str(tmp_path / "training"),
                                 session_dir=str(tmp_path / "sessions"), retain_context=False))
    widget.canvas.transition = "cut"
    widget.slideshow.timer.setInterval(60_000)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def ready_context(**overrides):
    return {"phase": "ready", "summary": "Travelers visit a frozen harbor.",
            "scene": "A boat crosses the harbor at night.", "visual_subjects": ["boat", "frozen harbor"],
            "segments_used": 1, "segments_recorded": 1, "method": "ollama", **overrides}


def test_context_button_shows_narrative_scene_subjects_and_image_title_then_restores(window, app, image_factory):
    image = image_factory()
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "images", [image])
    window.resize(980, 690)
    app.processEvents()
    normal = window.geometry()
    assert window.fullscreen_context_button.isVisible()
    window.fullscreen_context_button.click()
    app.processEvents()
    assert window.isFullScreen() and window._fullscreen_context and window._fullscreen_title
    assert window.canvas.context_overlay_visible and window.canvas.title_overlay_visible
    assert window.canvas.context_panel.isVisible()
    assert window.canvas.current_title == image.title
    assert window.canvas.context_text.toPlainText().split() == window.context_text.toPlainText().split()
    for heading in ("Narrative context", "Current scene", "Visual subjects"):
        assert heading in window.canvas.context_text.toPlainText()
    assert window.canvas.size() == window.size()
    assert not window.transport.isVisible() and not window.sidebar.isVisible()
    assert not window.source_label.isVisible() and not window.image_status_label.isVisible()
    window.leave_fullscreen()
    app.processEvents()
    assert not window._fullscreen and not window._fullscreen_context
    assert not window.canvas.context_overlay_visible and not window.canvas.title_overlay_visible
    assert not window.canvas.context_panel.isVisible()
    assert window.geometry() == normal
    assert window._displayed_candidate is image and window.source_label.isVisible()


def test_switching_three_modes_preserves_queue_history_frame_and_timer(window, app, image_factory):
    first, second = image_factory(), image_factory("two", "The Shore at Dawn", "cyan")
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "images", [first, second])
    timer_id = window.slideshow.timer.timerId()
    frame = window.canvas.current_pixmap.cacheKey()
    pending, history = list(window.slideshow.pending), list(window.slideshow.history)
    normal = window.geometry()
    for options, expected_title, expected_context in (
        ({}, False, False),
        ({"with_context": True}, True, True),
        ({"with_title": True}, True, False),
        ({"with_context": True}, True, True),
        ({"with_title": False}, False, False),
    ):
        window.toggle_fullscreen(**options)
        app.processEvents()
        assert window.isFullScreen()
        assert window.canvas.title_overlay_visible is expected_title
        assert window.canvas.context_overlay_visible is expected_context
        assert window._fullscreen_title is expected_title and window._fullscreen_context is expected_context
        assert window.slideshow.timer.timerId() == timer_id
        assert window.canvas.current_pixmap.cacheKey() == frame
        assert list(window.slideshow.pending) == pending and list(window.slideshow.history) == history
    window.leave_fullscreen()
    app.processEvents()
    assert window.geometry() == normal


def test_accepted_context_updates_live_while_title_stays_with_displayed_image(window, app, image_factory):
    image = image_factory()
    window._handle_event(0, "images", [image])
    window.toggle_fullscreen(with_context=True)
    window._handle_event(0, "context", ready_context())
    assert "Travelers visit a frozen harbor." in window.canvas.context_text.toPlainText()
    window._handle_event(0, "topic", Topic("Mountain pass", "Spoken narrative",
                                         summary="The travelers reach the mountains.", scene="They climb a steep pass.",
                                         visual_subjects=("mountain pass", "travelers"), context_used_segments=2))
    app.processEvents()
    context = window.canvas.context_text.toPlainText()
    assert "The travelers reach the mountains." in context and "They climb a steep pass." in context
    assert "mountain pass" in context and "frozen harbor" not in context
    assert window.canvas.current_title == image.title
    assert window._displayed_candidate is image
    window._handle_event(0, "stopped", None)
    window._handle_event(0, "context", ready_context(summary="A late worker result should be ignored."))
    assert window.canvas.context_text.toPlainText() == context


@pytest.mark.parametrize("phase", ["updating", "error", "retaining"])
def test_incomplete_context_keeps_established_overlay(window, app, phase):
    window._handle_event(0, "context", ready_context())
    window.toggle_fullscreen(with_context=True)
    before = window.canvas.context_text.toPlainText()
    window._handle_event(0, "context", {"phase": phase, "summary": "", "scene": "", "visual_subjects": [],
                                      "message": "Waiting for the local interpreter."})
    app.processEvents()
    assert window.canvas.context_text.toPlainText() == before
    assert window.canvas.context_panel.isVisible()


def test_context_is_plain_text_even_when_narrative_contains_markup(window, app):
    summary = '<b>Bold-looking narration</b> <img src="https://example.invalid/tracker.png">'
    scene = "<script>ignored()</script> & a harbor"
    subjects = ["<a href='https://example.invalid'>boat</a>", "ice & snow"]
    window._handle_event(0, "context", ready_context(summary=summary, scene=scene, visual_subjects=subjects))
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    text = window.canvas.context_text.toPlainText()
    assert summary in text and scene in text
    assert all(subject in text for subject in subjects)
    assert window.canvas.context_text.isReadOnly()


@pytest.mark.parametrize("exit_key", [Qt.Key.Key_Escape, Qt.Key.Key_F11])
def test_fullscreen_context_keeps_playback_and_exit_shortcuts(window, app, image_factory, exit_key):
    first, second = image_factory(), image_factory("two", "The Shore at Dawn", "cyan")
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "images", [first, second])
    window.toggle_fullscreen(with_context=True)
    window.activateWindow()
    window.canvas.context_text.setFocus()
    app.processEvents()
    QTest.keyClick(window.canvas.context_text, Qt.Key.Key_Space)
    app.processEvents()
    assert window.slideshow.paused
    QTest.keyClick(window.canvas.context_text, Qt.Key.Key_Right)
    app.processEvents()
    assert window._displayed_candidate is second
    assert window.canvas.current_title == second.title
    assert window.slideshow.paused
    QTest.keyClick(window.canvas.context_text, Qt.Key.Key_Space)
    app.processEvents()
    assert not window.slideshow.paused
    QTest.keyClick(window.canvas.context_text, exit_key)
    app.processEvents()
    assert not window._fullscreen and not window.canvas.context_overlay_visible
    assert not window.canvas.title_overlay_visible


@pytest.mark.parametrize("size", [(420, 260), (960, 540), (1600, 900)])
def test_long_narrative_scrolls_inside_canvas_above_title(app, image_factory, size):
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(*size)
    canvas.set_image(image_factory(title="A Very Long Museum Title " * 20))
    canvas.set_title_overlay(True)
    canvas.set_context("The travelers remember the story. " * 300,
                       "The boat crosses dark water. " * 100,
                       tuple(f"Visual subject {number}" for number in range(50)))
    canvas.set_context_overlay(True)
    canvas.show()
    app.processEvents()
    try:
        bounds = QRectF(canvas.context_panel.geometry())
        title_bounds = canvas._title_layout()[2]
        assert bounds.width() > 0 and bounds.height() > 0
        assert QRectF(canvas.rect()).contains(bounds)
        assert not bounds.intersects(title_bounds)
        scrollbar = canvas.context_text.verticalScrollBar()
        assert scrollbar.maximum() > 0
        scrollbar.setValue(scrollbar.maximum())
        app.processEvents()
        assert scrollbar.value() == scrollbar.maximum()
        assert "Visual subject 49" in canvas.context_text.toPlainText()
        canvas.resize(size[0] + 100, size[1] + 80)
        app.processEvents()
        resized = QRectF(canvas.context_panel.geometry())
        assert resized != bounds and QRectF(canvas.rect()).contains(resized)
        assert not resized.intersects(canvas._title_layout()[2])
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_empty_context_has_waiting_message_and_new_book_clears_previous_text(window, app, image_factory):
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    assert window.canvas.context_panel.isVisible()
    assert "waiting" in window.canvas.context_text.toPlainText().lower()
    assert window.canvas.current_pixmap.isNull() and not window.canvas.current_title
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "images", [image_factory()])
    window._reset_context()
    app.processEvents()
    assert window._fullscreen_context and window.canvas.context_overlay_visible
    text = window.canvas.context_text.toPlainText()
    assert "waiting" in text.lower() and "Travelers visit" not in text
    assert window.canvas.current_pixmap.isNull() and not window.canvas.current_title

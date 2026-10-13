"""Exact time ranges and live image filters leave the spoken story intact."""
from dataclasses import asdict
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QFontDatabase, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.range_slider import RangeSlider
from semantic_image_stream.temporal_filter import parse_year


class LocalFilterBoundary:
    def __init__(self):
        self.updates = []

    def update_image_filters(self, settings):
        self.updates.append(settings)

    def record_displayed(self, candidate):
        pass

    def get_feedback_vote(self, topic, candidate):
        return None

    def stop(self):
        pass


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    # The Windows offscreen plugin starts without system font enumeration.
    # Use installed fonts in this process so layout checks measure real text.
    if not QFontDatabase.families() and os.name == "nt":
        for name in ("segoeui.ttf", "georgia.ttf", "georgiai.ttf"):
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if path.is_file():
                assert QFontDatabase.addApplicationFont(str(path)) >= 0
    return application


@pytest.fixture
def window_factory(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    windows = []

    def make(**overrides):
        settings = dict(cache_dir=str(tmp_path / "cache"), training_dir=str(tmp_path / "training"),
                        session_dir=str(tmp_path / "sessions"), retain_context=False,
                        providers=["wikimedia", "archive"], relevance_threshold=.1,
                        book_context="The listener's own background", whisper_model="large-v3")
        settings.update(overrides)
        window = MainWindow(Settings(**settings), config_path=tmp_path / "config.toml")
        window.pipeline = LocalFilterBoundary()
        window.resize(980, 720)
        window.show()
        app.processEvents()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window._monitoring = False
        window._stopping = False
        window.close()
        window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture
def slider(app):
    widget = RangeSlider()
    widget.resize(420, 40)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def edit(widget, text):
    widget.setFocus()
    widget.selectAll()
    QTest.keyClicks(widget, text)


def test_saved_date_choices_and_strict_unknown_ai_defaults_are_visible(window_factory):
    window = window_factory(time_filter_enabled=True, time_filter_target="created",
                            time_filter_start=parse_year("500 BCE"), time_filter_end=1900,
                            time_filter_include_undated=True, exclude_ai_generated=False)
    assert window.time_filter_target.currentData() == "created"
    assert window.time_filter_start.text() == "500 BCE"
    assert window.time_filter_end.text() == "1900 CE"
    assert window.time_range.values == (-499, 1900)
    assert window.time_filter_include_undated.isChecked()
    assert not window.exclude_ai_generated.isChecked()
    default = window_factory()
    assert not default.time_filter_enabled.isChecked()
    assert not default.time_filter_include_undated.isChecked()
    assert default.exclude_ai_generated.isChecked()
    assert not default.time_filter_panel.isEnabled()
    assert "metadata" in default.exclude_ai_generated.toolTip()


def test_exact_deep_time_fields_survive_scale_changes_and_save_restart(window_factory):
    window = window_factory(time_filter_enabled=True)
    edit(window.time_filter_start, "4.6 Ga")
    edit(window.time_filter_end, "66 Ma")
    expected = (parse_year("4.6 Ga"), parse_year("66 Ma"))
    assert window.time_range.values == expected
    original_text = window.time_filter_start.text(), window.time_filter_end.text()
    for scale in ("geological", "calendar", "geological"):
        window.time_filter_scale.setCurrentIndex(window.time_filter_scale.findData(scale))
        assert window.time_range.values == expected
        assert (window.time_filter_start.text(), window.time_filter_end.text()) == original_text
        settings = window._read_settings(require_recording=False)
        assert (settings.time_filter_start, settings.time_filter_end) == expected
    window.save_settings()
    saved = Settings.load(window.config_path)
    assert (saved.time_filter_start, saved.time_filter_end) == expected
    assert saved.time_filter_scale == "geological"
    restored = window_factory(time_filter_enabled=saved.time_filter_enabled,
                              time_filter_start=saved.time_filter_start, time_filter_end=saved.time_filter_end,
                              time_filter_scale=saved.time_filter_scale)
    assert restored.time_range.values == expected
    assert restored._read_settings(False).time_filter_start == expected[0]


def test_subyear_slider_values_are_not_lost_to_formatted_labels(window_factory):
    window = window_factory(time_filter_enabled=True, time_filter_scale="geological")
    exact = (-66_123_456.123456, -11_700.987654)
    window.time_range.set_values(*exact)
    window._time_slider_changed(*exact)
    for scale in ("calendar", "geological"):
        window.time_filter_scale.setCurrentIndex(window.time_filter_scale.findData(scale))
        settings = window._read_settings(False)
        assert (settings.time_filter_start, settings.time_filter_end) == exact


def test_invalid_or_reversed_fields_do_not_update_running_filters(window_factory):
    window = window_factory(time_filter_enabled=True)
    window._monitoring = True
    edit(window.time_filter_start, "2100 CE")
    QTest.qWait(350)
    assert not window.pipeline.updates
    assert "older" in window.filter_status_label.text()
    edit(window.time_filter_start, "0")
    QTest.qWait(350)
    assert not window.pipeline.updates
    assert "year zero" in window.filter_status_label.text()
    valid_range = window._filter_last_valid_range
    window.time_filter_enabled.setChecked(False)
    QTest.qWait(350)
    assert len(window.pipeline.updates) == 1
    assert not window.pipeline.updates[0].time_filter_enabled
    assert (window.pipeline.updates[0].time_filter_start, window.pipeline.updates[0].time_filter_end) == valid_range


def test_live_edits_are_debounced_immutable_and_preserve_story_audio_and_saved_choices(window_factory):
    window = window_factory(time_filter_enabled=True)
    before = asdict(window.settings)
    topic = Topic("fossil coastline", "Private story", summary="A visit to the fossil coastline.",
                  scene="A fossil lies on the coastline.", visual_subjects=("fossil", "coastline"))
    window._handle_event(0, "context", {"phase": "ready", "topic_snapshot": topic,
                        "playback_key": "scene:1", "summary": topic.summary, "scene": topic.scene})
    window._handle_event(0, "transcript", [TranscriptSegment(0, 2, "Private story")])
    text, context, playback_key = window.transcript.toPlainText(), dict(window._context_payload), window._playback_key
    window._monitoring = True
    window._update_controls()
    assert not window.audio_mode.isEnabled()
    assert window.time_filter_target.isEnabled() and window.time_range.isEnabled()
    assert window.exclude_ai_generated.isEnabled()
    edit(window.time_filter_start, "1500 CE")
    edit(window.time_filter_end, "1800 CE")
    window.time_filter_target.setCurrentIndex(window.time_filter_target.findData("created"))
    window.time_filter_include_undated.setChecked(True)
    window.exclude_ai_generated.setChecked(False)
    QTest.qWait(350)
    assert len(window.pipeline.updates) == 1
    first = window.pipeline.updates[0]
    assert (first.time_filter_start, first.time_filter_end) == (1500, 1800)
    assert first.time_filter_target == "created" and first.time_filter_include_undated
    assert not first.exclude_ai_generated
    for field in ("audio_mode", "audio_device", "whisper_model", "providers", "book_context", "relevance_threshold"):
        assert getattr(first, field) == before[field]
    assert asdict(window.settings) == before
    assert window.transcript.toPlainText() == text and window._context_payload == context
    assert window._playback_key == playback_key
    edit(window.time_filter_end, "1900 CE")
    window.exclude_ai_generated.setChecked(True)
    QTest.qWait(350)
    assert len(window.pipeline.updates) == 2
    assert first.time_filter_end == 1800 and not first.exclude_ai_generated
    assert window.pipeline.updates[1].time_filter_end == 1900
    assert window.pipeline.updates[1].exclude_ai_generated
    window._stopping = True
    window._update_controls()
    assert not window.time_filter_enabled.isEnabled() and not window.exclude_ai_generated.isEnabled()
    assert not window.time_filter_panel.isEnabled()


def test_two_handles_have_keyboard_focus_and_cannot_cross(slider, app):
    slider.set_values(1000, 1500)
    first, second = slider._handles
    first.setFocus()
    app.processEvents()
    QTest.keyClick(first, Qt.Key.Key_Right)
    assert slider.values[0] > 1000 and slider.values[1] == 1500
    QTest.keyClick(first, Qt.Key.Key_Tab)
    assert second.hasFocus()
    QTest.keyClick(second, Qt.Key.Key_Left)
    assert slider.values[0] < slider.values[1] < 1500
    first.setValue(10_000)
    assert slider.values[0] == slider.values[1]
    second.setValue(0)
    assert slider.values[0] == slider.values[1]
    assert "Start" in first.accessibleName() and "End" in second.accessibleName()


def test_geological_slider_is_nonlinear_and_covers_the_supported_oldest_age(slider):
    exact = (parse_year("4.6 Ga"), parse_year("66 Ma"))
    slider.set_values(*exact)
    for scale in ("geological", "calendar", "geological"):
        slider.set_scale(scale)
        assert slider.values == exact
    assert slider._position(parse_year("66 Ma")) - slider._position(parse_year("4 Ga")) > .1
    assert slider._position(parse_year("1 ka")) > slider._position(parse_year("1 Ma"))
    assert slider._domain()[0] == parse_year("4.6 Ga")
    slider._handles[1].setValue(slider._handles[1].value() + 10)
    assert slider.values[0] == exact[0]
    assert exact[0] <= slider.values[1] < 1950
    slider.set_values(parse_year("14 Ga"), exact[1])
    domain = slider._domain()
    assert domain[0] == parse_year("14 Ga")
    QTest.keyClick(slider._handles[0], Qt.Key.Key_Right)
    assert parse_year("14 Ga") < slider.values[0] < exact[1]
    assert slider.values[1] == exact[1]
    assert slider._domain() == domain
    slider.set_scale("calendar")
    slider.set_scale("geological")
    assert slider._domain() == domain


def test_mouse_drag_applies_only_on_release_not_mid_drag(window_factory):
    window = window_factory(time_filter_enabled=True)
    window._monitoring = True
    slider = window.time_range
    window.findChild(QScrollArea).ensureWidgetVisible(slider)
    window.time_filter_include_undated.setChecked(True)
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(30, slider.height() // 2))
    assert slider.is_dragging
    QTest.mouseMove(slider, QPoint(60, slider.height() // 2))
    QTest.qWait(350)
    assert not window.pipeline.updates
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(60, slider.height() // 2))
    assert not slider.is_dragging
    QTest.qWait(350)
    assert len(window.pipeline.updates) == 1
    assert window.pipeline.updates[0].time_filter_start == slider.values[0]


def test_filter_ack_holds_frame_and_transcript_with_980_layout_and_fullscreen_escape(window_factory, app, tmp_path):
    window = window_factory(time_filter_enabled=True)
    images = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.png"
        pixels = QImage(128, 96, QImage.Format.Format_RGB32)
        pixels.fill(QColor("blue"))
        assert pixels.save(str(path))
        images.append(ImageCandidate(f"https://images.example/{name}", "https://source.example/image",
                                     name, "wikimedia", local_path=str(path)))
    topic = Topic("historic fossil", "Story", summary="The expedition continues.", scene="A fossil on a coast.")
    window._handle_event(0, "context", {"phase": "ready", "playback_key": "scene:1", "topic_snapshot": topic})
    window._handle_event(0, "transcript", [TranscriptSegment(0, 2, "Earlier recorded story")])
    window._handle_event(0, "images", images)
    window.slideshow.timer.setInterval(60_000)
    timer_id = window.slideshow.timer.timerId()
    text = window.transcript.toPlainText()
    assert window.slideshow.pending
    window._handle_event(0, "context", {"phase": "ready", "playback_key": "scene:2", "topic_snapshot": topic,
                        "message": "Image filters updated; the spoken story is retained."})
    assert not window.slideshow.pending
    assert window._displayed_candidate is images[0] and window.transcript.toPlainText() == text
    assert window.slideshow.timer.timerId() == timer_id
    scroll = window.findChild(QScrollArea)
    scroll.ensureWidgetVisible(window.time_filter_panel)
    app.processEvents()
    assert scroll.horizontalScrollBar().maximum() == 0
    assert window.width() == 980 and window.height() == 720
    assert window.time_range.width() >= 180
    assert window.time_filter_start.width() >= 80
    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    window.activateWindow()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    app.processEvents()
    assert not window._fullscreen and window.time_filter_enabled.isVisible()
    assert window.time_filter_start.text() == "1000 CE"
    assert window.slideshow.timer.timerId() == timer_id


def test_new_book_keeps_date_filter_and_ai_preferences(window_factory):
    window = window_factory(time_filter_enabled=True, time_filter_start=-499, time_filter_end=1900,
                            time_filter_target="created", exclude_ai_generated=False)
    before = window._collect_image_filters()
    window._handle_event(0, "transcript", [TranscriptSegment(0, 2, "Earlier book")])
    window.new_book()
    assert not window.transcript.toPlainText()
    assert window._collect_image_filters() == before

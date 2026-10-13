"""Offscreen Qt checks use real local fixtures, never a simulated runtime stream."""
import os
import json
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QDesktopServices, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow, safe_source_url
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.presets import AUDIOBOOK_PRESETS
from semantic_image_stream.slideshow import ImageCanvas, SlideshowController
from semantic_image_stream.audio_capture import AudioDevice
import semantic_image_stream.desktop_ui as desktop_ui


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def image_factory(tmp_path):
    def make(name="red", color="red"):
        path = tmp_path / f"{name}.png"
        image = QImage(100, 100, QImage.Format.Format_RGB32)
        image.fill(QColor(color))
        assert image.save(str(path))
        return ImageCandidate(
            image_url=f"https://images.example/{name}.png", source_url=f"https://sources.example/{name}",
            title=f"{name} & archive", provider="wikimedia", creator="Photographer", license="CC BY 4.0",
            local_path=str(path), score=0.82, authenticity="Source identifies historical imagery",
        )
    return make


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    # This UI test does not enumerate the developer's real audio devices.
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    widget = MainWindow(Settings(cache_dir=str(tmp_path / "cache")))
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_real_widgets_show_transcript_topic_source_and_history(window, app, image_factory):
    first, second = image_factory(), image_factory("blue", "blue")
    window._handle_event(0, "transcript", [TranscriptSegment(65, 70, "Galaxies form in dark matter halos.")])
    window._handle_event(0, "topic", Topic("galaxy formation", "Cosmology lecture", kind="scientific"))
    window._handle_event(0, "images", [first, second])
    app.processEvents()
    assert "[01:05] Galaxies" in window.transcript.toPlainText()
    assert window.topic_label.text() == "galaxy formation"
    assert window.topic_history.count() == 1
    assert "CC BY 4.0" in window.source_label.text()
    assert "red &amp; archive" in window.source_label.text()
    assert "https://sources.example/red" in window.source_label.text()
    assert window.save_button.isEnabled()
    assert not window.canvas.current_pixmap.isNull()
    assert window.image_history.count() == 1
    window.next_image()
    assert window.image_history.count() == 2
    window._select_history(window.image_history.item(1))
    assert window.slideshow.current is first


def test_paused_topic_prefetch_preserves_current_until_manual_advance(window, image_factory):
    first, second, third = image_factory(), image_factory("blue", "blue"), image_factory("green", "green")
    window._handle_event(0, "topic", Topic("cosmology", "lecture"))
    window._handle_event(0, "images", [first, second])
    window.toggle_pause()
    window._handle_event(0, "topic", Topic("botany", "next lecture"))
    window._handle_event(0, "images", [third])
    assert window.slideshow.current is first
    assert list(window.slideshow.pending) == [third]
    window.slideshow.advance()
    assert window.slideshow.current is first
    window.next_image()
    assert window.slideshow.current is third
    assert window.slideshow.paused
    assert "Resume" in window.pause_button.text()
    window.toggle_pause()
    assert window.slideshow.timer.isActive()


def test_queue_history_and_topic_history_are_bounded(window, app, image_factory):
    fixture = image_factory()
    images = []
    for index in range(90):
        image = ImageCandidate.from_dict(fixture.to_dict())
        image.image_url = f"https://images.example/image-{index}.png"
        images.append(image)
    window.slideshow.set_queue(images, "topic")
    assert len(window.slideshow.pending) == 59
    for _ in range(90):
        window.next_image()
    assert len(window.slideshow.history) == 60
    assert window.image_history.count() == 60
    assert not window.slideshow.pending
    window.slideshow.set_queue(images, "topic")
    assert not window.slideshow.pending  # Repeated batches do not requeue previously shown images.
    assert window.slideshow.replay_count == 60
    assert len(window.slideshow._seen) == 60
    for index in range(50):
        window._handle_event(0, "topic", Topic(f"topic {index}", "context"))
    assert window.topic_history.count() == 30
    assert window.topic_history.item(0).text().startswith("topic 49")


def test_broken_paths_are_skipped_and_empty_results_keep_current(app, image_factory, tmp_path):
    controller = SlideshowController()
    first = image_factory()
    missing = ImageCandidate("https://example/missing", "https://example/source", "missing", "wikimedia",
                             local_path=str(tmp_path / "missing.png"))
    controller.set_queue([missing, first], "one")
    assert controller.current is first
    controller.set_queue([], "two")
    controller.advance(manual=True)
    assert controller.current is first
    controller.timer.stop()


@pytest.mark.parametrize("transition", ["crossfade", "dissolve", "zoom", "cut"])
def test_transition_renders_real_image_and_finishes(app, image_factory, transition):
    canvas = ImageCanvas()
    canvas.resize(640, 480)
    canvas.show()
    canvas.transition = "cut"
    canvas.set_image(image_factory())
    canvas.transition = transition
    canvas.set_image(image_factory("blue", "blue"))
    canvas._started = time.monotonic() - (canvas.interval / 2 if transition == "zoom" else canvas.transition_seconds / 2)
    canvas.update()
    app.processEvents()
    rendered = canvas.grab().toImage()
    center = rendered.pixelColor(320, 240)
    if transition == "crossfade":
        assert center.red() > 80 and center.blue() > 80
    elif transition in {"zoom", "cut"}:
        assert center.blue() > 240 and center.red() < 10
    else:
        # Dissolve reveals a stable set of rectangles from the real next photograph.
        samples = {rendered.pixelColor(x, y).name() for x in (110, 220, 330, 440, 530) for y in (80, 160, 240, 320, 400)}
        assert "#ff0000" in samples and "#0000ff" in samples
    canvas._started = time.monotonic() - 200
    canvas._frame()
    app.processEvents()
    assert not canvas.animation_timer.isActive()
    assert canvas.previous_pixmap.isNull()
    assert canvas.grab().toImage().pixelColor(320, 240).blue() > 240
    canvas.close()


@pytest.mark.parametrize("unsafe", ["file:///C:/Windows/system32/cmd.exe", "javascript:alert(1)",
                                     "https://user:password@example.com/image", "https://example.com/\nimage", "not a url"])
def test_source_links_reject_unsafe_schemes_and_credentials(unsafe):
    assert safe_source_url(unsafe) is None


def test_presentation_controls_and_stale_background_events(window, app):
    window.interval.setValue(15)
    window.transition.setCurrentIndex(window.transition.findData("dissolve"))
    window.attribution.setChecked(False)
    assert window.slideshow.timer.interval() == 15000
    assert window.canvas.transition == "dissolve"
    assert window.source_label.isHidden()
    window._handle_event(999, "topic", Topic("outdated", "stale worker"))
    assert window.current_topic == ""
    window.toggle_fullscreen()
    app.processEvents()
    assert window._fullscreen and window.sidebar.isHidden()
    window.leave_fullscreen()
    assert not window._fullscreen and not window.sidebar.isHidden()


def test_settings_roundtrip_preserves_audio_and_provider_choices(window, tmp_path):
    destination = tmp_path / "chosen.toml"
    window.config_path = destination
    window.provider_boxes["duckduckgo"].setChecked(True)
    window.interval.setValue(12)
    window.save_settings()
    loaded = Settings.load(destination)
    assert loaded.providers == ["wikimedia", "duckduckgo"]
    assert loaded.slideshow_interval == 12


def test_source_url_allows_real_https_attribution():
    assert safe_source_url("https://commons.wikimedia.org/wiki/File:Galaxy.jpg") is not None


def test_source_click_and_save_call_real_attribution_export(window, image_factory, tmp_path, monkeypatch):
    image = image_factory()
    window.slideshow.set_queue([image], "topic")
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toString()))
    window.source_label.linkActivated.emit(image.source_url)
    window.source_label.linkActivated.emit("file:///C:/Windows/system32/cmd.exe")
    assert opened == [image.source_url]
    destination = tmp_path / "selected.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    window.save_current()
    assert destination.read_bytes() == image.path.read_bytes()
    metadata = json.loads(destination.with_suffix(".png.json").read_text(encoding="utf-8"))
    assert metadata["source_url"] == image.source_url
    assert metadata["license"] == "CC BY 4.0"


def test_real_pipeline_capture_error_restores_controls_without_hanging(window, app, tmp_path):
    recording = tmp_path / "invalid.wav"
    recording.write_bytes(b"This is intentionally not an audio file.")
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audio_file.setText(str(recording))
    window.start_monitoring()
    assert not window.start_button.isEnabled()
    assert not window.threshold.isEnabled()
    deadline = time.monotonic() + 15
    while window._monitoring and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window._monitoring, window.status_label.text()
    assert window.start_button.isEnabled()
    assert window.threshold.isEnabled()
    assert "capture" in window.status_label.text().lower() or "audio" in window.status_label.text().lower()


def test_decode_failure_remains_visible_after_ready_and_is_not_saved(window, image_factory, tmp_path):
    path = tmp_path / "unreadable.jpg"
    path.write_bytes(b"not an image")
    unreadable = ImageCandidate("https://example.com/broken.jpg", "https://example.com/source", "broken", "wikimedia",
                                local_path=str(path))
    recorded = []

    class RecordingPipeline:
        def record_displayed(self, image):
            recorded.append(image)

        def stop(self):
            pass

    window.pipeline = RecordingPipeline()
    window._handle_event(0, "images", [unreadable])
    assert window.canvas.current_pixmap.isNull()
    assert not window.save_button.isEnabled()
    assert recorded == []
    assert "Could not display image" in window.image_status_label.text()
    window._handle_event(0, "status", "Ready: 1 relevant images")
    assert "Could not display image" in window.image_status_label.text()
    assert "Could not display image" in window.canvas.empty_message
    valid = image_factory()
    window._handle_event(0, "images", [valid])
    assert not window.canvas.current_pixmap.isNull()
    assert window.save_button.isEnabled()
    assert "Could not display image" not in window.image_status_label.text()
    assert recorded == [valid]
    assert window._displayed_candidate is valid


def test_retrieval_explains_empty_preview_and_audio_status_does_not_replace_it(window):
    window._handle_event(0, "retrieval", {
        "phase": "empty", "message": "No images could be loaded from the selected sources.",
        "found": 9, "downloaded": 0, "rejected": 9, "accepted": 0,
        "threshold": 0.3, "best_score": None, "failures": {"Image links unavailable": 9},
    })
    text = window.image_status_label.text()
    assert "No images could be loaded" in text
    assert "Found 9" in text and "Loaded 0" in text and "Ready 0" in text
    assert "Image links unavailable: 9" in text
    assert window.canvas.empty_message == text
    window._handle_event(0, "status", "Listening: speakers")
    assert window.image_status_label.text() == text
    assert window.status_label.text() == "Listening: speakers"


def test_retrieval_shows_relevance_minimum_and_paused_queue_in_empty_preview(window, image_factory):
    window._handle_event(0, "retrieval", {
        "phase": "empty", "message": "No images reached the relevance minimum.",
        "found": 7, "downloaded": 5, "rejected": 5, "accepted": 0,
        "threshold": 0.3, "best_score": 0.24, "failures": {},
    })
    assert "Best relevance 0.24; minimum 0.30" in window.image_status_label.text()
    window.toggle_pause()
    window._handle_event(0, "images", [image_factory()])
    assert window.canvas.current_pixmap.isNull()
    assert window.queue_label.text() == "Paused · 1 image ready"
    assert "Images are paused" in window.image_status_label.text()
    assert "Resume images or Next" in window.canvas.empty_message
    window.next_image()
    assert not window.canvas.current_pixmap.isNull()
    assert window.queue_label.text() == "Paused · 0 images ready"


def test_missing_cached_files_are_visible_without_a_preview(window, tmp_path):
    missing = ImageCandidate("https://example.com/missing.jpg", "https://example.com/source", "missing", "wikimedia",
                             local_path=str(tmp_path / "missing.jpg"))
    window._handle_event(0, "images", [missing])
    window._handle_event(0, "status", "Ready: 1 relevant images")
    assert "image files are unavailable" in window.image_status_label.text()
    assert window.canvas.current_pixmap.isNull()
    assert not window.save_button.isEnabled()


def test_canvas_reports_success_or_failure(app, image_factory, tmp_path):
    canvas = ImageCanvas()
    assert canvas.set_image(image_factory()) is True
    original = canvas.current_pixmap.cacheKey()
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"bad")
    assert canvas.set_image(ImageCandidate("https://example.com/bad.jpg", "https://example.com/source", "bad", "wikimedia",
                                          local_path=str(path))) is False
    assert canvas.current_pixmap.cacheKey() == original
    assert canvas.set_image(None) is True
    assert canvas.current_pixmap.isNull()
    canvas.close()


@pytest.fixture
def delayed_devices_window(app, monkeypatch, tmp_path):
    """Leave enumeration pending, without opening a native audio endpoint."""
    workers = []
    pipelines = []

    class DeferredWorker:
        def __init__(self, **kwargs):
            self.target = kwargs["target"]

        def start(self):
            workers.append(self)

    class NoCapturePipeline:
        def __init__(self, settings, callback, session_state=None):
            self.settings = settings
            self.callback = callback
            self.session_state = session_state
            pipelines.append(self)

        def start(self):
            pass

        def stop(self):
            self.callback("stopped", None)

    from semantic_image_stream import engine
    monkeypatch.setattr(desktop_ui, "threading", SimpleNamespace(Thread=DeferredWorker))
    monkeypatch.setattr(engine, "Pipeline", NoCapturePipeline)
    widget = MainWindow(Settings(audio_mode="microphone", audio_device=1, cache_dir=str(tmp_path / "cache")))
    widget.show()
    app.processEvents()
    assert len(workers) == 1
    yield widget, pipelines, workers
    widget.close()
    app.processEvents()


def test_start_waits_for_saved_device_discovery_without_selecting_default(delayed_devices_window):
    window, pipelines, _ = delayed_devices_window
    assert window._devices_loading
    assert not window.start_button.isEnabled()
    assert window._read_settings().audio_device == 1
    window.start_monitoring()
    assert not pipelines
    assert window.settings.audio_device == 1
    devices = [AudioDevice(1, "Selected input", 48000, 2), AudioDevice(2, "Different default", 48000, 2, is_default=True)]
    window._devices_ready(window._device_request, (devices, 1), "")
    assert window.start_button.isEnabled()
    assert window.audio_device.currentData() == 1
    window.start_monitoring()
    assert len(pipelines) == 1
    assert pipelines[0].settings.audio_device == 1


def test_unavailable_saved_device_is_preserved_until_user_selects_another(delayed_devices_window):
    window, pipelines, _ = delayed_devices_window
    window._devices_ready(window._device_request, ([AudioDevice(2, "Other input", 48000, 2, is_default=True)], 1), "")
    assert not window.start_button.isEnabled()
    assert "unavailable" in window.status_label.text().lower()
    assert window.audio_device.currentData() == 1
    assert window._read_settings().audio_device == 1
    window.start_monitoring()
    assert not pipelines and window.settings.audio_device == 1
    window.audio_device.setCurrentIndex(window.audio_device.findData(2))
    assert window.start_button.isEnabled()
    window.start_monitoring()
    assert pipelines[0].settings.audio_device == 2


def test_discovery_error_preserves_selection_and_refresh_can_recover(delayed_devices_window):
    window, pipelines, workers = delayed_devices_window
    window._devices_ready(window._device_request, ([], 1), "Device service unavailable")
    assert not window.start_button.isEnabled()
    assert window._read_settings().audio_device == 1
    assert "Device service unavailable" in window.status_label.text()
    window._load_devices()
    assert len(workers) == 2 and window._devices_loading
    window._devices_ready(window._device_request, ([AudioDevice(1, "Selected input", 48000, 2)], 1), "")
    assert window.start_button.isEnabled()
    assert window._last_error == "" and window.status_label.objectName() == "status"
    window.start_monitoring()
    assert pipelines[0].settings.audio_device == 1


def test_file_mode_starts_immediately_and_ignores_late_device_discovery(delayed_devices_window, tmp_path):
    window, pipelines, _ = delayed_devices_window
    pending_token = window._device_request
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"Fixture is not decoded by the no-capture pipeline.")
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audio_file.setText(str(recording))
    assert not window._devices_loading and window.start_button.isEnabled()
    window.start_monitoring()
    assert pipelines[0].settings.audio_mode == "file"
    window._devices_ready(pending_token, ([AudioDevice(2, "Late input", 48000, 2)], 2), "")
    assert window.audio_device.currentData() == 1
    assert pipelines[0].settings.audio_device == 1


def test_stopped_mode_change_waits_for_new_devices_without_reusing_old_index(delayed_devices_window):
    window, pipelines, workers = delayed_devices_window
    window._devices_ready(window._device_request, ([AudioDevice(1, "Microphone", 48000, 2)], 1), "")
    window.start_monitoring()
    window.stop_monitoring()
    assert not window._monitoring and not window._stopping
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("loopback"))
    assert len(workers) == 2
    assert not window.start_button.isEnabled()
    assert window._read_settings().audio_device is None
    window.start_monitoring()
    assert len(pipelines) == 1
    window._devices_ready(window._device_request, ([AudioDevice(25, "Desktop audio", 48000, 2, True, True)], None), "")
    window.audio_device.setCurrentIndex(window.audio_device.findData(25))
    window.start_monitoring()
    assert len(pipelines) == 2
    assert pipelines[1].settings.audio_mode == "loopback"
    assert pipelines[1].settings.audio_device == 25


def test_fullscreen_canvas_fills_display_and_escape_restores_controls(window, app, image_factory, tmp_path):
    window.canvas.transition = "cut"
    image = image_factory()
    window._handle_event(0, "images", [image])
    original_rect = window.geometry()
    original_minimum = window.minimumSize()
    original_margins = window.centralWidget().layout().getContentsMargins()
    window.toggle_fullscreen()
    window.activateWindow()
    app.processEvents()
    assert window.isFullScreen()
    assert window.windowHandle().screen() is QApplication.screens()[window.display.currentData()]
    assert window.canvas.mapTo(window.centralWidget(), QPoint(0, 0)) == QPoint(0, 0)
    assert window.canvas.size() == window.centralWidget().size() == window.size()
    assert window.size() == window.windowHandle().screen().size()
    for widget in (window.header, window.sidebar, window.topic_label, window.source_label, window.transport,
                   window.status_label, window.image_status_label, window.pause_button, window.next_button,
                   window.save_button, window.fullscreen_button):
        assert not widget.isVisible()
    # A rendering assertion catches remaining controls, margins, or source bars.
    rendered = window.grab().toImage()
    assert rendered.pixelColor(0, 0).red() > 240
    assert rendered.pixelColor(rendered.width() - 1, rendered.height() - 1).red() > 240
    assert rendered.save(str(tmp_path / "fullscreen.png"))
    QTest.keyClick(window, Qt.Key.Key_Escape)
    app.processEvents()
    assert not window._fullscreen and not window.isFullScreen()
    assert window.geometry() == original_rect
    assert window.minimumSize() == original_minimum
    assert window.centralWidget().layout().getContentsMargins() == original_margins
    assert window.header.isVisible() and window.sidebar.isVisible() and window.transport.isVisible()
    assert window.source_label.isVisible()
    assert window._displayed_candidate is image
    assert "CC BY 4.0" in window.source_label.text()


def test_fullscreen_restores_hidden_attribution_and_maximized_state(window, app):
    window.attribution.setChecked(False)
    window.image_status_label.hide()
    window.showMaximized()
    app.processEvents()
    assert window.isMaximized()
    window.toggle_fullscreen()
    app.processEvents()
    assert window.canvas.size() == window.size()
    # Presentation changes must not reveal a source bar during fullscreen.
    window.interval.setValue(11)
    assert not window.source_label.isVisible()
    window.leave_fullscreen()
    app.processEvents()
    assert window.isMaximized()
    assert window.source_label.isHidden()
    assert window.image_status_label.isHidden()
    assert window.header.isVisible() and window.transport.isVisible()


def test_progressive_batches_merge_pending_and_prioritize_new_images(app, image_factory):
    controller = SlideshowController()
    first, second, third = image_factory(), image_factory("blue", "blue"), image_factory("green", "green")
    controller.set_queue([first, second], "astronomy")
    controller.set_queue([first, third], "astronomy")
    assert list(controller.pending) == [second, third]
    controller.advance(manual=True)
    assert controller.current is second
    controller.advance(manual=True)
    assert controller.current is third
    controller.advance(manual=True)
    assert controller.current is first
    controller.set_queue([first, second, third], "astronomy")
    assert not controller.pending and controller.replay_count == 3
    controller.timer.stop()


def test_real_timer_cycles_past_batch_and_presentation_does_not_reset_cadence(window, image_factory):
    first, second = image_factory(), image_factory("blue", "blue")
    observed = []
    window.slideshow.current_changed.connect(lambda image: observed.append(image))
    window.interval.setValue(1)
    window._handle_event(0, "topic", Topic("astronomy", "Stars and galaxies"))
    window._handle_event(0, "images", [first, second])
    QTest.qWait(350)
    timer_id = window.slideshow.timer.timerId()
    remaining = window.slideshow.timer.remainingTime()
    window.transition.setCurrentIndex(window.transition.findData("dissolve"))
    window.attribution.setChecked(False)
    assert window.slideshow.timer.timerId() == timer_id
    assert 0 < window.slideshow.timer.remainingTime() <= remaining
    QTest.qWait(2950)
    assert len(observed) >= 4
    assert observed[:4] == [first, second, first, second]
    assert all(left is not right for left, right in zip(observed, observed[1:]))
    assert "Cycling 2 current-topic images" == window.queue_label.text()


def test_new_subject_clears_playback_before_new_results_and_never_cycles_old_topic(window, image_factory):
    first, second, third = image_factory(), image_factory("blue", "blue"), image_factory("green", "green")
    window._handle_event(0, "topic", Topic("astronomy", "Stars"))
    window._handle_event(0, "images", [first, second])
    window._handle_event(0, "topic", Topic("botany", "Plants"))
    assert window.slideshow.current is first
    assert not window.slideshow.pending and window.slideshow.replay_count == 0
    for _ in range(3):
        window.slideshow.advance()
    assert window.slideshow.current is first and len(window.slideshow.history) == 1
    window._handle_event(0, "images", [third])
    window.slideshow.advance()
    assert window.slideshow.current is third
    for _ in range(3):
        window.slideshow.advance()
    assert window.slideshow.current is third and len(window.slideshow.history) == 2


def test_cycling_pause_manual_next_and_resume_use_real_timer(window, image_factory):
    first, second = image_factory(), image_factory("blue", "blue")
    window.interval.setValue(1)
    window._handle_event(0, "images", [first, second])
    window.next_image()
    assert not window.slideshow.pending
    window.toggle_pause()
    QTest.qWait(1150)
    assert window.slideshow.current is second and len(window.slideshow.history) == 2
    window.next_image()
    assert window.slideshow.current is first and window.slideshow.paused
    assert not window.slideshow.timer.isActive()
    window.toggle_pause()
    QTest.qWait(1150)
    assert window.slideshow.current is second and len(window.slideshow.history) == 4


def test_deleted_replay_files_are_skipped_and_single_image_does_not_repeat(app, image_factory):
    controller = SlideshowController()
    first, second = image_factory(), image_factory("blue", "blue")
    controller.set_queue([first, second], "astronomy")
    controller.advance(manual=True)
    first.path.unlink()
    for _ in range(4):
        controller.advance(manual=True)
    assert controller.current is second
    assert len(controller.history) == 2 and controller.replay_count == 1
    controller.timer.stop()


def test_new_session_discards_old_topic_playback(window, image_factory, monkeypatch):
    from semantic_image_stream import engine

    class IdlePipeline:
        def __init__(self, settings, callback, session_state=None):
            self.settings, self.callback = settings, callback
            self.session_state = session_state

        def start(self):
            pass

        def stop(self):
            self.callback("stopped", None)

    monkeypatch.setattr(engine, "Pipeline", IdlePipeline)
    first, second = image_factory(), image_factory("blue", "blue")
    window._handle_event(0, "topic", Topic("old book", "Prior recording"))
    window._handle_event(0, "images", [first, second])
    window.start_monitoring()
    assert window.current_topic == "" and window.topic_label.text() == "Waiting for a topic"
    assert not window.slideshow.pending and window.slideshow.replay_count == 0
    window.slideshow.advance()
    assert window.slideshow.current is first


def test_audiobook_preset_requires_apply_and_preserves_unsaved_input_choices(window, tmp_path):
    window.config_path = tmp_path / "listener.toml"
    initial_interval = window.interval.value()
    window.model.setCurrentText("turbo")
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("microphone"))
    window.audio_device.addItem("Chosen input", 77)
    window.audio_device.setCurrentIndex(window.audio_device.findData(77))
    window.whisper_device.setCurrentIndex(window.whisper_device.findData("cpu"))
    window.transition.setCurrentIndex(window.transition.findData("zoom"))
    window.chunk_interval.setValue(8)
    window.settings.language = "en"
    cache_dir = window.settings.cache_dir
    window.audiobook_preset.setCurrentIndex(window.audiobook_preset.findData("history"))
    assert window.apply_preset_button.isEnabled()
    assert window.interval.value() == initial_interval
    assert not window.config_path.exists()
    QTest.mouseClick(window.apply_preset_button, Qt.MouseButton.LeftButton)
    chosen = window._read_settings()
    assert chosen.audio_mode == "microphone" and chosen.audio_device == 77
    assert chosen.whisper_model == "turbo" and chosen.whisper_device == "cpu"
    assert chosen.language == "en" and chosen.cache_dir == cache_dir
    assert chosen.transition == "zoom" and chosen.chunk_seconds == 8
    assert chosen.slideshow_interval == AUDIOBOOK_PRESETS["history"].slideshow_interval
    assert chosen.transcript_window_seconds == 120 and chosen.prefetch_count == 8
    assert chosen.providers == ["wikimedia", "archive"] and chosen.ranking == "context_clip"
    assert chosen.semantic_provider == "ollama"
    assert not window.config_path.exists()
    window.save_settings()
    saved = Settings.load(window.config_path)
    assert saved.transcript_window_seconds == 120 and saved.audio_device == 77
    assert saved.whisper_model == "turbo" and saved.providers == ["wikimedia", "archive"]


def test_audiobook_preset_disabled_during_monitoring_and_stopping(window):
    assert window.audiobook_preset.currentData() is None
    assert not window.apply_preset_button.isEnabled()
    window.audiobook_preset.setCurrentIndex(window.audiobook_preset.findData("fiction"))
    original = window.interval.value()
    for monitoring, stopping in ((True, False), (False, True)):
        window._monitoring, window._stopping = monitoring, stopping
        window._update_controls()
        assert not window.audiobook_preset.isEnabled() and not window.apply_preset_button.isEnabled()
        window.apply_preset()
        assert window.interval.value() == original
    window._monitoring = window._stopping = False
    window._update_controls()
    assert window.audiobook_preset.isEnabled() and window.apply_preset_button.isEnabled()


def test_audiobook_preset_can_be_chosen_before_a_recording(window):
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audiobook_preset.setCurrentIndex(window.audiobook_preset.findData("science"))
    window.apply_preset()
    assert window.settings.audio_mode == "file" and window.settings.audio_file == ""
    assert window.settings.transcript_window_seconds == 90
    assert "preset applied" in window.status_label.text()


def test_decode_failed_image_is_not_replayed_and_preserves_displayed_frame(window, image_factory, tmp_path):
    first, second = image_factory(), image_factory("blue", "blue")
    bad_path = tmp_path / "unreadable.jpg"
    bad_path.write_bytes(b"Not a decodable image")
    bad = ImageCandidate("https://example.com/bad.jpg", "https://example.com/source", "bad", "wikimedia",
                         local_path=str(bad_path))
    recorded = []

    class RecordingPipeline:
        def record_displayed(self, image):
            recorded.append(image)

        def stop(self):
            pass

    window.pipeline = RecordingPipeline()
    window._handle_event(0, "images", [first, bad, second])
    pixmap_key = window.canvas.current_pixmap.cacheKey()
    window.next_image()
    assert window.slideshow.current is first
    assert window._displayed_candidate is first
    assert window.canvas.current_pixmap.cacheKey() == pixmap_key
    assert window.save_button.isEnabled()
    assert "Could not display image" in window.image_status_label.text()
    assert bad not in window.slideshow.history and bad not in window.slideshow.pending
    assert recorded == [first]
    window._handle_event(0, "status", "Ready: 3 relevant images")
    assert "Could not display image" in window.image_status_label.text()
    window.next_image()
    assert window.slideshow.current is second and window.slideshow.replay_count == 2
    for _ in range(4):
        window.next_image()
    assert recorded == [first, second, first, second, first, second]
    assert list(window.slideshow.history) == recorded
    assert window.image_history.count() == len(recorded)
    assert bad not in window.slideshow._playlist.values()
    assert "Could not display image" not in window.image_status_label.text()


def test_full_transcript_appends_after_160_lines_without_duplicates_or_rebuilds(window):
    segments = [TranscriptSegment(index * 6, index * 6 + 5, f"Narration segment {index}.") for index in range(350)]
    window._handle_event(0, "transcript", segments[:200])
    assert window.transcript.document().blockCount() == 200
    assert "Narration segment 0." in window.transcript.toPlainText()
    changes = []
    window.transcript.document().contentsChange.connect(lambda position, removed, added: changes.append((position, removed, added)))
    original_length = len(window.transcript.toPlainText())
    window._handle_event(0, "transcript", segments)
    assert window.transcript.document().blockCount() == 350
    assert changes and all(position >= original_length and removed == 0 for position, removed, _ in changes)
    text = window.transcript.toPlainText()
    changes.clear()
    window._handle_event(0, "transcript", segments)
    assert not changes and window.transcript.toPlainText() == text
    assert text.count("Narration segment 0.") == 1
    assert "350 awaiting context update" in window.context_status.text()
    replacement = [TranscriptSegment(0, 5, "Different fixture transcript.")]
    window._handle_event(0, "transcript", replacement)
    assert window.transcript.document().blockCount() == 1
    assert "Different fixture transcript." in window.transcript.toPlainText()


@pytest.fixture
def recording_context_pipeline(monkeypatch):
    from semantic_image_stream import engine
    instances = []

    class NoCapturePipeline:
        def __init__(self, settings, callback, session_state=None):
            self.settings, self.callback, self.session_state = settings, callback, session_state
            self.deleted_history = False
            instances.append(self)

        def start(self):
            self.callback("transcript", list(self.session_state.segments))

        def stop(self):
            self.callback("stopped", None)

        def clear_history(self):
            self.deleted_history = True

    monkeypatch.setattr(engine, "Pipeline", NoCapturePipeline)
    return instances


def test_stop_start_reuses_all_session_text_and_state(window, recording_context_pipeline):
    window.start_monitoring()
    first = recording_context_pipeline[-1]
    first.session_state.segments.append(TranscriptSegment(0, 5, "The ship entered orbit."))
    first.callback("transcript", list(first.session_state.segments))
    first.callback("context", {"phase": "ready", "summary": "A spacecraft entered orbit.",
                               "segments_used": 1, "segments_recorded": 1})
    window.stop_monitoring()
    assert not window._monitoring
    window.start_monitoring()
    second = recording_context_pipeline[-1]
    assert first is not second and first.session_state is second.session_state is window._session_state
    assert "A spacecraft entered orbit." in window.context_text.toPlainText()
    second.session_state.segments.append(TranscriptSegment(6, 11, "It deployed its antenna."))
    second.callback("transcript", list(second.session_state.segments))
    assert window.transcript.document().blockCount() == 2
    assert window.transcript.toPlainText().count("The ship entered orbit.") == 1
    assert "It deployed its antenna." in window.transcript.toPlainText()


def test_new_book_clears_local_context_and_seed_and_rejects_prior_events(window, recording_context_pipeline):
    window.book_context.setText("Example book background")
    window.start_monitoring()
    previous = recording_context_pipeline[-1]
    previous.session_state.segments.append(TranscriptSegment(0, 5, "An old book scene."))
    previous.callback("transcript", list(previous.session_state.segments))
    previous.callback("context", {"phase": "ready", "summary": "Old interpretation.", "segments_used": 1})
    window.stop_monitoring()
    window.new_book()
    assert window.book_context.text() == "" and window.settings.book_context == ""
    assert window._session_state is None and window.pipeline is None
    assert not window.transcript.toPlainText() and not window.context_text.toPlainText()
    assert not window._transcript_segments and not window._context_payload
    assert "cleared locally" in window.status_label.text()
    previous.callback("transcript", list(previous.session_state.segments))
    assert not window.transcript.toPlainText()
    window.start_monitoring()
    assert recording_context_pipeline[-1].session_state is not previous.session_state
    assert not recording_context_pipeline[-1].session_state.segments


def test_changing_book_seed_starts_a_fresh_explained_context(window, recording_context_pipeline):
    window.start_monitoring()
    previous = recording_context_pipeline[-1]
    previous.session_state.segments.append(TranscriptSegment(0, 5, "Previous context."))
    previous.callback("transcript", list(previous.session_state.segments))
    window.stop_monitoring()
    window.book_context.setText("A new historical book")
    window.start_monitoring()
    current = recording_context_pipeline[-1]
    assert current.session_state is not previous.session_state
    assert current.settings.book_context == "A new historical book"
    assert not window.transcript.toPlainText()
    assert "Book/context changed" in window.status_label.text()


def test_delete_history_also_resets_all_local_context(window, recording_context_pipeline):
    window.start_monitoring()
    previous = recording_context_pipeline[-1]
    previous.session_state.segments.append(TranscriptSegment(0, 5, "Recorded locally."))
    previous.callback("transcript", list(previous.session_state.segments))
    window.stop_monitoring()
    window.clear_history()
    assert previous.deleted_history
    assert window._session_state is None and not window._transcript_segments
    assert not window.transcript.toPlainText() and not window.context_text.toPlainText()


def test_context_panel_shows_full_coverage_backlog_and_model_error_without_stopping_audio(window):
    segments = [TranscriptSegment(index * 6, index * 6 + 5, f"Narration {index}") for index in range(5)]
    window._monitoring = True
    window._handle_event(0, "transcript", segments)
    window._handle_event(0, "context", {"phase": "ready", "summary": "A secret reconnaissance mission.",
                                      "scene": "A spacecraft passes over Earth.", "visual_subjects": ["reconnaissance satellite"],
                                      "segments_used": 3, "segments_recorded": 5, "method": "ollama", "model": "qwen2.5:7b"})
    assert "Using 3 of 5 recorded segments" in window.context_status.text()
    assert "2 awaiting context update" in window.context_status.text()
    assert "qwen2.5:7b" in window.context_status.text()
    assert "A secret reconnaissance mission." in window.context_text.toPlainText()
    assert "reconnaissance satellite" in window.context_text.toPlainText()
    window._handle_event(0, "context", {"phase": "error", "message": "The local model is unavailable."})
    assert window._monitoring
    assert "local model is unavailable" in window.context_status.text()
    assert window.tabs.tabText(window._context_tab_index) == "Context · error"
    assert "A secret reconnaissance mission." in window.context_text.toPlainText()
    window._handle_event(0, "context", {"phase": "ready", "message": "", "segments_used": 5})
    assert "Using 5 recorded segments · full session" in window.context_status.text()
    assert window.tabs.tabText(window._context_tab_index) == "Context"
    window._monitoring = False


def test_context_changes_even_when_search_query_stays_the_same(window):
    first = Topic("reconnaissance satellite", "First narrative", summary="The satellite is launched.",
                  scene="Launch from a coastal site.", visual_subjects=("launch vehicle",), context_used_segments=1)
    second = Topic("reconnaissance satellite", "Further narrative", summary="The satellite is now in orbit.",
                   scene="Earth seen from orbit.", visual_subjects=("Earth from orbit",), context_used_segments=2)
    window._handle_event(0, "topic", first)
    window._handle_event(0, "topic", second)
    assert window.topic_history.count() == 1
    assert "The satellite is now in orbit." in window.context_text.toPlainText()
    assert "Earth seen from orbit." in window.context_text.toPlainText()
    assert "launch vehicle" not in window.context_text.toPlainText()


def test_context_controls_persist_and_are_disabled_during_capture(window, tmp_path):
    window.config_path = tmp_path / "context.toml"
    window.interpreter.setCurrentIndex(window.interpreter.findData("ollama"))
    assert window.context_model.isEnabled()
    window.context_model.setText("qwen2.5:14b")
    window.book_context.setText("A book about a historical space program")
    window.ranking.setCurrentIndex(window.ranking.findData("context_clip"))
    window.save_settings()
    saved = Settings.load(window.config_path)
    assert saved.semantic_provider == "ollama" and saved.context_model == "qwen2.5:14b"
    assert saved.book_context == "A book about a historical space program" and saved.ranking == "context_clip"
    for monitoring, stopping in ((True, False), (False, True)):
        window._monitoring, window._stopping = monitoring, stopping
        window._update_controls()
        assert not window.interpreter.isEnabled() and not window.context_model.isEnabled()
        assert not window.book_context.isEnabled() and not window.new_book_button.isEnabled()
        window.new_book()
        assert window.book_context.text() == saved.book_context
    window._monitoring = window._stopping = False
    window._update_controls()
    assert window.interpreter.isEnabled() and window.context_model.isEnabled()
    window.interpreter.setCurrentIndex(window.interpreter.findData("contextual"))
    assert not window.context_model.isEnabled() and window.book_context.isEnabled()


def test_significant_scene_key_clears_same_query_playback_without_summary_churn(window, image_factory):
    first, second, third = image_factory(), image_factory("blue", "blue"), image_factory("green", "green")
    window._handle_event(0, "context", {"phase": "ready", "summary": "A satellite launch.",
                                      "scene": "A rocket lifts off.", "visual_subjects": ["rocket launch"],
                                      "segments_used": 2, "segments_recorded": 2, "playback_key": 1})
    window._handle_event(0, "topic", Topic("reconnaissance satellite", "Launch narrative"))
    window._handle_event(0, "images", [first, second])
    assert list(window.slideshow.pending) == [second] and window.slideshow.replay_count == 2
    window._handle_event(0, "context", {"phase": "ready", "summary": "The mission begins with a launch.",
                                      "scene": "The launch vehicle takes off.", "visual_subjects": ["rocket launch"],
                                      "segments_used": 3, "segments_recorded": 3, "playback_key": 1})
    assert list(window.slideshow.pending) == [second] and window.slideshow.replay_count == 2
    window._handle_event(0, "context", {"phase": "ready", "summary": "The same mission continues in orbit.",
                                      "scene": "Earth viewed from orbit.", "visual_subjects": ["Earth from orbit"],
                                      "segments_used": 4, "segments_recorded": 4, "playback_key": 2})
    assert window.current_topic == "reconnaissance satellite"
    assert window.slideshow.current is first
    assert not window.slideshow.pending and window.slideshow.replay_count == 0
    window.slideshow.advance()
    assert window.slideshow.current is first
    window._handle_event(0, "images", [third])
    window._handle_event(0, "images", [third, first])
    assert list(window.slideshow.pending) == [third, first]
    window.next_image()
    assert window.slideshow.current is third

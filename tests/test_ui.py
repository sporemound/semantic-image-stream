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
    assert len(window.slideshow.pending) == 30  # Only unseen members of this topic remain.
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
        def __init__(self, settings, callback):
            self.settings = settings
            self.callback = callback
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

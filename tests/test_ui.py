"""Offscreen Qt checks use real local fixtures, never a simulated runtime stream."""
import os
import json
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QColor, QDesktopServices, QImage
from PySide6.QtWidgets import QApplication, QFileDialog

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow, safe_source_url
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.slideshow import ImageCanvas, SlideshowController


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

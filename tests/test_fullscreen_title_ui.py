"""The title view paints the displayed image's title without changing fullscreen playback."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFontDatabase, QFontMetricsF, QImage, QRawFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate, Topic
from semantic_image_stream.slideshow import ImageCanvas


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    # Windows' offscreen platform does not enumerate the installed font
    # database. Register existing system fonts for this test process only.
    if not QFontDatabase.families() and os.name == "nt":
        for name in ("georgia.ttf", "georgiai.ttf"):
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
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def cream_pixels(rendered, bounds):
    result = []
    for y in range(max(0, int(bounds.top()) - 4), min(rendered.height(), int(bounds.bottom()) + 5)):
        for x in range(max(0, int(bounds.left()) - 4), min(rendered.width(), int(bounds.right()) + 5)):
            color = rendered.pixelColor(x, y)
            if color.red() > 180 and color.green() > 165 and color.blue() < 240:
                result.append((x, y))
    return result


def test_title_button_uses_actual_image_title_and_escape_restores_clean_normal_view(window, app, image_factory):
    image = image_factory()
    window.canvas.transition = "cut"
    window._handle_event(0, "topic", Topic("A different spoken book title", "Narration"))
    window._handle_event(0, "images", [image])
    normal = window.geometry()
    window.fullscreen_title_button.click()
    window.activateWindow()
    app.processEvents()
    assert window.isFullScreen() and window._fullscreen_title
    assert window.canvas.title_overlay_visible
    assert window.canvas.current_title == image.title
    assert window.canvas.current_title != window.current_topic
    assert window.canvas.size() == window.size()
    assert not window.transport.isVisible() and not window.sidebar.isVisible()
    assert not window.source_label.isVisible() and not window.image_status_label.isVisible()
    font, lines, bounds = window.canvas._title_layout()
    assert font.italic() and font.pixelSize() >= 20
    assert QRawFont.fromFont(font).supportsCharacter("M")
    assert cream_pixels(window.canvas.grab().toImage(), bounds)
    assert lines and bounds.left() >= 18 and bounds.bottom() < window.canvas.height()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    app.processEvents()
    assert not window._fullscreen and not window.canvas.title_overlay_visible
    assert window.geometry() == normal
    assert window._displayed_candidate is image and window.source_label.isVisible()


def test_switching_clean_and_title_views_preserves_image_queue_history_and_timer(window, app, image_factory):
    first, second = image_factory(), image_factory("two", "Second Image")
    window._handle_event(0, "images", [first, second])
    window.slideshow.timer.setInterval(60_000)
    timer_id = window.slideshow.timer.timerId()
    frame = window.canvas.current_pixmap.cacheKey()
    pending, history = list(window.slideshow.pending), list(window.slideshow.history)
    normal = window.geometry()
    window.toggle_fullscreen()
    app.processEvents()
    clean = window.canvas.grab().toImage()
    assert not window.canvas.title_overlay_visible
    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    assert window.isFullScreen() and window._fullscreen_title
    titled = window.canvas.grab().toImage()
    assert titled != clean
    assert window.slideshow.timer.timerId() == timer_id
    assert window.canvas.current_pixmap.cacheKey() == frame
    assert list(window.slideshow.pending) == pending and list(window.slideshow.history) == history
    window.toggle_fullscreen(with_title=False)
    app.processEvents()
    assert window.isFullScreen() and not window.canvas.title_overlay_visible
    assert window.canvas.grab().toImage() == clean
    window.leave_fullscreen()
    app.processEvents()
    assert window.geometry() == normal


def test_failed_decode_keeps_previous_image_title_and_next_transition_updates_it(window, image_factory, tmp_path, app):
    first = image_factory()
    next_image = image_factory("two", "The Shore at Dawn", "cyan")
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"Not a decodable raster")
    bad = ImageCandidate("https://images.example/bad", "https://source.example/bad", "Wrong failed title",
                         "wikimedia", local_path=str(bad_path))
    window._handle_event(0, "images", [first, bad, next_image])
    window.toggle_fullscreen(with_title=True)
    window.next_image()
    assert window._displayed_candidate is first
    assert window.canvas.current_title == first.title
    assert "Could not display image" in window.image_status_label.text()
    window.canvas.transition = "dissolve"
    window.next_image()
    app.processEvents()
    assert window._displayed_candidate is next_image
    assert window.canvas.current_title == next_image.title
    assert window.canvas.animation_timer.isActive()
    assert cream_pixels(window.canvas.grab().toImage(), window.canvas._title_layout()[2])


@pytest.mark.parametrize("size,title", [
    ((420, 260), "A Very Long Descriptive Museum Title " * 25),
    ((960, 540), "A" * 1000),
    ((1600, 900), "The River, the Mountains, and the Light at the End of the Day"),
])
def test_long_titles_wrap_fit_and_render_inside_margins(app, image_factory, size, title):
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(*size)
    canvas.set_title_overlay(True)
    canvas.set_image(image_factory(title=title))
    canvas.show()
    app.processEvents()
    try:
        font, lines, bounds = canvas._title_layout()
        assert 1 <= len(lines) <= 3
        assert bounds.left() > 0 and bounds.right() < canvas.width()
        assert bounds.top() > 0 and bounds.bottom() < canvas.height()
        assert all(QFontMetricsF(font).horizontalAdvance(line) <= bounds.width() + 1 for line in lines)
        positions = cream_pixels(canvas.grab().toImage(), bounds)
        assert len(positions) > 30
        assert min(x for x, _ in positions) >= bounds.left() - 4
        assert max(y for _, y in positions) < canvas.height()
        canvas.resize(size[0] + 100, size[1] + 80)
        app.processEvents()
        resized_bounds = canvas._title_layout()[2]
        assert resized_bounds != bounds and resized_bounds.bottom() < canvas.height()
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_title_option_is_available_at_small_window_and_blank_canvas_has_no_overlay(window, app):
    window.resize(980, 690)
    app.processEvents()
    assert window.fullscreen_title_button.isVisible()
    assert window.transport.width() <= window.stage.width()
    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    assert not window.canvas.current_title and window.canvas.current_pixmap.isNull()
    window.leave_fullscreen()
    assert not window.canvas.title_overlay_visible


def test_f11_keeps_its_toggle_behavior_in_title_mode(window, image_factory, app):
    window._handle_event(0, "images", [image_factory()])
    window.toggle_fullscreen(with_title=True)
    window.activateWindow()
    app.processEvents()
    QTest.keyClick(window, Qt.Key.Key_F11)
    app.processEvents()
    assert not window._fullscreen and not window.canvas.title_overlay_visible

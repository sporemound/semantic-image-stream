"""Fullscreen title mode includes artist, provider, and rights credit."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.slideshow import ImageCanvas


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_deviantart_is_selectable(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    window = MainWindow(Settings(
        cache_dir=str(tmp_path / "cache"),
        session_dir=str(tmp_path / "sessions"),
        retain_context=False,
    ))
    try:
        assert window.provider_boxes["deviantart"].text() == "DeviantArt"
        assert "Bing fallback" in window.provider_boxes["duckduckgo"].text()
        assert "artist, source, and rights" in window.fullscreen_title_button.toolTip()
    finally:
        window.close()
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_deviantart_credit_is_painted_below_title(app, tmp_path):
    path = tmp_path / "art.png"
    pixels = QImage(320, 180, QImage.Format.Format_RGB32)
    pixels.fill(QColor("#102b20"))
    assert pixels.save(str(path))
    image = ImageCandidate(
        image_url="https://images.example.org/art.png",
        source_url="https://www.deviantart.com/moonlitmoss/art/Ghost-Pipe-123",
        title="Ghost Pipe at Midnight",
        provider="deviantart",
        creator="MoonlitMoss",
        license="Artist copyright — see DeviantArt source for reuse terms",
        local_path=str(path),
    )
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(960, 540)
    canvas.set_title_overlay(True)
    assert canvas.set_image(image)
    canvas.show()
    app.processEvents()
    try:
        assert canvas.current_credit == (
            "By MoonlitMoss · DeviantArt · "
            "Artist copyright — see DeviantArt source for reuse terms"
        )
        credit_layout = canvas._credit_layout()
        assert credit_layout is not None
        _, text, bounds = credit_layout
        assert "MoonlitMoss" in text and "DeviantArt" in text
        rendered = canvas.grab().toImage()
        light_pixels = 0
        for y in range(max(0, int(bounds.top()) - 3), min(rendered.height(), int(bounds.bottom()) + 4)):
            for x in range(max(0, int(bounds.left()) - 3), min(rendered.width(), int(bounds.right()) + 4)):
                color = rendered.pixelColor(x, y)
                if color.red() > 145 and color.green() > 155 and color.blue() > 160:
                    light_pixels += 1
        assert light_pixels > 20
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

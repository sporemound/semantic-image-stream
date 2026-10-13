"""Automatic playback never repeats displayed URLs or perceptually similar images."""
from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from semantic_image_stream.configuration import Settings
from semantic_image_stream.models import ImageCandidate, TranscriptSegment
from semantic_image_stream.slideshow import SlideshowController


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def image_factory(tmp_path):
    def make(name, phash="", color="blue"):
        path = tmp_path / f"{name}.png"
        pixels = QImage(128, 96, QImage.Format.Format_RGB32)
        pixels.fill(QColor(color))
        assert pixels.save(str(path))
        return ImageCandidate(f"https://images.example/{name}.png", f"https://source.example/{name}",
                              name, "wikimedia", local_path=str(path), phash=phash)
    return make


@pytest.fixture
def controller(app):
    player = SlideshowController()
    yield player
    player.timer.stop()
    player.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_exhausted_queue_holds_frame_until_unseen_progressive_results(controller, image_factory):
    first, second, third = (image_factory(name) for name in ("first", "second", "third"))
    shown = []
    controller.current_changed.connect(shown.append)
    controller.set_queue([first, second], "scene:1")
    for _ in range(10):
        controller.advance()
    assert shown == [first, second]
    assert controller.current is second and not controller.pending
    controller.set_queue([first, second, third], "scene:1")
    controller.advance()
    assert shown == [first, second, third]
    for _ in range(10):
        controller.advance(manual=True)
    assert shown == [first, second, third]


def test_scene_change_and_reset_keep_all_displayed_urls(controller, image_factory):
    first, second, third = (image_factory(name) for name in ("first", "second", "third"))
    controller.set_queue([first, second], "scene:1")
    controller.set_topic("scene:2")
    assert not controller.pending and controller.current is first
    controller.set_queue([first, second, third], "scene:2")
    assert list(controller.pending) == [second, third]
    controller.advance()
    controller.reset_playback()
    controller.set_queue([first, second, third], "scene:3")
    assert list(controller.pending) == [third]
    controller.advance()
    assert list(controller.history) == [first, second, third]


def test_same_or_near_duplicate_pixels_at_different_url_are_suppressed(controller, image_factory):
    first = image_factory("first", "0000000000000000")
    copy = image_factory("another-url", "0000000000000000")
    near_copy = image_factory("resized-url", "000000000000003f")  # six different bits
    fresh = image_factory("fresh", "ffffffffffffffff")
    controller.set_queue([first, copy, near_copy, fresh], "scene:1")
    controller.advance()
    assert controller.current is fresh
    assert list(controller.history) == [first, fresh]
    controller.set_queue([copy, near_copy, first], "scene:2")
    assert not controller.pending


def test_restored_seen_filters_pending_and_invalid_hashes_safely(controller, image_factory):
    first = image_factory("first", "0000000000000000")
    fresh = image_factory("fresh", "ffffffffffffffff")
    controller.set_paused(True)
    controller.set_queue([first, fresh], "scene:1")
    controller.restore_seen([], ["0000000000000001", "not-a-hash", "-123456789123456", None])
    assert list(controller.pending) == [fresh]
    controller.restore_seen([fresh.image_url, "", None], [])
    assert not controller.pending
    controller.set_paused(False)
    assert controller.current is None


def test_explicit_history_revisit_never_puts_that_image_back_in_automatic_queue(controller, image_factory):
    first, second, third = (image_factory(name) for name in ("first", "second", "third"))
    controller.set_queue([first, second], "scene:1")
    controller.advance()
    controller.select_history(0)
    assert controller.current is first
    controller.set_queue([first, second, third], "scene:1")
    controller.advance()
    assert controller.current is third
    assert list(controller.history) == [first, second, third]


def test_only_explicit_clear_seen_allows_an_image_to_return(controller, image_factory):
    first = image_factory("first", "0000000000000000")
    controller.set_queue([first], "scene:1")
    controller.clear(clear_seen=False)
    controller.set_queue([first], "new-book")
    assert controller.current is None and not controller.pending
    controller.clear(clear_seen=True)
    controller.set_queue([first], "after-delete-history")
    assert controller.current is first


def test_missing_and_decode_rejected_candidates_do_not_poison_displayed_ledger(controller, image_factory):
    first = image_factory("first", "0000000000000000")
    rejected = image_factory("rejected", "ffffffffffffffff")
    missing = image_factory("missing")
    missing.path.unlink()
    controller.set_queue([first], "scene:1")

    def refuse(candidate):
        if candidate is rejected:
            controller.discard(candidate, first)

    controller.current_changed.connect(refuse)
    controller.set_queue([missing, rejected], "scene:1")
    controller.advance()
    assert controller.current is first
    assert rejected.image_url not in controller._seen
    assert int(rejected.phash, 16) not in controller._seen_hashes
    assert first.image_url in controller._seen
    assert list(controller.history) == [first]
    controller.current_changed.disconnect(refuse)
    controller.set_queue([rejected], "scene:1")
    controller.advance()
    assert controller.current is rejected


def test_pending_and_history_bound_does_not_forget_old_displayed_urls(controller, image_factory):
    template = image_factory("template")
    batch = [replace(template, image_url=f"https://images.example/{index}") for index in range(140)]
    for offset in (0, 60, 120):
        controller.set_queue(batch[offset:offset + 60], "scene:1")
        for _ in range(60):
            controller.advance()
    assert len(controller.history) == 60 and len(controller._seen) == 140
    assert not controller.pending
    controller.set_queue(batch, "scene:2")
    assert not controller.pending


def test_window_restores_durable_seen_and_new_book_preserves_it(app, monkeypatch, tmp_path, image_factory):
    from semantic_image_stream.cache import Cache
    from semantic_image_stream.desktop_ui import MainWindow

    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    settings = Settings(cache_dir=str(tmp_path / "cache"), session_dir=str(tmp_path / "sessions"),
                        training_dir=str(tmp_path / "training"), retain_context=False)
    first = image_factory("first", "0000000000000000")
    duplicate = image_factory("other-url", "0000000000000001")
    cache = Cache(settings)
    cache.record(first)
    cache.close()
    window = MainWindow(settings)
    try:
        window._handle_event(0, "images", [first, duplicate])
        assert not window.slideshow.pending and window.slideshow.current is None
        window._handle_event(0, "transcript", [TranscriptSegment(0, 5, "A different new book.")])
        window.new_book()
        window._handle_event(window._session, "images", [first, duplicate])
        assert not window.slideshow.pending and window.slideshow.current is None
        cache = Cache(settings)
        assert first.image_url in cache.seen_urls()
        cache.close()
        window.clear_cache()
        window._handle_event(window._session, "images", [first, duplicate])
        assert not window.slideshow.pending and window.slideshow.current is None
        window.clear_history()
        window._handle_event(window._session, "images", [first])
        assert window.slideshow.current is first
    finally:
        window.close()
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

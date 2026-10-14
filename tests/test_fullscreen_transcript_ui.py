"""Live speech subtitles remain readable without disturbing scene playback."""
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QRectF
from PySide6.QtGui import QFontDatabase, QFontMetricsF
from PySide6.QtWidgets import QApplication

from semantic_image_stream.models import TranscriptSegment
from semantic_image_stream.slideshow import ImageCanvas
from test_fullscreen_context_ui import app as app, image_factory as image_factory, window as window, ready_context


@pytest.fixture(scope="module", autouse=True)
def subtitle_font(app):
    # The offscreen Windows platform cannot discover system fonts by itself.
    if os.name == "nt":
        arial = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
        if arial.is_file():
            assert QFontDatabase.addApplicationFont(str(arial)) >= 0


def subtitle_pixels(canvas, bounds):
    """Find white caption strokes and their dark edge within the reserved lane."""
    pixels = canvas.grab().toImage()
    white = []
    for y in range(max(0, math.floor(bounds.top())), min(pixels.height(), math.ceil(bounds.bottom()))):
        for x in range(max(0, math.floor(bounds.left())), min(pixels.width(), math.ceil(bounds.right()))):
            color = pixels.pixelColor(x, y)
            if min(color.red(), color.green(), color.blue()) > 235:
                white.append((x, y))
    outlined = any(
        max(pixels.pixelColor(x + dx, y + dy).getRgb()[:3]) < 35
        for x, y in white
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3))
        if 0 <= x + dx < pixels.width() and 0 <= y + dy < pixels.height()
    )
    return white, outlined


def test_early_speech_renders_as_subtitle_before_context_or_image_is_ready(window, app):
    text = "The ship appeared beyond the harbor wall."
    window._handle_event(0, "transcript", [TranscriptSegment(65, 70, text)])
    window.toggle_fullscreen(with_context=True)
    app.processEvents()

    assert window.canvas.current_pixmap.isNull()
    assert "waiting" in window.canvas.context_text.toPlainText().lower()
    assert window.canvas.current_transcript == text
    _, lines, bounds = window.canvas._transcript_layout()
    assert " ".join(lines) == text
    assert "[01:05]" not in " ".join(lines)
    assert not bounds.intersects(QRectF(window.canvas.context_panel.geometry()))
    white, outlined = subtitle_pixels(window.canvas, bounds)
    assert len(white) > 20
    assert outlined


def test_live_subtitle_does_not_move_context_scroll_or_change_playback(window, app, image_factory):
    first, second = image_factory(), image_factory("two", "The Shore at Dawn", "cyan")
    window._handle_event(0, "context", ready_context(summary="The travelers remember their journey. " * 200))
    window._handle_event(0, "images", [first, second])
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    scrollbar = window.canvas.context_text.verticalScrollBar()
    assert scrollbar.maximum() > 0
    scrollbar.setValue(scrollbar.maximum() // 2)
    context_position = scrollbar.value()
    context_text = window.canvas.context_text.toPlainText()
    timer_id = window.slideshow.timer.timerId()
    frame = window.canvas.current_pixmap.cacheKey()
    pending, history = list(window.slideshow.pending), list(window.slideshow.history)

    first_passage = TranscriptSegment(0, 5, "The ship crossed the bay.")
    window._handle_event(0, "transcript", [first_passage])
    window._handle_event(0, "context", {"phase": "updating", "summary": "", "scene": "", "visual_subjects": []})
    latest = "A bell sounded from the harbor tower."
    window._handle_event(0, "transcript", [first_passage, TranscriptSegment(6, 11, latest)])
    app.processEvents()

    assert window.canvas.current_transcript == latest
    assert window.canvas.context_text.toPlainText() == context_text
    assert scrollbar.value() == context_position
    assert window.canvas.current_title == first.title
    assert window.canvas.current_pixmap.cacheKey() == frame
    assert window.slideshow.timer.timerId() == timer_id
    assert list(window.slideshow.pending) == pending
    assert list(window.slideshow.history) == history
    assert window.transcript.toPlainText().splitlines() == [
        "[00:00] The ship crossed the bay.", "[00:06] " + latest,
    ]


def test_subtitle_uses_latest_nonempty_plain_speech_and_rejects_stale_snapshots(window, app):
    first = TranscriptSegment(0, 5, "The first passage.")
    latest = '<b>The second passage.</b> & <img src="https://example.invalid/image.png">'
    passages = [first, TranscriptSegment(6, 11, "  " + latest + "\n"), TranscriptSegment(12, 13, "  ")]
    window._handle_event(0, "transcript", passages)
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    assert window.canvas.current_transcript == latest
    _, lines, _ = window.canvas._transcript_layout()
    assert "<b>" in " ".join(lines) or "<img" in " ".join(lines)

    window._handle_event(0, "transcript", passages)
    window._handle_event(0, "transcript", [])
    window._handle_event(0, "transcript", [first])
    window._handle_event(window._session - 1, "transcript", [TranscriptSegment(20, 25, "Stale worker text.")])
    assert window.canvas.current_transcript == latest


def test_stop_keeps_caption_accepts_final_speech_and_new_book_clears_it(window, app):
    first = TranscriptSegment(0, 5, "The ship entered the harbor.")
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "transcript", [first])
    window.toggle_fullscreen(with_context=True)
    window._handle_event(0, "stopped", None)
    established_context = window.canvas.context_text.toPlainText()
    window._handle_event(0, "transcript", [])
    window._handle_event(0, "transcript", [TranscriptSegment(0, 5, "A replacement after Stop.")])
    assert window.canvas.current_transcript == first.text

    final = TranscriptSegment(6, 11, "The crew lowered the anchor.")
    window._handle_event(0, "transcript", [first, final])
    window._handle_event(0, "context", ready_context(summary="A delayed interpretation."))
    assert window.canvas.current_transcript == final.text
    assert window.canvas.context_text.toPlainText() == established_context

    old_session = window._session
    window.new_book()
    app.processEvents()
    assert window._fullscreen_context and window.canvas.context_overlay_visible
    assert window.canvas.current_transcript == ""
    assert window.canvas._transcript_layout() is None
    window._handle_event(old_session, "transcript", [first, final])
    assert window.canvas.current_transcript == ""
    fresh = "A new story begins."
    window._handle_event(window._session, "transcript", [TranscriptSegment(0, 5, fresh)])
    assert window.canvas.current_transcript == fresh


def test_subtitle_is_visible_only_in_context_mode_and_renders_white_over_image(window, app, image_factory):
    window._handle_event(0, "images", [image_factory()])
    text = "The lights on the waterfront shimmered in the distance."
    window._handle_event(0, "transcript", [TranscriptSegment(0, 5, text)])
    assert window.canvas._transcript_layout() is None
    for options, visible in (({}, False), ({"with_title": True}, False),
                             ({"with_context": True}, True), ({"with_title": False}, False)):
        window.toggle_fullscreen(**options)
        app.processEvents()
        layout = window.canvas._transcript_layout()
        assert (layout is not None) is visible
        assert window.canvas.current_transcript == text
        if visible:
            white, outlined = subtitle_pixels(window.canvas, layout[2])
            assert len(white) > 20 and outlined
    window.leave_fullscreen()
    assert window.canvas._transcript_layout() is None
    assert window.canvas.current_transcript == text


def test_context_credit_footer_follows_paused_frame_and_history_selection(window, app, image_factory):
    first, second = image_factory(), image_factory("two", "The Shore at Dawn", "cyan")
    first.creator, first.license = "First artist", "CC BY 4.0"
    second.creator, second.license = "Second artist", "Public domain"
    window._handle_event(0, "context", ready_context())
    window._handle_event(0, "images", [first, second])
    window.toggle_fullscreen(with_context=True)
    window.slideshow.set_paused(True)
    app.processEvents()

    assert window.canvas.credit_text.isVisible()
    assert window.canvas.credit_text.toPlainText() == "Image credits\n" + window.canvas.current_credit
    assert "First artist" in window.canvas.credit_text.toPlainText()
    assert "Wikimedia Commons" in window.canvas.credit_text.toPlainText()
    assert "CC BY 4.0" in window.canvas.credit_text.toPlainText()
    window._handle_event(0, "transcript", [TranscriptSegment(0, 5, "A new passage is recognized.")])
    window._handle_event(0, "context", ready_context(summary="The story moves to a different location."))
    assert window._displayed_candidate is first
    assert "First artist" in window.canvas.credit_text.toPlainText()

    window.next_image()
    app.processEvents()
    assert window.slideshow.paused and window._displayed_candidate is second
    assert "Second artist" in window.canvas.credit_text.toPlainText()
    assert "First artist" not in window.canvas.credit_text.toPlainText()
    assert "Public domain" in window.canvas.credit_text.toPlainText()
    window.slideshow.select_history(0)
    app.processEvents()
    assert window.slideshow.paused and window._displayed_candidate is first
    assert "First artist" in window.canvas.credit_text.toPlainText()
    assert "Second artist" not in window.canvas.credit_text.toPlainText()
    assert window.canvas.current_transcript == "A new passage is recognized."


def test_credit_markup_stays_literal_and_bottom_credit_returns_in_title_mode(window, app, image_factory):
    image = image_factory()
    image.creator = '<b>Artist & Company</b> <a href="https://example.invalid/">Creator</a>'
    image.license = '<img src="https://example.invalid/tracker.png"> CC BY 4.0'
    window._handle_event(0, "images", [image])
    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    title_bottom = window.canvas._title_layout()[2].bottom()
    assert window.canvas._credit_layout() is not None
    window.toggle_fullscreen(with_context=True)
    app.processEvents()

    text = window.canvas.credit_text.toPlainText()
    assert image.creator in text and image.license in text
    assert window.canvas.credit_text.isReadOnly()
    assert "&lt;b&gt;" in window.canvas.credit_text.document().toHtml()
    assert "&lt;img" in window.canvas.credit_text.document().toHtml()
    assert window.canvas._credit_layout() is None
    assert window.canvas._title_layout()[2].bottom() > title_bottom

    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    assert not window.canvas.context_panel.isVisible()
    assert window.canvas._credit_layout() is not None
    assert window.canvas._title_layout()[2].bottom() == title_bottom
    assert window.canvas.current_credit in window.canvas.credit_text.toPlainText()


@pytest.mark.parametrize("size", [(420, 260), (960, 540), (1600, 900)])
def test_long_credits_scroll_independently_and_stay_inside_context_panel(app, image_factory, size):
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(*size)
    image = image_factory()
    image.creator = "Contributing artist " * 12
    image.license = "Source license details and attribution. " * 10 + "Final license clause."
    canvas.set_image(image)
    canvas.set_title_overlay(True)
    canvas.set_context("Narrative context remains independently readable. " * 150,
                       "The ship enters the harbor.", ["ship", "harbor"])
    canvas.set_transcript("The crew lowered the anchor.")
    canvas.set_context_overlay(True)
    canvas.show()
    app.processEvents()
    try:
        assert canvas.credit_text.toPlainText() == "Image credits\n" + canvas.current_credit
        assert "Final license clause." in canvas.credit_text.toPlainText()
        assert canvas.context_panel.rect().contains(canvas.credit_text.geometry())
        assert not canvas.credit_text.geometry().intersects(canvas.context_text.geometry())
        context_scroll = canvas.context_text.verticalScrollBar()
        context_scroll.setValue(context_scroll.maximum() // 2)
        context_position = context_scroll.value()
        credit_scroll = canvas.credit_text.verticalScrollBar()
        assert credit_scroll.maximum() > 0
        credit_scroll.setValue(credit_scroll.maximum())
        assert context_scroll.value() == context_position

        canvas.resize(size[0] + 100, size[1] + 80)
        app.processEvents()
        assert canvas.context_panel.rect().contains(canvas.credit_text.geometry())
        assert not canvas.credit_text.geometry().intersects(canvas.context_text.geometry())
        assert not QRectF(canvas.context_panel.geometry()).intersects(canvas._transcript_layout()[2])
        canvas.set_image(None)
        app.processEvents()
        assert not canvas.current_credit and not canvas.credit_text.toPlainText()
        assert not canvas.credit_text.isVisible()
        assert canvas.current_transcript == "The crew lowered the anchor."
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


@pytest.mark.parametrize("size", [(420, 260), (960, 540), (1600, 900)])
def test_long_speech_keeps_newest_words_and_fits_above_title_and_context(app, image_factory, size):
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(*size)
    canvas.set_image(image_factory(title="An Extremely Long Museum Image Title " * 25))
    canvas.set_title_overlay(True)
    canvas.set_context("The travelers remember the story. " * 100,
                       "They crossed the harbor at night.", ["boat", "frozen harbor"])
    tail = "The final words are the newest speech."
    canvas.set_transcript("Earlier spoken words. " * 300 + tail)
    canvas.set_context_overlay(True)
    canvas.show()
    app.processEvents()
    try:
        assert len(canvas.current_transcript) <= 1200
        assert canvas.current_transcript.endswith(tail)
        font, lines, bounds = canvas._transcript_layout()
        assert 1 <= len(lines) <= 2
        assert " ".join(lines).endswith(tail)
        assert QRectF(canvas.rect()).contains(bounds)
        assert abs(bounds.center().x() - canvas.rect().center().x()) <= 1
        assert not bounds.intersects(canvas._title_layout()[2])
        assert not bounds.intersects(QRectF(canvas.context_panel.geometry()))
        assert all(QFontMetricsF(font).horizontalAdvance(line) <= bounds.width() + 1 for line in lines)
        white, outlined = subtitle_pixels(canvas, bounds)
        assert len(white) > 20 and outlined
        canvas.resize(size[0] + 100, size[1] + 80)
        app.processEvents()
        resized = canvas._transcript_layout()[2]
        assert resized != bounds and QRectF(canvas.rect()).contains(resized)
        assert not resized.intersects(canvas._title_layout()[2])
        assert not resized.intersects(QRectF(canvas.context_panel.geometry()))
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

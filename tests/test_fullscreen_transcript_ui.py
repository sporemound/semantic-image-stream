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


def consume_subtitle_pages(canvas, app, limit=200):
    """Advance the reading clock deterministically and collect every shown word."""
    pages = []
    canvas._transcript_timer.stop()
    for _ in range(limit):
        layout = canvas._transcript_layout()
        assert layout is not None
        pages.append(" ".join(layout[1]))
        canvas._advance_transcript()
        canvas._transcript_timer.stop()
        app.processEvents()
        if canvas._transcript_idle:
            return pages
    pytest.fail("The subtitle queue did not drain within the expected number of pages.")


def displayed_image_details(canvas):
    return f"Image title\n{canvas.current_title}\n\nImage credits\n{canvas.current_credit}"


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
    consume_subtitle_pages(window.canvas, app)
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
    consume_subtitle_pages(window.canvas, app)

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
    assert not window.canvas._transcript_pending and not window.canvas._transcript_timer.isActive()
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
    assert window.canvas.credit_text.toPlainText() == displayed_image_details(window.canvas)
    assert first.title in window.canvas.credit_text.toPlainText()
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
    assert second.title in window.canvas.credit_text.toPlainText()
    assert first.title not in window.canvas.credit_text.toPlainText()
    assert "First artist" not in window.canvas.credit_text.toPlainText()
    assert "Public domain" in window.canvas.credit_text.toPlainText()
    window.slideshow.select_history(0)
    app.processEvents()
    assert window.slideshow.paused and window._displayed_candidate is first
    assert "First artist" in window.canvas.credit_text.toPlainText()
    assert first.title in window.canvas.credit_text.toPlainText()
    assert second.title not in window.canvas.credit_text.toPlainText()
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
    assert "Image title\n" + image.title in text
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
        assert canvas.credit_text.toPlainText() == displayed_image_details(canvas)
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


def test_live_batch_and_later_extensions_show_all_passages_once(window, app):
    window.toggle_fullscreen(with_context=True)
    initial = [TranscriptSegment(0, 2, "First accepted passage."),
               TranscriptSegment(2, 3, "   "),
               TranscriptSegment(3, 5, "Second accepted passage.")]
    window._handle_event(0, "transcript", initial)
    assert " ".join(window.canvas._transcript_layout()[1]) == initial[0].text
    window._handle_event(0, "transcript", initial)
    window._handle_event(0, "transcript", [])
    window._handle_event(0, "transcript", initial[:1])
    window._handle_event(-1, "transcript", [TranscriptSegment(0, 2, "Stale accepted passage.")])
    extension = [TranscriptSegment(6, 8, "Third accepted passage."),
                 TranscriptSegment(9, 11, "Fourth accepted passage.")]
    window._handle_event(0, "transcript", initial + extension)
    pages = consume_subtitle_pages(window.canvas, app)
    expected = " ".join(segment.text for segment in initial + extension).split()
    assert " ".join(pages).split() == expected
    assert "Stale" not in " ".join(pages)
    final_page = window.canvas._transcript_layout()[1]
    window.canvas._advance_transcript()
    assert window.canvas._transcript_layout()[1] == final_page
    assert window.canvas._transcript_idle and not window.canvas._transcript_pending
    assert not window.canvas._transcript_timer.isActive()


def test_incoming_speech_preserves_active_reading_page_and_its_timer(window, app):
    window.toggle_fullscreen(with_context=True)
    first = TranscriptSegment(0, 8, " ".join(f"Narrated{number}" for number in range(70)))
    window._handle_event(0, "transcript", [first])
    page = window.canvas._transcript_layout()[1]
    timer_id = window.canvas._transcript_timer.timerId()
    interval = window.canvas._transcript_timer.interval()
    assert window.canvas._transcript_timer.isActive()
    assert 1200 <= interval <= max(1200, len(" ".join(page).split()) * 250)
    extension = [TranscriptSegment(9, 12, "The next passage waits behind the unread words."),
                 TranscriptSegment(13, 16, "The final passage also remains available.")]
    window._handle_event(0, "transcript", [first] + extension)
    assert window.canvas._transcript_layout()[1] == page
    assert window.canvas._transcript_timer.timerId() == timer_id
    assert window.canvas._transcript_timer.interval() == interval
    pages = consume_subtitle_pages(window.canvas, app)
    assert " ".join(pages).split() == " ".join(segment.text for segment in [first] + extension).split()


def test_restored_or_hidden_history_starts_with_latest_passage_and_mode_switch_keeps_live_queue(window, app):
    history = [TranscriptSegment(0, 4, "Old recorded history."),
               TranscriptSegment(5, 9, "Latest restored passage.")]
    window._show_transcript(history, restored=True)
    window.toggle_fullscreen(with_context=True)
    assert consume_subtitle_pages(window.canvas, app) == [history[-1].text]
    extension = [TranscriptSegment(10, 14, " ".join(f"Current{number}" for number in range(60))),
                 TranscriptSegment(15, 18, "The queued next passage.")]
    window._handle_event(0, "transcript", history + extension)
    active = window.canvas.current_transcript
    pending = list(window.canvas._transcript_pending)
    current_page = window.canvas._transcript_layout()[1]
    window.toggle_fullscreen(with_title=True)
    app.processEvents()
    assert window.canvas._transcript_layout() is None
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    assert window.canvas.current_transcript == active
    assert list(window.canvas._transcript_pending) == pending
    assert window.canvas._transcript_layout()[1] == current_page
    assert " ".join(consume_subtitle_pages(window.canvas, app)).split() == (
        " ".join(segment.text for segment in extension).split()
    )


def test_resize_after_reading_first_page_preserves_all_unread_words(app):
    canvas = ImageCanvas()
    canvas.resize(420, 260)
    canvas.set_context_overlay(True)
    text = " ".join(f"Word{number}" for number in range(150))
    canvas.enqueue_transcript([text])
    canvas.show()
    app.processEvents()
    try:
        first_page = " ".join(canvas._transcript_layout()[1])
        canvas._advance_transcript()
        canvas._transcript_timer.stop()
        canvas.resize(960, 540)
        app.processEvents()
        rest = consume_subtitle_pages(canvas, app)
        assert (first_page + " " + " ".join(rest)).split() == text.split()
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_new_book_clears_active_and_queued_subtitles_before_new_speech(window, app):
    window.toggle_fullscreen(with_context=True)
    passages = [TranscriptSegment(0, 5, "The active caption from the old book."),
                TranscriptSegment(6, 11, "An unread caption from the old book.")]
    window._handle_event(0, "transcript", passages)
    assert window.canvas._transcript_pending
    old_session = window._session
    window.new_book()
    app.processEvents()
    assert window.canvas._transcript_layout() is None
    assert not window.canvas.current_transcript
    assert window.canvas._transcript_idle and not window.canvas._transcript_pending
    assert not window.canvas._transcript_timer.isActive()
    window._handle_event(old_session, "transcript", passages)
    assert not window.canvas.current_transcript and not window.canvas._transcript_pending
    fresh = "The first caption from a new book."
    window._handle_event(window._session, "transcript", [TranscriptSegment(0, 5, fresh)])
    assert consume_subtitle_pages(window.canvas, app) == [fresh]


def test_replacement_snapshot_shows_latest_correction_without_replaying_history(window, app):
    window.toggle_fullscreen(with_context=True)
    original = [TranscriptSegment(0, 5, "Original earlier passage."),
                TranscriptSegment(6, 11, "Original latest passage.")]
    window._handle_event(0, "transcript", original)
    assert window.canvas._transcript_pending
    replacement = [TranscriptSegment(0, 5, "Corrected earlier passage."),
                   TranscriptSegment(6, 11, "Corrected latest passage.")]
    window._handle_event(0, "transcript", replacement)
    assert window.transcript.toPlainText().splitlines() == [
        "[00:00] " + replacement[0].text, "[00:06] " + replacement[1].text,
    ]
    assert not window.canvas._transcript_pending
    assert consume_subtitle_pages(window.canvas, app) == [replacement[-1].text]


@pytest.mark.parametrize("size", [(420, 260), (960, 540), (1600, 900)])
def test_long_speech_shows_every_word_on_readable_pages_below_context(app, image_factory, size):
    canvas = ImageCanvas()
    canvas.transition = "cut"
    canvas.resize(*size)
    canvas.set_image(image_factory(title="An Extremely Long Museum Image Title " * 25))
    canvas.set_title_overlay(True)
    canvas.set_context("The travelers remember the story. " * 100,
                       "They crossed the harbor at night.", ["boat", "frozen harbor"])
    tail = "The final words are the newest speech."
    text = "Earlier spoken words. " * 300 + tail
    canvas.set_transcript(text)
    canvas.set_context_overlay(True)
    canvas.show()
    app.processEvents()
    try:
        assert len(canvas.current_transcript) > 1200
        font, lines, bounds = canvas._transcript_layout()
        assert 1 <= len(lines) <= 2
        assert " ".join(lines).startswith("Earlier spoken words.")
        assert "…" not in " ".join(lines)
        assert QRectF(canvas.rect()).contains(bounds)
        assert abs(bounds.center().x() - canvas.rect().center().x()) <= 1
        assert not bounds.intersects(QRectF(canvas.context_panel.geometry()))
        assert all(QFontMetricsF(font).horizontalAdvance(line) <= bounds.width() + 1 for line in lines)
        white, outlined = subtitle_pixels(canvas, bounds)
        assert len(white) > 20 and outlined
        canvas.resize(size[0] + 100, size[1] + 80)
        app.processEvents()
        resized = canvas._transcript_layout()[2]
        assert resized != bounds and QRectF(canvas.rect()).contains(resized)
        assert not resized.intersects(QRectF(canvas.context_panel.geometry()))
        pages = consume_subtitle_pages(canvas, app)
        assert " ".join(pages).split() == text.split()
        assert " ".join(pages).endswith(tail)
    finally:
        canvas.close()
        canvas.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

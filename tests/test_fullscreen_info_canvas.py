"""Fullscreen detail selections preserve source text and independent playback."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from semantic_image_stream.slideshow import ImageCanvas
from test_fullscreen_context_ui import app as app, image_factory as image_factory


ALL_DETAILS = ("narrative", "scene", "subjects", "title", "credits", "transcript")
STORY = ("The crew remembers a damn difficult voyage.",
         "A ship enters the frozen harbor.",
         ("ship", "frozen harbor"))
CAPTION = "The captain said, 'That was a fucking long voyage.'"


@pytest.fixture
def canvas(app, image_factory):
    widget = ImageCanvas()
    widget.transition = "cut"
    widget.resize(960, 540)
    image = image_factory(title="The Ship at the Frozen Harbor")
    image.creator = "Harbor photographer"
    image.license = "CC BY 4.0"
    assert widget.set_image(image)
    widget.set_context(*STORY)
    widget.set_transcript(CAPTION)
    widget.set_context_overlay(True)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.mark.parametrize("detail", ALL_DETAILS)
def test_each_detail_selection_is_independent_and_restores_source_text(canvas, app, detail):
    baseline_context = canvas.context_text.toPlainText()
    baseline_credit = canvas.credit_text.toPlainText()
    baseline_page = canvas._transcript_layout()[1]
    frame = canvas.current_pixmap.cacheKey()
    source = (canvas._context_content, canvas.current_title, canvas.current_credit, canvas.current_transcript)
    headings = {"narrative": "Narrative context", "scene": "Current scene", "subjects": "Visual subjects",
                "title": "Image title", "credits": "Image credits"}

    canvas.set_fullscreen_info({detail: False})
    app.processEvents()
    for key, heading in headings.items():
        displayed = canvas.context_text.toPlainText() if key in ("narrative", "scene", "subjects") else (
            canvas.credit_text.toPlainText()
        )
        assert (heading in displayed) == (key != detail)
    assert (canvas._transcript_layout() is not None) == (detail != "transcript")
    assert canvas.current_pixmap.cacheKey() == frame
    assert (canvas._context_content, canvas.current_title, canvas.current_credit, canvas.current_transcript) == source

    canvas.set_fullscreen_info({detail: True})
    app.processEvents()
    assert canvas.context_text.toPlainText() == baseline_context
    assert canvas.credit_text.toPlainText() == baseline_credit
    assert canvas._transcript_layout()[1] == baseline_page


def test_all_disabled_hides_panel_and_caption_without_changing_other_fullscreen_modes(canvas, app):
    canvas.set_fullscreen_info(dict.fromkeys(ALL_DETAILS, False))
    app.processEvents()
    assert not canvas.context_panel.isVisible()
    assert not canvas.context_text.toPlainText()
    assert not canvas.credit_text.toPlainText()
    assert canvas._transcript_layout() is None
    assert not canvas._transcript_timer.isActive()

    canvas.set_context_overlay(False)
    canvas.set_title_overlay(True)
    app.processEvents()
    assert canvas.current_title == "The Ship at the Frozen Harbor"
    assert canvas._title_layout()[1]
    assert canvas._credit_layout() is not None
    assert not canvas.context_panel.isVisible()
    assert canvas._transcript_layout() is None


def test_transcript_only_mode_shows_caption_without_an_empty_context_box(canvas, app):
    canvas.set_fullscreen_info(dict.fromkeys(ALL_DETAILS, False))
    canvas.set_fullscreen_info({"transcript": True})
    app.processEvents()
    assert not canvas.context_panel.isVisible()
    assert " ".join(canvas._transcript_layout()[1]) == CAPTION
    assert canvas._transcript_timer.isActive()


@pytest.mark.parametrize("detail,expected", [("scene", STORY[1]), ("subjects", "\n".join(STORY[2])),
                                             ("title", "The Ship at the Frozen Harbor"),
                                             ("credits", "Harbor photographer")])
def test_single_enabled_panel_section_is_visible_without_narrative(canvas, app, detail, expected):
    canvas.set_fullscreen_info(dict.fromkeys(ALL_DETAILS, False))
    canvas.set_fullscreen_info({detail: True})
    app.processEvents()
    assert canvas.context_panel.isVisible()
    text = canvas.context_text.toPlainText() + canvas.credit_text.toPlainText()
    assert expected in text
    assert "Narrative context" not in text
    assert "Waiting for interpreted speech" not in text
    assert canvas._transcript_layout() is None


def test_waiting_panel_only_appears_when_narrative_is_selected(canvas, app):
    canvas.set_image(None)
    canvas.set_context("", "", ())
    canvas.set_fullscreen_info(dict.fromkeys(ALL_DETAILS, False))
    canvas.set_fullscreen_info({"scene": True, "subjects": True, "title": True, "credits": True})
    app.processEvents()
    assert not canvas.context_panel.isVisible()
    canvas.set_fullscreen_info({"narrative": True})
    app.processEvents()
    assert canvas.context_panel.isVisible()
    assert "Waiting for interpreted speech" in canvas.context_text.toPlainText()


def test_hidden_caption_retains_latest_incoming_speech_without_backlog_or_censorship(canvas, app):
    canvas.set_transcript("Active passage before hiding captions.")
    canvas.enqueue_transcript(["An unread queued passage."])
    original = canvas.current_transcript
    canvas.set_fullscreen_info({"transcript": False})
    assert not canvas._transcript_timer.isActive()
    canvas._advance_transcript()
    assert canvas.current_transcript == original

    latest = "<b>She said 'fuck that' without hesitation.</b> & carried on."
    canvas.enqueue_transcript([f"Hidden earlier passage {number}." for number in range(100)] + [latest])
    assert canvas.current_transcript == latest
    assert not canvas._transcript_pending
    assert canvas._transcript_layout() is None
    canvas.set_fullscreen_info({"transcript": True})
    app.processEvents()
    assert " ".join(canvas._transcript_layout()[1]) == latest
    assert not canvas._transcript_pending
    assert canvas._transcript_timer.isActive()


def test_other_detail_changes_preserve_active_caption_queue_and_reading_timer(canvas):
    canvas.enqueue_transcript(["A second caption passage.", "A third caption passage."])
    active = canvas.current_transcript
    pending = list(canvas._transcript_pending)
    page = canvas._transcript_layout()[1]
    timer_id = canvas._transcript_timer.timerId()
    canvas.set_fullscreen_info({"title": False, "narrative": False})
    assert canvas.current_transcript == active
    assert list(canvas._transcript_pending) == pending
    assert canvas._transcript_layout()[1] == page
    assert canvas._transcript_timer.timerId() == timer_id


def test_unrelated_detail_changes_preserve_context_and_credit_scroll_positions(canvas, app, image_factory):
    image = image_factory(title="A detailed title about the harbor " * 30)
    image.creator = "Harbor photographer"
    image.license = "A long license clause for this photograph. " * 20
    canvas.set_image(image)
    canvas.set_context("The crew remembers every part of the voyage. " * 300, STORY[1], STORY[2])
    app.processEvents()
    context_scroll = canvas.context_text.verticalScrollBar()
    credit_scroll = canvas.credit_text.verticalScrollBar()
    assert context_scroll.maximum() > 0 and credit_scroll.maximum() > 0
    context_scroll.setValue(context_scroll.maximum() // 2)
    credit_scroll.setValue(credit_scroll.maximum() // 2)
    context_position, credit_position = context_scroll.value(), credit_scroll.value()
    canvas.set_fullscreen_info({"transcript": False})
    app.processEvents()
    assert context_scroll.value() == context_position
    assert credit_scroll.value() == credit_position

    canvas.set_fullscreen_info({"narrative": False})
    app.processEvents()
    assert credit_scroll.value() == credit_position
    canvas.set_fullscreen_info({"narrative": True})
    app.processEvents()
    context_scroll.setValue(context_scroll.maximum() // 2)
    context_position = context_scroll.value()
    canvas.set_fullscreen_info({"title": False})
    app.processEvents()
    assert context_scroll.value() == context_position


def test_hidden_story_updates_are_restored_as_latest_context_with_markup_kept_as_text(canvas):
    canvas.set_fullscreen_info({"narrative": False, "scene": False, "subjects": False})
    summary = "<script>Fuck the old story.</script> The new story starts here."
    scene = "The ship leaves the harbor & sails onward."
    canvas.set_context(summary, scene, ("open sea",))
    assert not canvas.context_text.toPlainText()
    canvas.set_fullscreen_info({"narrative": True, "scene": True, "subjects": True})
    text = canvas.context_text.toPlainText()
    assert summary in text and scene in text and "open sea" in text
    assert STORY[0] not in text

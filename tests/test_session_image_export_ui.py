"""Export controls use the complete book session independently of slideshow history."""
import csv
from dataclasses import replace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QFileDialog

from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import Topic, TranscriptSegment
from semantic_image_stream.session_image_export import record_session_images
from test_fullscreen_context_ui import app as app, image_factory as image_factory, window as window


def recorded_images(window, image_factory, count=2):
    topic = Topic("frozen harbor", "A boat arrived.", summary="A voyage reaches the ice.",
                  scene="A boat at an icy harbor.", visual_subjects=("boat", "harbor"),
                  transcript=("A boat arrived.",))
    state = SessionState(segments=[TranscriptSegment(0, 5, "A boat arrived.")], topic=topic)
    base = image_factory()
    images = [replace(base, image_url=f"https://images.example/{index}.png",
                      title=f"Harbor, image {index} — 北", score=.7 if index % 2 == 0 else .3)
              for index in range(count)]
    assert record_session_images(state, images, topic, .15, "context_clip")
    window._session_state = state
    window._handle_event(window._session, "topic", topic)
    window._handle_event(window._session, "images", images[:2])
    return state, images


def test_images_tab_exports_all_session_links_at_selected_minimum(window, app, image_factory, monkeypatch, tmp_path):
    state, images = recorded_images(window, image_factory, count=80)
    window.tabs.setCurrentWidget(window.images_panel)
    window.csv_minimum.setValue(.6)
    app.processEvents()
    assert window.export_images_button.isVisible() and window.export_images_button.isEnabled()
    assert "40 session links" in window.image_export_status.text()
    assert len(window.slideshow.history) < len(state.image_matches)
    destination = tmp_path / "book-links"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    window.export_images_button.click()
    exported = destination.with_suffix(".csv")
    assert exported.read_bytes().startswith(b"\xef\xbb\xbf")
    with exported.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert {row["image_url"] for row in rows} == {candidate.image_url for candidate in images[::2]}
    assert rows[0]["scene"] == state.topic.scene and "北" in rows[0]["title"]
    assert "Exported 40 image links" in window.image_export_status.text()
    window.csv_minimum.setValue(.8)
    assert not window.export_images_button.isEnabled()
    window.csv_minimum.setValue(.2)
    assert "80 session links" in window.image_export_status.text()


def test_stop_retains_export_and_new_book_discards_previous_links(window, image_factory):
    state, images = recorded_images(window, image_factory)
    old_session = window._session
    window._handle_event(old_session, "stopped", None)
    assert window.export_images_button.isEnabled() and window._session_state is state
    window.new_book()
    assert window._session_state is None and not window.export_images_button.isEnabled()
    window._handle_event(old_session, "images", images)
    assert not window.export_images_button.isEnabled()


def test_manual_refresh_keeps_frame_context_and_export_but_discards_pending(window, image_factory):
    state, images = recorded_images(window, image_factory)

    class Pipeline:
        calls = 0
        image_request_revision = 0

        def refresh_images(self):
            self.calls += 1
            return True

    pipeline = window.pipeline = Pipeline()
    window._monitoring = True
    window._update_controls()
    assert window.refresh_images_button.isEnabled() and window.slideshow.pending
    frame = window.canvas.current_pixmap.cacheKey()
    current = window._displayed_candidate
    history = list(window.slideshow.history)
    scene = window.context_text.toPlainText()
    window.refresh_images_button.click()
    assert pipeline.calls == 1 and not window.slideshow.pending
    assert window.canvas.current_pixmap.cacheKey() == frame and window._displayed_candidate is current
    assert list(window.slideshow.history) == history and window.context_text.toPlainText() == scene
    assert len(state.image_matches) == len(images) and window.export_images_button.isEnabled()
    old_status = window._retrieval_message
    window._handle_event(window._session, "images", (0, images))
    window._handle_event(window._session, "retrieval", (0, {"message": "Obsolete search"}))
    assert not window.slideshow.pending and window._retrieval_message == old_status
    window._handle_event(window._session, "images", (1, images))
    assert list(window.slideshow.pending) == [images[1]]
    window._monitoring = False
    window.pipeline = None


def test_imported_context_restores_csv_rows_instead_of_previous_book(window, image_factory, monkeypatch, tmp_path):
    from semantic_image_stream.session_store import export_context

    state, _ = recorded_images(window, image_factory)
    destination = tmp_path / "context.json"
    export_context(state, destination)
    window.new_book()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(destination), ""))
    window.import_context()
    assert window._session_state.image_matches == state.image_matches
    assert window.export_images_button.isEnabled()
    assert "2 session links" in window.image_export_status.text()
    assert window._context_frozen and not window.slideshow.history

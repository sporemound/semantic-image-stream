"""Fullscreen information choices control visible details, never the story."""
from dataclasses import asdict

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint
from PySide6.QtWidgets import QApplication

from semantic_image_stream import session_store
from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import Topic, TranscriptSegment
from test_context_transfer_ui import choose_import, write_story
from test_fullscreen_context_ui import app as app, image_factory as image_factory, window as window
from test_fullscreen_transcript_ui import subtitle_font as subtitle_font, subtitle_pixels


KEYS = ("transcript", "narrative", "scene", "subjects", "title", "credits")


def selections(window):
    return {key: action.isChecked() for key, action in window.fullscreen_info_actions.items()}


def choose_only(window, key):
    for name, action in window.fullscreen_info_actions.items():
        action.setChecked(name == key)


def establish_information(window, image_factory):
    segment = TranscriptSegment(0, 5, "The lights on the shore begin to shimmer.")
    topic = Topic("harbor", segment.text, summary="An expedition approaches a harbor.",
                  scene="A ship rounds an icy cape.", visual_subjects=("ship", "harbor"), context_used_segments=1)
    state = SessionState(segments=[segment], topic=topic, book_context="An expedition book.",
                         analyzer_checkpoint={"processed_count": 1})
    window._session_state = state
    window._handle_event(window._session, "transcript", list(state.segments))
    window._handle_event(window._session, "topic", topic)
    image = image_factory(title="The Harbor at Night")
    image.creator, image.license = "Fixture artist", "CC BY 4.0"
    window._handle_event(window._session, "images", [image, image_factory("two", "The Icy Cape", "cyan")])
    window.toggle_fullscreen(with_context=True)
    return state, topic, image, segment


def test_all_information_defaults_on_and_choices_save_without_an_audio_recording(window, app, tmp_path):
    assert set(window.fullscreen_info_actions) == set(KEYS)
    assert all(selections(window).values())
    assert all(getattr(Settings(), "fullscreen_" + key) is True for key in KEYS)
    assert window.fullscreen_info_button.menu() is window.fullscreen_info_menu
    assert window.form.labelForField(window.fullscreen_info_button).text() == "Fullscreen information"
    for key, action in window.fullscreen_info_actions.items():
        assert action.isCheckable()
        action.setChecked(key in {"transcript", "scene", "credits"})

    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audio_file.setText(str(tmp_path / "not-selected-yet.wav"))
    window.chunk_interval.setValue(3)
    window.config_path = tmp_path / "fullscreen-choices.toml"
    expected = selections(window)
    window.save_settings()
    loaded = Settings.load(window.config_path)
    assert loaded.audio_mode == "file"
    assert loaded.audio_file == str(tmp_path / "not-selected-yet.wav")
    assert loaded.chunk_seconds == 3
    assert all(getattr(loaded, "fullscreen_" + key) == enabled for key, enabled in expected.items())
    assert window.pipeline is None and not window._monitoring

    fresh = MainWindow(loaded, window.config_path)
    fresh.show()
    app.processEvents()
    try:
        assert selections(fresh) == expected
        assert fresh.canvas._fullscreen_info == expected
        assert fresh.pipeline is None and not fresh._monitoring
    finally:
        fresh.close()
        fresh.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


@pytest.mark.parametrize("key", KEYS)
def test_each_information_choice_independently_controls_rendered_content(window, app, image_factory, key):
    _, topic, image, segment = establish_information(window, image_factory)
    choose_only(window, key)
    app.processEvents()
    context = window.canvas.context_text.toPlainText()
    footer = window.canvas.credit_text.toPlainText()
    if key in {"narrative", "scene", "subjects"}:
        expected = {
            "narrative": "Narrative context\n" + topic.summary,
            "scene": "Current scene\n" + topic.scene,
            "subjects": "Visual subjects\n" + "\n".join(topic.visual_subjects),
        }[key]
        assert context.split() == expected.split()
        assert window.canvas.context_text.isVisible()
        assert footer == "" and not window.canvas.credit_text.isVisible()
        assert window.canvas.context_panel.isVisible()
    elif key in {"title", "credits"}:
        expected = ("Image title\n" + image.title if key == "title"
                    else "Image credits\n" + window.canvas.current_credit)
        assert footer == expected
        assert window.canvas.credit_text.isVisible()
        assert context == "" and not window.canvas.context_text.isVisible()
        assert window.canvas.context_panel.isVisible()
    else:
        assert context == "" and footer == ""
        assert not window.canvas.context_panel.isVisible()
        layout = window.canvas._transcript_layout()
        assert " ".join(layout[1]) == segment.text
        white, outlined = subtitle_pixels(window.canvas, layout[2])
        assert len(white) > 20 and outlined
    assert (window.canvas._transcript_layout() is not None) is (key == "transcript")
    assert window.isFullScreen() and window._fullscreen_context
    assert all(getattr(window.settings, "fullscreen_" + name) == enabled
               for name, enabled in selections(window).items())


def test_turning_every_detail_off_hides_the_panel_without_exiting_fullscreen(window, app, image_factory):
    establish_information(window, image_factory)
    choose_only(window, None)
    app.processEvents()
    assert window.isFullScreen() and window._fullscreen_context
    assert window.canvas.context_overlay_visible
    assert not window.canvas.context_panel.isVisible()
    assert window.canvas._transcript_layout() is None
    assert window.canvas.context_text.toPlainText() == ""
    assert window.canvas.credit_text.toPlainText() == ""
    window.fullscreen_info_actions["transcript"].setChecked(True)
    app.processEvents()
    assert window.canvas._transcript_layout() is not None
    assert not window.canvas.context_panel.isVisible()
    window.fullscreen_info_actions["title"].setChecked(True)
    app.processEvents()
    assert window.canvas.context_panel.isVisible()
    assert "Image title\nThe Harbor at Night" == window.canvas.credit_text.toPlainText()


def test_live_information_toggles_preserve_story_frame_queue_history_and_slideshow_timer(window, app, image_factory):
    state, _, _, _ = establish_information(window, image_factory)
    window._monitoring = True
    window._update_controls()
    assert window.fullscreen_info_button.isEnabled()
    window._handle_event(window._session, "transcript", list(state.segments) + [
        TranscriptSegment(6, 11, "The next spoken passage remains queued."),
    ])
    app.processEvents()
    session = window._session
    snapshot = session_store.snapshot_session(state)
    snapshot.pop("saved_at")
    accepted_context = dict(window._context_payload)
    raw_transcript = window.transcript.toPlainText()
    raw_context = window.context_text.toPlainText()
    frame = window.canvas.current_pixmap.cacheKey()
    pending, history = list(window.slideshow.pending), list(window.slideshow.history)
    timer_id = window.slideshow.timer.timerId()
    caption, caption_queue = window.canvas.current_transcript, list(window.canvas._transcript_pending)
    runtime = asdict(window._read_settings(require_recording=False))
    for key in KEYS:
        window.fullscreen_info_actions[key].trigger()
        app.processEvents()
        current = session_store.snapshot_session(state)
        current.pop("saved_at")
        assert current == snapshot and window._session_state is state and window._session == session
        assert window._context_payload == accepted_context
        assert window.transcript.toPlainText() == raw_transcript
        assert window.context_text.toPlainText() == raw_context
        assert window.canvas.current_pixmap.cacheKey() == frame
        assert list(window.slideshow.pending) == pending and list(window.slideshow.history) == history
        assert window.slideshow.timer.timerId() == timer_id
        assert window.canvas.current_transcript == caption
        assert list(window.canvas._transcript_pending) == caption_queue
        assert window.isFullScreen() and window._fullscreen_context
    after = asdict(window._read_settings(require_recording=False))
    assert {name: value for name, value in after.items() if not name.startswith("fullscreen_")} == {
        name: value for name, value in runtime.items() if not name.startswith("fullscreen_")
    }
    window._monitoring = False
    window._update_controls()


def test_right_click_menu_is_available_only_in_context_fullscreen(window, app, monkeypatch):
    invoked = []
    monkeypatch.setattr(window.fullscreen_info_menu, "exec", lambda point: invoked.append(point))
    point = QPoint(40, 50)
    window.canvas.customContextMenuRequested.emit(point)
    assert invoked == []
    window.toggle_fullscreen()
    app.processEvents()
    window.canvas.customContextMenuRequested.emit(point)
    assert invoked == []
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    window.canvas.customContextMenuRequested.emit(point)
    assert invoked == [window.canvas.mapToGlobal(point)]
    window.leave_fullscreen()
    window.canvas.customContextMenuRequested.emit(point)
    assert len(invoked) == 1


def test_imported_story_preserves_fullscreen_information_choices(window, app, monkeypatch, tmp_path):
    for name, action in window.fullscreen_info_actions.items():
        action.setChecked(name in {"subjects", "credits"})
    expected = selections(window)
    path = tmp_path / "story-context.json"
    imported = write_story(path)
    choose_import(monkeypatch, path)
    window.toggle_fullscreen(with_context=True)
    window.import_context()
    app.processEvents()
    assert selections(window) == expected
    assert window.canvas._fullscreen_info == expected
    assert all(getattr(window.settings, "fullscreen_" + key) == enabled for key, enabled in expected.items())
    assert window.canvas.context_text.toPlainText().split() == (
        "Visual subjects\n" + "\n".join(imported.topic.visual_subjects)
    ).split()
    assert "Narrative context" not in window.canvas.context_text.toPlainText()
    assert window.canvas._transcript_layout() is None

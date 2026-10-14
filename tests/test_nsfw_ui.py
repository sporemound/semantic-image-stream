"""NSFW image choices use harmless colored fixtures and preserve spoken context."""
from dataclasses import asdict

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox

from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.session_store import snapshot_session
from test_time_filter_ui import app as app, window_factory as window_factory


@pytest.fixture
def image_factory(tmp_path):
    def make(name="mature-fixture", *, mature=True, color="red"):
        path = tmp_path / f"{name}.png"
        pixels = QImage(128, 96, QImage.Format.Format_RGB32)
        pixels.fill(QColor(color))
        assert pixels.save(str(path))
        return ImageCandidate(f"https://images.example/{name}.png", f"https://source.example/{name}",
                              f"{name} colored square", "Fixture source", creator="Fixture artist",
                              license="CC BY 4.0", local_path=str(path), mature_content=mature)
    return make


def test_checkbox_defaults_off_and_stored_choice_is_visible(window_factory):
    default = window_factory()
    assert isinstance(default.allow_nsfw, QCheckBox)
    assert default.allow_nsfw.text() == "Allow nudity (NSFW)"
    assert not default.allow_nsfw.isChecked() and default.allow_nsfw.isEnabled()
    assert default._collect_image_filters()["allow_nsfw"] is False
    assert Settings().allow_nsfw is False
    enabled = window_factory(allow_nsfw=True)
    assert enabled.allow_nsfw.isChecked()
    assert enabled._collect_image_filters()["allow_nsfw"] is True


def test_saved_nsfw_choice_restores_without_recording_or_changing_three_second_chunks(window_factory, tmp_path):
    window = window_factory(chunk_seconds=3, fullscreen_title=False, fullscreen_transcript=True)
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audio_file.clear()
    window.allow_nsfw.setChecked(True)
    window.save_settings()
    saved = Settings.load(window.config_path)
    assert saved.allow_nsfw is True and saved.chunk_seconds == 3
    assert saved.audio_mode == "file" and saved.audio_file == ""
    assert saved.whisper_model == "large-v3" and saved.providers == ["wikimedia", "archive"]
    assert saved.fullscreen_title is False and saved.fullscreen_transcript is True
    assert not window._monitoring and window.pipeline.updates == []
    restored = window_factory(allow_nsfw=saved.allow_nsfw, chunk_seconds=saved.chunk_seconds)
    assert restored.allow_nsfw.isChecked() and restored.chunk_interval.value() == 3
    window.allow_nsfw.setChecked(False)
    window.save_settings()
    assert Settings.load(window.config_path).allow_nsfw is False

    old_config = tmp_path / "older-config.toml"
    old_config.write_text('chunk_seconds = 3.0\nproviders = ["archive"]\n', encoding="utf-8")
    old = Settings.load(old_config)
    assert old.allow_nsfw is False and old.chunk_seconds == 3.0 and old.providers == ["archive"]


@pytest.mark.parametrize("invalid", [1, "false", None])
def test_nsfw_setting_requires_an_actual_boolean(invalid):
    with pytest.raises(ValueError, match="allow_nsfw must be true or false"):
        Settings(allow_nsfw=invalid).validate()


def test_live_nsfw_choice_is_debounced_and_keeps_complete_story_and_other_preferences(window_factory):
    window = window_factory(chunk_seconds=3)
    narration = "The adult nude model stood in the studio. The narration mentions nudity and sex."
    segment = TranscriptSegment(0, 3, narration, "en")
    topic = Topic("art studio", narration, summary="The artist studies an adult nude model.",
                  scene="An adult nude model poses in an art studio.", visual_subjects=("adult nude model", "studio"),
                  context_used_segments=1)
    state = SessionState(segments=[segment], topic=topic, book_context=window.settings.book_context,
                         analyzer_checkpoint={"version": 1, "cursor": 1})
    window._session_state = state
    window._handle_event(window._session, "transcript", [segment])
    window._handle_event(window._session, "topic", topic)
    before = asdict(window.settings)
    saved_story = snapshot_session(state)
    saved_story.pop("saved_at")
    raw_transcript = window.transcript.toPlainText()
    raw_context = window.context_text.toPlainText()
    accepted_context = dict(window._context_payload)
    selection = {name: action.isChecked() for name, action in window.fullscreen_info_actions.items()}
    session, playback = window._session, window._playback_key
    window._monitoring = True
    window._update_controls()
    assert window.allow_nsfw.isEnabled() and not window.audio_mode.isEnabled()
    window.allow_nsfw.setChecked(True)
    window.allow_nsfw.setChecked(False)
    window.allow_nsfw.setChecked(True)
    QTest.qWait(100)
    assert window.pipeline.updates == []
    QTest.qWait(300)
    assert len(window.pipeline.updates) == 1
    first = window.pipeline.updates[0]
    assert first.allow_nsfw is True
    first_dict = asdict(first)
    first_dict["allow_nsfw"] = before["allow_nsfw"]
    assert first_dict == before
    window.allow_nsfw.setChecked(False)
    QTest.qWait(350)
    assert len(window.pipeline.updates) == 2 and window.pipeline.updates[1].allow_nsfw is False
    assert first.allow_nsfw is True and asdict(window.settings) == before
    after_story = snapshot_session(state)
    after_story.pop("saved_at")
    assert after_story == saved_story and window._session_state is state and window._session == session
    assert window.transcript.toPlainText() == raw_transcript and narration in raw_transcript
    assert window.context_text.toPlainText() == raw_context and topic.scene in raw_context
    assert window._context_payload == accepted_context and window._playback_key == playback
    assert {name: action.isChecked() for name, action in window.fullscreen_info_actions.items()} == selection
    window._stopping = True
    window._update_controls()
    assert not window.allow_nsfw.isEnabled()
    window._stopping = False
    window._monitoring = False
    window._update_controls()
    assert window.allow_nsfw.isEnabled()


def test_turning_nsfw_off_immediately_clears_known_mature_frame_and_fullscreen_title(window_factory,
                                                                                app, image_factory):
    window = window_factory(allow_nsfw=True)
    candidate = image_factory()
    window._handle_event(window._session, "images", [candidate])
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    assert window._displayed_candidate is candidate and window.slideshow.current is candidate
    assert "Image title\n" + candidate.title in window.canvas.credit_text.toPlainText()
    assert window.save_button.isEnabled()
    window.allow_nsfw.setChecked(False)
    assert window.pipeline.updates == [], "The frame guard must act before the live-filter debounce"
    assert window._displayed_candidate is None and window.slideshow.current is None
    assert window.canvas.current_pixmap.isNull() and window.canvas.current_title == ""
    assert candidate.title not in window.canvas.credit_text.toPlainText()
    assert not window.save_button.isEnabled()
    assert candidate not in window.slideshow.history and candidate not in window.slideshow.pending
    assert window.isFullScreen() and window._fullscreen_context


def test_known_mature_history_cannot_replace_current_frame_after_disabling_nsfw(window_factory,
                                                                              image_factory):
    window = window_factory(allow_nsfw=True)
    mature, safe = image_factory(), image_factory("safe-fixture", mature=False, color="blue")
    window._handle_event(window._session, "images", [mature, safe])
    window.next_image()
    assert window._displayed_candidate is safe and mature in window.slideshow.history
    history_item = next(window.image_history.item(index) for index in range(window.image_history.count())
                        if window.image_history.item(index).data(Qt.ItemDataRole.UserRole) == 0)
    window.allow_nsfw.setChecked(False)
    window._select_history(history_item)
    assert window._displayed_candidate is safe and window.slideshow.current is safe
    assert window.canvas.current_title == safe.title and window.save_button.isEnabled()
    assert mature not in window.slideshow.history


def test_pending_and_late_old_backend_mature_results_are_guarded_by_current_checkbox(window_factory, image_factory):
    window = window_factory(allow_nsfw=True)
    safe = image_factory("safe-first", mature=False, color="blue")
    queued_mature = image_factory("queued-mature", color="red")
    late_mature = image_factory("old-backend-mature", color="magenta")
    window._handle_event(window._session, "images", [safe, queued_mature])
    assert window._displayed_candidate is safe and list(window.slideshow.pending) == [queued_mature]
    window.allow_nsfw.setChecked(False)
    window.next_image()
    assert window._displayed_candidate is safe and window.slideshow.current is safe
    assert queued_mature not in window.slideshow.pending and queued_mature not in window.slideshow.history
    window._handle_event(window._session, "images", [late_mature])
    window.next_image()
    assert window._displayed_candidate is safe and window.canvas.current_title == safe.title
    assert late_mature not in window.slideshow.pending and late_mature not in window.slideshow.history
    window._show_image(late_mature)
    assert window._displayed_candidate is safe and window.canvas.current_title == safe.title
    window._handle_event(window._session - 1, "images", [late_mature])
    assert window._displayed_candidate is safe


def test_unknown_maturity_is_usable_and_initial_mature_results_do_not_hide_safe_fallback(window_factory,
                                                                                     image_factory):
    window = window_factory()
    mature = image_factory()
    unknown = image_factory("unknown-fixture", mature=None, color="cyan")
    window._handle_event(window._session, "images", [mature, unknown])
    assert window._displayed_candidate is unknown and window.slideshow.current is unknown
    assert unknown.mature_content is None and not window.canvas.current_pixmap.isNull()
    assert window.canvas.current_title == unknown.title and window.save_button.isEnabled()
    assert mature not in window.slideshow.history and mature not in window.slideshow.pending


def test_enabling_nsfw_admits_labeled_fixture_and_its_title_in_context_fullscreen(window_factory,
                                                                              app, image_factory):
    window = window_factory()
    candidate = image_factory()
    window.allow_nsfw.setChecked(True)
    window._handle_event(window._session, "images", [candidate])
    window.toggle_fullscreen(with_context=True)
    app.processEvents()
    assert window._displayed_candidate is candidate and window.slideshow.current is candidate
    assert candidate in window.slideshow.history and not window.canvas.current_pixmap.isNull()
    assert "Image title\n" + candidate.title in window.canvas.credit_text.toPlainText()
    assert "Fixture artist" in window.canvas.credit_text.toPlainText()
    assert window.canvas.context_panel.isVisible() and window.save_button.isEnabled()


def test_new_book_keeps_nsfw_choice_and_fullscreen_selectors(window_factory):
    window = window_factory(allow_nsfw=True, fullscreen_title=False)
    before = window._collect_image_filters()
    fullscreen = {name: action.isChecked() for name, action in window.fullscreen_info_actions.items()}
    window._handle_event(window._session, "transcript", [TranscriptSegment(0, 3, "A previous book passage.")])
    window.new_book()
    assert window.transcript.toPlainText() == "" and window.allow_nsfw.isChecked()
    assert window._collect_image_filters() == before
    assert {name: action.isChecked() for name, action in window.fullscreen_info_actions.items()} == fullscreen

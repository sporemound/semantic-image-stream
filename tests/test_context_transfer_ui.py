"""Context files transfer the full story while preserving the chosen runtime."""
from dataclasses import asdict

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QFileDialog

from semantic_image_stream import session_store
from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import Topic, TranscriptSegment
from test_fullscreen_context_ui import app as app, image_factory as image_factory, window as window
from test_fullscreen_transcript_ui import consume_subtitle_pages
from test_ui import recording_context_pipeline as recording_context_pipeline


def story_state():
    book = "An imported voyage through the polar seas."
    segments = [TranscriptSegment(0, 5, "The expedition sailed toward the ice."),
                TranscriptSegment(6, 11, "A frozen harbor appeared beyond the cape.")]
    return SessionState(
        segments=segments, book_context=book, book_guidance_strength=.42,
        guidance_revision=7, revision=12,
        provider_key=("ollama", "qwen2.5:7b", book, "f" * 64),
        analyzer_checkpoint={"running_summary": "The expedition approaches a harbor.", "processed_count": 2},
        topic=Topic("frozen harbor", " ".join(segment.text for segment in segments),
                    summary="An expedition approaches a frozen harbor.", scene="A ship rounds an icy cape.",
                    visual_subjects=("ship", "frozen harbor"), transcript=tuple(segment.text for segment in segments),
                    context_used_segments=2, image_guidance=book, guidance_strength=.42),
    )


def choose_import(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))


def choose_export(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))


def write_story(path):
    state = story_state()
    session_store.export_context(state, path)
    return state


def visible_story(window):
    state_snapshot = session_store.snapshot_session(window._session_state) if window._session_state else None
    if state_snapshot is not None:
        state_snapshot.pop("saved_at")
    return {
        "state": window._session_state, "session": window._session,
        "state_contents": state_snapshot,
        "transcript": window.transcript.toPlainText(), "context": window.context_text.toPlainText(),
        "title": window.canvas.current_title, "credit": window.canvas.credit_text.toPlainText(),
        "caption": window.canvas.current_transcript, "caption_queue": list(window.canvas._transcript_pending),
        "frame": window.canvas.current_pixmap.cacheKey(),
        "pending": list(window.slideshow.pending), "history": list(window.slideshow.history),
        "book": window.book_context.text(), "strength": window.book_guidance_strength.value(),
    }


def establish_old_story(window, image_factory):
    state = SessionState(book_context="The previous book.",
                         segments=[TranscriptSegment(0, 5, "The old story begins."),
                                   TranscriptSegment(6, 11, "The old story continues.")])
    window._session_state = state
    window._handle_event(window._session, "transcript", list(state.segments))
    window._handle_event(window._session, "context", {"phase": "ready", "summary": "The previous interpretation.",
                                                     "scene": "A previous scene.", "visual_subjects": ["old harbor"]})
    window._handle_event(window._session, "images", [image_factory(), image_factory("two", "Older image", "cyan")])
    return state


def test_export_works_with_retention_off_and_keeps_full_applied_state(window, monkeypatch, tmp_path):
    state = story_state()
    window._session_state = state
    window._applied_book_guidance = state.book_context, state.book_guidance_strength
    window._show_transcript(list(state.segments), restored=True)
    window._handle_event(window._session, "topic", state.topic)
    window.book_context.setText("An unapplied editor draft.")
    window.book_guidance_strength.setValue(.9)
    path = tmp_path / "exported-context.json"
    choose_export(monkeypatch, path)
    assert not window.retain_context.isChecked() and window._session_store is None
    window.tabs.setCurrentWidget(window.context_panel)
    assert window.export_context_button.isVisible() and window.export_context_button.isEnabled()
    window.export_context_button.click()

    exported = session_store.import_context(path)
    assert exported.session_id == state.session_id
    assert exported.segments == state.segments
    assert exported.topic.summary == state.topic.summary
    assert exported.topic.scene == state.topic.scene
    assert exported.topic.visual_subjects == state.topic.visual_subjects
    assert exported.analyzer_checkpoint == state.analyzer_checkpoint
    assert exported.provider_key == state.provider_key
    assert exported.book_context == state.book_context
    assert exported.book_guidance_strength == .42 and exported.guidance_revision == 7
    assert window.book_context.text() == "An unapplied editor draft."
    assert window._session_state is state and window._session_store is None
    assert not (tmp_path / "sessions" / "last-session.json").exists()


def test_export_ui_only_story_includes_raw_speech_and_accepted_context(window, monkeypatch, tmp_path):
    state = story_state()
    window._show_transcript(list(state.segments))
    window._handle_event(window._session, "topic", state.topic)
    window._applied_book_guidance = state.book_context, state.book_guidance_strength
    window._guidance_revision = state.guidance_revision
    assert window._session_state is None
    path = tmp_path / "ui-context.json"
    choose_export(monkeypatch, path)
    window.export_context()
    exported = session_store.import_context(path)
    assert exported.segments == state.segments
    assert exported.topic.summary == state.topic.summary
    assert exported.topic.visual_subjects == state.topic.visual_subjects
    assert exported.book_context == state.book_context and exported.book_guidance_strength == .42
    assert window._session_state is None


def test_import_replaces_old_story_and_rejects_prior_worker_events_without_changing_runtime(
        window, app, image_factory, monkeypatch, tmp_path):
    old = establish_old_story(window, image_factory)
    window.chunk_interval.setValue(3)
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    window.audio_file.setText(str(tmp_path / "current-audio.wav"))
    window.model.setCurrentText("turbo")
    window.whisper_device.setCurrentIndex(window.whisper_device.findData("cpu"))
    window.context_model.setText("qwen2.5:14b")
    window.ranking.setCurrentIndex(window.ranking.findData("context_clip"))
    window.threshold.setValue(.18)
    for name, check in window.provider_boxes.items():
        check.setChecked(name == "archive")
    before = asdict(window._read_settings(require_recording=False))
    old_session = window._session
    path = tmp_path / "imported-context.json"
    imported = write_story(path)
    choose_import(monkeypatch, path)
    window.tabs.setCurrentWidget(window.context_panel)
    assert window.import_context_button.isVisible() and window.import_context_button.isEnabled()
    window.import_context_button.click()
    app.processEvents()

    state = window._session_state
    assert state is not old and state.session_id == imported.session_id
    assert state.segments == imported.segments
    assert state.analyzer_checkpoint == imported.analyzer_checkpoint
    assert state.provider_key == imported.provider_key
    assert window.transcript.toPlainText().splitlines() == [
        "[00:00] " + imported.segments[0].text, "[00:06] " + imported.segments[1].text,
    ]
    assert imported.topic.summary in window.context_text.toPlainText()
    assert imported.topic.scene in window.context_text.toPlainText()
    assert all(subject in window.context_text.toPlainText() for subject in imported.topic.visual_subjects)
    assert window.book_context.text() == window.settings.book_context == imported.book_context
    assert window.book_guidance_strength.value() == window.settings.book_guidance_strength == .42
    assert window._applied_book_guidance == (imported.book_context, .42)
    assert window._guidance_revision == 7
    assert window._context_frozen and not window._monitoring and window.pipeline is None
    assert not window.slideshow.pending and not window.slideshow.history
    assert not window.canvas.current_title and window.canvas.current_pixmap.isNull()
    assert window.canvas.current_transcript == imported.segments[-1].text
    assert not window.canvas._transcript_pending
    after = asdict(window._read_settings(require_recording=False))
    for key in before:
        if key not in {"book_context", "book_guidance_strength"}:
            assert after[key] == before[key], key
    assert window._session_store is None
    snapshot = visible_story(window)
    window._handle_event(old_session, "transcript", list(old.segments))
    window._handle_event(old_session, "context", {"phase": "ready", "summary": "A stale old summary."})
    window._handle_event(old_session, "images", [image_factory("stale", "A stale title")])
    assert visible_story(window) == snapshot


def test_start_uses_imported_session_and_checkpoint_without_opening_hardware(
        window, recording_context_pipeline, monkeypatch, tmp_path):
    path = tmp_path / "resumable-context.json"
    imported = write_story(path)
    choose_import(monkeypatch, path)
    window.import_context()
    in_memory = window._session_state
    window.audio_mode.setCurrentIndex(window.audio_mode.findData("file"))
    recording = tmp_path / "fixture.wav"
    recording.write_bytes(b"A fixture pipeline does not decode this file.")
    window.audio_file.setText(str(recording))
    window.start_monitoring()
    assert recording_context_pipeline
    pipeline = recording_context_pipeline[-1]
    assert pipeline.session_state is in_memory
    assert pipeline.session_state.session_id == imported.session_id
    assert pipeline.session_state.segments == imported.segments
    assert pipeline.session_state.analyzer_checkpoint == imported.analyzer_checkpoint
    assert pipeline.session_state.provider_key == imported.provider_key
    assert pipeline.settings.book_context == imported.book_context
    assert pipeline.settings.book_guidance_strength == imported.book_guidance_strength
    assert window.transcript.document().blockCount() == len(imported.segments)
    assert window._monitoring and not window._context_frozen
    window.stop_monitoring()


def test_import_with_retention_on_saves_the_imported_story_immediately(window, monkeypatch, tmp_path):
    window.retain_context.setChecked(True)
    path = tmp_path / "saved-import.json"
    imported = write_story(path)
    choose_import(monkeypatch, path)
    window.import_context()
    assert window._session_store.path.is_file()
    saved = window._session_store.load()
    assert saved.session_id == imported.session_id and saved.segments == imported.segments
    assert saved.analyzer_checkpoint == imported.analyzer_checkpoint
    assert saved.provider_key == imported.provider_key
    assert saved.book_context == imported.book_context and saved.book_guidance_strength == .42


@pytest.mark.parametrize("operation", ["import", "export"])
def test_canceled_context_dialog_preserves_the_visible_story(window, image_factory, monkeypatch, operation):
    establish_old_story(window, image_factory)
    snapshot = visible_story(window)
    if operation == "import":
        choose_import(monkeypatch, "")
    else:
        choose_export(monkeypatch, "")
    getattr(window, operation + "_context")()
    assert visible_story(window) == snapshot


@pytest.mark.parametrize("failure", ["invalid", "missing"])
def test_import_file_error_preserves_current_story_and_image_queue(
        window, image_factory, monkeypatch, tmp_path, failure):
    establish_old_story(window, image_factory)
    snapshot = visible_story(window)
    path = tmp_path / (failure + ".json")
    if failure == "invalid":
        path.write_text('{"version":1,"segments":"damaged"}', encoding="utf-8")
    choose_import(monkeypatch, path)
    window.import_context()
    assert visible_story(window) == snapshot
    assert "could not be imported" in window.session_status_label.text().lower()


def test_export_file_error_preserves_current_story_and_image_queue(window, image_factory, monkeypatch, tmp_path):
    establish_old_story(window, image_factory)
    snapshot = visible_story(window)
    path = tmp_path / "blocked.json"
    path.mkdir()
    choose_export(monkeypatch, path)
    window.export_context()
    assert visible_story(window) == snapshot
    assert "could not be exported" in window.session_status_label.text().lower()


@pytest.mark.parametrize("flag", ["_monitoring", "_stopping"])
def test_import_is_disabled_during_monitoring_or_stopping(window, monkeypatch, flag):
    setattr(window, flag, True)
    window._update_controls()
    assert not window.import_context_button.isEnabled()

    def unexpected_dialog(*args, **kwargs):
        pytest.fail("Import must not prompt for a file while monitoring or stopping.")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", unexpected_dialog)
    window.import_context()
    assert window._session_state is None
    setattr(window, flag, False)
    window._update_controls()
    assert window.import_context_button.isEnabled()


def test_valid_import_stays_visible_when_immediate_local_save_fails(window, monkeypatch, tmp_path):
    window.retain_context.setChecked(True)
    store = window._session_store
    original_flush = store.flush

    def failed_flush(*args, **kwargs):
        raise session_store.SessionStoreError("Could not save imported context locally.")

    monkeypatch.setattr(store, "flush", failed_flush)
    path = tmp_path / "valid-import.json"
    imported = write_story(path)
    choose_import(monkeypatch, path)
    window.import_context()
    assert window._session_state.session_id == imported.session_id
    assert window._session_state.segments == imported.segments
    assert imported.segments[-1].text in window.transcript.toPlainText()
    assert imported.topic.summary in window.context_text.toPlainText()
    assert window._context_frozen and window.isVisible()
    assert "could not save" in window.session_status_label.text().lower()
    monkeypatch.setattr(store, "flush", original_flush)


def test_recognized_profanity_punctuation_and_unicode_survive_caption_pages_and_context_transfer(
        window, app, monkeypatch, tmp_path):
    text = ('“Fuck this shit,” he said. Damn it. 🌊 Merde — mierda; FUCK! '
            "This is uncensored speech. ") * 35
    segment = TranscriptSegment(0, 12, text)
    state = SessionState(segments=[segment],
                         topic=Topic("storm", text, summary="A character curses during a storm.",
                                     scene="A boat is caught in a storm.", context_used_segments=1))
    window._session_state = state
    window.toggle_fullscreen(with_context=True)
    window._handle_event(window._session, "transcript", list(state.segments))
    window._handle_event(window._session, "topic", state.topic)
    pages = consume_subtitle_pages(window.canvas, app)
    assert " ".join(pages).split() == text.split()
    assert "Fuck" in " ".join(pages) and "shit" in " ".join(pages)
    assert "*" not in " ".join(pages)

    path = tmp_path / "uncensored-context.json"
    choose_export(monkeypatch, path)
    window.export_context()
    exported = session_store.import_context(path)
    assert exported.segments == [segment]
    choose_import(monkeypatch, path)
    window.import_context()
    assert window._session_state.segments == [segment]
    assert window.transcript.toPlainText() == "[00:00] " + text.strip()
    assert " ".join(consume_subtitle_pages(window.canvas, app)).split() == text.split()

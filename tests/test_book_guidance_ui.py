"""Explicit book guidance has separate draft, accepted, and durable local state."""
from dataclasses import asdict, replace
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFontDatabase, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QScrollArea

from semantic_image_stream.configuration import Settings
from semantic_image_stream.desktop_ui import MainWindow
from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.session_store import SessionStore


class LocalGuidanceBoundary:
    def __init__(self, settings=None, on_event=None, session_state=None):
        self.settings, self.on_event, self.session_state = settings, on_event, session_state
        self.requests = []
        self.filter_requests = []
        self.started = self.stopped = False

    def update_book_guidance(self, settings):
        self.requests.append(settings)

    def update_image_filters(self, settings):
        self.filter_requests.append(settings)

    def record_displayed(self, candidate):
        pass

    def get_feedback_vote(self, topic, candidate):
        return None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def complete(self, window, revision=1, index=-1, emit=True):
        settings = self.requests[index]
        topic = window._feedback_topic or Topic("coast", "Spoken story", scene="A boat on the coast.")
        snapshot = replace(topic, image_guidance=settings.book_context, guidance_strength=settings.book_guidance_strength)
        if self.session_state is not None:
            with self.session_state.lock:
                self.session_state.book_context = settings.book_context
                self.session_state.book_guidance_strength = settings.book_guidance_strength
                self.session_state.guidance_revision = revision
                self.session_state.guidance_draft_marker = ""
                self.session_state.guidance_draft_revision = revision
                self.session_state.topic = snapshot
        payload = {"phase": "ready", "playback_key": f"guidance:{revision}", "playback_changed": True,
                   "retaining": False, "book_context": settings.book_context,
                   "book_guidance_strength": settings.book_guidance_strength, "guidance_revision": revision,
                   "summary": snapshot.summary, "scene": snapshot.scene, "topic_snapshot": snapshot}
        if emit:
            window._handle_event(window._session, "context", payload)
        return payload


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    if not QFontDatabase.families() and os.name == "nt":
        for name in ("segoeui.ttf", "georgia.ttf", "georgiai.ttf"):
            path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
            if path.is_file():
                assert QFontDatabase.addApplicationFont(str(path)) >= 0
    return application


@pytest.fixture
def window_factory(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    windows = []

    def make(**overrides):
        settings = dict(cache_dir=str(tmp_path / "cache"), training_dir=str(tmp_path / "training"),
                        session_dir=str(tmp_path / "sessions"), retain_context=False,
                        book_context="Established coastal background", book_guidance_strength=.65,
                        providers=["wikimedia", "archive"], whisper_model="large-v3", relevance_threshold=.1)
        settings.update(overrides)
        window = MainWindow(Settings(**settings), config_path=tmp_path / "config.toml")
        window.pipeline = LocalGuidanceBoundary()
        window.resize(980, 720)
        window.show()
        app.processEvents()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window._monitoring = window._stopping = False
        window.close()
        window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def established_story(window, tmp_path):
    topic = Topic("coastal boat", "Spoken story", summary="A journey along a cold coast.",
                  scene="A boat arrives at the coast.", visual_subjects=("boat", "coast"),
                  image_guidance=window.settings.book_context, guidance_strength=window.settings.book_guidance_strength)
    state = SessionState(segments=[TranscriptSegment(0, 2, "Earlier recorded passage")], topic=topic,
                         book_context=window.settings.book_context,
                         book_guidance_strength=window.settings.book_guidance_strength, guidance_revision=0)
    window._session_state = state
    window._guidance_revision = 0
    window.pipeline = LocalGuidanceBoundary(window.settings, session_state=state)
    window._handle_event(0, "context", {"phase": "ready", "playback_key": "scene:1", "topic_snapshot": topic,
                        "summary": topic.summary, "scene": topic.scene,
                        "book_context": state.book_context, "book_guidance_strength": state.book_guidance_strength,
                        "guidance_revision": 0})
    window._handle_event(0, "transcript", state.segments)
    images = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.png"
        pixels = QImage(128, 96, QImage.Format.Format_RGB32)
        pixels.fill(QColor("blue"))
        assert pixels.save(str(path))
        images.append(ImageCandidate(f"https://images.example/{name}", "https://source.example/coast",
                                     name, "wikimedia", local_path=str(path)))
    window._handle_event(0, "images", images)
    window.slideshow.timer.setInterval(60_000)
    return state, topic, images


def saved_state(settings):
    store = SessionStore(settings)
    try:
        return store.load()
    finally:
        store.close()


def test_editor_keeps_long_multiline_plain_text_and_strength_in_saved_config(window_factory):
    window = window_factory()
    text = ("Cold coastline, sea ice, human figures, mountain paths, village interiors.\n" * 8).strip()
    assert len(text) > 550
    assert isinstance(window.book_context, QPlainTextEdit)
    window.book_context.setText(text + "\n<literal background>")
    window.book_guidance_strength.setValue(.8)
    assert window.book_context.text() == text + "\n<literal background>"
    assert window.book_context.height() >= 100
    assert not window.pipeline.requests
    window.save_settings()
    saved = Settings.load(window.config_path)
    assert saved.book_context == window.book_context.text()
    assert saved.book_guidance_strength == .8
    assert window._applied_book_guidance == ("Established coastal background", .65)


def test_live_apply_uses_immutable_snapshot_and_only_ack_clears_pending_images(window_factory, tmp_path):
    window = window_factory()
    state, topic, images = established_story(window, tmp_path)
    window._monitoring = True
    window._update_controls()
    before = asdict(window.settings)
    transcript, context = window.transcript.toPlainText(), window.context_text.toPlainText()
    timer_id = window.slideshow.timer.timerId()
    window.book_context.setText("Glacial coast\nLow winter light, boats, ice, mountain villages.")
    window.book_guidance_strength.setValue(.75)
    assert window.book_context.isEnabled() and window.book_guidance_strength.isEnabled()
    assert not window.interpreter.isEnabled() and not window.audio_mode.isEnabled()
    assert not window.pipeline.requests
    window.apply_book_guidance_button.click()
    assert len(window.pipeline.requests) == 1 and not window.apply_book_guidance_button.isEnabled()
    request = window.pipeline.requests[0]
    assert request.book_context.startswith("Glacial coast\n") and request.book_guidance_strength == .75
    assert window._applied_book_guidance == (before["book_context"], .65)
    assert state.book_context == before["book_context"]
    assert window.slideshow.pending and window.context_text.toPlainText() == context
    window.pipeline.complete(window)
    assert window._applied_book_guidance == (request.book_context, .75)
    assert not window.slideshow.pending and window._displayed_candidate is images[0]
    assert window.slideshow.timer.timerId() == timer_id
    assert window.transcript.toPlainText() == transcript
    assert window._feedback_topic.scene == topic.scene
    assert not window.pipeline.started and not window.pipeline.stopped
    for field in ("audio_mode", "audio_device", "whisper_model", "providers", "relevance_threshold",
                  "time_filter_enabled", "exclude_ai_generated", "cache_dir"):
        assert getattr(request, field) == before[field]


def test_edits_while_apply_is_pending_remain_a_draft_and_older_ack_cannot_replace_acceptance(window_factory):
    window = window_factory()
    window._monitoring = True
    window.book_context.setText("First requested background")
    window.apply_book_guidance_button.click()
    window.book_context.setText("A later unsubmitted draft")
    window.book_guidance_strength.setValue(.9)
    window.pipeline.complete(window, revision=2)
    assert window._applied_book_guidance == ("First requested background", .65)
    assert window.book_context.text() == "A later unsubmitted draft"
    assert window.book_guidance_strength.value() == .9
    assert window.apply_book_guidance_button.isEnabled()
    assert "draft" in window.book_guidance_status.text()
    window._context_frozen = True
    window._handle_event(0, "context", {"phase": "ready", "book_context": "Stale background",
                        "book_guidance_strength": .1, "guidance_revision": 1})
    assert window._applied_book_guidance == ("First requested background", .65)
    window.book_context.setText("Changed after the request")
    assert window.pipeline.requests[0].book_context == "First requested background"


def test_stop_flush_preserves_worker_acceptance_before_queued_ack_and_freezes_scene(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    state, _, _ = established_story(window, tmp_path)
    window._monitoring = True
    transcript, context = window.transcript.toPlainText(), window.context_text.toPlainText()
    key, pending = window._playback_key, list(window.slideshow.pending)
    window.book_context.setText("Accepted new coast guidance")
    window.book_guidance_strength.setValue(.8)
    window.apply_book_guidance_button.click()
    ack = window.pipeline.complete(window, emit=False)
    assert window._applied_book_guidance[0] == "Established coastal background"
    window.stop_button.setEnabled(True)
    window.stop_button.click()
    assert not window.book_context.isEnabled() and not window.book_guidance_strength.isEnabled()
    loaded = saved_state(window.settings)
    assert loaded.book_context == state.book_context == "Accepted new coast guidance"
    assert loaded.book_guidance_strength == .8
    window._handle_event(0, "context", ack)
    assert window._applied_book_guidance == ("Accepted new coast guidance", .8)
    assert window._playback_key == key and list(window.slideshow.pending) == pending
    assert window.context_text.toPlainText() == context and window.transcript.toPlainText() == transcript
    window._handle_event(0, "stopped", None)
    assert window.book_context.isEnabled() and window._guidance_pending is None


def test_save_and_close_keep_applied_session_guidance_separate_from_draft_config(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    established_story(window, tmp_path)
    window.book_context.setText("Draft meant for the next Start")
    window.book_guidance_strength.setValue(.9)
    window.save_settings()
    config = Settings.load(window.config_path)
    assert (config.book_context, config.book_guidance_strength) == ("Draft meant for the next Start", .9)
    window._save_story_metadata(flush=True)
    loaded = saved_state(window.settings)
    assert (loaded.book_context, loaded.book_guidance_strength) == ("Established coastal background", .65)
    window.close()
    loaded = saved_state(config)
    assert (loaded.book_context, loaded.book_guidance_strength) == ("Established coastal background", .65)


def test_stopped_apply_and_relaunch_restore_strength_without_starting_audio(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    state, _, _ = established_story(window, tmp_path)
    transcript = window.transcript.toPlainText()
    window.book_context.setText("New explicitly applied background\nWinter sea, coast, ice.")
    window.book_guidance_strength.setValue(.85)
    window.apply_book_guidance_button.click()
    window._save_story_metadata(flush=True)
    assert not window.pipeline.requests and not window.pipeline.started
    assert window.transcript.toPlainText() == transcript and state.segments
    restored = window_factory(retain_context=True)
    assert restored.book_context.text() == window.book_context.text()
    assert restored.book_guidance_strength.value() == .85
    assert restored._applied_book_guidance == (window.book_context.text(), .85)
    assert restored.transcript.toPlainText() == transcript
    assert not restored._monitoring and not restored.pipeline.started


def test_explicitly_saved_draft_returns_to_editor_on_relaunch_without_overwriting_active_story(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    _, topic, _ = established_story(window, tmp_path)
    window.book_context.setText("Saved pending background\nA different visual atmosphere.")
    window.book_guidance_strength.setValue(.9)
    window.save_settings()
    window._save_story_metadata(flush=True)
    config = Settings.load(window.config_path)
    state = saved_state(window.settings)
    assert len(state.guidance_draft_marker) == 64 and state.guidance_draft_revision == state.guidance_revision
    assert state.book_context == "Established coastal background" and state.topic.scene == topic.scene
    restored = window_factory(retain_context=True, book_context=config.book_context,
                              book_guidance_strength=config.book_guidance_strength)
    assert restored.book_context.text() == config.book_context and restored.book_guidance_strength.value() == .9
    assert restored._applied_book_guidance == ("Established coastal background", .65)
    assert restored._session_state.topic.scene == topic.scene and restored._session_state.segments
    assert restored.apply_book_guidance_button.isEnabled()
    assert "draft" in restored.session_status_label.text()
    assert not restored.pipeline.started


def test_saved_empty_guidance_draft_is_distinguished_from_unchanged_default_settings(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    established_story(window, tmp_path)
    window.book_context.clear()
    window.save_settings()
    window._save_story_metadata(flush=True)
    restored = window_factory(retain_context=True, book_context="")
    assert restored.book_context.text() == "" and restored.apply_book_guidance_button.isEnabled()
    assert restored._applied_book_guidance == ("Established coastal background", .65)


def test_later_explicit_apply_invalidates_previously_saved_draft_without_editing_its_config(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    established_story(window, tmp_path)
    window.book_context.setText("An earlier saved draft")
    window.book_guidance_strength.setValue(.9)
    window.save_settings()
    window.book_context.setText("A later applied background")
    window.book_guidance_strength.setValue(.8)
    window.apply_book_guidance_button.click()
    window._save_story_metadata(flush=True)
    config = Settings.load(window.config_path)
    assert config.book_context == "An earlier saved draft" and config.book_guidance_strength == .9
    assert not window._session_state.guidance_draft_marker
    restored = window_factory(retain_context=True, book_context=config.book_context,
                              book_guidance_strength=config.book_guidance_strength)
    assert restored.book_context.text() == "A later applied background"
    assert restored.book_guidance_strength.value() == .8
    assert restored._applied_book_guidance == ("A later applied background", .8)


def test_draft_marker_from_an_older_accepted_revision_does_not_override_restored_story(window_factory, tmp_path):
    window = window_factory(retain_context=True)
    state, _, _ = established_story(window, tmp_path)
    window.book_context.setText("A saved draft from the previous guidance")
    window.save_settings()
    with state.lock:
        state.guidance_revision += 1
    window._save_story_metadata(flush=True)
    config = Settings.load(window.config_path)
    restored = window_factory(retain_context=True, book_context=config.book_context,
                              book_guidance_strength=config.book_guidance_strength)
    assert restored.book_context.text() == "Established coastal background"
    assert restored._applied_book_guidance == ("Established coastal background", .65)


def test_valid_guidance_and_live_date_filters_are_independent_of_unfinished_other_drafts(window_factory):
    window = window_factory(time_filter_enabled=True)
    window._monitoring = True
    window.time_filter_start.setText("unfinished date")
    window.book_context.setText("A valid winter coast")
    window.apply_book_guidance_button.click()
    assert len(window.pipeline.requests) == 1
    window.pipeline.complete(window)
    window.book_context.setText("x" * 2001)
    window.time_filter_start.setText("1500 CE")
    window._time_fields_edited()
    window._apply_image_filters_live()
    assert len(window.pipeline.filter_requests) == 1
    assert window.pipeline.filter_requests[0].time_filter_start == 1500
    assert window.pipeline.filter_requests[0].book_context == "A valid winter coast"
    assert not window.apply_book_guidance_button.isEnabled()
    assert "2,000" in window.book_guidance_status.text()


def test_start_uses_current_draft_without_losing_retained_transcript(window_factory, monkeypatch, tmp_path):
    import semantic_image_stream.engine as engine

    window = window_factory()
    state, _, _ = established_story(window, tmp_path)
    transcript = window.transcript.toPlainText()
    window.book_context.setText("Draft applied by Start")
    window.book_guidance_strength.setValue(.7)
    monkeypatch.setattr(engine, "Pipeline", LocalGuidanceBoundary)
    window.start_monitoring()
    assert window.pipeline.started and window.pipeline.session_state is state
    assert (window.pipeline.settings.book_context, window.pipeline.settings.book_guidance_strength) == (
            "Draft applied by Start", .7)
    assert window._applied_book_guidance == ("Draft applied by Start", .7)
    assert window.transcript.toPlainText() == transcript and state.segments


def test_failed_apply_keeps_prior_scene_and_allows_retry(window_factory, monkeypatch, tmp_path):
    window = window_factory()
    _, _, images = established_story(window, tmp_path)
    window._monitoring = True
    before = window._applied_book_guidance
    window.book_context.setText("A new draft")

    def unavailable(settings):
        raise RuntimeError("The local worker is unavailable. Try again.")

    monkeypatch.setattr(window.pipeline, "update_book_guidance", unavailable)
    window.apply_book_guidance_button.click()
    assert window._applied_book_guidance == before and window._guidance_pending is None
    assert window._displayed_candidate is images[0] and window.slideshow.pending
    assert window.apply_book_guidance_button.isEnabled()
    assert "Try again" in window.book_guidance_status.text()


def test_large_editor_fits_scroll_settings_and_escape_preserves_draft(window_factory, app):
    window = window_factory()
    window.book_context.setText("Coastal winter, ice, villages, sea, boats.\n" * 12)
    window.book_guidance_strength.setValue(.8)
    scroll = window.findChild(QScrollArea)
    scroll.ensureWidgetVisible(window.apply_book_guidance_button)
    app.processEvents()
    assert scroll.horizontalScrollBar().maximum() == 0 and window.width() == 980
    assert window.book_context.width() >= 300
    draft = window.book_context.text(), window.book_guidance_strength.value()
    window.toggle_fullscreen(with_title=True)
    window.activateWindow()
    app.processEvents()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    app.processEvents()
    assert not window._fullscreen and window.book_context.isVisible()
    assert (window.book_context.text(), window.book_guidance_strength.value()) == draft
    assert not window.pipeline.requests

"""Local training choices preserve working capture settings and saved book data."""
from dataclasses import asdict, replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.presets import AUDIOBOOK_PRESETS, apply_audiobook_preset
from semantic_image_stream.training_paths import default_training_dir, resolve_training_dir, training_file


def test_training_preferences_roundtrip_without_changing_cutoff_or_audio(tmp_path):
    settings = Settings(training_dir=str(tmp_path / "training"), use_story_examples=False,
                        ranking="learned_context", relevance_threshold=.43, audio_device=141,
                        whisper_model="large-v3", language="en", retain_context=True,
                        session_dir=str(tmp_path / "saved-book"), cache_dir=str(tmp_path / "cache"))
    path = tmp_path / "config.toml"
    settings.save(path)
    assert Settings.load(path) == settings.validate()
    assert settings.relevance_threshold == .43
    assert settings.audio_device == 141


@pytest.mark.parametrize("field,value", [
    ("training_dir", 12), ("training_dir", "bad\x00path"),
    ("training_dir", "bad\npath"), ("training_dir", "x" * 4097),
    ("use_story_examples", "true"), ("use_story_examples", 1),
])
def test_training_preferences_reject_invalid_types_and_paths(field, value):
    with pytest.raises(ValueError, match=field):
        replace(Settings(), **{field: value}).validate()


def test_training_files_resolve_separately_without_creating_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "app-data"))
    monkeypatch.setenv("SIS_TRAINING_TEST_ROOT", str(tmp_path / "chosen"))
    path = default_training_dir()
    assert path == tmp_path / "app-data" / "SemanticImageStream" / "training"
    assert not path.exists()
    settings = Settings(training_dir="%SIS_TRAINING_TEST_ROOT%/training",
                        session_dir=str(tmp_path / "sessions"), cache_dir=str(tmp_path / "cache"))
    # Windows expands percent-style environment variables; POSIX uses $NAME.
    if os.name != "nt":
        settings.training_dir = "$SIS_TRAINING_TEST_ROOT/training"
    assert resolve_training_dir(settings) == (tmp_path / "chosen" / "training").resolve()
    assert not resolve_training_dir(settings).exists()


def test_bundled_training_is_only_used_without_explicit_folder(monkeypatch, tmp_path):
    import semantic_image_stream.training_paths as paths

    local = tmp_path / "local-training"
    bundled = tmp_path / "package" / "data" / "training"
    bundled.mkdir(parents=True)
    model = bundled / "relevance-model.json"
    model.write_text("bundled model", encoding="utf-8")
    monkeypatch.setattr(paths, "default_training_dir", lambda: local)
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "package" / "training_paths.py"))
    assert training_file(SimpleNamespace(), model.name) == model
    explicit = tmp_path / "chosen-training"
    assert training_file(Settings(training_dir=str(explicit)), model.name) == explicit / model.name
    local.mkdir()
    (local / model.name).write_text("custom model", encoding="utf-8")
    assert training_file(Settings(), model.name) == local / model.name
    assert training_file(Settings(), "missing.json") == local / "missing.json"
    assert not explicit.exists()


@pytest.mark.parametrize("name", ["../other.json", "..", "/other.json", ""])
def test_training_file_rejects_path_traversal(name):
    with pytest.raises(ValueError, match="single file"):
        training_file(Settings(), name)


@pytest.mark.parametrize("key", AUDIOBOOK_PRESETS)
def test_presets_preserve_installed_training_choice_and_book_memory(tmp_path, key):
    settings = Settings(ranking="learned_context", training_dir=str(tmp_path / "trained"),
                        use_story_examples=False, audio_mode="microphone", audio_device=73,
                        whisper_model="turbo", book_context="A private ongoing book", retain_context=True,
                        session_dir=str(tmp_path / "saved-book"), cache_dir=str(tmp_path / "cache"))
    before = asdict(settings)
    updated = apply_audiobook_preset(settings, key)
    for field in ("ranking", "training_dir", "use_story_examples", "audio_mode", "audio_device",
                  "whisper_model", "book_context", "retain_context", "session_dir", "cache_dir"):
        assert getattr(updated, field) == before[field]
    assert asdict(settings) == before


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def local_window(app, monkeypatch, tmp_path):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    from semantic_image_stream.desktop_ui import MainWindow
    import semantic_image_stream.learned_relevance as learned_relevance
    import semantic_image_stream.session_store as session_store
    import semantic_image_stream.story_references as story_references

    monkeypatch.setattr(session_store, "default_session_dir", lambda: tmp_path / "sessions")
    monkeypatch.setattr(MainWindow, "_load_devices", lambda *args: None)
    calls = []

    def inspect_model(settings):
        calls.append(("model", settings.ranking))
        return {"status": "ready", "model": settings.clip_model, "images": 96, "pairs": 480,
                "message": "Local training pack ready."}

    def inspect_examples(settings):
        calls.append(("references", settings.use_story_examples))
        return {"status": "ready" if settings.use_story_examples else "disabled", "examples": 12}

    monkeypatch.setattr(learned_relevance, "inspect_training_pack", inspect_model)
    monkeypatch.setattr(story_references, "inspect_story_examples", inspect_examples)
    settings = Settings(training_dir=str(tmp_path / "training"), ranking="context_clip",
                        relevance_threshold=.43, audio_device=141, whisper_model="turbo",
                        book_context="Private narration stays local", retain_context=False,
                        session_dir=str(tmp_path / "saved-book"), cache_dir=str(tmp_path / "cache"))
    widget = MainWindow(settings, tmp_path / "settings.toml")
    widget.audio_device.addItem("Saved microphone", 141)
    widget.audio_device.setCurrentIndex(widget.audio_device.findData(141))
    widget.show()
    app.processEvents()
    yield widget, calls
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_training_mode_status_and_local_examples_save_working_settings(local_window, tmp_path):
    window, calls = local_window
    assert window.ranking.findData("learned_context") >= 0
    assert "ready (off)" in window.training_status_label.text()
    assert "96 images" in window.training_status_label.text()
    assert "480 training examples" in window.training_status_label.text()
    assert "12 examples" in window.training_status_label.text()
    assert window.settings.clip_model in window.training_status_label.text()
    assert str(tmp_path / "training") in window.training_status_label.toolTip()
    window.ranking.setCurrentIndex(window.ranking.findData("learned_context"))
    window.use_story_examples.setChecked(False)
    assert window.threshold.value() == .43
    assert "ready (off)" not in window.training_status_label.text()
    assert "Story examples off" in window.training_status_label.text()
    window.save_settings()
    saved = Settings.load(tmp_path / "settings.toml")
    assert saved.ranking == "learned_context"
    assert saved.use_story_examples is False
    assert saved.training_dir == str(tmp_path / "training")
    assert saved.relevance_threshold == .43
    assert saved.audio_device == 141
    assert saved.whisper_model == "turbo"
    assert saved.book_context == "Private narration stays local"
    assert calls[-1] == ("references", False)


def test_audiobook_apply_preserves_unsaved_training_controls(local_window):
    window, _ = local_window
    window.ranking.setCurrentIndex(window.ranking.findData("learned_context"))
    window.use_story_examples.setChecked(False)
    window.audiobook_preset.setCurrentIndex(window.audiobook_preset.findData("history"))
    window.apply_preset_button.click()
    assert window.ranking.currentData() == "learned_context"
    assert window.settings.ranking == "learned_context"
    assert window.use_story_examples.isChecked() is False
    assert window.settings.use_story_examples is False
    assert window.model.currentText() == "turbo"
    assert window.book_context.text() == "Private narration stays local"


def test_training_choices_freeze_during_monitoring_and_stopping(local_window):
    window, _ = local_window
    window._monitoring = True
    window._update_controls()
    assert not window.ranking.isEnabled()
    assert not window.use_story_examples.isEnabled()
    assert window.training_status_label.isEnabled()
    window._monitoring = False
    window._stopping = True
    window._update_controls()
    assert not window.use_story_examples.isEnabled()
    window._stopping = False
    window._update_controls()
    assert window.ranking.isEnabled()
    assert window.use_story_examples.isEnabled()


def test_start_passes_training_choice_and_retained_book_without_preparing_a_pack(local_window, monkeypatch, app):
    import semantic_image_stream.engine as engine
    from semantic_image_stream.models import TranscriptSegment

    window, calls = local_window
    states = []

    class CapturePipeline:
        def __init__(self, settings, callback, session_state=None):
            states.append((settings, session_state))
            self.callback = callback

        def start(self):
            pass

        def stop(self):
            self.callback("stopped", None)

    monkeypatch.setattr(engine, "Pipeline", CapturePipeline)
    retained = engine.SessionState(segments=[TranscriptSegment(0, 6, "The ship crossed the frozen harbor.")])
    window._session_state = retained
    window._show_transcript(retained.segments)
    window.ranking.setCurrentIndex(window.ranking.findData("learned_context"))
    window.use_story_examples.setChecked(False)
    window.start_button.click()
    assert states[0][0].ranking == "learned_context"
    assert states[0][0].use_story_examples is False
    assert states[0][0].relevance_threshold == .43
    assert states[0][1] is retained
    assert "frozen harbor" in window.transcript.toPlainText()
    assert not window.use_story_examples.isEnabled()
    assert not Path(window.settings.training_dir).exists()
    assert calls[-2:] == [("model", "learned_context"), ("references", False)]
    window.stop_button.click()
    app.processEvents()
    assert "frozen harbor" in window.transcript.toPlainText()
    assert window.use_story_examples.isEnabled()


def test_new_book_and_cache_deletion_keep_training_files(local_window, tmp_path):
    from semantic_image_stream.models import TranscriptSegment

    window, _ = local_window
    training = Path(window.settings.training_dir)
    training.mkdir()
    model_path = training / "relevance-model.json"
    reference_path = training / "story_examples.json"
    model_path.write_text("local trained artifact", encoding="utf-8")
    reference_path.write_text("public reference examples", encoding="utf-8")
    window.pipeline = SimpleNamespace(clear_cache=lambda: None, clear_history=lambda: None)
    window._handle_event(0, "transcript", [TranscriptSegment(0, 6, "Private recorded passage.")])
    window.clear_cache()
    assert "Private recorded passage" in window.transcript.toPlainText()
    window.new_book()
    assert not window.transcript.toPlainText()
    assert model_path.read_text(encoding="utf-8") == "local trained artifact"
    assert reference_path.read_text(encoding="utf-8") == "public reference examples"
    assert window.settings.training_dir == str(training)


def test_missing_or_invalid_pack_status_does_not_change_selected_cutoff(local_window, monkeypatch):
    import semantic_image_stream.learned_relevance as learned_relevance
    import semantic_image_stream.story_references as story_references

    window, _ = local_window
    monkeypatch.setattr(learned_relevance, "inspect_training_pack", lambda settings: {
        "status": "invalid", "message": "Trained relevance file could not be read."})
    monkeypatch.setattr(story_references, "inspect_story_examples", lambda settings: {
        "status": "missing", "examples": 0, "message": "No local story examples installed."})
    window.ranking.setCurrentIndex(window.ranking.findData("learned_context"))
    assert "could not be read" in window.training_status_label.text()
    assert "No local story examples" in window.training_status_label.text()
    assert window.ranking.currentData() == "learned_context"
    assert window.threshold.value() == .43


def test_cosine_only_model_status_discloses_calibration_without_changing_cutoff(local_window, monkeypatch):
    import semantic_image_stream.learned_relevance as learned_relevance

    window, _ = local_window
    monkeypatch.setattr(learned_relevance, "inspect_training_pack", lambda settings: {
        "status": "ready", "model": settings.clip_model, "images": 320, "pairs": 960,
        "chosen_head": "cosine_only", "recommended_threshold": .3890187047012057})
    window.ranking.setCurrentIndex(window.ranking.findData("learned_context"))
    assert "Trained calibration ready" in window.training_status_label.text()
    assert "320 images" in window.training_status_label.text()
    assert "960 training examples" in window.training_status_label.text()
    assert "Caption calibration suggestion: 0.39" in window.training_status_label.toolTip()
    assert "not an audiobook accuracy target" in window.training_status_label.toolTip()
    assert window.threshold.value() == .43
    assert window.settings.relevance_threshold == .43

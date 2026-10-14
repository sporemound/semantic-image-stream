"""Portable story snapshots preserve raw narration and safe resumption progress."""
from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from semantic_image_stream import narrative_context as narrative
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import Topic, TranscriptSegment
from semantic_image_stream.session_store import SessionStoreError, export_context, import_context, snapshot_session
from test_narrative_context import install_local_model, segments


def story(count=350):
    recording = [TranscriptSegment(index * 3 + .2, index * 3 + 2.8,
                                   f"Recorded passage {index}: a wooden ship crosses pack ice. Café.", "en")
                 for index in range(count)]
    state = SessionState(segments=recording, revision=8, book_context="A wooden ship's voyage through pack ice.",
                         book_guidance_strength=.42, guidance_revision=6,
                         guidance_draft_marker="a" * 64, guidance_draft_revision=6,
                         provider_key=("ollama", "qwen2.5:7b", "A wooden ship's voyage through pack ice.", "b" * 64))
    state.topic = Topic("wooden ship pack ice", "Rebuilt from recorded speech", kind="historical",
                        summary="A wooden ship moves through pack ice.", scene="A wooden ship among ice floes.",
                        visual_subjects=("wooden ship", "pack ice"), queries=("wooden ship pack ice", "pack ice"),
                        context_used_segments=min(3, count), visual_update="retain", image_guidance=state.book_context,
                        guidance_strength=.42, guidance_only=False)
    state.analyzer_checkpoint = {"version": 1, "cursor": min(3, count), "partial_offset": 0,
                                 "accepted_state": {"summary": "A wooden ship moves through pack ice."}}
    return state


def test_portable_roundtrip_retains_guidance_progress_and_complete_timestamped_transcript(tmp_path):
    state = story()
    changes = []
    state.on_change = changes.append
    before = snapshot_session(state)
    destination = tmp_path / "exports" / "book.context.json"

    export_context(state, destination)
    restored = import_context(destination)

    assert restored is not state and restored.session_id == state.session_id and restored.revision == 8
    assert restored.segments == state.segments and len(restored.segments) == 350
    assert restored.book_context == state.book_context and restored.book_guidance_strength == .42
    assert restored.guidance_revision == 6 and restored.guidance_draft_revision == 6
    assert restored.guidance_draft_marker == "a" * 64 and restored.provider_key == state.provider_key
    assert restored.analyzer_checkpoint == state.analyzer_checkpoint
    assert restored.analyzer_checkpoint is not state.analyzer_checkpoint
    assert restored.topic.summary == state.topic.summary and restored.topic.scene == state.topic.scene
    assert restored.topic.visual_subjects == state.topic.visual_subjects and restored.topic.queries == state.topic.queries
    assert restored.topic.visual_update == "retain" and restored.topic.context_used_segments == 3
    assert restored.topic.image_guidance == state.book_context and restored.topic.guidance_strength == .42
    assert restored.topic.transcript == tuple(segment.text for segment in state.segments)
    assert restored.topic.context == " ".join(restored.topic.transcript)
    assert restored.analyzer is None and restored.on_change is None and not changes
    assert restored.lock is not state.lock and restored.analysis_lock is not state.analysis_lock
    after = snapshot_session(state)
    before.pop("saved_at")
    after.pop("saved_at")
    assert before == after
    raw = json.loads(destination.read_bytes())
    assert "Café" in destination.read_text(encoding="utf-8")
    assert not {"lock", "analyzer", "on_change", "audio", "image_path"}.intersection(raw)
    assert "context" not in raw["topic"] and "transcript" not in raw["topic"]
    assert not list(destination.parent.glob(".context-*.tmp"))


def test_imported_lagging_checkpoint_resumes_only_unprocessed_recording(monkeypatch, tmp_path):
    config = Settings(semantic_provider="ollama", context_model="qwen2.5:7b", book_context="",
                      cache_dir=str(tmp_path / "images"), session_dir=str(tmp_path / "sessions"))
    recording = segments(6)
    built = []

    def factory(settings, **kwargs):
        analyzer = narrative.LocalNarrativeAnalyzer(settings, **kwargs)
        analyzer._batch_chars = len(recording[0].text) + 1
        prompts, _ = install_local_model(monkeypatch, analyzer)
        built.append((analyzer, prompts))
        return analyzer

    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", factory)
    state = SessionState(segments=list(recording))
    initial = Pipeline(config, lambda *args: None, session_state=state)
    analyzer = initial._session_analyzer()
    state.topic = initial._analyze_session(analyzer, recording)
    assert state.topic.context_used_segments == 1 and state.analyzer_checkpoint["cursor"] == 1
    destination = tmp_path / "story.json"
    export_context(state, destination)

    restored = import_context(destination)
    resumed_pipeline = Pipeline(config, lambda *args: None, session_state=restored)
    resumed = resumed_pipeline._session_analyzer()
    assert resumed.has_pending_segments and built[-1][1] == []
    for _ in range(len(recording)):
        if not resumed.has_pending_segments:
            break
        restored.topic = resumed_pipeline._analyze_session(resumed, restored.segments)
    assert not resumed.has_pending_segments and restored.topic.context_used_segments == len(recording)
    assert restored.analyzer_checkpoint["cursor"] == len(recording)
    assert [prompt["new_transcript"] for prompt in built[-1][1]] == [item.text for item in recording[1:]]
    assert restored.segments == recording


@pytest.mark.parametrize("damage", [
    lambda data: data.update(version=2),
    lambda data: data.update(session_id="../../other-file"),
    lambda data: data.update(revision=True),
    lambda data: data["segments"][0].update(end=-1),
    lambda data: data["segments"][0].update(start=float("nan")),
    lambda data: data["segments"][0].update(text={"unexpected": "data"}),
])
def test_malformed_core_data_is_rejected_without_changing_selected_file(tmp_path, damage):
    data = deepcopy(snapshot_session(story(2)))
    damage(data)
    destination = tmp_path / "bad.json"
    raw = json.dumps(data).encode()
    destination.write_bytes(raw)
    with pytest.raises(SessionStoreError, match="valid context JSON"):
        import_context(destination)
    assert destination.read_bytes() == raw


def test_corrupt_json_missing_file_and_oversized_source_are_rejected_without_changes(tmp_path, monkeypatch):
    import semantic_image_stream.session_store as module

    destination = tmp_path / "bad.json"
    destination.write_bytes(b'{"version":1, "incomplete":')
    raw = destination.read_bytes()
    with pytest.raises(SessionStoreError):
        import_context(destination)
    assert destination.read_bytes() == raw
    with pytest.raises(SessionStoreError):
        import_context(tmp_path / "missing.json")
    monkeypatch.setattr(module, "MAX_SESSION_BYTES", 32)
    destination.write_bytes(b"x" * 33)
    with pytest.raises(SessionStoreError, match="256 MiB"):
        import_context(destination)
    assert destination.read_bytes() == b"x" * 33


def test_damaged_derived_context_keeps_intact_recorded_speech(tmp_path):
    data = snapshot_session(story(2))
    data["topic"]["context_used_segments"] = 999
    data["analyzer_checkpoint"] = {"summary": "x" * 20_000}
    destination = tmp_path / "book.json"
    destination.write_text(json.dumps(data), encoding="utf-8")
    restored = import_context(destination)
    assert restored.segments == story(2).segments
    assert restored.topic is None and restored.analyzer_checkpoint is None


@pytest.mark.parametrize("growing", [False, True])
def test_bounded_read_rejects_source_that_changes_while_reading(tmp_path, monkeypatch, growing):
    destination = tmp_path / "story.json"
    export_context(story(2), destination)
    raw = destination.read_bytes()
    size = len(raw)
    original = Path.open

    class Reader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, requested=-1):
            assert requested <= size + 1, "Read the file size, not the whole supported limit"
            result = self.source.read(requested)
            if growing and requested > 1:
                with original(destination, "ab") as append:
                    append.write(b" ")
            return result

    def controlled_open(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        return Reader(source) if path == destination and args == ("rb",) else source

    monkeypatch.setattr(Path, "open", controlled_open)
    if growing:
        with pytest.raises(SessionStoreError, match="left unchanged"):
            import_context(destination)
        assert destination.read_bytes() == raw + b" "
    else:
        assert import_context(destination).segments == story(2).segments
        assert destination.read_bytes() == raw


def test_failed_atomic_export_preserves_previous_file_and_current_context(tmp_path, monkeypatch):
    import semantic_image_stream.session_store as module

    destination = tmp_path / "book.json"
    export_context(story(1), destination)
    previous = destination.read_bytes()
    state = story(2)
    calls = []

    def denied(*args):
        calls.append(args)
        raise PermissionError("private-path-or-text must never be exposed")

    monkeypatch.setattr(module.os, "replace", denied)
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    with pytest.raises(SessionStoreError, match="previous file") as error:
        export_context(state, destination)
    assert "private-path" not in str(error.value)
    assert len(calls) == 5 and len(set(calls)) == 1
    assert destination.read_bytes() == previous and len(state.segments) == 2
    assert not list(tmp_path.glob(".context-*.tmp"))


def test_oversized_export_preserves_existing_file(tmp_path, monkeypatch):
    import semantic_image_stream.session_store as module

    destination = tmp_path / "book.json"
    export_context(story(1), destination)
    previous = destination.read_bytes()
    monkeypatch.setattr(module, "MAX_SESSION_BYTES", 32)
    with pytest.raises(SessionStoreError, match="left unchanged"):
        export_context(story(2), destination)
    assert destination.read_bytes() == previous and not list(tmp_path.glob(".context-*.tmp"))


def test_live_export_uses_cached_checkpoint_without_waiting_for_analysis_lock(tmp_path):
    state = story(2)
    destination = tmp_path / "live.json"
    errors = []

    def export():
        try:
            export_context(state, destination)
        except Exception as exc:
            errors.append(exc)

    state.analysis_lock.acquire()
    worker = threading.Thread(target=export, daemon=True)
    try:
        state.segments.append(TranscriptSegment(6.2, 8.8, "A later recorded passage.", "en"))
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive() and not errors
        restored = import_context(destination)
        assert restored.segments == state.segments and len(restored.segments) == 3
        assert restored.analyzer_checkpoint["cursor"] == 2
    finally:
        state.analysis_lock.release()
        worker.join(timeout=2)

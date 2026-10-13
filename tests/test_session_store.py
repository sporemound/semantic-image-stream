"""Durable story memory across fresh objects and concurrent save/delete work."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.session_store import SessionStore, SessionStoreError, restore_session, snapshot_session


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path / "images-cache"), session_dir=str(tmp_path / "story-memory"))


def story(count=350):
    state = SessionState(book_context="Public fixture about a space mission",
                         provider_key=("ollama", "qwen2.5:7b", "Public fixture about a space mission"))
    state.segments.extend(TranscriptSegment(index * 6, index * 6 + 6, f"Recorded fixture passage {index}.", "en")
                          for index in range(count))
    state.topic = Topic("satellite orbit", "old context is rebuilt from raw speech", summary="A satellite enters orbit.",
                        scene="A satellite above Earth.", visual_subjects=("satellite above Earth",),
                        queries=("satellite orbit",), context_used_segments=count)
    return state


def test_full_transcript_and_context_survive_close_and_fresh_store(settings):
    state = story()
    messages = []
    store = SessionStore(settings, messages.append)
    store.bind(state)
    state.notify_changed()
    store.flush(state)
    store.close()
    fresh = SessionStore(settings)
    try:
        loaded = fresh.load()
        assert loaded is not state and loaded.session_id == state.session_id
        assert loaded.revision == state.revision and loaded.segments == state.segments
        assert len(loaded.segments) == 350
        assert loaded.book_context == state.book_context and loaded.provider_key == state.provider_key
        assert loaded.topic.summary == state.topic.summary and loaded.topic.scene == state.topic.scene
        assert loaded.topic.transcript == tuple(segment.text for segment in state.segments)
        assert loaded.topic.context == " ".join(segment.text for segment in state.segments)
        assert loaded.analyzer is None and loaded.on_change is None
        assert loaded.lock is not state.lock and loaded.analysis_lock is not state.analysis_lock
        raw = json.loads(fresh.path.read_text(encoding="utf-8"))
        assert "context" not in raw["topic"] and "transcript" not in raw["topic"]
        assert "analyzer" not in raw and "on_change" not in raw and "lock" not in raw
        assert any("350 recorded segments" in message for message in messages)
    finally:
        fresh.close()


def test_save_never_waits_for_inflight_analysis_and_can_keep_a_lagging_checkpoint(settings):
    state = story(2)
    state.analyzer_checkpoint = {"version": 1, "cursor": 1}
    store = SessionStore(settings)
    store.bind(state)
    state.analysis_lock.acquire()
    try:
        state.segments.append(TranscriptSegment(12, 18, "New speech while the model is busy."))
        started = time.monotonic()
        state.notify_changed()
        store.flush(state)
        assert time.monotonic() - started < 2
        saved = store.load()
        assert len(saved.segments) == 3 and saved.analyzer_checkpoint["cursor"] == 1
    finally:
        state.analysis_lock.release()
        store.close()


def test_clear_fences_pending_old_snapshot_and_keeps_new_session(settings, monkeypatch):
    import semantic_image_stream.session_store as module

    old, new = story(2), story(1)
    store = SessionStore(settings)
    store.bind(old)
    store.flush(old)
    entered, release = threading.Event(), threading.Event()
    original = module.json.dumps

    def delayed(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("session_id") == old.session_id and value.get("revision") == 1:
            entered.set()
            assert release.wait(3)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module.json, "dumps", delayed)
    try:
        old.notify_changed()
        assert entered.wait(2)
        store.clear()
        assert old.on_change is None and not store.path.exists()
        assert store.save(old) is None
        store.bind(new)
        store.save(new)
        release.set()
        store.flush(new)
        loaded = store.load()
        assert loaded.session_id == new.session_id and loaded.segments == new.segments
    finally:
        release.set()
        store.close()


def test_transient_windows_file_lock_retries_without_losing_prior_or_new_context(settings, monkeypatch):
    import semantic_image_stream.session_store as module

    state = story(1)
    store = SessionStore(settings)
    store.bind(state)
    store.flush(state)
    previous = store.path.read_bytes()
    actual_replace = module.os.replace
    attempts, waits = [], []

    def briefly_locked(source, destination):
        attempts.append((source, destination))
        if len(attempts) <= 2:
            assert store.path.read_bytes() == previous
            raise PermissionError("Controlled sharing violation")
        actual_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", briefly_locked)
    monkeypatch.setattr(module.time, "sleep", waits.append)
    try:
        state.segments.append(TranscriptSegment(6, 12, "A new recorded fixture passage."))
        store.flush(state)
        assert len(attempts) == 3 and waits == [.02, .04]
        assert len(set(attempts)) == 1, "Retry the same complete atomic snapshot"
        assert store.load().segments == state.segments
        assert not list(store.root.glob(".session-*.tmp"))
    finally:
        store.close()


def test_failed_atomic_replace_preserves_previous_save(settings, monkeypatch):
    import semantic_image_stream.session_store as module

    state = story(1)
    store = SessionStore(settings)
    store.bind(state)
    store.flush(state)
    original = store.path.read_bytes()

    def denied(*args):
        raise PermissionError("private-path-or-text must never be exposed")

    monkeypatch.setattr(module.os, "replace", denied)
    try:
        state.segments.append(TranscriptSegment(6, 12, "Unsaved new fixture speech."))
        state.notify_changed()
        with pytest.raises(SessionStoreError, match="current transcript remains") as error:
            store.flush(state)
        assert "private-path" not in str(error.value)
        assert store.path.read_bytes() == original
        assert len(store.load().segments) == 1
        assert not list(store.root.glob(".session-*.tmp"))
        assert len(state.segments) == 2
    finally:
        store.close()


def test_disable_saving_keeps_last_saved_file_and_drops_future_old_callbacks(settings):
    state = story(1)
    store = SessionStore(settings)
    store.bind(state)
    store.flush(state)
    raw = store.path.read_bytes()
    store.detach()
    state.segments.append(TranscriptSegment(6, 12, "Unretained speech."))
    assert state.on_change is None and store.save(state) is None
    assert store.path.read_bytes() == raw
    store.bind(state)
    store.flush(state)
    assert len(store.load().segments) == 2
    store.close()


def test_obsolete_write_failure_cannot_fail_a_newer_final_flush(settings, monkeypatch):
    import semantic_image_stream.session_store as module

    state = story(1)
    store = SessionStore(settings)
    store.bind(state)
    store.flush(state)
    old_entered, old_release = threading.Event(), threading.Event()
    new_entered, new_release = threading.Event(), threading.Event()
    original_json, original_write = module.json.dumps, store._atomic_write
    errors = []

    def encode(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("revision") == 1:
            old_entered.set()
            assert old_release.wait(3)
            raise OSError("Obsolete snapshot encoding failure")
        return original_json(value, *args, **kwargs)

    def write(raw):
        if json.loads(raw).get("revision") == 2:
            new_entered.set()
            assert new_release.wait(3)
        original_write(raw)

    def flush():
        try:
            store.flush(state, timeout=4)
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(module.json, "dumps", encode)
    monkeypatch.setattr(store, "_atomic_write", write)
    flusher = threading.Thread(target=flush)
    try:
        state.notify_changed()
        assert old_entered.wait(2)
        state.segments.append(TranscriptSegment(6, 12, "Current accepted speech."))
        state.notify_changed()
        flusher.start()
        with store._condition:
            assert store._condition.wait_for(lambda: store._latest_ticket >= 4, timeout=2)
        old_release.set()
        assert new_entered.wait(2)
        assert flusher.is_alive() and not errors
        new_release.set()
        flusher.join(timeout=2)
        assert not flusher.is_alive() and not errors
        assert len(store.load().segments) == 2
    finally:
        old_release.set()
        new_release.set()
        if flusher.ident is not None:
            flusher.join(timeout=3)
        store.close()


def test_corrupt_file_is_preserved_until_explicit_clear(settings):
    store = SessionStore(settings)
    store.root.mkdir(parents=True)
    broken = b'{"version":1, "incomplete":'
    store.path.write_bytes(broken)
    try:
        with pytest.raises(SessionStoreError, match="left untouched"):
            store.load()
        state = story(1)
        store.bind(state)
        with pytest.raises(SessionStoreError, match="unreadable"):
            store.flush(state)
        assert store.path.read_bytes() == broken
        store.clear()
        assert not store.path.exists()
        store.bind(state)
        store.flush(state)
        assert store.load().segments == state.segments
    finally:
        store.close()


@pytest.mark.parametrize("change", [
    lambda data: data.update(version=2),
    lambda data: data.update(session_id="../../other-file"),
    lambda data: data.update(revision=True),
    lambda data: data.update(provider_key=["remote", "model", "background"]),
    lambda data: data["segments"][0].update(start=float("nan")),
    lambda data: data["segments"][0].update(start=10 ** 400, end=10 ** 401),
    lambda data: data["segments"][0].update(end=-1),
    lambda data: data["segments"][0].update(text={"unexpected": "object"}),
])
def test_invalid_saved_core_data_is_rejected(change):
    data = deepcopy(snapshot_session(story(1)))
    change(data)
    with pytest.raises(ValueError):
        restore_session(data)


def test_bad_derived_fields_do_not_discard_recorded_speech():
    data = snapshot_session(story(2))
    data["topic"]["context_used_segments"] = 999
    data["analyzer_checkpoint"] = {"summary": "x" * 20_000}
    loaded = restore_session(data)
    assert len(loaded.segments) == 2 and loaded.topic is None and loaded.analyzer_checkpoint is None


def test_retained_scene_and_full_transcript_survive_session_restore():
    state = story(3)
    state.topic = replace(state.topic, visual_update="retain")
    loaded = restore_session(snapshot_session(state))
    assert loaded.topic.visual_update == "retain"
    assert loaded.topic.scene == state.topic.scene
    assert loaded.topic.queries == state.topic.queries
    assert loaded.topic.transcript == tuple(segment.text for segment in state.segments)
    assert loaded.topic.context_used_segments == 3


def test_older_saved_context_remains_compatible_without_visual_update():
    data = snapshot_session(story(2))
    data["topic"].pop("visual_update")
    loaded = restore_session(data)
    assert loaded.topic.visual_update == "replace"
    assert loaded.topic.query == "satellite orbit"
    assert len(loaded.topic.transcript) == 2


def test_invalid_visual_update_preserves_original_recorded_speech():
    state = story(2)
    data = snapshot_session(state)
    data["topic"]["visual_update"] = "unsupported"
    loaded = restore_session(data)
    assert loaded.topic is None
    assert loaded.segments == state.segments


def test_small_saved_story_does_not_allocate_the_whole_session_limit(settings, monkeypatch):
    store = SessionStore(settings)
    state = story(2)
    store.bind(state)
    store.flush(state)
    size = store.path.stat().st_size
    original = Path.open

    class SizedReader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.source.close()

        def fileno(self):
            return self.source.fileno()

        def read(self, requested=-1):
            if requested > size + 1:
                raise MemoryError("Avoid an unnecessary large allocation beside native models")
            return self.source.read(requested)

    def open_saved(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        return SizedReader(source) if path == store.path and args == ("rb",) else source

    monkeypatch.setattr(Path, "open", open_saved)
    try:
        assert store.load().segments == state.segments
    finally:
        store.close()


def test_clear_image_cache_preserves_saved_story_in_separate_folder(settings):
    store = SessionStore(settings)
    state = story(2)
    store.bind(state)
    store.flush(state)
    cache = Cache(settings)
    image = ImageCandidate(image_url="https://photos.example.org/fixture.png", source_url="https://museum.example.org/fixture",
                           title="Local fixture", provider="wikimedia")
    cache.put_query("a cached query", [image])
    cache.close()
    try:
        Pipeline(settings, lambda *args: None).clear_cache()
        assert store.load().segments == state.segments
    finally:
        store.close()

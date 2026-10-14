"""Session-wide ranking evidence survives cache eviction and portable transfer."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.session_image_export import (
    CSV_COLUMNS, ImageExportError, count_session_images, export_session_images, record_session_images,
)
from semantic_image_stream.session_store import (
    SessionStore, export_context, import_context, restore_session, snapshot_session,
)


def story():
    state = SessionState()
    state.segments = [TranscriptSegment(0, 6, "The spacecraft leaves Earth."),
                      TranscriptSegment(6, 12, "It approaches the Moon.")]
    state.topic = Topic("later UI topic", "Later UI context")
    return state


def topic(query="lunar spacecraft"):
    return Topic(query, "A spacecraft approaches the Moon.", summary="A spacecraft reaches lunar orbit.",
                 scene="A spacecraft above the Moon.", visual_subjects=("spacecraft", "Moon"),
                 transcript=("Earlier unrelated speech", "The spacecraft leaves Earth.",
                             "It approaches the Moon.", "A lunar landing begins."))


def image(name="moon", score=.85, **kwargs):
    values = dict(image_url=f"https://photos.example.org/{name}.png",
                  source_url=f"https://museum.example.org/{name}", title="A lunar spacecraft",
                  provider="wikimedia", creator="Public fixture creator", license="CC BY 4.0",
                  license_url="https://creativecommons.org/licenses/by/4.0/", description="Lunar orbit",
                  width=1280, height=720, score=score, local_path="missing-or-evicted.png")
    values.update(kwargs)
    return ImageCandidate(**values)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return reader.fieldnames, list(reader)


def test_blank_start_has_no_exportable_matches():
    assert count_session_images(None, .3) == 0
    assert count_session_images(SessionState(), .3) == 0
    with pytest.raises(ImageExportError, match="Start or load a session"):
        export_session_images(None, "unused.csv", .3)


def test_unique_best_score_keeps_original_ranking_context_and_not_live_objects():
    state, candidate = story(), image()
    updates = []
    state.on_change = lambda session: updates.append(session.revision)
    assert record_session_images(state, [candidate], topic(), .8, SimpleNamespace(name="context_clip"))
    first = deepcopy(state.image_matches[candidate.image_url])
    assert first["query"] == "lunar spacecraft" != state.topic.query
    assert first["ranker"] == "context_clip" and first["audio_position_seconds"] == 12
    assert first["acceptance_threshold"] == .8 and first["discovered_at_utc"].endswith("Z")
    assert first["relevant_passage"] == "The spacecraft leaves Earth.\nIt approaches the Moon.\nA lunar landing begins."
    assert "local_path" not in first
    candidate.score, candidate.title = .82, "Changed candidate title"
    assert not record_session_images(state, [candidate, candidate], topic("less relevant scene"), .8, "clip")
    assert state.image_matches[candidate.image_url] == first and updates == [1]
    candidate.score = .95
    state.segments.append(TranscriptSegment(12, 18, "The spacecraft lands."))
    assert record_session_images(state, [candidate], topic("lunar landing"), .8, "clip")
    best = state.image_matches[candidate.image_url]
    assert len(state.image_matches) == 1 and updates == [1, 2]
    assert best["relevance_score"] == .95 and best["title"] == "Changed candidate title"
    assert best["query"] == "lunar landing" and best["audio_position_seconds"] == 18
    candidate.score, candidate.title = .1, "Changed after recording"
    assert best["relevance_score"] == .95 and best["title"] == "Changed candidate title"


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf"), True, None, "0.95", 10 ** 400, 1.1])
def test_nonfinite_or_non_numeric_candidates_are_not_recorded(bad_score):
    state = story()
    assert not record_session_images(state, [image(score=bad_score)], topic(), .8, "clip")
    assert count_session_images(state, 0) == 0


@pytest.mark.parametrize("bad_url", ["file:///private.png", "javascript:alert(1)", "https:///empty.png",
                                    "https://user:password@photos.example.org/a.png", "https://photos.example.org/a\n.png"])
def test_only_public_http_image_links_are_recorded(bad_url):
    state = story()
    assert not record_session_images(state, [image(image_url=bad_url)], topic(), .8, "clip")


def test_export_filters_all_session_matches_without_requiring_cached_files(tmp_path):
    state = story()
    old = image("z-first", .95)
    assert record_session_images(state, [old, image("below", .79)], topic(), .8, "context_clip")
    assert record_session_images(state, [image("a-next", .95), image("c-accepted", .85)],
                                 topic("later scene"), .8, "context_clip")
    state.topic = topic("unrelated current UI scene")
    destination = tmp_path / "nested" / "links.csv"
    assert count_session_images(state, .9) == 2
    assert export_session_images(state, destination, .9) == 2
    columns, rows = read_csv(destination)
    assert tuple(columns) == CSV_COLUMNS and destination.read_bytes().startswith(b"\xef\xbb\xbf")
    assert [row["image_url"] for row in rows] == ["https://photos.example.org/a-next.png", old.image_url]
    assert rows[1]["source_url"] == old.source_url and rows[1]["query"] == "lunar spacecraft"
    assert rows[1]["creator"] == old.creator and float(rows[1]["relevance_score"]) == .95
    assert rows[1]["session_id"] == state.session_id and count_session_images(state, .8) == 3
    assert export_session_images(state, destination, .99) == 0
    assert read_csv(destination)[1] == []


def test_csv_preserves_unicode_quotes_multiline_text_and_neutralizes_spreadsheet_formulas(tmp_path):
    state = story()
    candidate = image(title='Moon, "été" 🌙\nsecond line', description='  =HYPERLINK("https://example.org")',
                      creator="@SUM(1+1)", license="-1+2", authenticity="+SUM(1+1)")
    exact_topic = replace(topic(), summary="\t=1+1", scene="\n@malicious", transcript=("normal", "=1+1"))
    assert record_session_images(state, [candidate], exact_topic, .8, "context_clip")
    destination = tmp_path / "images.csv"
    export_session_images(state, destination, .8)
    row = read_csv(destination)[1][0]
    assert row["title"] == candidate.title
    assert row["description"] == "'" + candidate.description
    assert row["creator"] == "'@SUM(1+1)" and row["license"] == "'-1+2"
    assert row["authenticity"] == "'+SUM(1+1)"
    assert row["summary"] == "'\t=1+1" and row["scene"] == "'\n@malicious"
    assert row["relevant_passage"] == "normal\n=1+1"
    assert state.image_matches[candidate.image_url]["description"] == candidate.description


def test_matches_survive_real_autosave_reload_and_portable_context_transfer(tmp_path):
    state = story()
    record_session_images(state, [image()], topic(), .8, "context_clip")
    settings = Settings(session_dir=str(tmp_path / "session"))
    store = SessionStore(settings)
    try:
        store.bind(state)
        store.flush(state)
    finally:
        store.close()
    fresh = SessionStore(settings)
    try:
        loaded = fresh.load()
        assert loaded.image_matches == state.image_matches and loaded.segments == state.segments
        assert loaded.image_matches is not state.image_matches
        assert count_session_images(loaded, .8) == 1
        assert export_session_images(loaded, tmp_path / "after-restart.csv", .8) == 1
        destination = tmp_path / "context.json"
        export_context(loaded, destination)
        transferred = import_context(destination)
        assert transferred.image_matches == state.image_matches and transferred.segments == state.segments
        assert json.loads(destination.read_text(encoding="utf-8"))["image_matches"] == state.image_matches
    finally:
        fresh.close()


def test_missing_or_damaged_derived_rows_do_not_discard_speech_or_other_matches():
    state = story()
    record_session_images(state, [image("one"), image("two")], topic(), .8, "clip")
    snapshot = snapshot_session(state)
    snapshot["image_matches"][image("one").image_url]["relevance_score"] = float("nan")
    snapshot["image_matches"]["invalid"] = {"image_url": "file:///other.png"}
    restored = restore_session(snapshot)
    assert restored.segments == state.segments and set(restored.image_matches) == {image("two").image_url}
    old = snapshot_session(state)
    old.pop("image_matches")
    assert restore_session(old).image_matches == {}
    snapshot["image_matches"] = "malformed derived data"
    assert restore_session(snapshot).segments == state.segments
    assert restore_session(snapshot).image_matches == {}


def test_foreign_session_matches_are_omitted_without_discarding_current_speech(tmp_path):
    current, foreign = story(), story()
    record_session_images(current, [image("current")], topic(), .8, "clip")
    record_session_images(foreign, [image("foreign")], topic(), .8, "clip")
    data = snapshot_session(current)
    data["image_matches"].update(deepcopy(foreign.image_matches))
    restored = restore_session(data)
    assert restored.segments == current.segments and restored.session_id == current.session_id
    assert set(restored.image_matches) == {image("current").image_url}
    current.image_matches.update(deepcopy(foreign.image_matches))
    assert count_session_images(current, .8) == 1
    assert set(snapshot_session(current)["image_matches"]) == {image("current").image_url}
    path = tmp_path / "own-session.csv"
    assert export_session_images(current, path, .8) == 1
    assert read_csv(path)[1][0]["session_id"] == current.session_id


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(source_url="javascript:alert(1)"),
    lambda row: row.update(license_url="file:///private"),
    lambda row: row.update(audio_position_seconds=float("inf")),
    lambda row: row.update(relevance_score=.1),
    lambda row: row.update(acceptance_threshold=True),
    lambda row: row.update(discovered_at_utc="2026-01-01T00:00:00"),
    lambda row: row.update(relevant_passage="x" * 6001),
    lambda row: row.update(ai_generated="false"),
    lambda row: row.update(width=-1),
])
def test_each_imported_match_is_independently_validated(mutation):
    state = story()
    record_session_images(state, [image()], topic(), .8, "clip")
    data = snapshot_session(state)
    mutation(next(iter(data["image_matches"].values())))
    loaded = restore_session(data)
    assert loaded.image_matches == {} and loaded.segments == state.segments


def test_context_snapshot_detaches_match_rows_and_limits_stored_passage():
    state = story()
    candidate = image(description="x" * 4000)
    record_session_images(state, [candidate], replace(topic(), transcript=("a" * 10_000,)), .8, "clip")
    snapshot = snapshot_session(state)
    saved = next(iter(snapshot["image_matches"].values()))
    assert len(saved["description"]) == 2000 and len(saved["relevant_passage"]) == 6000
    saved["title"] = "Modified snapshot"
    assert next(iter(state.image_matches.values()))["title"] == candidate.title


def test_failed_atomic_replace_leaves_prior_csv_and_session_intact(tmp_path, monkeypatch):
    import semantic_image_stream.session_image_export as module

    state = story()
    record_session_images(state, [image()], topic(), .8, "clip")
    before = deepcopy(state.image_matches)
    destination = tmp_path / "existing.csv"
    destination.write_bytes(b"Complete previous export")

    def denied(source, target):
        assert Path(source).read_bytes().startswith(b"\xef\xbb\xbf")
        assert destination.read_bytes() == b"Complete previous export"
        raise PermissionError("private transcript or path must not be exposed")

    monkeypatch.setattr(module.os, "replace", denied)
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    with pytest.raises(ImageExportError, match="previous file and session were left unchanged") as error:
        export_session_images(state, destination, .8)
    assert "private transcript" not in str(error.value)
    assert destination.read_bytes() == b"Complete previous export" and state.image_matches == before
    assert not list(tmp_path.glob(".images-*.tmp"))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.8", -1, 1.1])
def test_bad_export_minimum_does_not_write_a_file(tmp_path, score):
    destination = tmp_path / "should-not-exist.csv"
    with pytest.raises(ImageExportError, match="between 0 and 1"):
        export_session_images(story(), destination, score)
    assert not destination.exists()

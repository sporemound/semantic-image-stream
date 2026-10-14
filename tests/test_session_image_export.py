import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from semantic_image_stream.session_image_export import (
    CSV_FIELDS, SessionImageLog, default_export_threshold,
)


@dataclass
class Candidate:
    image_url: str = "https://images.example.org/one.jpg"
    source_url: str = "https://example.org/art/one"
    title: str = "An attributed image"
    provider: str = "deviantart"
    creator: str = "Artist"
    license: str = "All rights reserved"
    license_url: str = "https://example.org/license"
    score: float = 0.8
    local_path: str = "/private/cache/image.jpg"
    description: str = "PRIVATE DESCRIPTION"


@pytest.fixture
def ledger():
    with SessionImageLog() as value:
        yield value


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == CSV_FIELDS
        return list(reader)


def test_inclusive_threshold_and_stable_best_first_export(ledger, tmp_path):
    ledger.record([Candidate(score=0.699999), Candidate(image_url="https://x.test/b", score=.7),
                   Candidate(image_url="https://x.test/a", score=.7),
                   Candidate(image_url="https://x.test/c", score=.95)], ranking="context_clip")
    assert ledger.count() == 4
    assert ledger.count(.7) == 3
    path, count = ledger.export_csv(tmp_path / "chosen.csv", .7)
    rows = read_csv(path)
    assert count == 3
    assert [row["image_url"] for row in rows] == ["https://x.test/c", "https://x.test/a", "https://x.test/b"]
    assert {row["export_min_score"] for row in rows} == {"0.700000"}
    assert {row["session_id"] for row in rows} == {ledger.session_id}
    assert all(row["first_seen_utc"].endswith("Z") for row in rows)


def test_highest_score_metadata_snapshot_not_latest_mutable_candidate(ledger, tmp_path):
    first = Candidate(score=.4)
    topic = SimpleNamespace(query="first query", visual_subjects=("forest",),
                            context="PRIVATE TRANSCRIPT", transcript=("SECRET",))
    ledger.record([first], ranking="metadata", topic=topic)
    best = replace(first, score=.9, title="Best match", creator="Original artist")
    ledger.record([best], ranking="context_clip", topic=SimpleNamespace(query="best query", visual_subjects=("moon",)))
    best.score, best.title = .01, "Mutated after event"
    ledger.record([replace(first, score=.8, title="Lower match")], ranking="clip", topic=topic)
    path, count = ledger.export_csv(tmp_path / "best.csv")
    row, = read_csv(path)
    assert count == ledger.count() == 1
    assert row["relevance_score"] == "0.900000"
    assert row["title"] == "Best match"
    assert row["creator"] == "Original artist"
    assert row["ranking_mode"] == "context_clip"
    assert row["query"] == "best query"
    assert row["visual_subjects"] == '["moon"]'
    assert row["last_seen_utc"] >= row["first_seen_utc"]
    assert all(secret not in path.read_text(encoding="utf-8-sig")
               for secret in ("PRIVATE", "SECRET", "/private/cache", "description", "local_path"))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), -.1, 1.01, None, True, "bad"])
def test_invalid_scores_never_qualify(ledger, score):
    ledger.record([Candidate(score=score)], ranking="metadata")
    assert ledger.count() == 0


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -.1, 1.1, None, True, "bad"])
def test_bad_export_threshold_cannot_overwrite(ledger, tmp_path, score):
    path = tmp_path / "existing.csv"
    path.write_text("KEEP", encoding="utf-8")
    with pytest.raises(ValueError, match="finite number"):
        ledger.export_csv(path, score)
    assert path.read_text() == "KEEP"
    assert not list(tmp_path.glob(".sis-csv-*"))


@pytest.mark.parametrize("url", ["file:///private/image", "javascript:alert(1)", "=HYPERLINK(1)",
                                 "https://user:pass@example.org/a", "https:///missing", "",
                                 "https://example.org:bad/a", "https://example.org/a\nb", "https://x.test/a b",
                                 "https://x.test\\@evil.test/a"])
def test_invalid_direct_links_are_excluded(ledger, url):
    ledger.record([Candidate(image_url=url)], ranking="metadata")
    assert ledger.count() == 0


def test_signed_urls_not_merged_or_rewritten_and_unsafe_secondary_links_blank(ledger, tmp_path):
    urls = ["https://images.test/a.jpg?token=one&size=full", "https://images.test/a.jpg?token=two&size=full"]
    ledger.record([Candidate(image_url=url, source_url="file:///secret", license_url="javascript:alert(1)")
                   for url in urls], ranking="metadata")
    path, _ = ledger.export_csv(tmp_path / "urls")
    rows = read_csv(path)
    assert path.suffix == ".csv"
    assert {row["image_url"] for row in rows} == set(urls)
    assert all(row["source_url"] == row["license_url"] == "" for row in rows)


@pytest.mark.parametrize("text", ['=HYPERLINK("https://evil.test")', "+1+1", "-1+1", "@SUM(1,2)",
                                  "  =1+1", "\t=1+1", "\r=1+1", "\ufeff=1+1"])
def test_csv_formula_injection_is_neutralized(ledger, tmp_path, text):
    ledger.record([Candidate(title=text, creator=text, license=text, provider=text)], ranking=text,
                  topic=SimpleNamespace(query=text, visual_subjects=(text,)))
    path, _ = ledger.export_csv(tmp_path / "safe.csv")
    row, = read_csv(path)
    for key in ("title", "creator", "license", "provider", "ranking_mode", "query"):
        assert row[key] == "'" + text
    assert row["image_url"].startswith("https://")


def test_unicode_commas_quotes_newlines_and_credit_metadata_round_trip(ledger, tmp_path):
    title = 'Moon, "forest"\n見えないものの庭'
    ledger.record([Candidate(title=title, creator="Zoë 森", license="CC BY 4.0")], ranking="learned_context")
    path, _ = ledger.export_csv(tmp_path / "credits.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    row, = read_csv(path)
    assert row["title"] == title
    assert row["creator"] == "Zoë 森"
    assert row["provider"] == "deviantart"
    assert row["source_url"] == Candidate().source_url
    assert row["license"] == "CC BY 4.0"
    assert row["license_url"] == Candidate().license_url


def test_no_matches_is_header_only_and_export_is_repeatable(ledger, tmp_path):
    ledger.record([Candidate(score=.3)], ranking="metadata")
    path, count = ledger.export_csv(tmp_path / "empty.csv")
    assert count == 0
    assert read_csv(path) == []
    assert ledger.count(.2) == 1
    ledger.export_csv(path, .2)
    first = path.read_bytes()
    ledger.export_csv(path, .2)
    assert path.read_bytes() == first


def test_failed_replace_preserves_destination_and_recorded_links(ledger, tmp_path, monkeypatch):
    ledger.record([Candidate()], ranking="metadata")
    path = tmp_path / "existing.csv"
    path.write_text("KEEP", encoding="utf-8")

    def denied(*_args):
        raise PermissionError("locked")

    monkeypatch.setattr("semantic_image_stream.session_image_export.os.replace", denied)
    with pytest.raises(PermissionError):
        ledger.export_csv(path)
    assert path.read_text() == "KEEP"
    assert ledger.count() == 1
    assert not list(tmp_path.glob(".sis-csv-*"))


def test_missing_directory_is_reported_without_losing_records(ledger, tmp_path):
    ledger.record([Candidate()], ranking="metadata")
    with pytest.raises(OSError):
        ledger.export_csv(tmp_path / "missing" / "file.csv")
    assert ledger.count() == 1


def test_more_links_than_slideshow_history_are_exportable(ledger, tmp_path):
    ledger.record((Candidate(image_url=f"https://x.test/{index:04d}.jpg") for index in range(1000)),
                  ranking="context_clip")
    assert ledger.count(.7) == 1000
    path, count = ledger.export_csv(tmp_path / "all.csv")
    assert count == len(read_csv(path)) == 1000


def test_close_is_idempotent_and_sessions_are_isolated():
    first, second = SessionImageLog(), SessionImageLog()
    try:
        first.record([Candidate()], ranking="metadata")
        assert second.count() == 0
        assert first.session_id != second.session_id
        assert "UTC" in first.suggested_filename
        assert first.suggested_filename.endswith(".csv")
        first.close()
        first.close()
        with pytest.raises(sqlite3.ProgrammingError):
            first.count()
    finally:
        first.close()
        second.close()


def test_default_cutoffs_respect_raw_clip_scale():
    assert default_export_threshold("clip") == .35
    for mode in ("metadata", "context_clip", "learned_context"):
        assert default_export_threshold(mode) == .7

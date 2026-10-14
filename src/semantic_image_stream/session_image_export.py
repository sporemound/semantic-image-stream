"""Private, window-session image ledger and spreadsheet-safe CSV export.

Only image metadata from accepted pipeline batches is recorded. Audio, transcript,
full narrative context, manual guidance, and cache paths are never stored here.
"""
from __future__ import annotations

from collections.abc import Iterable
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

if TYPE_CHECKING:
    from .models import ImageCandidate, Topic

DEFAULT_HIGH_RELEVANCE = 0.70
CSV_FIELDS = (
    "session_id", "first_seen_utc", "last_seen_utc", "relevance_score",
    "export_min_score", "ranking_mode", "image_url", "source_url", "title",
    "provider", "creator", "license", "license_url", "query", "visual_subjects",
)


def default_export_threshold(ranking: str) -> float:
    """Raw CLIP uses a narrower cosine scale than the normalized rankers."""
    return 0.35 if ranking == "clip" else DEFAULT_HIGH_RELEVANCE


def _threshold(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Export minimum score must be a finite number from 0 to 1.") from None
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Export minimum score must be a finite number from 0 to 1.")
    return result


def _web_url(value: str) -> str:
    """Keep only credential-free HTTP(S) links; never interpret a formula as a URL."""
    if not isinstance(value, str):
        return ""
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c == "\\" for c in value)):
            return ""
        parsed.port  # Reject malformed ports as well as malformed hostnames.
    except ValueError:
        return ""
    return value


def _csv_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    # CSV quoting alone does not stop Excel/Calc from evaluating cell formulas.
    value = value.replace("\x00", "")
    stripped = value.lstrip(" \t\r\n\v\f\ufeff")
    if stripped.startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


class SessionImageLog:
    """One row per direct image URL; best score and its metadata are kept together.

    SQLite's empty filename creates a private temporary database, deleted on
    close. Its page cache is bounded, unlike an ever-growing in-memory history.
    Calls belong to the UI thread; no model work or network requests are made.
    """

    def __init__(self) -> None:
        self.session_id = uuid4().hex
        self.started_utc = datetime.now(timezone.utc)
        self._db = sqlite3.connect("")
        try:
            self._db.execute("PRAGMA cache_size = -1024")
            self._db.execute("PRAGMA temp_store = FILE")
            self._db.execute("""CREATE TABLE images (
                image_url TEXT PRIMARY KEY, score REAL NOT NULL,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, metadata TEXT NOT NULL
            )""")
            self._db.execute("CREATE INDEX image_scores ON images(score DESC, image_url)")
        except Exception:
            self._db.close()
            raise
        self._closed = False

    @property
    def suggested_filename(self) -> str:
        stamp = self.started_utc.strftime("%Y%m%d-%H%M%S-UTC")
        return f"session-images-high-relevance-{stamp}-{self.session_id[:8]}.csv"

    def record(self, candidates: Iterable[ImageCandidate], *, ranking: str,
               topic: Topic | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self._db:
            for candidate in candidates:
                image_url = _web_url(candidate.image_url)
                if not image_url:
                    continue
                try:
                    score = _threshold(candidate.score)
                except ValueError:
                    continue
                data = {name: str(getattr(candidate, name, "") or "")
                        for name in ("title", "provider", "creator", "license")}
                data.update(
                    source_url=_web_url(candidate.source_url),
                    license_url=_web_url(candidate.license_url),
                    ranking_mode=str(ranking),
                    query=str(topic.query) if topic is not None else "",
                    visual_subjects=json.dumps(list(topic.visual_subjects), ensure_ascii=False)
                    if topic is not None else "[]",
                )
                self._db.execute("""INSERT INTO images VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_url) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        metadata = CASE WHEN excluded.score > images.score
                            THEN excluded.metadata ELSE images.metadata END,
                        score = MAX(images.score, excluded.score)
                    """, (image_url, score, now, now, json.dumps(data, ensure_ascii=False)))

    def count(self, minimum_score: float = 0.0) -> int:
        return self._db.execute("SELECT COUNT(*) FROM images WHERE score >= ?",
                                (_threshold(minimum_score),)).fetchone()[0]

    def export_csv(self, destination: str | Path,
                   minimum_score: float = DEFAULT_HIGH_RELEVANCE) -> tuple[Path, int]:
        """Stream a filtered snapshot to an atomic UTF-8/BOM CSV, best scores first.

        A zero-match export contains the header only. The caller may disable its
        save button instead. Existing files survive any failed write/replace.
        """
        threshold = _threshold(minimum_score)
        target = Path(destination)
        if target.suffix.lower() != ".csv":
            target = target.with_name(target.name + ".csv")
        temporary = None
        count = 0
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                             prefix=".sis-csv-", suffix=".tmp", dir=target.parent,
                                             delete=False) as output:
                temporary = Path(output.name)
                writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
                writer.writeheader()
                cursor = self._db.execute("""SELECT image_url, score, first_seen, last_seen, metadata
                    FROM images WHERE score >= ? ORDER BY score DESC, image_url ASC""", (threshold,))
                try:
                    for image_url, score, first_seen, last_seen, metadata in cursor:
                        row = json.loads(metadata)
                        row.update(session_id=self.session_id, image_url=image_url,
                                   first_seen_utc=first_seen, last_seen_utc=last_seen,
                                   relevance_score=format(score, ".6f"),
                                   export_min_score=format(threshold, ".6f"))
                        writer.writerow({name: _csv_cell(row.get(name, "")) for name in CSV_FIELDS})
                        count += 1
                finally:
                    cursor.close()
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target, count

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True

    def __enter__(self) -> SessionImageLog:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

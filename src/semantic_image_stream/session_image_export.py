"""Durable, point-in-time image matches and spreadsheet-safe session exports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit


MAX_SAVED_IMAGE_MATCHES = 1_000_000
CSV_COLUMNS = (
    "session_id", "image_url", "source_url", "title", "description", "provider", "creator",
    "license", "license_url", "authenticity", "ai_generated", "mature_content", "width", "height",
    "relevance_score", "acceptance_threshold", "ranker", "discovered_at_utc", "audio_position_seconds",
    "query", "summary", "scene", "visual_subjects", "relevant_passage",
)
_TEXT_LIMITS = {
    "session_id": 32, "title": 1000, "description": 2000, "provider": 100, "creator": 1000,
    "license": 1000, "authenticity": 1000, "ranker": 100, "query": 180, "summary": 400,
    "scene": 280, "visual_subjects": 2000, "relevant_passage": 6000,
}


class ImageExportError(RuntimeError):
    """A CSV could not be written; the existing file and session remain intact."""


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_url(value, optional=False):
    if not isinstance(value, str) or len(value) > 8192:
        return False
    if not value:
        return optional
    if any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return False
    try:
        parts = urlsplit(value)
        return (parts.scheme.lower() in {"http", "https"} and bool(parts.hostname)
                and parts.username is None and parts.password is None and parts.port != 0)
    except ValueError:
        return False


def _bounded_text(value, limit):
    return value[:limit] if isinstance(value, str) else ""


def _validated_match(row, image_url=None):
    if not isinstance(row, dict):
        return None
    if not _valid_url(row.get("image_url")) or (image_url is not None and image_url != row["image_url"]):
        return None
    if not all(_valid_url(row.get(field, ""), optional=True) for field in ("source_url", "license_url")):
        return None
    for field, limit in _TEXT_LIMITS.items():
        if not isinstance(row.get(field), str) or len(row[field]) > limit:
            return None
    identity = row["session_id"]
    if len(identity) != 32 or any(char not in "0123456789abcdef" for char in identity):
        return None
    for field in ("relevance_score", "acceptance_threshold", "audio_position_seconds"):
        if not _finite_number(row.get(field)) or row[field] < 0:
            return None
    if (row["acceptance_threshold"] > 1 or row["relevance_score"] > 1
            or row["relevance_score"] < row["acceptance_threshold"]):
        return None
    for field in ("width", "height"):
        if type(row.get(field)) is not int or not 0 <= row[field] <= 1_000_000:
            return None
    for field in ("ai_generated", "mature_content"):
        if row.get(field) is not None and type(row[field]) is not bool:
            return None
    discovered = row.get("discovered_at_utc")
    if not isinstance(discovered, str) or len(discovered) > 40:
        return None
    try:
        instant = datetime.fromisoformat(discovered.replace("Z", "+00:00"))
        if instant.utcoffset() is None or instant.utcoffset().total_seconds() != 0:
            return None
    except (ValueError, OverflowError):
        return None
    return {field: row.get(field) for field in CSV_COLUMNS}


def restore_image_matches(value, session_id=None):
    """Discard damaged derived rows independently, without rejecting recorded speech."""
    if not isinstance(value, dict) or len(value) > MAX_SAVED_IMAGE_MATCHES:
        return {}
    matches = {}
    for url, row in value.items():
        match = _validated_match(row, url)
        if match is not None and (session_id is None or match["session_id"] == session_id):
            matches[url] = match
    return matches


def snapshot_image_matches(state, maximum=None):
    """Return detached validated records; caller holds the session lock."""
    records = getattr(state, "image_matches", {})
    if not isinstance(records, dict):
        return {}
    if maximum is not None and len(records) > maximum:
        raise ValueError("Recorded image matches exceed the supported context size")
    return {url: match for url, row in records.items()
            if (match := _validated_match(row, url)) is not None and match["session_id"] == state.session_id}


def record_session_images(state, candidates, topic, threshold, ranker) -> bool:
    """Retain each accepted URL's best ranking and its original topic snapshot.

    Candidates are mutable and may be rescored later. Stored rows deliberately
    contain no cache paths or references to those candidates or the live topic.
    """
    if not _finite_number(threshold) or not 0 <= threshold <= 1 or topic is None:
        return False
    changed = False
    with state.lock:
        if not isinstance(getattr(state, "image_matches", None), dict):
            state.image_matches = {}
        position = state.segments[-1].end if state.segments else 0.0
        if not _finite_number(position) or position < 0:
            position = 0.0
        transcript = getattr(topic, "transcript", ())
        passage = "\n".join(text for text in transcript[-3:] if isinstance(text, str))[-6000:]
        subjects = getattr(topic, "visual_subjects", ())
        subject_text = "; ".join(subject for subject in subjects if isinstance(subject, str))[:2000]
        context = {
            "session_id": state.session_id,
            "acceptance_threshold": float(threshold),
            "ranker": _bounded_text(ranker if isinstance(ranker, str) else getattr(ranker, "name", ""), 100),
            "discovered_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "audio_position_seconds": float(position),
            "query": _bounded_text(getattr(topic, "query", ""), 180),
            "summary": _bounded_text(getattr(topic, "summary", ""), 400),
            "scene": _bounded_text(getattr(topic, "scene", ""), 280),
            "visual_subjects": subject_text,
            "relevant_passage": passage,
        }
        for candidate in candidates:
            score = getattr(candidate, "score", None)
            url = getattr(candidate, "image_url", None)
            if not _finite_number(score) or not threshold <= score <= 1 or not _valid_url(url):
                continue
            previous = _validated_match(state.image_matches.get(url), url)
            if previous is not None and previous["relevance_score"] >= score:
                continue
            row = dict(context, image_url=url, relevance_score=float(score))
            for field in ("source_url", "license_url"):
                value = getattr(candidate, field, "")
                row[field] = value if _valid_url(value, optional=True) else ""
            for field in ("title", "description", "provider", "creator", "license", "authenticity"):
                row[field] = _bounded_text(getattr(candidate, field, ""), _TEXT_LIMITS[field])
            for field in ("width", "height"):
                value = getattr(candidate, field, 0)
                row[field] = value if type(value) is int and 0 <= value <= 1_000_000 else 0
            for field in ("ai_generated", "mature_content"):
                value = getattr(candidate, field, None)
                row[field] = value if type(value) is bool else None
            state.image_matches[url] = row
            changed = True
    if changed:
        state.notify_changed()
    return changed


def _spreadsheet_cell(value):
    if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                  or value.startswith(("\t", "\r", "\n"))):
        return "'" + value
    return value


def count_session_images(state, minimum_score) -> int:
    """Count exportable unique links using the same validation as the CSV writer."""
    if state is None or not _finite_number(minimum_score) or not 0 <= minimum_score <= 1:
        return 0
    with state.lock:
        records = getattr(state, "image_matches", {})
        if not isinstance(records, dict):
            return 0
        # Counting is used by frequent GUI updates. Validate each row in place
        # without retaining another complete copy of a long session's matches.
        return sum(1 for url, row in records.items()
                   if (match := _validated_match(row, url)) is not None
                   and match["session_id"] == state.session_id and match["relevance_score"] >= minimum_score)


def export_session_images(state, destination, minimum_score) -> int:
    """Atomically write session matches at or above the selected relevance score."""
    if state is None:
        raise ImageExportError("Start or load a session before exporting image links.")
    if not _finite_number(minimum_score) or not 0 <= minimum_score <= 1:
        raise ImageExportError("Choose a relevance minimum between 0 and 1 before exporting images.")
    temporary = None
    try:
        with state.lock:
            matches = snapshot_image_matches(state)
        rows = sorted((row for row in matches.values() if row["relevance_score"] >= minimum_score),
                      key=lambda row: (-row["relevance_score"], row["image_url"]))
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=destination.parent,
                                         prefix=".images-", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            writer = csv.DictWriter(target, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _spreadsheet_cell(value) for field, value in row.items()})
            target.flush()
            os.fsync(target.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.02 * (attempt + 1))
        return len(rows)
    except (OSError, ValueError, TypeError, RecursionError, MemoryError):
        raise ImageExportError("Image links could not be exported. Close the CSV if it is open in another app "
                               "and choose a writable location. The previous file and session were left unchanged.") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

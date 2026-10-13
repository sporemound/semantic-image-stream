"""Provenance-aware calendar and geological period filtering.

Dates are astronomical years internally: 1 BCE is 0, 2 BCE is -1.
User-facing calendar input has no year zero. Explicit BP/ka/Ma/Ga ages use
1950 CE as their fixed reference, as in NOAA's paleoclimate search:
https://www.ncei.noaa.gov/access/paleo-search/

Selected numerical geological boundaries come from ICS v2026/06, not a
generated inference: https://stratigraphy.org/ICSchart/ChronostratChart2026-06.pdf
ICS notes that numerical ages are subject to revision and do not define most
formal boundaries. They are approximate age ranges for this image filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import re

BP_REFERENCE = 1950.0
MIN_YEAR = BP_REFERENCE - 14_000_000_000
MAX_YEAR = 100_000.0
ICS_VERSION = "2026/06"
ICS_SOURCE = "https://stratigraphy.org/ICSchart/ChronostratChart2026-06.pdf"
PRESENT_YEAR = float(datetime.now().year)


def _year(value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or not MIN_YEAR <= value <= MAX_YEAR):
        raise ValueError("Enter a supported finite calendar year or an age up to 14 Ga.")
    return float(value)


@dataclass(frozen=True)
class TemporalRange:
    start: float
    end: float

    def __post_init__(self):
        _year(self.start)
        _year(self.end)
        if self.start > self.end:
            raise ValueError("The start must be older than or equal to the end.")

    def overlaps(self, other):
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class TemporalMatch:
    matched: bool
    reason: str
    source: str = ""
    range: TemporalRange | None = None


# name, older Ma boundary, younger Ma boundary, formal rank. Selected units,
# not the chart artwork. Numerical bounds are labeled approximate in evidence.
_ICS_UNITS = (
    ("Hadean", 4567, 4031, "eon"), ("Archean", 4031, 2500, "eon"),
    ("Proterozoic", 2500, 538.8, "eon"), ("Phanerozoic", 538.8, 0, "eon"),
    ("Precambrian", 4567, 538.8, "informal"),
    ("Eoarchean", 4031, 3600, "era"), ("Paleoarchean", 3600, 3200, "era"),
    ("Mesoarchean", 3200, 2800, "era"), ("Neoarchean", 2800, 2500, "era"),
    ("Paleoproterozoic", 2500, 1600, "era"), ("Mesoproterozoic", 1600, 1000, "era"),
    ("Neoproterozoic", 1000, 538.8, "era"),
    ("Paleozoic", 538.8, 251.902, "era"), ("Mesozoic", 251.902, 66, "era"),
    ("Cenozoic", 66, 0, "era"),
    ("Tonian", 1000, 720, "period"), ("Cryogenian", 720, 635, "period"),
    ("Ediacaran", 635, 538.8, "period"),
    ("Cambrian", 538.8, 486.85, "period"), ("Ordovician", 486.85, 443.1, "period"),
    ("Silurian", 443.1, 419.62, "period"), ("Devonian", 419.62, 358.86, "period"),
    ("Carboniferous", 358.86, 298.9, "period"), ("Permian", 298.9, 251.902, "period"),
    ("Triassic", 251.902, 201.4, "period"), ("Jurassic", 201.4, 143.1, "period"),
    ("Cretaceous", 143.1, 66, "period"), ("Paleogene", 66, 23.04, "period"),
    ("Neogene", 23.04, 2.58, "period"), ("Quaternary", 2.58, 0, "period"),
    ("Paleocene", 66, 56, "epoch"), ("Eocene", 56, 33.9, "epoch"),
    ("Oligocene", 33.9, 23.04, "epoch"), ("Miocene", 23.04, 5.333, "epoch"),
    ("Pliocene", 5.333, 2.58, "epoch"), ("Pleistocene", 2.58, .0117, "epoch"),
    ("Holocene", .0117, 0, "epoch"),
)
GEOLOGICAL_PERIODS = {name: TemporalRange(BP_REFERENCE - older * 1e6,
                                       PRESENT_YEAR if younger == 0 else BP_REFERENCE - younger * 1e6)
                      for name, older, younger, _ in _ICS_UNITS}
# ICS specifies the Holocene/Greenlandian boundary in years before AD2000:
# https://stratigraphy.org/news/125 . Explicit BP input still references1950.
GEOLOGICAL_PERIODS["Holocene"] = TemporalRange(2000 - 11700, PRESENT_YEAR)
GEOLOGICAL_PERIODS["Pleistocene"] = TemporalRange(BP_REFERENCE - 2.58e6, 2000 - 11700)
_PERIOD_NAMES = {name.casefold(): name for name in GEOLOGICAL_PERIODS}
_NUMBER = r"(?:\d{1,12}(?:\.\d{1,17})?|\.\d{1,17})(?:[eE][+-]?\d{1,3})?"
_AGE_UNIT = (r"(?:ga|gyr|billion\s+years?|ma|myr|million\s+years?|ka|kyr|thousand\s+years?|"
             r"bp|ybp|years?\s+(?:bp|before\s+present)|b2k)")
_ERA = r"(?:BCE|BC|CE|AD)"


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Enter a year, BCE/CE year, or an age with BP, ka, Ma or Ga units.")
    result = " ".join(value.replace("−", "-").split())
    result = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", lambda match: match[0].replace(",", ""), result)
    if result.startswith("[") and result.endswith("]"):
        result = result[1:-1].strip()
    result = re.sub(r"\bB\.\s*C\.\s*E\.?(?=\s|$)", "BCE", result, flags=re.I)
    result = re.sub(r"\bB\.\s*C\.?(?=\s|$)", "BC", result, flags=re.I)
    result = re.sub(r"\bA\.\s*D\.?(?=\s|$)", "AD", result, flags=re.I)
    result = re.sub(r"\bC\.\s*E\.?(?=\s|$)", "CE", result, flags=re.I)
    return re.sub(r"^(?:circa\s*|c\.?\s*a\.?\s*|c\.\s*|~\s*)", "", result, flags=re.I)


def parse_year(text) -> float:
    """Parse strict field input. Negative plain integers mean historical BCE."""
    value = _text(text)
    age = re.fullmatch(rf"({_NUMBER})\s*({_AGE_UNIT})(?:\s*(?:bp|ago|before present))?", value, re.I)
    if age:
        number, unit = Decimal(age[1]), age[2].casefold()
        factor = (1e9 if unit.startswith(("ga", "gyr", "billion")) else
                  1e6 if unit.startswith(("ma", "myr", "million")) else
                  1e3 if unit.startswith(("ka", "kyr", "thousand")) else 1)
        reference = 2000.0 if unit == "b2k" else BP_REFERENCE
        return _year(float(Decimal(str(reference)) - number * Decimal(str(factor))))
    calendar = re.fullmatch(rf"(?:(CE|AD)\s*)?([+-]?{_NUMBER})(?:\s*({_ERA}))?", value, re.I)
    if not calendar:
        raise ValueError("Use a whole calendar year, BCE/CE, or an age with BP, ka, Ma or Ga units.")
    number = Decimal(calendar[2])
    era = (calendar[3] or calendar[1] or "").upper()
    if abs(number) < 1 or (calendar[1] and calendar[3]) or (number < 0 and era):
        raise ValueError("Calendar years have no year zero; use 1 BCE or 1 CE.")
    if number != number.to_integral_value() and not era:
        raise ValueError("A fractional calendar year needs an explicit BCE or CE label.")
    return _year(float(1 - abs(number) if number < 0 or era in {"BC", "BCE"} else number))


def _iso_year(value):
    # Creation-field input only: a strict date, never digits within an ID.
    if re.fullmatch(r"\d{4}:\d{2}:\d{2}(?: \d{2}:\d{2}:\d{2})?", value):
        value = value[:10].replace(":", "-") + value[10:]
    if not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?)?", value):
        return None
    if len(value) == 7:
        value += "-01"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return float(parsed.year)


def parse_period(text) -> TemporalRange:
    value = _text(text)
    name = re.sub(r"\s+(?:period|epoch|era|eon)$", "", value, flags=re.I).casefold()
    if name in _PERIOD_NAMES:
        return GEOLOGICAL_PERIODS[_PERIOD_NAMES[name]]
    uncertainty = re.fullmatch(rf"({_NUMBER})\s*(?:±|\+/-)\s*({_NUMBER})\s*"
                               rf"({_AGE_UNIT}(?:\s*(?:bp|ago|before present))?)", value, re.I)
    if uncertainty:
        center, error, unit = float(uncertainty[1]), float(uncertainty[2]), uncertainty[3]
        if error > center:
            raise ValueError("An age uncertainty cannot extend past its reference year.")
        years = (parse_year(f"{center + error:.9f} {unit}"), parse_year(f"{center - error:.9f} {unit}"))
        return TemporalRange(min(years), max(years))
    century = re.fullmatch(rf"(\d{{1,3}})(?:st|nd|rd|th)\s+century(?:\s*({_ERA}))?", value, re.I)
    if century:
        number = int(century[1])
        if number < 1:
            raise ValueError("Centuries begin with the first century.")
        if (century[2] or "CE").upper() in {"BCE", "BC"}:
            return TemporalRange(1 - number * 100, -(number - 1) * 100)
        return TemporalRange((number - 1) * 100 + 1, number * 100)
    iso = _iso_year(value)
    if iso is not None:
        return TemporalRange(iso, iso)
    if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2}(?:[T ].*)?)?", value):
        raise ValueError("Enter a valid source calendar date.")
    parts = re.split(r"\s+(?:to|through)\s+|\s*[–—]\s*", value, maxsplit=1, flags=re.I)
    if len(parts) == 1:
        hyphen = re.fullmatch(r"(-?\d[^-]*?)-(-?\d[^-]*)", value)
        if hyphen:
            parts = [hyphen[1].strip(), hyphen[2].strip()]
    if len(parts) == 2:
        left, right = parts
        # A trailing shared unit applies to both bounds: 66–56 Ma, 500–200 BCE.
        for source, destination in ((right, left), (left, right)):
            unit = re.search(rf"\s*({_AGE_UNIT}(?:\s*(?:bp|ago|before present))?|{_ERA})$", source, re.I)
            if unit and re.fullmatch(rf"[+-]?{_NUMBER}", destination):
                if destination == left:
                    left += " " + unit[1]
                else:
                    right += " " + unit[1]
        years = (parse_year(left), parse_year(right))
        return TemporalRange(min(years), max(years))
    year = parse_year(value)
    return TemporalRange(year, year)


def format_year(year, scale="calendar") -> str:
    year = _year(year)
    if scale not in {"calendar", "geological"}:
        raise ValueError("Choose calendar or geological years.")
    if scale == "geological" and year <= BP_REFERENCE:
        age = Decimal(str(BP_REFERENCE)) - Decimal(str(year))
        factor, unit = ((1e9, "Ga") if age >= 1e9 else (1e6, "Ma") if age >= 1e6 else
                        (1e3, "ka") if age >= 1e3 else (1, "BP"))
        number = format(age / Decimal(str(factor)), "f")
        if "." in number:
            number = number.rstrip("0").rstrip(".")
        return f"{number} {unit}" + (" BP" if unit != "BP" else "")
    if not year.is_integer():
        return format_year(year, "geological") if year <= BP_REFERENCE else f"{year:.17g} CE"
    integer = int(year)
    return f"{1 - integer} BCE" if integer <= 0 else f"{integer} CE"


def _evidence(period, target, source, precision):
    return {"target": target, "start": float(period.start), "end": float(period.end),
            "source": str(source)[:512], "precision": str(precision)[:160]}


def creation_evidence(text, source) -> list[dict]:
    """Parse a provider's explicit original creation/publication date field."""
    try:
        # A creation date cannot be guessed from a geological unit name.
        if re.sub(r"\s+(?:period|epoch|era|eon)$", "", _text(text), flags=re.I).casefold() in _PERIOD_NAMES:
            return []
        period = parse_period(text)
    except (ValueError, TypeError):
        return []
    approximate = bool(re.match(r"\s*(?:\[|circa|c\.?\s*a\.?|c\.|~)", text, re.I))
    return [_evidence(period, "created", source, "approximate source creation date; year/range" if approximate else
                      "source creation date; year/range")]


def validated_evidence(rows) -> list[dict]:
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows[:32]:
        if (not isinstance(row, dict) or set(row) != {"target", "start", "end", "source", "precision"}
                or not isinstance(row.get("target"), str) or row["target"] not in {"depicted", "created"}
                or not isinstance(row["source"], str) or not row["source"].strip() or len(row["source"]) > 512
                or not isinstance(row["precision"], str) or len(row["precision"]) > 160):
            continue
        try:
            period = TemporalRange(row["start"], row["end"])
        except (ValueError, TypeError, OverflowError):
            continue
        result.append(_evidence(period, row["target"], " ".join(row["source"].split()), row["precision"]))
    return result


def _depicted_clauses(text):
    """Keep creation-date fragments in their original scope across punctuation."""
    creation_scope = False
    creation_words = re.compile(r"\b(?:photographed|captured|uploaded|digitized|published|copyright|"
                                r"created|printed|drawn|born|died|lifespan)\b", re.I)
    subject_words = re.compile(r"\b(?:fossil|artifact|specimen|object|event|battle|subject|strata|"
                               r"sediment|rock|formation|dinosaur|diagram|depicts?|depicted)\b", re.I)
    for clause in re.split(r"[\n;,]|(?<=[.!?])\s+", text):
        if creation_scope:
            named_period = any(re.search(rf"\b{re.escape(name)}\b", clause, re.I)
                               for name in GEOLOGICAL_PERIODS)
            explicit_subject = subject_words.search(clause) and (named_period or re.search(
                r"\b(?:dated|age|from|formed|lived|depicts?|depicted|in)\b", clause, re.I))
            if not explicit_subject:
                continue
            creation_scope = False
        creation = creation_words.search(clause)
        if creation:
            creation_scope = True
            clause = clause[:creation.start()]
        if clause.strip():
            yield clause


def _affirmative_date_text(text):
    # Keep positive corrections without turning the rejected date/period into
    # evidence. This withholds unknown inequality bounds instead of inventing
    # a range from "older than" or "before" a named unit.
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\bnot\s+(?:only|just|merely)\b", "", text, flags=re.I)
    positive = []
    for piece in re.split(r"\b(?:but|however|rather|instead)\b", text, flags=re.I):
        rejected = re.search(
            r"\b(?:no|not|never|without|neither|nor|isn't|aren't|wasn't|weren't|"
            r"didn't|doesn't|don't|non[-\s]|older than|younger than|earlier than|later than|"
            r"more than|less than|at least|at most|before(?!\s+present)|after)\b|[<>≤≥]",
            piece, re.I)
        positive.append(piece[:rejected.start()] if rejected else piece)
    return " . ".join(positive)


def extract_depicted_evidence(title, description, categories="", source="source subject metadata") -> list[dict]:
    """Use only explicit depicted-subject periods, never incidental year digits."""
    texts = [text[:8000] for text in (title, description, categories) if isinstance(text, str)]
    result = []
    geological_context = re.compile(r"\b(?:fossil|fossils|dinosaur|dinosaurs|geolog\w*|stratigraph\w*|"
                                    r"palaeo\w*|paleo\w*|strata|sediment\w*|formation|outcrop|"
                                    r"period|epoch|era|eon|ammonite|trilobite|stromatolite|"
                                    r"prehistoric|tectonic|reconstruction|diagram|chart|plot|map|"
                                    r"biota|artifact|archaeological|rock|rocks)\b", re.I)
    scientific_context = any(geological_context.search(text) for text in texts)
    for text in texts:
        text = re.sub(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", lambda match: match[0].replace(",", ""), text)
        for clause in _depicted_clauses(text):
            clause = _affirmative_date_text(clause)
            # Creator lifespan, publication/capture/upload dates belong to a
            # different target and cannot silently date the depicted subject.
            if re.search(r"\b(?:born|died|lifespan|ISBN|ISSN|specimen\s*(?:id|number|no)|"
                         r"catalog(?:ue)?\s*(?:id|number|no)|accession)\b", clause, re.I):
                continue
            if scientific_context and not re.search(r"\bJurassic\s+(?:Park|World)\b", clause, re.I):
                for name, period in GEOLOGICAL_PERIODS.items():
                    if re.search(rf"\b{re.escape(name)}\b", clause, re.I):
                        result.append(_evidence(period, "depicted", f"{source}; ICS {ICS_VERSION}",
                                                "named geological unit; approximate numerical boundaries"))
            patterns = (
                rf"(?<![\w.]){_NUMBER}(?:\s*(?:[–—-]|±|\+/-)\s*{_NUMBER})?\s*{_AGE_UNIT}(?:\s*(?:bp|ago|before present))?\b",
                rf"(?<![\w-])\d{{1,8}}(?:\s*[–—-]\s*\d{{1,8}})?\s*{_ERA}\b",
                rf"\b\d{{1,3}}(?:st|nd|rd|th)\s+century(?:\s*{_ERA})?\b",
                r"\b(?:in the year|depicts? (?:the year|events? (?:of|in))|dated? to)\s+"
                r"(\d{4}-\d{2}-\d{2}|\d{3,4})\b",
            )
            for pattern in patterns:
                for match in re.finditer(pattern, clause, re.I):
                    # An unmatched uncertainty must not be interpreted as a
                    # separate much-younger age merely because it has units.
                    if re.search(r"(?:±|\+/-)\s*$", clause[:match.start()]):
                        continue
                    if match.lastindex and re.match(
                            rf"\s*(?:{_AGE_UNIT}|{_ERA}|±|\+/-|[–—:-])", clause[match.end():], re.I):
                        # A labeled scientific age/BCE date cannot also be
                        # accepted as a bare CE calendar year.
                        continue
                    value = match[1] if match.lastindex else match[0]
                    try:
                        period = parse_period(value)
                    except ValueError:
                        continue
                    result.append(_evidence(period, "depicted", source, "explicit subject date/age; year/range"))
    unique = []
    for row in validated_evidence(result):
        if row not in unique:
            unique.append(row)
    return unique[:32]


def filter_match(settings, candidate) -> TemporalMatch:
    if not getattr(settings, "time_filter_enabled", False):
        return TemporalMatch(True, "The time filter is off.")
    selected = TemporalRange(settings.time_filter_start, settings.time_filter_end)
    target = settings.time_filter_target
    rows = validated_evidence(getattr(candidate, "temporal_evidence", []))
    if target == "depicted":
        rows += extract_depicted_evidence(candidate.title, candidate.description, source=f"{candidate.provider} subject metadata")
    relevant = [row for row in rows if row["target"] == target]
    for row in relevant:
        period = TemporalRange(row["start"], row["end"])
        if period.overlaps(selected):
            return TemporalMatch(True, "The recorded period overlaps the selected range.", row["source"], period)
    if relevant:
        return TemporalMatch(False, "The recorded period is outside the selected range.", relevant[0]["source"])
    included = getattr(settings, "time_filter_include_undated", False)
    return TemporalMatch(included, "No supported period is recorded; undated images are included." if included else
                         "No supported period is recorded for the selected date type.")


def filter_identity(settings) -> str:
    enabled = getattr(settings, "time_filter_enabled", False)
    data = {"enabled": enabled}
    if enabled:
        data.update(target=settings.time_filter_target, start=float(settings.time_filter_start),
                    end=float(settings.time_filter_end), undated=settings.time_filter_include_undated,
                    geological_table=ICS_VERSION)
    # Display scale alone does not change what matches or restart playback.
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def search_hints(settings) -> tuple[str, ...]:
    if not getattr(settings, "time_filter_enabled", False):
        return ()
    selected = TemporalRange(settings.time_filter_start, settings.time_filter_end)
    if settings.time_filter_target == "depicted" and selected.start < -100_000:
        containing = [(name, period) for name, period in GEOLOGICAL_PERIODS.items()
                      if period.start <= selected.start and selected.end <= period.end]
        if containing:
            return (min(containing, key=lambda item: item[1].end - item[1].start)[0],)
        periods = [(name, period) for name, _, _, rank in _ICS_UNITS if rank == "period"
                   for period in (GEOLOGICAL_PERIODS[name],) if period.overlaps(selected)]
        periods.sort(key=lambda item: min(item[1].end, selected.end) - max(item[1].start, selected.start), reverse=True)
        if periods:
            return tuple(name for name, _ in periods[:2])
        return (f"{format_year(selected.start, 'geological')} to {format_year(selected.end, 'geological')}",)
    start, end = format_year(selected.start), format_year(selected.end)
    return (start if start == end else f"{start} to {end}",)

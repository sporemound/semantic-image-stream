from dataclasses import replace
import json

import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.temporal_filter import (
    BP_REFERENCE, GEOLOGICAL_PERIODS, ICS_VERSION, MIN_YEAR, TemporalRange,
    creation_evidence, extract_depicted_evidence, filter_identity, filter_match,
    format_year, parse_period, parse_year, search_hints, validated_evidence,
)


def image(title="Original diagram", description="", evidence=None):
    return ImageCandidate("https://example.test/image.jpg", "https://example.test/source", title, "test",
                          description=description, temporal_evidence=evidence or [])


def selected(start="1800", end="1900", target="depicted", **kwargs):
    return Settings(time_filter_enabled=True, time_filter_start=parse_year(start),
                    time_filter_end=parse_year(end), time_filter_target=target, **kwargs)


@pytest.mark.parametrize("text,expected", [
    ("2026", 2026), ("2026 CE", 2026), ("AD 45", 45), ("1 BCE", 0), ("2 BC", -1),
    ("500 B.C.E.", -499), ("-500", -499), ("−1", 0), ("c. 1890", 1890), ("500.5 CE", 500.5),
    ("0 BP", 1950), ("500 BP", 1450), ("10 ka", -8050), ("10 ka BP", -8050),
    ("66 Ma", 1950 - 66e6), ("2.5 Ga", 1950 - 2.5e9),
    ("13.8 billion years ago", 1950 - 13.8e9), ("11.7 ka BP", 1950 - 11700),
    ("11700 b2k", 2000 - 11700), ("14 Ga", MIN_YEAR),
])
def test_year_parser_calendar_and_declared_bp_reference(text, expected):
    assert parse_year(text) == expected


@pytest.mark.parametrize("text", ["0", "0 BCE", "0 CE", "-500 BCE", "500.5", "nan", "inf",
                                  "15 Ga", "-5 Ma", "Specimen 1998", "ISBN 9781998", "", None])
def test_invalid_or_identifier_year_is_rejected(text):
    with pytest.raises(ValueError):
        parse_year(text)


@pytest.mark.parametrize("text,start,end", [
    ("500–200 BCE", -499, -199), ("500 BCE to 200 CE", -499, 200),
    ("1850-1900", 1850, 1900), ("66–56 Ma", 1950 - 66e6, 1950 - 56e6),
    ("2.5 to 2 Ga", 1950 - 2.5e9, 1950 - 2e9),
    ("19th century", 1801, 1900), ("5th century BCE", -499, -400),
    ("2020-02-29T12:30:00Z", 2020, 2020), ("2020:02:29 12:30:00", 2020, 2020),
    ("66 ± 0.6 Ma", 1950 - 66.6e6, 1950 - 65.4e6),
    ("66+/-2 Ma", 1950 - 68e6, 1950 - 64e6),
])
def test_ranges_shared_units_centuries_and_source_creation_dates(text, start, end):
    assert parse_period(text) == TemporalRange(start, end)


def test_latest_ics_table_specific_revised_boundaries_and_provenance():
    assert ICS_VERSION == "2026/06"
    assert parse_period("Cretaceous period") == TemporalRange(1950 - 143.1e6, 1950 - 66e6)
    assert parse_period("Jurassic") == TemporalRange(1950 - 201.4e6, 1950 - 143.1e6)
    assert parse_period("Permian") == TemporalRange(1950 - 298.9e6, 1950 - 251.902e6)
    assert parse_period("Hadean") == TemporalRange(1950 - 4567e6, 1950 - 4031e6)
    assert parse_period("Holocene").start == -9700
    assert parse_period("Pleistocene").end == -9700
    assert parse_period("Holocene").end >= 2026
    assert all(period.start <= period.end for period in GEOLOGICAL_PERIODS.values())
    evidence = extract_depicted_evidence("Cretaceous fossil diagram", "", source="Museum source")
    assert evidence[0]["target"] == "depicted"
    assert "ICS 2026/06" in evidence[0]["source"]
    assert "approximate" in evidence[0]["precision"]


def test_bce_formatter_never_displays_calendar_year_zero():
    assert format_year(0) == "1 BCE"
    assert format_year(-499) == "500 BCE"
    assert format_year(1) == "1 CE"
    for year in (MIN_YEAR, 1950 - 2.5e9, 1950 - 66e6, -8050, 1450, 1950, 2026):
        assert parse_year(format_year(year, "geological")) == pytest.approx(year)
    for year in (-12345.6789, 1850.125, 2026.125):
        assert parse_year(format_year(year, "calendar")) == pytest.approx(year)
    assert format_year(parse_year("4.6 Ga"), "geological") == "4.6 Ga BP"
    assert format_year(parse_year("66.6 Ma"), "geological") == "66.6 Ma BP"


def test_depicted_and_created_are_never_substituted_for_each_other():
    candidate = image("Cretaceous fossil diagram", evidence=creation_evidence("2020-01-15", "Original creation field"))
    assert filter_match(selected("150 Ma", "60 Ma"), candidate).matched
    assert not filter_match(selected("150 Ma", "60 Ma", "created"), candidate).matched
    assert filter_match(selected("2010", "2026", "created"), candidate).matched
    assert not filter_match(selected("2010", "2026"), candidate).matched


@pytest.mark.parametrize("title,description", [
    ("Specimen 1998", "Collection number 1850; ISBN 9781234567890."),
    ("Object 2001", "Photographed in 2020 CE. Uploaded in 2024 CE."),
    ("Portrait", "Artist born 1800 CE and died 1870 CE."),
    ("Book cover", "Published 1850 CE; copyright 1900 CE."),
    ("Jurassic Park poster", "Dinosaur movie poster."),
    ("Jurassic World poster", "Dinosaur movie poster."),
])
def test_incidental_numbers_creator_lifetimes_and_publication_are_not_subject_dates(title, description):
    assert extract_depicted_evidence(title, description) == []
    assert not filter_match(selected("1800", "2026"), image(title, description)).matched


def test_explicit_subject_age_uncertainty_and_scientific_category():
    rows = extract_depicted_evidence("Ammonite fossil", "Specimen dated to 66 ± 0.6 Ma.", "Cretaceous")
    assert any(row["start"] == 1950 - 66.6e6 and row["end"] == 1950 - 65.4e6 for row in rows)
    assert any("ICS" in row["source"] for row in rows)
    assert all(row["target"] == "depicted" for row in rows)


@pytest.mark.parametrize("title", ["Jurassic climate diagram", "Cretaceous biota map", "Holocene climate plot"])
def test_scientific_named_period_titles_and_present_endpoint(title):
    evidence = extract_depicted_evidence(title, "")
    assert evidence and all(row["target"] == "depicted" for row in evidence)
    if "Holocene" in title:
        assert filter_match(selected("2022", "2026"), image(title)).matched


def test_mixed_subject_age_and_photo_creation_do_not_erase_or_conflate_each_other():
    for description in ("Fossil dated 66 Ma, photographed 2020 CE.", "Fossil dated 66 Ma photographed 2020 CE."):
        evidence = extract_depicted_evidence("Fossil diagram", description)
        assert len(evidence) == 1
        assert evidence[0]["start"] == BP_REFERENCE - 66e6
        assert all(row["target"] == "depicted" for row in evidence)
    explicit = extract_depicted_evidence("Fossil", "Age 66±2 Ma.")
    assert len(explicit) == 1
    assert explicit[0]["start"] == BP_REFERENCE - 68e6
    assert explicit[0]["end"] == BP_REFERENCE - 64e6
    assert extract_depicted_evidence("Fossil", "Unusable ID66 ±2 Ma.") == []


@pytest.mark.parametrize("value", ["[1930]", "c.a.1930", "circa1930"])
def test_creation_field_approximate_dates_remain_provenance_labeled(value):
    evidence = creation_evidence(value, "Archive original publication date")
    assert evidence[0]["start"] == 1930
    assert "approximate" in evidence[0]["precision"]


@pytest.mark.parametrize("value", ["2020-99", "1900-02-29", "0000-01-01"])
def test_malformed_creation_dates_do_not_become_broad_year_ranges(value):
    assert creation_evidence(value, "Original date") == []


@pytest.mark.parametrize("description,expected", [
    ("Fossil dated to 150 Ma.", BP_REFERENCE - 150e6),
    ("Fossil dated to 1000 BP.", 950),
    ("Artifact dated to 500 BCE.", -499),
    ("Artifact dated to 2020-01-15.", 2020),
])
def test_labeled_subject_dates_are_not_also_reinterpreted_as_bare_ce_years(description, expected):
    rows = extract_depicted_evidence("Original subject", description)
    assert len(rows) == 1
    assert rows[0]["start"] == rows[0]["end"] == expected
    if "150 Ma" in description:
        assert not filter_match(selected("100", "200"), image("Fossil", description)).matched
    if "500 BCE" in description:
        assert not filter_match(selected("400", "600"), image("Artifact", description)).matched


@pytest.mark.parametrize("description", [
    "Photographed on June 1, 2020 CE.",
    "Published on January 1, 2020 CE; 2021 CE.",
    "Artist born on January 1, 1800 CE.",
])
def test_creation_and_lifespan_dates_keep_scope_across_comma_fragments(description):
    assert extract_depicted_evidence("Original image", description) == []


def test_explicit_subject_age_can_restart_after_creation_date_fragment():
    rows = extract_depicted_evidence("Fossil diagram", "Photographed in 2020 CE, fossil dated to 150 Ma.")
    assert len(rows) == 1
    assert rows[0]["start"] == BP_REFERENCE - 150e6


def test_negated_geological_period_is_excluded_and_positive_correction_is_kept():
    description = "Fossil not from the Jurassic period but from the Cretaceous period."
    rows = extract_depicted_evidence("Fossil diagram", description)
    assert len(rows) == 1
    assert rows[0]["start"] == GEOLOGICAL_PERIODS["Cretaceous"].start
    candidate = image("Fossil diagram", description)
    assert not filter_match(selected("180 Ma", "150 Ma"), candidate).matched
    assert filter_match(selected("100 Ma", "90 Ma"), candidate).matched


@pytest.mark.parametrize("description,expected", [
    ("Fossil not dated to 150 Ma but dated to 66 Ma.", BP_REFERENCE - 66e6),
    ("Artifact wasn't dated to 500 BCE but dated to 200 BCE.", -199),
    ("Jurassic fossil is not from the Cretaceous period.", GEOLOGICAL_PERIODS["Jurassic"].start),
])
def test_negated_date_age_or_comparison_does_not_hide_positive_subject(description, expected):
    rows = extract_depicted_evidence("Fossil artifact diagram", description)
    assert len(rows) == 1 and rows[0]["start"] == expected


@pytest.mark.parametrize("description", [
    "Fossil was not dated to 150 Ma.", "Fossil was older than 150 Ma.",
    "Fossil formed before the Jurassic period.", "Fossil age is >150 Ma.",
])
def test_negated_or_inequality_age_does_not_invent_a_filter_range(description):
    assert extract_depicted_evidence("Fossil diagram", description) == []


def test_explicit_event_and_century_dates_are_accepted_with_metadata_source():
    assert extract_depicted_evidence("Battle in 500 BCE", "")[0]["start"] == -499
    assert extract_depicted_evidence("19th century landscape", "")[0]["start"] == 1801
    assert extract_depicted_evidence("Battle diagram", "Depicts events of 1863.")[0]["start"] == 1863


def test_unknown_exclusion_and_opt_in_inclusion_are_explicit():
    settings = selected()
    unknown = image()
    result = filter_match(settings, unknown)
    assert result.matched is False and "No supported period" in result.reason
    included = filter_match(replace(settings, time_filter_include_undated=True), unknown)
    assert included.matched and "undated images are included" in included.reason
    dated = image(evidence=creation_evidence("2020", "Original date"))
    assert not filter_match(replace(selected(target="created"), time_filter_include_undated=True), dated).matched
    assert filter_match(Settings(), unknown).matched


def test_filter_overlap_is_inclusive_and_reports_date_source():
    candidate = image("Depicted event 1900 CE")
    match = filter_match(selected(), candidate)
    assert match.matched and match.range == TemporalRange(1900, 1900)
    assert "subject metadata" in match.source
    assert not filter_match(selected("1800", "1899"), candidate).matched


def test_identity_ignores_disabled_bounds_and_display_scale():
    settings = Settings()
    assert filter_identity(settings) == filter_identity(replace(settings, time_filter_start=-1e9,
                                                                time_filter_target="created", time_filter_include_undated=True))
    enabled = replace(settings, time_filter_enabled=True)
    assert filter_identity(enabled) == filter_identity(replace(enabled, time_filter_scale="geological"))
    assert filter_identity(enabled) != filter_identity(replace(enabled, time_filter_end=2000))
    assert filter_identity(enabled) != filter_identity(replace(enabled, time_filter_target="created"))
    assert filter_identity(enabled) == filter_identity(replace(enabled, exclude_ai_generated=False))


def test_bounded_search_hints_for_calendar_geological_and_astrophysical_periods():
    assert search_hints(Settings()) == ()
    assert search_hints(selected()) == ("1800 CE to 1900 CE",)
    assert search_hints(selected("180 Ma", "150 Ma")) == ("Jurassic",)
    hints = search_hints(selected("13.8 Ga", "13 Ga"))
    assert len(hints) == 1 and "Ga BP" in hints[0]
    assert all(len(hint.split()) <= 8 for hint in hints)


@pytest.mark.parametrize("row", [
    {"target": "created", "start": True, "end": 1900, "source": "test", "precision": "year"},
    {"target": [], "start": 1850, "end": 1900, "source": "test", "precision": "year"},
    {"target": "created", "start": 1900, "end": 1850, "source": "test", "precision": "year"},
    {"target": "created", "start": float("nan"), "end": 1900, "source": "test", "precision": "year"},
    {"target": "depicted", "start": 1850, "end": 1900, "source": "", "precision": "year"},
])
def test_malformed_cached_evidence_is_ignored_without_trusting_dates(row):
    assert validated_evidence([row]) == []
    data = image().to_dict() | {"temporal_evidence": [row], "ai_generated": "false"}
    restored = ImageCandidate.from_dict(data)
    assert restored.temporal_evidence == []
    assert restored.ai_generated is None


def test_new_candidate_fields_cache_roundtrip_and_old_cache_defaults():
    candidate = image(evidence=creation_evidence("1850–1900", "Original document date"))
    candidate.ai_generated = False
    assert ImageCandidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate
    old = candidate.to_dict()
    del old["ai_generated"]
    del old["temporal_evidence"]
    restored = ImageCandidate.from_dict(old)
    assert restored.ai_generated is None and restored.temporal_evidence == []


def test_settings_roundtrip_defaults_and_invalid_temporal_controls(tmp_path):
    original = selected("500 BCE", "2026", time_filter_scale="geological", exclude_ai_generated=False)
    path = tmp_path / "settings.toml"
    original.save(path)
    assert Settings.load(path).time_filter_start == -499
    assert Settings.load(path).exclude_ai_generated is False
    assert Settings().time_filter_enabled is False and Settings().exclude_ai_generated is True
    for field, value in (("time_filter_target", "upload"), ("time_filter_scale", "deep_time"),
                         ("time_filter_enabled", 1), ("time_filter_include_undated", "true"),
                         ("time_filter_start", float("nan")), ("time_filter_end", True),
                         ("exclude_ai_generated", "false")):
        with pytest.raises(ValueError):
            replace(original, **{field: value}).validate()
    with pytest.raises(ValueError):
        replace(original, time_filter_start=2027, time_filter_end=2026).validate()

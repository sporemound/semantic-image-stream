"""Offline regressions for retaining grounded visual context between observations."""
from types import SimpleNamespace

import pytest

from semantic_image_stream import narrative_context as narrative
from semantic_image_stream.models import TranscriptSegment


CAMP = "Tents stood on the flat Arctic landscape in sunlight."


def response(text, *, scene=None, subjects=(), queries=(), mode=None):
    quote = text.split("\n")[-1]
    data = {"summary": text, "current_scene": scene or quote,
            "visual_subjects": list(subjects), "search_queries": list(queries),
            "kind": "general", "evidence": [quote]}
    if mode:
        data["visual_update"] = {"mode": mode, "evidence": quote}
    return data


def analyzer_with(monkeypatch, responses):
    settings = SimpleNamespace(context_model="qwen2.5:7b", book_context="",
                               context_cpu_threads=6, context_timeout=180)
    analyzer = narrative.LocalNarrativeAnalyzer(settings)
    monkeypatch.setattr(analyzer, "_ensure_service", lambda *args: None)
    pending = iter(responses)
    calls = []

    def generate(text, *args, **kwargs):
        calls.append(text)
        return next(pending)

    monkeypatch.setattr(analyzer, "_generate", generate)
    return analyzer, calls


def camp_response():
    return response(CAMP, subjects=("tents", "flat Arctic landscape"),
                    queries=("tents flat Arctic landscape", "flat Arctic landscape"))


def recording(*texts):
    return [TranscriptSegment(index * 6, index * 6 + 5, text) for index, text in enumerate(texts)]


@pytest.mark.parametrize("text", [
    "They waited and wondered what might happen.",
    "They discussed the journey and asked questions.",
    "Yes.",
    "He felt uneasy; his face was stinging.",
])
def test_nonvisual_and_sensory_passages_retain_visuals_and_all_new_context(monkeypatch, text):
    next_state = response(text)
    if "stinging" in text:
        next_state = response(text, subjects=("stinging face",), queries=("stinging face",))
    analyzer, calls = analyzer_with(monkeypatch, [camp_response(), next_state])
    before = analyzer.analyze(recording(CAMP))
    after = analyzer.analyze(recording(CAMP, text))
    assert after.visual_update == "retain"
    assert (after.query, after.scene, after.queries, after.visual_subjects, after.concepts, after.kind) == (
        before.query, before.scene, before.queries, before.visual_subjects, before.concepts, before.kind)
    assert after.summary.startswith(before.summary)
    assert text in after.summary
    assert after.context == CAMP + " " + text
    assert after.transcript == (CAMP, text)
    assert after.context_used_segments == 2
    assert calls == [CAMP, text]
    assert analyzer._state["evidence"] == [text]


def test_source_confirmed_move_overrides_model_retain(monkeypatch):
    text = "They entered the cave. Rocks covered the cave floor."
    new = response(text, subjects=("cave", "rocks"), queries=("cave rocks",), mode="retain")
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "replace"
    assert topic.query == "cave rocks"


def test_new_blank_physical_scene_clears_instead_of_reusing_camp(monkeypatch):
    text = "They walked into an empty room. Nothing was visible."
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), response(text)])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "clear"
    assert topic.query == ""
    assert topic.context_used_segments == 2


def test_new_physical_scene_cannot_keep_historical_only_query(monkeypatch):
    text = "They entered the cave."
    new = response(text, scene=CAMP, subjects=("tents",), queries=("tents",))
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "clear"
    assert not topic.queries


def test_attack_with_sensory_reaction_is_a_new_observed_subject(monkeypatch):
    text = "A wolf attacked him beside the tents. He felt pain."
    new = response(text, subjects=("wolf", "tents"), queries=("wolf tents",), mode="retain")
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "replace"
    assert "wolf" in topic.query


def test_new_grounded_object_without_a_transition_verb_still_updates(monkeypatch):
    text = "A bear stood beside the tents."
    new = response(text, subjects=("bear", "tents"), queries=("bear tents",), mode="retain")
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "replace"
    assert "bear" in topic.query


@pytest.mark.parametrize("text", ["There were no tents.", "The tents vanished.", "The tents were no longer there.",
                                  "There wasn't a tent.", "The tents weren't there."])
def test_explicit_removed_visual_cannot_be_retained(monkeypatch, text):
    # The old positive fact remains in full history, so simple vocabulary
    # grounding alone would incorrectly approve this stale model candidate.
    new = response(text, scene=CAMP, subjects=("tents",), queries=("tents",), mode="retain")
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, text))
    assert topic.visual_update == "clear"
    assert not topic.query


def test_closed_door_does_not_keep_open_door_query(monkeypatch):
    first = "An open wooden door stood on the flat landscape."
    last = "He closed the wooden door."
    old = response(first, subjects=("open wooden door", "flat landscape"),
                   queries=("open wooden door flat landscape",))
    new = response(last, scene="Closed wooden door.", subjects=("closed wooden door",),
                   queries=("closed wooden door",), mode="retain")
    analyzer, _ = analyzer_with(monkeypatch, [old, new])
    analyzer.analyze(recording(first))
    topic = analyzer.analyze(recording(first, last))
    assert topic.visual_update == "replace"
    assert topic.query == "closed wooden door"


def test_explicit_negated_door_polarity_cannot_reuse_open_door(monkeypatch):
    first = "An open wooden door stood on the flat landscape."
    last = "The wooden door wasn't open."
    old = response(first, subjects=("open wooden door",), queries=("open wooden door",))
    stale = response(last, scene=first, subjects=("open wooden door",), queries=("open wooden door",))
    analyzer, _ = analyzer_with(monkeypatch, [old, stale])
    analyzer.analyze(recording(first))
    topic = analyzer.analyze(recording(first, last))
    assert topic.visual_update == "clear"
    assert not topic.query


def test_retained_checkpoint_uses_legacy_state_shape_and_revalidates(monkeypatch):
    filler = "They waited and wondered what might happen."
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), response(filler, mode="retain")])
    analyzer.analyze(recording(CAMP))
    topic = analyzer.analyze(recording(CAMP, filler))
    checkpoint = analyzer.export_checkpoint()
    assert set(checkpoint["accepted_state"]) == set(narrative.RESPONSE_SCHEMA["required"])
    restored, calls = analyzer_with(monkeypatch, [])
    assert restored.restore_checkpoint(checkpoint, recording(CAMP, filler))
    resumed = restored.analyze(recording(CAMP, filler))
    assert not calls
    assert resumed.query == topic.query
    assert resumed.summary == topic.summary


def test_invalid_control_evidence_does_not_advance_cursor(monkeypatch):
    filler = "They waited."
    new = response(filler, mode="retain")
    new["visual_update"]["evidence"] = "Imagined reference facts."
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new, new])
    analyzer.analyze(recording(CAMP))
    with pytest.raises(narrative.EvidenceGroundingError):
        analyzer.analyze(recording(CAMP, filler))
    assert analyzer.context_used_segments == 1
    assert analyzer.scene == CAMP


def test_initial_empty_passage_has_no_visual_fallback(monkeypatch):
    text = "They waited and wondered what might happen."
    analyzer, _ = analyzer_with(monkeypatch, [response(text)])
    topic = analyzer.analyze(recording(text))
    assert topic.visual_update == "clear"
    assert not topic.query


def test_awakening_clears_a_dream_without_an_observed_waking_depiction(monkeypatch):
    dream = "He dreamed of a wooden door on a flat landscape."
    awake = "He woke from the dream."
    old = response(dream, subjects=("wooden door", "flat landscape"), queries=("wooden door flat landscape",))
    analyzer, _ = analyzer_with(monkeypatch, [old, response(awake, mode="retain")])
    analyzer.analyze(recording(dream))
    topic = analyzer.analyze(recording(dream, awake))
    assert topic.visual_update == "clear"
    assert not topic.query


def test_new_scene_then_filler_in_one_analyze_call_is_replace(monkeypatch):
    move = "They entered the cave. Rocks covered the cave floor."
    filler = "They waited and wondered what might happen."
    new = response(move, subjects=("cave", "rocks"), queries=("cave rocks",))
    analyzer, _ = analyzer_with(monkeypatch, [camp_response(), new, response(filler)])
    analyzer.analyze(recording(CAMP))
    # Ensure separate chronological batches in the same call.
    analyzer._batch_chars = len(move) + 1
    topic = analyzer.analyze(recording(CAMP, move, filler))
    assert topic.visual_update == "replace"
    assert topic.query == "cave rocks"
    assert topic.context_used_segments == 3


def test_repeated_retained_summaries_keep_setting_and_latest_information(monkeypatch):
    first = CAMP + " " + "They waited quietly and considered the journey. " * 6
    one = "They felt uneasy and waited quietly."
    two = "They felt afraid and waited together."
    old = camp_response()
    old["summary"] = first
    analyzer, _ = analyzer_with(monkeypatch, [old, response(one), response(two)])
    analyzer.analyze(recording(first))
    first_retained = analyzer.analyze(recording(first, one))
    second_retained = analyzer.analyze(recording(first, one, two))
    assert first_retained.visual_update == second_retained.visual_update == "retain"
    assert first_retained.summary.startswith(CAMP)
    assert second_retained.summary.startswith(CAMP)
    assert one in first_retained.summary
    assert two in second_retained.summary
    assert first_retained.summary != second_retained.summary
    assert len(second_retained.summary) <= 400

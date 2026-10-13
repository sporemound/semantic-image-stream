"""Mock-local tests for persistent manual image guidance and factual grounding."""
from copy import deepcopy

import pytest

from semantic_image_stream import narrative_context as narrative
from semantic_image_stream.models import TranscriptSegment
from test_narrative_context import install_local_model, settings, stream_state


BRIEF = "polar tundra, rocky coastline, wooden cabin, psychological horror, darkness"
PASSAGE = "He waited beside the wooden cabin."


def state(text=PASSAGE, scene="Wooden cabin on polar tundra.", subjects=("wooden cabin", "polar tundra"),
          queries=("wooden cabin polar tundra",)):
    return {"summary": text, "current_scene": scene, "visual_subjects": list(subjects),
            "search_queries": list(queries), "kind": "general", "evidence": [text]}


def recording(*texts):
    return [TranscriptSegment(index * 6, index * 6 + 5, text) for index, text in enumerate(texts)]


def test_manual_brief_is_persistent_separate_data_and_evidence_is_recorded_only(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    filler = "He considered what might happen."
    prompts, requests = install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(
        state() if call == 1 else state(filler)))
    before = analyzer.analyze(recording(PASSAGE))
    after = analyzer.analyze(recording(PASSAGE, filler))
    assert before.query == after.query == "wooden cabin polar tundra"
    assert after.visual_update == "retain"
    for prompt in prompts:
        assert prompt["book_background"] == BRIEF
        assert "polar tundra" in prompt["persistent_visual_brief"]["setting_anchors"]
        assert "wooden cabin" in prompt["persistent_visual_brief"]["subjects"]
        assert "darkness" in prompt["persistent_visual_brief"]["themes"]
        assert prompt["book_guidance_strength"] == .65
        assert all(quote in prompt["latest_passage"] for quote in prompt["evidence_options"])
        assert all("polar tundra" not in quote for quote in prompt["evidence_options"])
    assert all(request.url.host == "127.0.0.1" for request in requests)
    assert "an explicit latest observed physical setting" in narrative.SYSTEM_PROMPT
    assert "A book title does not establish a plot" in narrative.SYSTEM_PROMPT


def test_explicit_new_recorded_environment_is_not_replaced_by_manual_setting(monkeypatch):
    text = "They entered the desert. Sand dunes surrounded the camp."
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state(
        text, scene="Desert sand dunes.", subjects=("desert", "sand dunes"), queries=("desert sand dunes",))))
    topic = analyzer.analyze(recording(text))
    assert topic.query == "desert sand dunes"
    assert topic.scene == "Desert sand dunes."
    assert "polar" not in topic.query
    assert topic.context == text


def test_abstract_theme_cannot_authorize_physical_lighting_fact(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state(
        scene="Dark wooden cabin.", subjects=("dark wooden cabin",), queries=("dark wooden cabin",))))
    with pytest.raises(narrative.VisualGroundingError):
        analyzer.analyze(recording(PASSAGE))
    assert analyzer.context_used_segments == 0


def test_same_theme_word_is_permitted_when_independently_recorded(monkeypatch):
    text = "Darkness filled the wooden cabin."
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state(
        text, scene="Dark wooden cabin.", subjects=("dark wooden cabin",), queries=("dark wooden cabin",))))
    assert analyzer.analyze(recording(text)).query == "dark wooden cabin"


def test_title_does_not_supply_a_physical_object_or_plot(monkeypatch):
    text = "He waited outside."
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context="Title: The Hidden Temple"))
    install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state(
        text, scene="Hidden temple.", subjects=("hidden temple",), queries=("hidden temple",))))
    with pytest.raises(narrative.VisualGroundingError):
        analyzer.analyze(recording(text))


def test_zero_strength_does_not_add_manual_qualifiers(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF, book_guidance_strength=0.0))
    prompts, _ = install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state(
        scene="Wooden cabin.", subjects=("wooden cabin",), queries=("wooden cabin",))))
    topic = analyzer.analyze(recording(PASSAGE))
    assert topic.query == "wooden cabin"
    assert prompts[0]["persistent_visual_brief"] == {"setting_anchors": [], "subjects": [], "themes": []}
    assert prompts[0]["book_guidance_strength"] == 0


def test_ignored_zero_strength_qualifiers_cannot_pass_source_guard(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF, book_guidance_strength=0.0))
    install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state()))
    topic = analyzer.analyze(recording(PASSAGE))
    # The model's independently grounded cabin subject can repair its query;
    # the unsupported manual setting is never retained as a visual fact.
    assert topic.query == "wooden cabin"
    assert "polar" not in topic.scene
    assert topic.context_used_segments == 1


def test_compatible_pre_brief_checkpoint_migrates_without_rereading(monkeypatch):
    original = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    install_local_model(monkeypatch, original, lambda prompt, call: stream_state(state()))
    original.analyze(recording(PASSAGE))
    checkpoint = original.export_checkpoint()
    current_key = checkpoint["context_key"]
    checkpoint["context_key"] = original._pre_brief_context_key()
    assert current_key != checkpoint["context_key"]
    resumed = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    prompts, _ = install_local_model(monkeypatch, resumed)
    assert resumed.restore_checkpoint(checkpoint, recording(PASSAGE))
    assert resumed.analyze(recording(PASSAGE)).query == "wooden cabin polar tundra"
    assert not prompts
    assert resumed.export_checkpoint()["context_key"] == current_key
    assert resumed.context_used_segments == 1


def test_legacy_theme_based_scene_is_not_migrated_as_a_fact(monkeypatch):
    original = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    install_local_model(monkeypatch, original, lambda prompt, call: stream_state(state()))
    original.analyze(recording(PASSAGE))
    checkpoint = original.export_checkpoint()
    checkpoint["context_key"] = original._pre_brief_context_key()
    checkpoint["accepted_state"].update(current_scene="Dark wooden cabin.", visual_subjects=["dark wooden cabin"],
                                         search_queries=["dark wooden cabin"])
    checkpoint["processed_digest"] = original._checkpoint_digest(
        (PASSAGE,), 1, 0, checkpoint["accepted_state"], checkpoint["summary_method"])
    resumed = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    assert not resumed.restore_checkpoint(checkpoint, recording(PASSAGE))
    assert resumed.context_used_segments == 0


def test_weight_only_update_preserves_cursor_checkpoint_and_private_settings(monkeypatch):
    configuration = settings(book_context=BRIEF, book_guidance_strength=.65)
    analyzer = narrative.LocalNarrativeAnalyzer(configuration)
    prompts, _ = install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state()))
    analyzer.analyze(recording(PASSAGE))
    checkpoint = analyzer.export_checkpoint()
    analyzer.update_book_guidance(BRIEF, .9)
    updated = analyzer.export_checkpoint()
    assert updated["cursor"] == 1
    assert updated["accepted_state"] == checkpoint["accepted_state"]
    assert updated["context_key"] == checkpoint["context_key"]
    assert len(prompts) == 1
    assert analyzer.settings.book_guidance_strength == .9
    assert configuration.book_guidance_strength == .65
    resumed = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF, book_guidance_strength=.9))
    assert resumed.restore_checkpoint(updated, recording(PASSAGE))


def test_live_brief_change_repairs_only_existing_grounded_state_without_replay(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    prompts, _ = install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(state()))
    analyzer.analyze(recording(PASSAGE))
    analyzer.update_book_guidance("tropical desert", .8)
    checkpoint = analyzer.export_checkpoint()
    assert checkpoint["cursor"] == 1
    assert checkpoint["accepted_state"]["search_queries"] == ["wooden cabin"]
    assert "polar" not in checkpoint["accepted_state"]["current_scene"]
    assert len(prompts) == 1
    assert analyzer._history_snapshot == (PASSAGE,)


def test_irreparable_old_manual_scene_preserves_raw_cursor_and_waits_for_next_passage(monkeypatch):
    first = "He paused and wondered what might happen."
    next_passage = "Sand dunes stretched across the desert."
    analyzer = narrative.LocalNarrativeAnalyzer(settings(book_context=BRIEF))
    prompts, _ = install_local_model(monkeypatch, analyzer, lambda prompt, call: stream_state(
        state(first, "Polar tundra.", ("polar tundra",), ("polar tundra",)) if call == 1 else
        state(next_passage, "Desert sand dunes.", ("desert", "sand dunes"), ("desert sand dunes",))))
    analyzer.analyze(recording(first))
    original_state = deepcopy(analyzer._state)
    with pytest.raises(narrative.NarrativeContextError, match="next recorded passage"):
        analyzer.update_book_guidance("tropical desert", .8)
    assert analyzer._state == original_state
    assert analyzer.context_used_segments == 1 and analyzer._history_snapshot == (first,)
    with pytest.raises(narrative.NarrativeContextError, match="next grounded story passage"):
        analyzer.export_checkpoint()
    with pytest.raises(narrative.NarrativeContextError, match="waiting for the next recorded passage"):
        analyzer.analyze(recording(first))
    assert analyzer.context_used_segments == 1 and analyzer._state == original_state
    assert len(prompts) == 1
    recovered = analyzer.analyze(recording(first, next_passage))
    assert recovered.query == "desert sand dunes"
    assert recovered.context_used_segments == 2
    assert analyzer.export_checkpoint()["cursor"] == 2
    assert prompts[-1]["book_background"] == "tropical desert"
    assert prompts[-1]["new_transcript"] == next_passage

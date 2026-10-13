import asyncio
from dataclasses import replace
import math
import threading

import pytest

from semantic_image_stream.book_guidance import (guidance_for, guidance_identity, guidance_seed,
                                                guided_queries, refresh_guidance, snapshot_guidance, visual_brief)
from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import Topic, TranscriptSegment
from semantic_image_stream.session_store import restore_session, snapshot_session

# Original synthetic regression brief; the listener's complete manual text is
# tested separately in a private workspace fixture and never packaged.
BRIEF = ("Fiction anthology, Arctic landscapes, remote communities, tundra, rocky coastlines, sea ice, "
         "psychological suspense, mythical journeys, fictional disappearances, personal memory, "
         "fragmented recollections, emotional distance, uncertain motives, tension, recollection, "
         "hope, doubt, identity, impressions, reflections, dreams, storytelling, imaginative themes")


def scene(query="door", text="A door stood alone.", **extra):
    return Topic(query, text, scene=text, visual_subjects=(query,), transcript=(text,), **extra)


def test_supplied_anthology_guidance_keeps_concrete_places_separate_from_themes():
    brief = visual_brief(BRIEF)
    assert brief["setting_anchors"][0] == "Arctic landscapes"
    assert {"tundra", "rocky coastlines", "sea ice", "remote communities"} <= set(brief["setting_anchors"])
    assert not brief["subjects"]
    assert not {"darkness", "psychological suspense", "colonial trauma"} & set(brief["setting_anchors"])
    assert "Fiction anthology" in brief["themes"]


def test_search_subject_gets_arctic_environment_and_fallback_is_still_from_supplied_brief():
    queries = guided_queries(scene(), Settings(book_context=BRIEF))
    assert queries == ["door Arctic landscapes", "door tundra", "Arctic landscapes"]
    assert all(len(query.split()) <= 8 for query in queries)
    assert not any("horror anthology" in query or "trauma" in query for query in queries)


@pytest.mark.parametrize("text", ["Book: Imaginary Arctic Castle", "Mood: psychological suspense",
                                  "Setting: supernatural horror", "Focus: psychological suspense",
                                  "Avoid: modern city streets", "No urban photographs"])
def test_title_theme_or_negative_preference_cannot_invent_a_physical_setting(text):
    brief = visual_brief(text)
    assert not brief["setting_anchors"] and not brief["subjects"]
    assert guided_queries(scene(), Settings(book_context=text)) == ["door"]
    assert guidance_seed(Settings(book_context=text)) is None


def test_current_explicit_recorded_location_wins_without_place_in_object_query():
    topic = scene("brick building", "A brick building in London.")
    settings = Settings(book_context=BRIEF)
    guide, _ = guidance_for(topic, settings)
    assert "Arctic" not in guide and "tundra" not in guide
    assert "psychological suspense" in guide
    assert guided_queries(topic, settings) == ["brick building"]


def test_contradictory_recorded_environment_wins_but_manual_place_is_not_observed_evidence():
    settings = Settings(book_context=BRIEF)
    assert guided_queries(scene("camp", "A camp in the tropical jungle."), settings) == ["camp"]
    # A model carrying manual vocabulary forward cannot make it an independent
    # new observation that overrides a later supplied brief.
    topic = replace(scene(), scene="A door in the Arctic tundra.")
    assert guided_queries(topic, Settings(book_context="Setting: Paris"))[0] == "door Paris"


def test_actor_possessive_is_not_a_new_geographic_setting():
    topic = scene("door", "He stopped at Jomi's door.")
    assert guided_queries(topic, Settings(book_context=BRIEF))[0] == "door Arctic landscapes"


@pytest.mark.parametrize("text", ["A cabin stands beside a coastal forest.", "A cabin stands in a rural village."])
def test_compatible_environment_descriptors_do_not_erase_the_supplied_region(text):
    assert guided_queries(scene("cabin", text), Settings(book_context=BRIEF))[0] == "cabin Arctic landscapes"


def test_provider_word_bound_preserves_the_current_concrete_subject():
    topic = scene("solitary unattached mysterious weathered wooden cabin Arctic door")
    topic = replace(topic, visual_subjects=("wooden door",))
    query = guided_queries(topic, Settings(book_context="rocky coastlines"))[0]
    assert len(query.split()) <= 8 and "door" in query and "wooden" in query and "coastlines" in query


def test_immutable_delivered_guidance_does_not_follow_later_settings_or_clear():
    settings = Settings(book_context=BRIEF)
    delivered = snapshot_guidance(scene(), settings)
    changed = replace(settings, book_context="tropical jungle", book_guidance_strength=1)
    assert guidance_for(delivered, changed) == (BRIEF, .65)
    assert guided_queries(delivered, changed)[0] == "door Arctic landscapes"
    cleared = snapshot_guidance(delivered, replace(changed, book_context=""))
    assert guidance_for(cleared, settings) == ("", 0)
    assert snapshot_guidance(scene(), Settings()).image_guidance is None


def test_no_guidance_or_zero_preserves_existing_discovery_and_cache_identity():
    topic = scene()
    for settings in (Settings(), Settings(book_context=BRIEF, book_guidance_strength=0)):
        assert guided_queries(topic, settings) == ["door"]
        assert guidance_identity(topic, settings) == ""
        assert guidance_seed(settings) is None


def test_negative_image_directives_cannot_reward_excluded_objects_in_positive_ranking():
    settings = Settings(book_context="Arctic landscapes; Avoid hospitals; Exclude book covers; do not show modern cities")
    text, strength = guidance_for(scene(), settings)
    assert text == "Arctic landscapes" and strength == .65
    assert guided_queries(scene(), settings)[0] == "door Arctic landscapes"


def test_only_an_explicit_manual_setting_can_seed_images_without_fabricated_narration():
    topic = guidance_seed(Settings(book_context=BRIEF))
    assert topic.query == "Arctic landscapes"
    assert not topic.context and not topic.transcript and not topic.summary and not topic.scene
    assert topic.context_used_segments == 0
    assert topic.image_guidance == BRIEF
    assert topic.guidance_only


def test_manual_seed_changes_or_clears_and_keeps_all_recorded_words_for_ranking():
    settings = Settings(book_context="Arctic landscapes")
    seed = guidance_seed(settings, ("A wooden cabin stood alone.",))
    changed = refresh_guidance(seed, replace(settings, book_context="desert landscape"))
    assert changed.query == "desert landscape" and changed.guidance_only
    assert changed.transcript == seed.transcript and not changed.scene and not changed.summary
    assert refresh_guidance(seed, replace(settings, book_context="")) is None
    assert refresh_guidance(seed, replace(settings, book_guidance_strength=0)) is None


def test_expired_manual_only_qualifier_is_removed_without_rewriting_recorded_history():
    original = snapshot_guidance(scene("polar cabin", "A wooden cabin stood alone."), Settings(book_context="polar landscape"))
    original = replace(original, scene="A polar cabin stood alone.", summary="A polar cabin stands.")
    changed = refresh_guidance(original, Settings(book_context="desert landscape"))
    assert "polar" not in changed.scene and "polar" not in changed.query and "polar" not in changed.summary
    assert changed.transcript == original.transcript
    assert "polar" in original.scene


def test_session_roundtrip_preserves_guidance_and_accepts_old_missing_fields():
    state = SessionState(book_context=BRIEF, book_guidance_strength=.9, guidance_revision=3)
    state.segments = [TranscriptSegment(0, 4, "A door stood alone.")]
    state.topic = snapshot_guidance(scene(), Settings(book_context=BRIEF, book_guidance_strength=.9))
    payload = snapshot_session(state)
    restored = restore_session(payload)
    assert restored.book_guidance_strength == .9 and restored.guidance_revision == 3
    assert restored.topic.image_guidance == BRIEF and restored.topic.guidance_strength == .9
    del payload["book_guidance_strength"], payload["guidance_revision"]
    del payload["topic"]["image_guidance"], payload["topic"]["guidance_strength"]
    legacy = restore_session(payload)
    assert legacy.book_guidance_strength == .65 and legacy.guidance_revision == 0
    assert legacy.topic.image_guidance is None


@pytest.mark.parametrize("value", [True, "strong", -1, 1.1, math.nan, math.inf])
def test_strength_rejects_invalid_or_nonfinite_values(value):
    with pytest.raises(ValueError, match="book_guidance_strength"):
        Settings(book_guidance_strength=value).validate()


def test_live_guidance_updates_image_scope_but_keeps_recording_votes_cutoff_and_filters(tmp_path):
    events = []
    settings = Settings(cache_dir=str(tmp_path), relevance_threshold=.1, time_filter_enabled=True,
                        time_filter_start=1800, time_filter_end=2026, exclude_ai_generated=False)
    state = SessionState(segments=[TranscriptSegment(0, 4, "A door stood alone.")], topic=scene())
    pipeline = Pipeline(settings, lambda *event: events.append(event), state)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline._displayed_urls.add("https://images.example/already-seen")
    pipeline._approved_images["pending"] = object()
    pipeline.update_book_guidance(replace(settings, book_context=BRIEF, book_guidance_strength=.9))
    assert state.segments[0].text == "A door stood alone."
    assert state.topic.image_guidance == BRIEF and state.topic.guidance_strength == .9
    assert not pipeline._approved_images and pipeline._displayed_urls
    assert pipeline.settings.relevance_threshold == .1 and pipeline.settings.time_filter_enabled
    assert not pipeline.settings.exclude_ai_generated
    ack = [value for kind, value in events if kind == "context"][-1]
    assert ack["phase"] == "ready" and not ack["retaining"] and ack["playback_changed"]
    assert ack["book_context"] == BRIEF and ack["guidance_revision"] == 1
    assert pipeline.topic_queue.get_nowait()[1] is state.topic


@pytest.mark.asyncio
async def test_inflight_provider_results_cannot_fill_or_cache_after_live_guidance_change(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    pipeline = Pipeline(settings, lambda *args: None)
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            entered.set()
            await release.wait()
            return []

    cache = Cache(settings)
    try:
        task = asyncio.create_task(pipeline._query_candidates("door", [Provider()], cache))
        await entered.wait()
        pipeline.update_book_guidance(replace(settings, book_context=BRIEF))
        release.set()
        assert await task == [] and not pipeline._last_query_succeeded
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_guidance_revision_rechecks_same_processed_window_without_publishing_obsolete_model_result(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), book_context="polar landscape", semantic_interval=.01)
    old = snapshot_guidance(scene("cabin", "A wooden cabin stood alone."), settings)
    old = replace(old, scene="A polar cabin stood alone.", context_used_segments=1)
    state = SessionState(segments=[TranscriptSegment(0, 4, "A wooden cabin stood alone.")], topic=old)
    events, accepted = [], threading.Event()

    def on_event(kind, value):
        events.append((kind, value))
        if kind == "context" and value.get("book_context") == "desert landscape":
            accepted.set()
        elif kind == "context" and value["phase"] == "ready" and value["segments_used"] == 1:
            pipeline.stop_event.set()

    pipeline = Pipeline(settings, on_event, state)
    pipeline._loop = asyncio.get_running_loop()
    pipeline.topic_queue = asyncio.Queue(maxsize=1)

    class Analyzer:
        def __init__(self):
            self.settings = replace(settings)
            self.calls = 0

        def update_book_guidance(self, text, strength):
            self.settings.book_context, self.settings.book_guidance_strength = text, strength

        def export_checkpoint(self):
            return {}

        def analyze(self, segments):
            self.calls += 1
            if self.calls == 1:
                pipeline.update_book_guidance(replace(settings, book_context="desert landscape"))
                assert accepted.wait(1)
                return old
            return replace(old, scene="A wooden cabin stood alone.")

    analyzer = state.analyzer = Analyzer()
    await asyncio.wait_for(pipeline._interpret(analyzer), 3)
    ready = [value for kind, value in events if kind == "context" and value["phase"] == "ready"]
    assert analyzer.calls == 2 and ready
    assert all("polar" not in value["scene"] for value in ready)
    assert pipeline._latest_topic.image_guidance == "desert landscape"
    assert len(state.segments) == 1


def test_saved_compatible_book_guidance_change_restores_processed_cursor_without_model_replay(tmp_path):
    from semantic_image_stream.engine import _interpreter_key
    from semantic_image_stream.narrative_context import LocalNarrativeAnalyzer

    old_settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama", book_context="Arctic landscapes")
    analyzer = LocalNarrativeAnalyzer(old_settings)
    raw = "A wooden cabin stood alone beside the rocky coast."
    data = {"summary": raw, "current_scene": raw, "visual_subjects": ["wooden cabin", "rocky coast"],
            "search_queries": ["wooden cabin rocky coast"], "kind": "general", "evidence": [raw]}
    checked = analyzer._validate(data, old_settings.book_context + " " + raw, raw)
    analyzer.summary_method = checked.pop("_summary_method", "local model")
    analyzer._state, analyzer._cursor, analyzer._history_snapshot = checked, 1, (raw,)
    state = SessionState(segments=[TranscriptSegment(0, 4, raw)], book_context=old_settings.book_context,
                         provider_key=_interpreter_key(old_settings), analyzer_checkpoint=analyzer.export_checkpoint(),
                         topic=snapshot_guidance(scene("wooden cabin rocky coast", raw, context_used_segments=1), old_settings))
    settings = replace(old_settings, book_context="rocky coastlines, remote communities", book_guidance_strength=.9)
    pipeline = Pipeline(settings, lambda *args: None, state)
    restored = pipeline._session_analyzer()
    assert restored._cursor == 1 and restored._history_snapshot == (raw,)
    assert restored.settings.book_context == settings.book_context
    assert state.analyzer_checkpoint is not None and len(state.segments) == 1
    assert pipeline._latest_topic.image_guidance == settings.book_context

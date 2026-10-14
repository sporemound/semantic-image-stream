"""Mature-content choices govern live discovery and cache reuse independently."""
import asyncio
from collections import Counter
from dataclasses import asdict, replace
import json

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment


def candidate(name, title="A museum artwork", **fields):
    return ImageCandidate(f"https://images.example/{name}.jpg", "https://source.example/artwork",
                          title, "wikimedia", **fields)


class Ranker:
    name = "context_clip"
    model = object()

    def rank(self, topic, candidates, recent):
        return [replace(image, score=.9) for image in candidates]


class NoDownload:
    async def download(self, image):
        pytest.fail("Validated cached pixels must not be downloaded again")


async def evaluate(pipeline, topic, images, cache, ranker=None):
    return await pipeline._evaluate_candidates(topic, images, None, None, NoDownload(), ranker or Ranker(),
        cache, {}, Counter(), {"found": 0, "downloaded": 0, "best_score": None})


def test_mature_filter_uses_explicit_marks_without_censoring_narration_or_disabling_other_filters(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    state = SessionState(segments=[TranscriptSegment(0, 5, "Fuck this shit, he said. Damn it.")])
    pipeline = Pipeline(settings, lambda *args: None, session_state=state)
    marked = candidate("marked", mature_content=True)
    unknown = candidate("unknown", title="A nude figure in classical sculpture", mature_content=None)
    assert pipeline._image_filter_match(marked) == (False, "Marked as mature content")
    assert pipeline._image_filter_match(unknown)[0]
    assert pipeline._image_filter_match(candidate("not-marked", mature_content=False))[0]
    pipeline.update_image_filters(replace(settings, allow_nsfw=True))
    assert pipeline._image_filter_match(marked)[0]
    assert pipeline._image_filter_match(candidate("generated", mature_content=True, ai_generated=True)) == (
        False, "Marked as AI-generated",
    )
    assert state.segments[0].text == "Fuck this shit, he said. Damn it."


def test_live_mature_change_restarts_discovery_but_preserves_story_models_and_runtime(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), chunk_seconds=3, audio_device=77, whisper_model="turbo")
    events = []
    topic = Topic("museum", "The full story.", summary="An expedition visits a museum.")
    state = SessionState(segments=[TranscriptSegment(0, 5, "The visitors studied a painting.")], topic=topic,
                         provider_key=("contextual", "", ""), analyzer_checkpoint={"processed_count": 1})
    pipeline = Pipeline(settings, lambda *event: events.append(event), session_state=state)
    pipeline._latest_topic = topic
    pipeline._approved_images["pending"] = candidate("pending")
    pipeline._query_pages["museum"] = 4
    pipeline._refresh_queries.add("museum")
    pipeline._fresh_query_cycles.add("museum")
    pipeline._fallback_positions["museum"] = 2
    pipeline._displayed_urls.add("https://images.example/displayed.jpg")
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, Topic("old scene", "Old context.")))
    pipeline._capture, pipeline._transcriber, pipeline._ranker = object(), object(), object()
    resources = pipeline._capture, pipeline._transcriber, pipeline._ranker
    before = asdict(pipeline.settings)
    revision = pipeline._image_filter_revision
    generation = state.playback_generation
    update = replace(settings, allow_nsfw=True, audio_mode="file", audio_file="different.wav",
                     whisper_model="tiny", providers=["archive"], ranking="clip", relevance_threshold=.8)
    pipeline.update_image_filters(update)

    assert pipeline.settings.allow_nsfw
    assert pipeline._image_filter_revision == revision + 1 and state.playback_generation == generation + 1
    assert not pipeline._approved_images and not pipeline._query_pages
    assert not pipeline._refresh_queries and not pipeline._fresh_query_cycles and not pipeline._fallback_positions
    assert pipeline.topic_queue.get_nowait()[1] is topic
    assert pipeline._displayed_urls == {"https://images.example/displayed.jpg"}
    assert pipeline.session_state is state and state.topic is topic
    assert state.segments[0].text == "The visitors studied a painting."
    assert state.provider_key == ("contextual", "", "") and state.analyzer_checkpoint == {"processed_count": 1}
    assert (pipeline._capture, pipeline._transcriber, pipeline._ranker) == resources
    after = asdict(pipeline.settings)
    assert {key: value for key, value in after.items() if key != "allow_nsfw"} == {
        key: value for key, value in before.items() if key != "allow_nsfw"
    }
    assert any(kind == "context" and value["playback_changed"] for kind, value in events)
    pipeline.update_image_filters(update)
    assert pipeline._image_filter_revision == revision + 1
    pipeline.update_image_filters(settings)
    assert not pipeline.settings.allow_nsfw and pipeline._image_filter_revision == revision + 2


@pytest.mark.asyncio
async def test_query_cache_separates_both_modes_and_ignores_legacy_queries(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    pipeline = Pipeline(settings, lambda *args: None)
    calls = []

    class Provider:
        name = "wikimedia"

        def __init__(self):
            self.settings = pipeline.settings

        async def search(self, query, limit):
            return await self.search_page(query, limit, 0)

        async def search_page(self, query, limit, page):
            calls.append((self.settings.allow_nsfw, page))
            return [candidate(f"{self.settings.allow_nsfw}-{page}", mature_content=self.settings.allow_nsfw)]

    cache = Cache(settings)
    provider = Provider()
    try:
        legacy_key = json.dumps(["museum", settings.providers, settings.results_per_provider])
        cache.put_query(legacy_key, [candidate("legacy-mature", mature_content=True)])
        restricted = await pipeline._query_candidates("museum", [provider], cache)
        assert restricted[0].image_url != "https://images.example/legacy-mature.jpg"
        for page in (0, 1, 1):
            await pipeline._query_candidates("museum", [provider], cache, page=page)
        assert calls == [(False, 0), (False, 1)]
        pipeline.update_image_filters(replace(settings, allow_nsfw=True))
        for page in (0, 1, 0):
            allowed = await pipeline._query_candidates("museum", [provider], cache, page=page)
            assert allowed[0].mature_content is True
        assert calls == [(False, 0), (False, 1), (True, 0), (True, 1)]
        pipeline.update_image_filters(settings)
        for page in (0, 1):
            restricted = await pipeline._query_candidates("museum", [provider], cache, page=page)
            assert restricted[0].mature_content is False
        assert len(calls) == 4
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_inflight_permissive_query_cannot_cache_after_live_restriction(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), allow_nsfw=True)
    pipeline = Pipeline(settings, lambda *args: None)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            selected = pipeline.settings.allow_nsfw
            calls.append(selected)
            entered.set()
            await release.wait()
            return [candidate(str(selected), mature_content=selected)]

    cache = Cache(settings)
    try:
        task = asyncio.create_task(pipeline._query_candidates("museum", [Provider()], cache))
        await entered.wait()
        pipeline.update_image_filters(replace(settings, allow_nsfw=False))
        release.set()
        assert await task == []
        assert not pipeline._last_query_succeeded
        assert cache.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 0
        result = await pipeline._query_candidates("museum", [Provider()], cache)
        assert calls == [True, False] and result[0].mature_content is False
    finally:
        cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed,cached_mark,fresh_mark", [(False, True, None), (True, True, False), (True, None, True)])
async def test_cached_mature_marks_are_merged_persisted_and_rechecked(tmp_path, allowed, cached_mark, fresh_mark):
    settings = Settings(cache_dir=str(tmp_path), allow_nsfw=allowed, min_width=32, min_height=24)
    events = []
    pipeline = Pipeline(settings, lambda *event: events.append(event))
    topic = pipeline._latest_topic = Topic("museum artwork", "A museum artwork is described.")
    path = tmp_path / "images" / "validated.png"
    path.parent.mkdir()
    Image.new("RGB", (80, 60), "green").save(path)
    cached = candidate("cached", local_path=str(path), width=80, height=60,
                       phash="a" * 16, mature_content=cached_mark)
    fresh = candidate("cached", mature_content=fresh_mark)
    cache = Cache(settings)
    try:
        cache.put_image(cached)
        published, failed = await evaluate(pipeline, topic, [fresh], cache)
        assert not failed and published is allowed
        assert cache.get_image(fresh.image_url).mature_content is True
        assert bool([value for kind, value in events if kind == "images"]) is allowed
        if allowed:
            pipeline.update_image_filters(replace(settings, allow_nsfw=False))
            events.clear()
            published, failed = await evaluate(pipeline, topic, [candidate("cached")], cache)
            assert not published and not failed
            assert not any(kind == "images" for kind, _ in events)
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_live_restriction_during_ranking_cannot_publish_mature_image(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), allow_nsfw=True, min_width=32, min_height=24)
    events = []
    pipeline = Pipeline(settings, lambda *event: events.append(event))
    pipeline._loop = asyncio.get_running_loop()
    topic = pipeline._latest_topic = Topic("museum artwork", "A museum artwork is described.")
    path = tmp_path / "images" / "validated.png"
    path.parent.mkdir()
    Image.new("RGB", (80, 60), "green").save(path)
    image = candidate("mature", local_path=str(path), width=80, height=60, phash="a" * 16, mature_content=True)

    class ChangingRanker(Ranker):
        calls = 0

        def rank(self, topic, images, recent):
            self.calls += 1
            if self.calls == 1:
                pipeline.update_image_filters(replace(settings, allow_nsfw=False))
            return super().rank(topic, images, recent)

    cache = Cache(settings)
    try:
        cache.put_image(image)
        published, failed = await evaluate(pipeline, topic, [image], cache, ChangingRanker())
        assert not published and not failed
        assert not any(kind == "images" for kind, _ in events)
        assert not pipeline.settings.allow_nsfw
    finally:
        cache.close()

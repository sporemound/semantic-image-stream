"""Time and AI filters govern cached pixels, live requests, and queue refill."""
import asyncio
from collections import Counter
from dataclasses import replace

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.temporal_filter import creation_evidence, parse_year


def image(name, title="Jurassic ammonite fossil", **values):
    return ImageCandidate(f"https://images.example/{name}.jpg", "https://museum.example/item",
                          title, "wikimedia", **values)


class Ranker:
    name = "context_clip"
    model = object()

    def rank(self, topic, candidates, recent):
        return [replace(candidate, score=.9) for candidate in candidates]


@pytest.mark.asyncio
async def test_geological_and_ai_filters_reject_before_downloading_and_preserve_cutoff(tmp_path, monkeypatch):
    events, downloads = [], []
    settings = Settings(cache_dir=str(tmp_path / "cache"), min_width=32, min_height=24, prefetch_count=1,
                        search_interval=.01, time_filter_enabled=True, time_filter_target="depicted",
                        time_filter_start=parse_year("200 Ma"), time_filter_end=parse_year("150 Ma"),
                        relevance_threshold=.40)
    pipeline = Pipeline(settings, lambda kind, value: events.append((kind, value)))
    topic = Topic("ammonite fossil", "The story describes an ammonite fossil.")
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])
    candidates = [image("ai", ai_generated=True), image("old", "Triassic ammonite fossil"),
                  image("unknown", "Ammonite specimen accession123456"), image("match")]

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return candidates

    class Downloader:
        async def download(self, candidate):
            downloads.append(candidate.image_url)
            path = tmp_path / "valid.png"
            Image.new("RGB", (80, 60), "green").save(path)
            pipeline.stop_event.set()
            return replace(candidate, local_path=str(path), width=80, height=60, phash="a" * 16)

    cache = Cache(settings)
    try:
        await asyncio.wait_for(pipeline._search([Provider()], Downloader(), Ranker(), cache), 3)
        assert downloads == [candidates[-1].image_url]
        assert pipeline.settings.relevance_threshold == .40
        assert any("selected time/AI filters" in value.get("message", "")
                   for kind, value in events if kind == "retrieval") is False
    finally:
        cache.close()


def test_live_filter_change_discards_pending_only_and_preserves_story_and_history(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), relevance_threshold=.30)
    events = []
    pipeline = Pipeline(settings, lambda *event: events.append(event))
    topic = Topic("fossil", "All the recorded story context.", scene="An ammonite fossil")
    pipeline._latest_topic = pipeline.session_state.topic = topic
    pipeline.session_state.segments.append(TranscriptSegment(0, 6, "A fossil was examined."))
    pipeline._displayed_urls.add("https://images.example/previous.jpg")
    pipeline._approved_images["pending"] = image("pending")
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    previous = pipeline.session_state.playback_generation
    update = replace(settings, time_filter_enabled=True, time_filter_start=parse_year("200 Ma"),
                     time_filter_end=parse_year("150 Ma"))
    pipeline.update_image_filters(update)
    assert not pipeline._approved_images
    assert pipeline.session_state.playback_generation == previous + 1
    assert pipeline.session_state.topic is topic and len(pipeline.session_state.segments) == 1
    assert pipeline._displayed_urls == {"https://images.example/previous.jpg"}
    assert pipeline.settings.relevance_threshold == .30
    assert pipeline.topic_queue.get_nowait()[1] is topic
    generation = pipeline.session_state.playback_generation
    pipeline.update_image_filters(replace(update, time_filter_scale="geological"))
    assert pipeline.session_state.playback_generation == generation


@pytest.mark.asyncio
async def test_query_cache_is_separate_for_time_target_and_ai_option(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    pipeline = Pipeline(settings, lambda *args: None)
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            return []

    cache = Cache(settings)
    try:
        for _ in range(2):
            await pipeline._query_candidates("fossil", [Provider()], cache)
        assert len(calls) == 1
        update = replace(settings, time_filter_enabled=True, time_filter_start=1800.0, time_filter_end=1900.0)
        pipeline.update_image_filters(update)
        for _ in range(2):
            await pipeline._query_candidates("fossil", [Provider()], cache)
        assert len(calls) == 2
        pipeline.update_image_filters(replace(update, time_filter_target="created"))
        await pipeline._query_candidates("fossil", [Provider()], cache)
        assert len(calls) == 3
        pipeline.update_image_filters(replace(update, time_filter_target="created", exclude_ai_generated=False))
        await pipeline._query_candidates("fossil", [Provider()], cache)
        assert len(calls) == 4
        assert calls == ["fossil"] * 4
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_old_inflight_request_cannot_cache_after_live_range_change(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    pipeline = Pipeline(settings, lambda *args: None)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            entered.set()
            await release.wait()
            return []

    cache = Cache(settings)
    try:
        task = asyncio.create_task(pipeline._query_candidates("fossil", [Provider()], cache))
        await entered.wait()
        pipeline.update_image_filters(replace(settings, exclude_ai_generated=False))
        release.set()
        assert await task == []
        assert not pipeline._last_query_succeeded
        pipeline.update_image_filters(settings)
        await pipeline._query_candidates("fossil", [Provider()], cache)
        assert len(calls) == 2
    finally:
        cache.close()


def test_creation_and_embedded_ai_filters_are_reapplied_to_approved_cached_images(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), time_filter_enabled=True, time_filter_target="created",
                        time_filter_start=1900.0, time_filter_end=1950.0)
    pipeline = Pipeline(settings, lambda *args: None)
    topic = Topic("fossil", "fossil")
    pipeline._update_playback_scope(topic)
    pipeline._sync_image_pool(topic)
    path = tmp_path / "raster.png"
    Image.new("RGB", (80, 60), "green").save(path)
    rows = [image("dated", local_path=str(path), temporal_evidence=creation_evidence("1920", "publication field")),
            image("late", local_path=str(path), temporal_evidence=creation_evidence("2020", "publication field")),
            image("ai", local_path=str(path), temporal_evidence=creation_evidence("1920", "publication field"), ai_generated=True)]
    pipeline._approved_images = {candidate.image_url: candidate for candidate in rows}
    pipeline._sync_image_pool(topic)
    assert list(pipeline._approved_images) == [rows[0].image_url]


@pytest.mark.asyncio
@pytest.mark.parametrize("embedded_ai", [False, True])
async def test_cached_pixels_use_fresh_date_metadata_but_keep_embedded_ai_mark(tmp_path, embedded_ai):
    settings = Settings(cache_dir=str(tmp_path), min_width=32, min_height=24,
                        time_filter_enabled=True, time_filter_target="created",
                        time_filter_start=1900., time_filter_end=1950.)
    events, ranked = [], []
    pipeline = Pipeline(settings, lambda *event: events.append(event))
    topic = pipeline._latest_topic = Topic("ammonite", "An ammonite in a museum.")
    path = tmp_path / "images" / "validated.png"
    path.parent.mkdir()
    Image.new("RGB", (80, 60), "green").save(path)
    old = image("cached", "Old title", local_path=str(path), width=80, height=60,
                phash="a" * 16, ai_generated=embedded_ai)
    fresh = image("cached", "Ammonite museum photo", temporal_evidence=creation_evidence("1920", "source date"))

    class InspectRanker(Ranker):
        def rank(self, topic, candidates, recent):
            ranked.extend(candidates)
            return super().rank(topic, candidates, recent)

    class NoDownload:
        async def download(self, candidate):
            pytest.fail("Validated cached pixels must not be downloaded again")

    cache = Cache(settings)
    try:
        cache.put_image(old)
        published, failed = await pipeline._evaluate_candidates(topic, [fresh], None, None,
            NoDownload(), InspectRanker(), cache, {}, Counter(), {"found": 0, "downloaded": 0, "best_score": None})
        assert not failed and published is not embedded_ai
        assert ranked[0].title == fresh.title and ranked[0].temporal_evidence == fresh.temporal_evidence
        assert ranked[0].ai_generated is (True if embedded_ai else None)
        assert bool([value for kind, value in events if kind == "images"]) is not embedded_ai
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_live_range_change_during_ranking_cannot_publish_old_image(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), min_width=32, min_height=24,
                        time_filter_enabled=True, time_filter_target="created",
                        time_filter_start=1900., time_filter_end=1950.)
    events = []
    pipeline = Pipeline(settings, lambda *event: events.append(event))
    pipeline._loop = asyncio.get_running_loop()
    topic = pipeline._latest_topic = Topic("ammonite", "The same museum scene.")
    pipeline.session_state.segments.append(TranscriptSegment(0, 3, "The same museum scene."))
    path = tmp_path / "images" / "dated.png"
    path.parent.mkdir()
    Image.new("RGB", (80, 60), "green").save(path)
    candidate = image("dated", local_path=str(path), width=80, height=60, phash="a" * 16,
                      temporal_evidence=creation_evidence("1920", "source date"))

    class ChangingRanker(Ranker):
        calls = 0

        def rank(self, topic, candidates, recent):
            self.calls += 1
            if self.calls == 1:
                pipeline.update_image_filters(replace(settings, time_filter_start=2000., time_filter_end=2030.))
            return super().rank(topic, candidates, recent)

    cache = Cache(settings)
    try:
        cache.put_image(candidate)
        ranker = ChangingRanker()
        published, failed = await pipeline._evaluate_candidates(topic, [candidate], None, None,
            None, ranker, cache, {}, Counter(), {"found": 0, "downloaded": 0, "best_score": None})
        assert not published and not failed and ranker.calls == 2
        assert not any(kind == "images" for kind, _ in events)
        assert pipeline._latest_topic is topic and len(pipeline.session_state.segments) == 1
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_period_hint_does_not_hide_source_results_when_index_has_no_period_terms(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), time_filter_enabled=True,
                        time_filter_start=parse_year("200 Ma"), time_filter_end=parse_year("150 Ma"))
    pipeline = Pipeline(settings, lambda *args: None)
    calls = []
    unknown, match = image("unknown", "Undated ammonite"), image("dated")

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            return [match] if query == "ammonite" else [unknown]

    cache = Cache(settings)
    try:
        found = await pipeline._query_candidates("ammonite", [Provider()], cache)
        assert calls == ["Jurassic ammonite", "ammonite"]
        assert found == [unknown, match] and pipeline._image_filter_match(match)[0]
        await pipeline._query_candidates("ammonite", [Provider()], cache)
        assert len(calls) == 2
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_geological_hint_survives_provider_query_word_limit(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), time_filter_enabled=True,
                        time_filter_start=parse_year("200 Ma"), time_filter_end=parse_year("150 Ma"))
    pipeline = Pipeline(settings, lambda *args: None)
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            return [image("dated")]

    cache = Cache(settings)
    try:
        await pipeline._query_candidates("ammonite fossil shells marine seafloor sediment rocks strata", [Provider()], cache)
        assert len(calls) == 1 and calls[0].startswith("Jurassic ")
        assert len(calls[0].split()) == 8
    finally:
        cache.close()

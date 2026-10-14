"""Source failures and consumed caches must not stall current-scene discovery."""
import asyncio
from collections import Counter
from dataclasses import replace
import threading
from types import SimpleNamespace

from PIL import Image
import httpx
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment


def candidate(name="camp"):
    return ImageCandidate(f"https://images.example/{name}.jpg", "https://source.example/item",
                          name, "wikimedia", phash="f" * 16)


@pytest.mark.asyncio
async def test_consumed_cache_page_fetches_fresh_head_results(tmp_path):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    cache = Cache(pipeline.settings)
    old, fresh = candidate("old"), candidate("fresh")
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            return [old] if len(calls) == 1 else [fresh]

    try:
        assert await pipeline._query_candidates("camp", [Provider()], cache) == [old]
        assert await pipeline._query_candidates("camp", [Provider()], cache) == [old]
        assert len(calls) == 1
        pipeline._displayed_urls.add(old.image_url)
        assert await pipeline._query_candidates("camp", [Provider()], cache) == [fresh]
        assert calls == ["camp", "camp"]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_one_unavailable_provider_keeps_healthy_source_search_cadence(tmp_path, monkeypatch):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), search_interval=10), lambda *args: None)
    topic = Topic("camp", "The camp remains beside the lake.", scene="camp")
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, topic))
    calls = []
    retry_delays = []
    cache = Cache(pipeline.settings)
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: [])

    class Provider:
        def __init__(self, name):
            self.name = name

        async def search(self, query, limit):
            calls.append(self.name)
            if self.name == "broken":
                raise OSError("temporarily unavailable")
            return []

        async def search_page(self, query, limit, page):
            return await self.search(query, limit)

    async def retry(generation, topic):
        retry_delays.append(pipeline._search_retry_delay)
        if len(retry_delays) == 2:
            pipeline.stop_event.set()
        else:
            pipeline._last_search = 0
            pipeline.topic_queue.put_nowait((generation, topic))

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)
    try:
        await asyncio.wait_for(pipeline._search([Provider("broken"), Provider("available")],
                                              None, SimpleNamespace(name="fixture"), cache), 2)
        assert retry_delays == [10, 10]
        assert calls == ["broken", "available", "available"]
        assert pipeline._source_retry_at["broken"] > 0
    finally:
        cache.close()


def test_explicit_refresh_preserves_story_and_rate_limit_deadlines(tmp_path):
    topic = Topic("camp", "The camp remains beside the lake.", scene="camp")
    segments = [TranscriptSegment(0, 6, "A camp stands beside the lake.")]
    state = SessionState(segments=segments, topic=topic, playback_generation=7)
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None, state)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, topic))
    pipeline._query_pages["camp"] = 8
    pipeline._source_retry_at["duckduckgo"] = 999999999
    pipeline._displayed_urls.add(candidate("displayed").image_url)
    pipeline._approved_images[candidate().image_url] = candidate()
    assert pipeline.refresh_images()
    assert pipeline._query_pages["camp"] == 0 and "camp" in pipeline._refresh_queries
    assert pipeline._source_retry_at == {"duckduckgo": 999999999}
    assert pipeline._displayed_urls == {candidate("displayed").image_url}
    assert state.segments is segments and state.topic is topic and state.playback_generation == 7
    assert not pipeline._approved_images
    assert pipeline.topic_queue.get_nowait() == (1, pipeline._latest_topic)
    assert pipeline._image_filter_revision == 1


@pytest.mark.asyncio
async def test_refresh_during_slow_ranking_fences_obsolete_image_and_csv_row(tmp_path):
    entered, release = threading.Event(), threading.Event()
    events = []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache")), lambda *event: events.append(event))
    topic = Topic("camp", "A camp beside a lake.", scene="camp")
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    cache = Cache(pipeline.settings)

    class Downloader:
        async def download(self, image):
            path = tmp_path / "camp.png"
            Image.new("RGB", (800, 600), "green").save(path)
            return replace(image, local_path=str(path), width=800, height=600)

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            entered.set()
            assert release.wait(2)
            return [replace(image, score=.9) for image in images]

    task = asyncio.create_task(pipeline._evaluate_candidates(topic, [candidate()], None, None,
                              Downloader(), Ranker(), cache, {}, Counter(),
                              {"found": 0, "downloaded": 0, "best_score": None}))
    try:
        while not entered.is_set():
            await asyncio.sleep(.005)
        assert pipeline.refresh_images()
        release.set()
        assert await asyncio.wait_for(task, 2) == (False, False)
        assert not pipeline._approved_images and not pipeline.session_state.image_matches
        assert not any(kind == "images" for kind, _ in events)
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cache.close()


@pytest.mark.asyncio
async def test_fast_provider_image_is_published_before_slow_source_finishes(tmp_path, monkeypatch):
    release = asyncio.Event()
    slow_finished = False
    published = []
    topic = Topic("camp", "A camp beside a lake.", scene="camp")
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache"), prefetch_count=2), lambda *args: None)
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, topic))
    cache = Cache(pipeline.settings)
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: [])

    class Provider:
        def __init__(self, name):
            self.name = name

        async def search(self, query, limit):
            nonlocal slow_finished
            if self.name == "slow":
                await release.wait()
                slow_finished = True
            image = candidate(self.name)
            image.phash = "f" * 16 if self.name == "fast" else "0" * 16
            return [image]

    class Downloader:
        async def download(self, image):
            path = tmp_path / (image.title + ".png")
            Image.new("RGB", (800, 600), "green").save(path)
            return replace(image, local_path=str(path), width=800, height=600)

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            return [replace(image, score=.9) for image in images]

    def receive(kind, images):
        if kind == "images":
            if not published:
                assert not slow_finished
                release.set()
            published[:] = images
            if len(images) == 2:
                pipeline.stop_event.set()

    pipeline.on_event = receive
    try:
        await asyncio.wait_for(pipeline._search([Provider("slow"), Provider("fast")],
                                              Downloader(), Ranker(), cache), 2)
        assert {image.title for image in published} == {"fast", "slow"}
        assert len(pipeline.session_state.image_matches) == 2
    finally:
        release.set()
        pipeline.stop_event.set()
        cache.close()


@pytest.mark.asyncio
async def test_early_full_buffer_preserves_completed_source_retry_after(tmp_path):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    cache = Cache(pipeline.settings)
    calls = []

    class Provider:
        def __init__(self, name):
            self.name = name

        async def search(self, query, limit):
            calls.append(self.name)
            if self.name == "limited":
                response = httpx.Response(429, headers={"Retry-After": "3600"},
                                          request=httpx.Request("GET", "https://source.example/search"))
                response.raise_for_status()
            return [candidate()]

    async def enough(images):
        # Let the other immediate source's response finish while this batch is
        # being checked, then signal that no more pending work is needed.
        await asyncio.sleep(0)
        return True

    providers = [Provider("fast"), Provider("limited")]
    try:
        assert await pipeline._query_candidates("camp", providers, cache, on_candidates=enough)
        assert pipeline._last_query_failed and pipeline._last_query_incomplete
        assert pipeline._source_retry_at["limited"] > 0
        await pipeline._query_candidates("lake", providers, cache)
        assert calls == ["fast", "limited", "fast"]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_refresh_during_inflight_source_request_keeps_force_flag(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    topic = Topic("camp", "A camp beside a lake.", scene="camp")
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache")), lambda *args: None)
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, topic))
    cache = Cache(pipeline.settings)
    calls = []
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: [])

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            if len(calls) == 1:
                entered.set()
                await release.wait()
            return [candidate("new")]

    async def evaluate(topic, images, *args):
        assert [image.title for image in images] == ["new"]
        pipeline.stop_event.set()
        return False, False

    monkeypatch.setattr(pipeline, "_evaluate_candidates", evaluate)
    task = asyncio.create_task(pipeline._search([Provider()], None, SimpleNamespace(name="fixture"), cache))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # Install an otherwise valid non-consumed cached head page while the
        # obsolete request is in flight, then explicitly force a fresh request.
        identity = '["camp", ["wikimedia"], 10, {"content_policy": "mature-v1", "allow_nsfw": false}]'
        cache.put_query(identity, [candidate("cached")])
        assert pipeline.refresh_images()
        release.set()
        await asyncio.wait_for(task, 2)
        assert calls == ["camp", "camp"]
        assert pipeline.image_request_revision == 1
    finally:
        release.set()
        pipeline.stop_event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cache.close()

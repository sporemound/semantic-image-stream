import asyncio

from PIL import Image
import httpx
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import ImageCandidate, Topic


def test_verified_reference_pack_addition_preserves_valid_saved_context(tmp_path, monkeypatch):
    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama")
    topic = Topic("lake", "Recorded lake scene", scene="lake")
    checkpoint = {"recorded": "legacy valid checkpoint"}
    state = SessionState(provider_key=("ollama", settings.context_model, ""), topic=topic,
                         analyzer_checkpoint=checkpoint)
    class Analyzer:
        def restore_checkpoint(self, value, recording):
            assert value is checkpoint
            return True
    monkeypatch.setattr("semantic_image_stream.engine.build_analyzer", lambda *args, **kwargs: Analyzer())
    pipeline = Pipeline(settings, lambda *args: None, session_state=state)
    pipeline._session_analyzer()
    assert state.topic is topic and pipeline._latest_topic is topic
    assert state.provider_key[:3] == ("ollama", settings.context_model, "")
    assert len(state.provider_key) == 4  # Bundled examples become part of future checkpoints.


def test_display_fingerprints_survive_history_limits_cache_clear_and_relaunch(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), history_enabled=False, history_limit=1)
    cache = Cache(settings)
    image = ImageCandidate("https://photos.example/old.jpg", "https://source.example/file", "Displayed", "wikimedia",
                           phash="ffffffffffffffff")
    cache.record(image)
    assert not cache.history()
    cache.clear()
    cache.close()
    cache = Cache(settings)
    try:
        assert cache.seen_urls() == {image.image_url}
        assert cache.seen_phashes() == {image.phash}
        cache.clear_history()
        assert not cache.seen_urls() and not cache.seen_phashes()
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_search_refills_same_scene_from_later_pages_without_new_transcript(tmp_path):
    shown, pages, events = [], [], []
    topic = Topic("lake", "A lake beside a forest", scene="lake forest", visual_subjects=("lake", "forest"))
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache"), prefetch_count=1,
                                 search_interval=.01, relevance_threshold=.4), lambda *args: None)
    pipeline._latest_topic = topic
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    cache = Cache(pipeline.settings)
    pipeline._cache = cache
    def receive(name, value):
        events.append((name, value))
        if name == "images":
            for image in value:
                shown.append(image.image_url)
                pipeline.record_displayed(image)
            if len(shown) == 3:
                pipeline.stop_event.set()
    pipeline.on_event = receive
    class Provider:
        name = "wikimedia"
        async def search(self, query, limit):
            return await self.search_page(query, limit, 0)
        async def search_page(self, query, limit, page):
            pages.append(page)
            return [ImageCandidate(f"https://photos.example/{page}.jpg", "https://source.example/file", "lake forest", "wikimedia")]
    class Downloader:
        async def download(self, image):
            path = tmp_path / (str(len(shown)) + ".jpg")
            Image.new("RGB", (800, 600), "blue").save(path)
            image.local_path, image.width, image.height = str(path), 800, 600
            image.phash = ["0000000000000000", "ffffffffffffffff", "aaaaaaaaaaaaaaaa"][len(shown)]
            return image
    class Ranker:
        name = "fixture"
        def rank(self, topic, images, recent):
            for image in images:
                image.score = .9
            return images
    try:
        pipeline.topic_queue.put_nowait((0, topic))
        await asyncio.wait_for(pipeline._search([Provider()], Downloader(), Ranker(), cache), 3)
        assert pages == [0, 1, 2]
        assert len(shown) == len(set(shown)) == 3
        assert cache.seen_urls() == set(shown)
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_paged_queries_have_distinct_cache_entries_and_refresh_is_explicit(tmp_path):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    cache = Cache(pipeline.settings)
    calls = []
    class Provider:
        name = "wikimedia"
        async def search(self, query, limit):
            return await self.search_page(query, limit, 0)
        async def search_page(self, query, limit, page):
            calls.append(page)
            return [ImageCandidate(f"https://photos.example/{page}.jpg", "https://source.example/file", "lake", "wikimedia")]
    try:
        for page in (0, 1, 0, 1):
            await pipeline._query_candidates("lake", [Provider()], cache, page=page)
        assert calls == [0, 1]
        await pipeline._query_candidates("lake", [Provider()], cache, refresh=True)
        assert calls == [0, 1, 0]
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_source_retry_after_pauses_only_rate_limited_provider(tmp_path):
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *args: None)
    cache = Cache(pipeline.settings)
    calls = []
    class Provider:
        def __init__(self, name):
            self.name = name
        async def search(self, query, limit):
            calls.append(self.name)
            if self.name == "limited":
                request = httpx.Request("GET", "https://source.example/api")
                response = httpx.Response(429, headers={"Retry-After": "3600"}, request=request)
                response.raise_for_status()
            return [ImageCandidate("https://photos.example/new.jpg", "https://source.example/file", "lake", "wikimedia")]
    try:
        providers = [Provider("limited"), Provider("available")]
        assert await pipeline._query_candidates("lake", providers, cache)
        assert pipeline._last_query_failed
        assert await pipeline._query_candidates("forest", providers, cache)
        assert calls == ["limited", "available", "available"]
        assert not pipeline._last_query_failed
    finally:
        cache.close()

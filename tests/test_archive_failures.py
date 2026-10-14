"""Source outages must not become cached zero-result Archive searches."""
import json
import time

import httpx
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.search_providers import InternetArchiveProvider, ProviderError


def search_response(*identifiers):
    return httpx.Response(200, json={"response": {"docs": [{"identifier": value} for value in identifiers]}})


def image_response():
    return httpx.Response(200, json={"metadata": {"title": "Coastal cave"},
                                    "files": [{"name": "cave.jpg", "source": "original"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [403, 429, 503, "invalid_json"])
async def test_all_failed_item_metadata_is_a_provider_failure(tmp_path, failure):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/advancedsearch.php":
            return search_response("cave-one", "cave-two")
        if failure == "invalid_json":
            return httpx.Response(200, content=b"unavailable")
        return httpx.Response(failure, headers={"Retry-After": "120"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path)), client=client)
        with pytest.raises(ProviderError, match="item metadata unavailable") as caught:
            await provider.search_page("coastal cave", 5, 1)
    assert len(requests) == 3
    assert "coastal cave" not in str(caught.value)
    if failure == 429:
        assert caught.value.__cause__.response.status_code == 429
        assert caught.value.__cause__.response.headers["Retry-After"] == "120"


@pytest.mark.asyncio
async def test_mixed_metadata_failures_preserve_rate_limit_cause(tmp_path):
    def handle(request):
        if request.url.path == "/advancedsearch.php":
            return search_response("unavailable", "limited")
        return httpx.Response(429, headers={"Retry-After": "240"}) if request.url.path.endswith("limited") else httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path)), client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.search("coastal cave", 5)
    assert caught.value.__cause__.response.status_code == 429
    assert caught.value.__cause__.response.headers["Retry-After"] == "240"


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [False, True])
async def test_genuine_empty_search_or_successful_empty_metadata_stays_empty(tmp_path, items):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/advancedsearch.php":
            return search_response(*(["cave"] if items else []))
        return httpx.Response(200, json={"metadata": {"title": "Coastal cave"}, "files": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path)), client=client)
        assert await provider.search("coastal cave", 5) == []
    assert len(requests) == 1 + int(items)


@pytest.mark.asyncio
async def test_partial_success_keeps_original_images_without_retries(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/advancedsearch.php":
            return search_response("good-cave", "unavailable")
        return image_response() if request.url.path.endswith("good-cave") else httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path)), client=client)
        images = await provider.search("coastal cave", 5)
    assert len(images) == 1
    assert images[0].image_url == "https://archive.org/download/good-cave/cave.jpg"
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_metadata_rate_limit_is_not_negative_cached_and_recovers_after_cooldown(tmp_path, monkeypatch):
    settings = Settings(cache_dir=str(tmp_path), providers=["archive"])
    pipeline = Pipeline(settings, lambda *args: None)
    cache = Cache(settings)
    requests = []
    limited = True

    def handle(request):
        requests.append(request)
        if request.url.path == "/advancedsearch.php":
            return search_response("cave")
        return httpx.Response(429, headers={"Retry-After": "120"}) if limited else image_response()

    clock = [time.monotonic()]
    monkeypatch.setattr("semantic_image_stream.engine.time.monotonic", lambda: clock[0])
    key = json.dumps(["coastal cave", ["archive"], settings.results_per_provider,
                     {"content_policy": "mature-v1", "allow_nsfw": settings.allow_nsfw}])
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            providers = [InternetArchiveProvider(settings, client=client)]
            assert await pipeline._query_candidates("coastal cave", providers, cache) == []
            assert pipeline._last_query_failed and not pipeline._last_query_succeeded
            assert cache.get_query(key) is None
            assert pipeline._source_retry_at["archive"] == clock[0] + 120
            assert await pipeline._query_candidates("coastal cave", providers, cache) == []
            assert pipeline._last_query_all_paused and len(requests) == 2
            assert cache.get_query(key) is None
            limited = False
            clock[0] += 121
            images = await pipeline._query_candidates("coastal cave", providers, cache)
            assert len(images) == 1 and len(requests) == 4
            assert not pipeline._last_query_failed and pipeline._last_query_succeeded
            assert cache.get_query(key)[0].image_url == images[0].image_url
    finally:
        cache.close()

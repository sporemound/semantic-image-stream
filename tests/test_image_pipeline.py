"""Regressions for a blank preview while audio continues to transcribe."""
import asyncio
from dataclasses import replace
import json

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.image_ranking import MetadataRanker
from semantic_image_stream.models import ImageCandidate, Topic


class QueryProvider:
    name = "wikimedia"

    def __init__(self, responses, name="wikimedia"):
        self.responses = responses
        self.name = name
        self.queries = []

    async def search(self, query, limit):
        self.queries.append(query)
        result = self.responses.get(query, [])
        if isinstance(result, Exception):
            raise result
        return [replace(candidate) for candidate in result]


def query_key(pipeline, query):
    return json.dumps([query.casefold(), pipeline.settings.providers, pipeline.settings.results_per_provider])


def stop_on_empty(pipeline, monkeypatch, retries=0):
    async def retry(generation, topic):
        nonlocal retries
        if retries:
            retries -= 1
            pipeline.topic_queue.put_nowait((generation, topic))
        else:
            pipeline.stop_event.set()

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)


class Provider:
    name = "wikimedia"

    def __init__(self, images, delay=None):
        self.images = images
        self.delay = delay
        self.called = asyncio.Event()

    async def search(self, query, limit):
        self.called.set()
        if self.delay:
            await self.delay.wait()
        return [replace(c) for c in self.images]


class Downloader:
    def __init__(self, directory, slow=None):
        self.directory = directory
        self.slow = slow
        self.slow_started = asyncio.Event()

    async def download(self, candidate):
        if candidate.image_url.endswith("slow"):
            self.slow_started.set()
            await self.slow.wait()
        path = self.directory / ("a" * 64 + ".png")
        Image.new("RGB", (800, 600), "red").save(path)
        candidate.local_path = str(path)
        candidate.width, candidate.height = 800, 600
        candidate.phash = "aaaaaaaaaaaaaaaa"
        return candidate


def pipeline_fixture(tmp_path):
    settings = Settings(cache_dir=str(tmp_path), semantic_interval=0.01, search_interval=0.01)
    events = []
    ready = asyncio.Event()

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "images":
            ready.set()

    pipeline = Pipeline(settings, receive)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    cache = Cache(settings)
    directory = tmp_path / "images"
    directory.mkdir(exist_ok=True)
    return pipeline, cache, directory, events, ready


async def cleanup(task, pipeline, cache):
    pipeline.stop_event.set()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    cache.close()


@pytest.mark.asyncio
async def test_first_preview_does_not_wait_for_slow_downloads(tmp_path):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    topic = Topic("black hole event horizon", "Black holes and event horizons", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((1, topic))
    candidates = [ImageCandidate("https://image.example/fast", "https://source.example/file", "Black hole event horizon", "wikimedia"),
                  ImageCandidate("https://image.example/slow", "https://source.example/file", "Black hole event horizon", "wikimedia")]
    slow = asyncio.Event()
    downloader = Downloader(directory, slow)
    task = asyncio.create_task(pipeline._search([Provider(candidates)], downloader, MetadataRanker(), cache))
    try:
        await asyncio.wait_for(downloader.slow_started.wait(), 2)
        await asyncio.wait_for(ready.wait(), 2)
        assert not slow.is_set(), "Preview should arrive before the blocked second download"
        image = next(payload[0] for kind, payload in events if kind == "images")
        assert image.score >= pipeline.settings.relevance_threshold
        assert image.path.is_file()
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_topic_changes_rerank_inflight_results_instead_of_discarding(tmp_path):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    initial = Topic("black hole event horizon", "Black holes", kind="science")
    latest = Topic("black hole spacetime", "Black holes curve spacetime", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = initial
    pipeline.topic_queue.put_nowait((1, initial))
    provider_release = asyncio.Event()
    provider = Provider([ImageCandidate("https://image.example/fast", "https://source.example/file",
                                       "Black hole event horizon spacetime", "wikimedia")], provider_release)
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache))
    try:
        await asyncio.wait_for(provider.called.wait(), 2)
        pipeline._generation = 2
        pipeline._latest_topic = latest
        provider_release.set()
        await asyncio.wait_for(ready.wait(), 2)
        image = next(payload[0] for kind, payload in events if kind == "images")
        assert image.score >= pipeline.settings.relevance_threshold
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_unrelated_old_topic_images_are_not_published(tmp_path):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    initial = Topic("black hole event horizon", "Black holes", kind="science")
    latest = Topic("orchid pollination", "Orchids and pollinating insects", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = initial
    pipeline.topic_queue.put_nowait((1, initial))
    release = asyncio.Event()
    provider = Provider([ImageCandidate("https://image.example/fast", "https://source.example/file",
                                       "Black hole event horizon", "wikimedia")], release)
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache))
    try:
        await asyncio.wait_for(provider.called.wait(), 2)
        pipeline._generation = 2
        pipeline._latest_topic = latest
        release.set()
        for _ in range(100):
            if any(kind == "retrieval" and data["phase"] == "empty" for kind, data in events):
                break
            await asyncio.sleep(0.01)
        assert any(kind == "retrieval" and data["phase"] == "empty" for kind, data in events)
        assert not ready.is_set()
        assert not any(kind == "images" for kind, _ in events)
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_new_topic_wakes_failed_search_without_thirty_second_wait(tmp_path):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    first = Topic("black hole event horizon", "Black holes", kind="science")
    latest = Topic("orchid pollination", "Orchids", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = first
    pipeline.topic_queue.put_nowait((1, first))
    provider = Provider([])
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache))
    try:
        for _ in range(100):
            if any(kind == "retrieval" and data["phase"] == "empty" for kind, data in events):
                break
            await asyncio.sleep(0.01)
        provider.called.clear()
        provider.images = [ImageCandidate("https://image.example/fast", "https://source.example/file", "Orchid pollination", "wikimedia")]
        pipeline._generation = 2
        pipeline._latest_topic = latest
        pipeline.topic_queue.put_nowait((2, latest))
        await asyncio.wait_for(ready.wait(), 2)
        assert provider.called.is_set()
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_empty_search_tries_only_two_distinct_alternatives_and_caches_empty_results(tmp_path, monkeypatch):
    pipeline, cache, directory, events, _ = pipeline_fixture(tmp_path)
    topic = Topic("garbled narration phrase", "Corona satellites and Mexico City", kind="history")
    pipeline._generation = 1
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((1, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries",
                        lambda _: ["Corona satellite", "Mexico City", "Unbounded third query"])
    provider = QueryProvider({})
    stop_on_empty(pipeline, monkeypatch, retries=1)
    try:
        await asyncio.wait_for(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache), 2)
        assert provider.queries == [topic.query, "Corona satellite", "Mexico City"]
        assert all(cache.get_query(query_key(pipeline, query)) == [] for query in provider.queries)
        assert cache.get_query(query_key(pipeline, "Unbounded third query")) is None
        attempts = [data for kind, data in events if kind == "retrieval" and data["phase"] == "searching"]
        assert attempts
        assert all("shorter query from the current spoken context" in data["message"] for data in attempts)
        assert all(topic.query not in data["message"] for data in attempts)
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_fallback_matches_keep_full_context_and_selected_relevance_minimum(tmp_path, monkeypatch):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    pipeline.settings.relevance_threshold = 0.50
    topic = Topic("andale cagey officer didn't crew chanted",
                  "Corona satellite imagery and a KGB mission in Mexico City",
                  ("satellite", "KGB"), "history")
    pipeline._generation = 1
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((1, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: ["Corona satellite", "Mexico City"])
    image = ImageCandidate("https://image.example/fast", "https://source.example/file",
                           "Corona satellite", "wikimedia")
    provider = QueryProvider({"Corona satellite": [image]})

    class RecordingRanker(MetadataRanker):
        def __init__(self):
            self.topics = []

        def rank(self, current, candidates, recent):
            self.topics.append(current)
            return super().rank(current, candidates, recent)

    ranker = RecordingRanker()
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), ranker, cache))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert provider.queries == [topic.query, "Corona satellite"]
        assert all(current == replace(topic, query="Corona satellite") for current in ranker.topics)
        assert pipeline.settings.relevance_threshold == 0.50
        shown = next(images[0] for kind, images in events if kind == "images")
        assert shown.score >= 0.50
        status = [data for kind, data in events if kind == "retrieval" and data["phase"] == "ready"]
        assert status and all(data["threshold"] == 0.50 for data in status)
        assert "using a shorter query from the current spoken context" in status[-1]["message"]
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_partial_provider_failure_still_tries_alternatives_without_negative_caching(tmp_path, monkeypatch):
    pipeline, cache, directory, _, _ = pipeline_fixture(tmp_path)
    pipeline.settings.providers = ["wikimedia", "archive"]
    topic = Topic("garbled narration phrase", "Corona satellites", kind="history")
    pipeline._generation = 1
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((1, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: ["Corona satellite"])
    unavailable = QueryProvider({topic.query: TimeoutError(), "Corona satellite": TimeoutError()})
    available = QueryProvider({}, name="archive")
    stop_on_empty(pipeline, monkeypatch, retries=1)
    try:
        await asyncio.wait_for(pipeline._search([unavailable, available], Downloader(directory),
                                               MetadataRanker(), cache), 2)
        expected = [topic.query, "Corona satellite"] * 2
        assert unavailable.queries == available.queries == expected
        assert all(cache.get_query(query_key(pipeline, query)) is None for query in expected)
    finally:
        cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_primary_results_never_trigger_alternative_queries(tmp_path, monkeypatch, cached):
    pipeline, cache, directory, _, ready = pipeline_fixture(tmp_path)
    topic = Topic("Corona satellite", "Corona satellite imagery", kind="history")
    pipeline._generation = 1
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((1, topic))

    def forbidden(_):
        raise AssertionError("A primary match must not perform alternative lookup")

    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", forbidden)
    image = ImageCandidate("https://image.example/fast", "https://source.example/file",
                           "Corona satellite imagery", "wikimedia")
    provider = QueryProvider({topic.query: [image]})
    if cached:
        cache.put_query(query_key(pipeline, topic.query), [image])
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert provider.queries == ([] if cached else [topic.query])
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_topic_change_during_fallback_rejects_obsolete_matches(tmp_path, monkeypatch):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    initial = Topic("garbled narration phrase", "Corona satellite imagery", kind="history")
    latest = Topic("orchid pollination", "Orchid pollination and insects", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = initial
    pipeline.topic_queue.put_nowait((1, initial))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: ["Corona satellite"])
    called, release = asyncio.Event(), asyncio.Event()

    class DelayedFallback(QueryProvider):
        async def search(self, query, limit):
            if query == "Corona satellite":
                called.set()
                await release.wait()
            return await super().search(query, limit)

    image = ImageCandidate("https://image.example/fast", "https://source.example/file",
                           "Corona satellite imagery", "wikimedia")
    provider = DelayedFallback({"Corona satellite": [image]})
    stop_on_empty(pipeline, monkeypatch)
    task = asyncio.create_task(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache))
    try:
        await asyncio.wait_for(called.wait(), 2)
        pipeline._generation = 2
        pipeline._latest_topic = latest
        release.set()
        await asyncio.wait_for(task, 2)
        assert not ready.is_set()
        assert not any(kind == "images" for kind, _ in events)
        empty = [data for kind, data in events if kind == "retrieval" and data["phase"] == "empty"]
        assert empty and "using a shorter query" not in empty[-1]["message"]
    finally:
        await cleanup(task, pipeline, cache)


@pytest.mark.asyncio
async def test_topic_change_abandons_remaining_alternatives_from_old_context(tmp_path, monkeypatch):
    pipeline, cache, directory, _, _ = pipeline_fixture(tmp_path)
    initial = Topic("garbled narration phrase", "Corona satellite imagery", kind="history")
    latest = Topic("orchid pollination", "Orchid pollination and insects", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = initial
    pipeline.topic_queue.put_nowait((1, initial))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries",
                        lambda current: ["Corona satellite", "KGB mission"] if current is initial
                        else ["pollinating insects", "orchid flowers"])

    class ChangingProvider(QueryProvider):
        async def search(self, query, limit):
            if query == "Corona satellite":
                pipeline._generation = 2
                pipeline._latest_topic = latest
            return await super().search(query, limit)

    provider = ChangingProvider({})
    stop_on_empty(pipeline, monkeypatch)
    try:
        await asyncio.wait_for(pipeline._search([provider], Downloader(directory), MetadataRanker(), cache), 2)
        assert provider.queries == [initial.query, "Corona satellite", "pollinating insects"]
        assert "KGB mission" not in provider.queries
        assert "orchid flowers" not in provider.queries, "Topic changes must not reset the two-query budget"
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_topic_change_while_fallback_is_ranked_rechecks_current_context(tmp_path, monkeypatch):
    pipeline, cache, directory, events, ready = pipeline_fixture(tmp_path)
    initial = Topic("garbled narration phrase", "Corona satellite imagery", kind="history")
    latest = Topic("orchid pollination", "Orchid pollination and insects", kind="science")
    pipeline._generation = 1
    pipeline._latest_topic = initial
    pipeline.topic_queue.put_nowait((1, initial))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: ["Corona satellite"])
    image = ImageCandidate("https://image.example/fast", "https://source.example/file",
                           "Corona satellite imagery", "wikimedia")
    provider = QueryProvider({"Corona satellite": [image]})

    class ChangingRanker:
        name = "clip"
        model = object()

        def __init__(self):
            self.topics = []

        def rank(self, current, candidates, recent):
            self.topics.append(current)
            if len(self.topics) == 1:
                pipeline._generation = 2
                pipeline._latest_topic = latest
                return [replace(candidate, score=0.95) for candidate in candidates]
            return []

    ranker = ChangingRanker()
    stop_on_empty(pipeline, monkeypatch)
    try:
        await asyncio.wait_for(pipeline._search([provider], Downloader(directory), ranker, cache), 2)
        assert ranker.topics == [replace(initial, query="Corona satellite"), latest]
        assert not ready.is_set()
        assert not any(kind == "images" for kind, _ in events)
    finally:
        cache.close()

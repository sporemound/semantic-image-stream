"""Regressions for a blank preview while audio continues to transcribe."""
import asyncio
from dataclasses import replace

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.image_ranking import MetadataRanker
from semantic_image_stream.models import ImageCandidate, Topic


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

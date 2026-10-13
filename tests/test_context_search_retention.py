"""Established story context keeps discovery alive without additional ASR."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.narrative_context import LocalNarrativeAnalyzer


def fixture(tmp_path, callback):
    settings = Settings(cache_dir=str(tmp_path / "cache"), providers=["wikimedia"], prefetch_count=1,
                        min_width=32, min_height=24, semantic_interval=0.01, search_interval=0.001,
                        relevance_threshold=0.40, use_story_examples=False)
    pipeline = Pipeline(settings, callback)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    cache = Cache(settings)
    pipeline._cache = cache
    return pipeline, cache


def candidate(name, title="Arctic camp"):
    return ImageCandidate(f"https://images.example.org/{name}.jpg", "https://source.example.org/item",
                          title, "wikimedia", phash="0" * 16 if name == "old" else "f" * 16)


class Downloader:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    async def download(self, image):
        path = self.directory / (Path(image.image_url).stem + ".png")
        Image.new("RGB", (120, 80), "green").save(path)
        return replace(image, local_path=str(path), width=120, height=80)


class Ranker:
    name = "context_clip"
    model = object()

    def rank(self, topic, images, recent):
        return [replace(image, score=0.9 if image.title == topic.query else 0.1) for image in images]


def key(pipeline, query, page=0):
    identity = [query.casefold(), pipeline.settings.providers, pipeline.settings.results_per_provider]
    if page:
        identity.append({"page": page})
    return json.dumps(identity)


async def close(task, pipeline, cache):
    pipeline.stop_event.set()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    pipeline._cache = None
    cache.close()


@pytest.mark.asyncio
async def test_nonvisual_new_passage_retains_established_visual_context_and_playback(tmp_path, monkeypatch):
    camp = "Three camp tents stand on Arctic ice."
    filler = "He wondered whether the waiting would ever end."
    events, ready = [], []
    pipeline = None

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] in {"ready", "retaining"}:
            ready.append((pipeline._latest_topic, payload["playback_key"]))
            if len(ready) == 1:
                with pipeline.session_state.lock:
                    pipeline.session_state.segments.append(TranscriptSegment(6, 12, filler))
            else:
                pipeline.stop_event.set()

    settings = Settings(cache_dir=str(tmp_path), semantic_provider="ollama", semantic_interval=0.01,
                        use_story_examples=False)
    pipeline = Pipeline(settings, receive)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.session_state.segments.append(TranscriptSegment(0, 6, camp))
    analyzer = LocalNarrativeAnalyzer(settings)
    states = iter([
        {"summary": camp, "current_scene": camp, "visual_subjects": ["camp tents", "Arctic ice"],
         "search_queries": ["camp tents Arctic ice", "camp tents"], "kind": "general", "evidence": [camp]},
        {"summary": camp, "current_scene": filler, "visual_subjects": [],
         "search_queries": [], "kind": "general", "evidence": [filler]},
    ])
    monkeypatch.setattr(analyzer, "_ensure_service", lambda *args: None)
    monkeypatch.setattr(analyzer, "_generate", lambda *args, **kwargs: next(states))
    await asyncio.wait_for(pipeline._interpret(analyzer), 3)
    assert len(ready) == 2
    first, second = ready[0][0], ready[1][0]
    assert second.query == first.query and second.query
    assert second.scene == first.scene and second.visual_subjects == first.visual_subjects
    assert ready[0][1] == ready[1][1]
    assert second.transcript == (camp, filler) and second.context_used_segments == 2
    assert pipeline.session_state.segments[0].text == camp
    assert sum(kind == "topic" for kind, _ in events) == 1
    final = [payload for kind, payload in events if kind == "context"][-1]
    assert final["phase"] == "retaining" and final["retaining"] is True
    assert not final["playback_changed"]


@pytest.mark.asyncio
async def test_consumed_cached_queue_refills_from_retained_topic_without_new_transcript(tmp_path, monkeypatch):
    pipeline = None
    seen = []

    def receive(kind, images):
        if kind == "images":
            seen.extend(image.image_url for image in images)
            for image in images:
                pipeline.record_displayed(image)
            if len(seen) == 2:
                pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    topic = Topic("Arctic camp", "The explorers remain at the Arctic camp.",
                  scene="Arctic camp", visual_subjects=("Arctic camp",))
    pipeline._latest_topic = topic
    pipeline._generation = 1
    pipeline._update_playback_scope(topic)
    pipeline.topic_queue.put_nowait((1, topic))
    cache.put_query(key(pipeline, topic.query), [candidate("old")])
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])
    pages = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            raise AssertionError("The first query is already cached")

        async def search_page(self, query, limit, page):
            assert query == topic.query
            pages.append(page)
            return [candidate("fresh")]

    task = asyncio.create_task(pipeline._search([Provider()], Downloader(tmp_path / "pixels"), Ranker(), cache))
    try:
        await asyncio.wait_for(task, 3)
        assert seen == [candidate("old").image_url, candidate("fresh").image_url]
        assert pages == [1]
        assert pipeline._latest_topic is topic and not pipeline.session_state.segments
        assert pipeline.settings.relevance_threshold == 0.40
    finally:
        await close(task, pipeline, cache)


@pytest.mark.asyncio
async def test_old_delayed_result_cannot_publish_or_restore_context_after_grounded_scene_change(tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    published = []
    pipeline = None

    def receive(kind, images):
        if kind == "images":
            published.extend(images)
            pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    camp = Topic("Arctic camp", "Camp was established earlier.", scene="Tents on Arctic ice",
                 visual_subjects=("camp tents", "Arctic ice"))
    door = Topic("Wooden door", "A wooden door now stands on Arctic ice.", scene="A wooden door on Arctic ice",
                 visual_subjects=("wooden door", "Arctic ice"))
    pipeline._latest_topic = camp
    pipeline._generation = 1
    pipeline._update_playback_scope(camp)
    pipeline.topic_queue.put_nowait((1, camp))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return [candidate("old" if query == camp.query else "fresh", query)]

    class DelayedDownloader(Downloader):
        async def download(self, image):
            if image.title == camp.query:
                entered.set()
                await release.wait()
            return await super().download(image)

    task = asyncio.create_task(pipeline._search([Provider()], DelayedDownloader(tmp_path / "pixels"), Ranker(), cache))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        pipeline._latest_topic = pipeline.session_state.topic = door
        pipeline._generation = 2
        assert pipeline._update_playback_scope(door)
        generation = pipeline.session_state.playback_generation
        pipeline.topic_queue.put_nowait((2, door))
        release.set()
        await asyncio.wait_for(task, 3)
        assert [image.title for image in published] == [door.query]
        assert pipeline._latest_topic is door and pipeline.session_state.topic is door
        assert pipeline.session_state.playback_generation == generation
    finally:
        release.set()
        await close(task, pipeline, cache)


@pytest.mark.asyncio
async def test_provider_in_cooldown_is_not_a_successful_empty_query(tmp_path):
    pipeline, cache = fixture(tmp_path, lambda *args: None)
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            return []

    pipeline._source_retry_at["wikimedia"] = time.monotonic() + 60
    try:
        assert await pipeline._query_candidates("Arctic camp", [Provider()], cache) == []
        assert calls == []
        assert not pipeline._last_query_failed
        assert pipeline._last_query_incomplete and pipeline._last_query_all_paused
        assert not pipeline._last_query_succeeded
        assert cache.get_query(key(pipeline, "Arctic camp")) is None
    finally:
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_exhausted_nonhead_page_eventually_refreshes_without_new_speech(tmp_path, monkeypatch):
    published, pages = [], []
    pipeline = None

    def receive(kind, images):
        if kind == "images":
            published.extend(images)
            pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    topic = Topic("Arctic camp", "The same camp remains established.")
    pipeline._latest_topic = topic
    pipeline._generation = 1
    pipeline._query_pages[topic.query.casefold()] = 9
    pipeline.topic_queue.put_nowait((1, topic))
    old = candidate("old")
    pipeline._displayed_urls.add(old.image_url)
    for page in range(10):
        cache.put_query(key(pipeline, topic.query, page), [old])
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])
    retries = 0

    async def retry(generation, current):
        nonlocal retries
        retries += 1
        if retries > 12:
            pipeline.stop_event.set()
        else:
            pipeline.topic_queue.put_nowait((generation, current))

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            pages.append(0)
            return [old]

        async def search_page(self, query, limit, page):
            pages.append(page)
            return [candidate("fresh")] if page == 1 else [old]

    task = asyncio.create_task(pipeline._search([Provider()], Downloader(tmp_path / "pixels"), Ranker(), cache))
    try:
        await asyncio.wait_for(task, 3)
        assert [image.image_url for image in published] == [candidate("fresh").image_url]
        assert 1 in pages and retries <= 12
        assert pipeline._latest_topic is topic and not pipeline.session_state.segments
    finally:
        await close(task, pipeline, cache)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "network_error", "ranking_error", "one_paused", "all_paused"])
async def test_retry_cadence_distinguishes_empty_pages_from_failed_or_paused_sources(tmp_path, monkeypatch, outcome):
    pipeline, cache = fixture(tmp_path, lambda *args: None)
    topic = Topic("Arctic camp", "The Arctic camp remains in view.")
    pipeline._latest_topic = topic
    pipeline._generation = 1
    pipeline.topic_queue.put_nowait((1, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])
    calls, retries = [], []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(self.name)
            if outcome == "network_error":
                raise TimeoutError()
            return [candidate("fresh")] if outcome == "ranking_error" else []

    class Other(Provider):
        name = "archive"

    class FailedRanker(Ranker):
        def rank(self, topic, images, recent):
            raise RuntimeError("substituted local model failure")

    providers = [Provider()]
    if outcome == "one_paused":
        pipeline.settings.providers.append("archive")
        providers.append(Other())
        pipeline._source_retry_at["archive"] = time.monotonic() + 90
    if outcome == "all_paused":
        pipeline._source_retry_at["wikimedia"] = time.monotonic() + 90

    async def retry(generation, current):
        retries.append(pipeline._search_retry_delay)
        pipeline.stop_event.set()

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)
    task = asyncio.create_task(pipeline._search(providers, Downloader(tmp_path / "pixels"),
                                               FailedRanker() if outcome == "ranking_error" else Ranker(), cache))
    try:
        await asyncio.wait_for(task, 3)
        assert len(retries) == 1
        if outcome in {"network_error", "ranking_error"}:
            assert retries[0] >= 30
            assert pipeline._query_pages[topic.query.casefold()] == 0
        elif outcome == "all_paused":
            assert retries[0] > 89 and calls == []
            assert pipeline._query_pages[topic.query.casefold()] == 0
            assert cache.get_query(key(pipeline, topic.query)) is None
        else:
            assert retries == [1.0]
            assert pipeline._query_pages[topic.query.casefold()] == 1
            assert calls == ["wikimedia"]
            if outcome == "one_paused":
                assert pipeline._last_query_incomplete and not pipeline._last_query_all_paused
                assert not pipeline._last_query_failed
                assert cache.get_query(key(pipeline, topic.query)) is None
            else:
                assert cache.get_query(key(pipeline, topic.query)) == []
        assert pipeline._latest_topic is topic
        assert pipeline.settings.relevance_threshold == 0.40
    finally:
        await close(task, pipeline, cache)


@pytest.mark.asyncio
async def test_http_retry_after_blocks_new_requests_without_negative_cache(tmp_path):
    import httpx

    pipeline, cache = fixture(tmp_path, lambda *args: None)
    calls = []

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            calls.append(query)
            response = httpx.Response(429, headers={"Retry-After": "120"},
                                      request=httpx.Request("GET", "https://source.example.org/search"))
            response.raise_for_status()

    try:
        assert await pipeline._query_candidates("Arctic camp", [Provider()], cache) == []
        assert pipeline._last_query_failed and pipeline._last_query_incomplete
        assert not pipeline._last_query_succeeded
        assert pipeline._source_retry_at["wikimedia"] - time.monotonic() > 119
        assert await pipeline._query_candidates("camp tents", [Provider()], cache) == []
        assert calls == ["Arctic camp"]
        assert pipeline._last_query_all_paused and not pipeline._last_query_failed
        assert cache.get_query(key(pipeline, "Arctic camp")) is None
        assert cache.get_query(key(pipeline, "camp tents")) is None
    finally:
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_explicit_retention_preserves_visual_fields_even_if_empty_response_is_supplied(tmp_path):
    camp = Topic("Arctic camp", "The camp was established.", concepts=("tent",), kind="nature",
                 summary="Camp remains established.", scene="Tents on Arctic ice",
                 visual_subjects=("camp tents", "Arctic ice"), queries=("Arctic camp", "camp tents"))
    update = Topic("", "The waiting continues without new visible information.", kind="general",
                   summary="The explorers wait at their established camp.", context_used_segments=2,
                   visual_update="retain")
    contexts = []
    pipeline = None

    def receive(kind, payload):
        if kind == "context" and payload["phase"] in {"ready", "retaining"}:
            contexts.append(payload)
            pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    pipeline._latest_topic = pipeline.session_state.topic = camp
    pipeline._update_playback_scope(camp)
    previous_generation = pipeline.session_state.playback_generation
    pipeline.session_state.segments.extend([TranscriptSegment(0, 6, "Tents stand on Arctic ice."),
                                            TranscriptSegment(6, 12, "The waiting continues.")])

    class Analyzer:
        def analyze(self, segments):
            return update

    try:
        await asyncio.wait_for(pipeline._interpret(Analyzer()), 3)
        retained = pipeline._latest_topic
        assert (retained.query, retained.concepts, retained.kind, retained.scene, retained.visual_subjects, retained.queries) == (
            camp.query, camp.concepts, camp.kind, camp.scene, camp.visual_subjects, camp.queries)
        assert retained.summary == update.summary and retained.context == update.context
        assert retained.transcript == tuple(segment.text for segment in pipeline.session_state.segments)
        assert retained.context_used_segments == 2 and retained.visual_update == "retain"
        assert pipeline.session_state.playback_generation == previous_generation
        assert contexts[0]["phase"] == "retaining" and contexts[0]["retaining"]
    finally:
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_rejected_full_prefetch_raster_frees_slot_and_refills_without_marking_it_seen(tmp_path, monkeypatch):
    bad, fresh = candidate("old"), candidate("fresh")
    published, downloads, pages = [], [], []
    pipeline = None

    def receive(kind, images):
        if kind != "images":
            return
        published.extend(image.image_url for image in images)
        for image in images:
            if image.image_url == bad.image_url:
                # This is the UI decoder boundary: the prepared/cached raster
                # was not shown, and the full queue must not block its refill.
                pipeline.reject_image(image)
            else:
                pipeline.record_displayed(image)
                pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    topic = Topic("Arctic camp", "The same camp remains visible.", scene="Arctic camp")
    pipeline._latest_topic = topic
    pipeline._generation = 1
    pipeline.topic_queue.put_nowait((1, topic))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda _: [])

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return [bad]

        async def search_page(self, query, limit, page):
            pages.append(page)
            return [bad, fresh]

    class TracedDownloader(Downloader):
        async def download(self, image):
            downloads.append(image.image_url)
            return await super().download(image)

    task = asyncio.create_task(pipeline._search([Provider()], TracedDownloader(tmp_path / "pixels"), Ranker(), cache))
    try:
        await asyncio.wait_for(task, 3)
        assert published == [bad.image_url, fresh.image_url]
        assert downloads == [bad.image_url, fresh.image_url] and pages == [1]
        assert bad.image_url not in pipeline._displayed_urls
        assert bad.image_url not in pipeline._approved_images
        assert [image.image_url for image in cache.history()] == [fresh.image_url]
        assert not pipeline.session_state.segments and pipeline._latest_topic is topic
        assert pipeline.settings.relevance_threshold == 0.40
    finally:
        await close(task, pipeline, cache)


@pytest.mark.asyncio
async def test_newer_analyzer_retained_scene_catches_up_after_previous_batch_failure(tmp_path):
    camp = Topic("Arctic camp", "The group pitched tents on Arctic ice.", scene="Tents on Arctic ice",
                 visual_subjects=("camp tents", "Arctic ice"))
    cave = Topic("Cave chamber", "The group left camp and entered a rocky underground chamber.",
                 summary="The group now waits inside the cave.", scene="A dark rocky underground chamber",
                 visual_subjects=("rocky cave", "underground chamber"), queries=("Cave chamber",),
                 context_used_segments=3, visual_update="retain")
    events = []
    pipeline = None

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "context" and payload["phase"] in {"ready", "retaining"}:
            pipeline.stop_event.set()

    pipeline, cache = fixture(tmp_path, receive)
    pipeline._latest_topic = pipeline.session_state.topic = camp
    pipeline._update_playback_scope(camp)
    previous_generation = pipeline.session_state.playback_generation
    pipeline.session_state.segments.extend([
        TranscriptSegment(0, 6, "Tents stood on Arctic ice."),
        TranscriptSegment(6, 12, "They left camp and entered a rocky underground chamber."),
        TranscriptSegment(12, 18, "They waited without speaking."),
    ])

    class Analyzer:
        def analyze(self, segments):
            # The cave batch was accepted internally before a later batch
            # failed. The retry retains that newer accepted internal state.
            return cave

    try:
        await asyncio.wait_for(pipeline._interpret(Analyzer()), 3)
        assert pipeline._latest_topic.query == cave.query
        assert pipeline._latest_topic.scene == cave.scene
        assert pipeline._latest_topic.visual_subjects == cave.visual_subjects
        assert pipeline._latest_topic.visual_update == "replace"
        assert pipeline.session_state.playback_generation == previous_generation + 1
        context = [payload for kind, payload in events if kind == "context"][-1]
        assert context["phase"] == "ready" and context["playback_changed"] and not context["retaining"]
        assert [payload.query for kind, payload in events if kind == "topic"] == [cave.query]
    finally:
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_newer_retained_alternative_queries_are_not_lost_inside_same_scene(tmp_path):
    published = Topic("Arctic camp", "The tents remain on Arctic ice.", scene="Tents on Arctic ice",
                      visual_subjects=("camp tents", "Arctic ice"), queries=("Arctic camp",))
    current = replace(published, queries=("Arctic camp", "canvas tents"), visual_update="retain")
    pipeline, cache = fixture(tmp_path, lambda *args: None)
    pipeline._latest_topic = pipeline.session_state.topic = published
    pipeline._update_playback_scope(published)
    previous_generation = pipeline.session_state.playback_generation
    pipeline._last_fill_signature = pipeline._fill_signature(published)
    pipeline._segments.append(TranscriptSegment(0, 6, "Their canvas tents still stand on Arctic ice."))

    class Analyzer:
        def analyze(self, segments):
            return current

    def receive(kind, payload):
        if kind == "context" and payload["phase"] in {"ready", "retaining"}:
            pipeline.stop_event.set()

    pipeline.on_event = receive
    try:
        await asyncio.wait_for(pipeline._interpret(Analyzer()), 3)
        assert pipeline._latest_topic.queries == current.queries
        assert pipeline._latest_topic.visual_update == "replace"
        assert pipeline.session_state.playback_generation == previous_generation
        generation, pending = pipeline.topic_queue.get_nowait()
        assert pending.queries == current.queries and generation == pipeline._generation
    finally:
        pipeline._cache = None
        cache.close()

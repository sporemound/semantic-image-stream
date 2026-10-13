"""Image discovery recovers without resetting a long-running spoken session."""
import asyncio
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import threading

from PIL import Image
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import Pipeline, SessionState
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["return", "cancel", "exception"])
async def test_interrupted_search_restarts_with_backoff_and_retains_long_session(tmp_path, monkeypatch, failure):
    clock = [6 * 3600.0]
    sleep = asyncio.sleep
    pauses = []

    async def advance(seconds):
        pauses.append(seconds)
        clock[0] += seconds
        await sleep(0)

    monkeypatch.setattr("semantic_image_stream.engine.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("semantic_image_stream.engine.asyncio.sleep", advance)
    segments = [TranscriptSegment(0, 6, "The tents remain beside the lake.")]
    topic = Topic("lake tents", "The camp remains beside the lake.", scene="lake tents")
    checkpoint = {"cursor": 1}
    state = SessionState(segments=segments, topic=topic, analyzer_checkpoint=checkpoint,
                         playback_generation=37)
    events = []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), relevance_threshold=.4),
                        lambda *event: events.append(event), state)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((pipeline._generation, topic))
    pipeline._displayed_urls.add("https://images.example/already-shown.jpg")
    pipeline._recent_hashes.add("0" * 16)
    resources = (object(), object(), object(), object())
    calls = []

    async def search(*args):
        calls.append((clock[0], args))
        if len(calls) == 1:
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "exception":
                raise OSError("private input must not appear in recovery diagnostics")
            return
        pipeline.stop_event.set()

    monkeypatch.setattr(pipeline, "_search", search)
    await asyncio.wait_for(pipeline._supervise_image_search(*resources), 2)
    assert len(calls) == 2 and calls[1][0] - calls[0][0] >= 30
    assert all(args == resources for _, args in calls) and pauses
    assert state.segments is segments and state.topic is topic
    assert state.analyzer_checkpoint is checkpoint and state.playback_generation == 37
    assert pipeline._displayed_urls == {"https://images.example/already-shown.jpg"}
    assert pipeline._recent_hashes == {"0" * 16} and pipeline.settings.relevance_threshold == .4
    assert len(events) == 1 and events[0][0] == "retrieval"
    assert "private input" not in str(events)


@pytest.mark.asyncio
async def test_intentional_stop_does_not_restart_cancelled_search(tmp_path, monkeypatch):
    events, calls = [], []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path)), lambda *event: events.append(event))

    async def search(*args):
        calls.append(1)
        pipeline.stop_event.set()
        raise asyncio.CancelledError

    monkeypatch.setattr(pipeline, "_search", search)
    with pytest.raises(asyncio.CancelledError):
        await pipeline._supervise_image_search([], None, None, None)
    assert calls == [1] and not events


def image(query):
    return ImageCandidate(f"https://images.example/{query}.jpg", "https://source.example/item",
                          query, "wikimedia", phash="f" * 16)


class Downloader:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    async def download(self, candidate):
        path = self.directory / (candidate.title + ".png")
        Image.new("RGB", (120, 80), "green").save(path)
        return replace(candidate, local_path=str(path), width=120, height=80)


@pytest.mark.asyncio
async def test_repeated_scenes_during_slow_ranking_yield_to_current_source_query(tmp_path, monkeypatch):
    events, queries, rankings = [], [], []
    topics = [Topic(word, word, scene=word, visual_subjects=(word,)) for word in ("camp", "cave", "coast")]
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache"), search_interval=.001,
                                 prefetch_count=1, min_width=32, min_height=24, relevance_threshold=.4),
                        lambda *args: None)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline._latest_topic = pipeline.session_state.topic = topics[0]
    pipeline._update_playback_scope(topics[0])
    pipeline.topic_queue.put_nowait((0, topics[0]))
    pipeline._displayed_urls.add("https://images.example/old.jpg")
    pipeline._recent_hashes.add("0" * 16)
    cache = Cache(pipeline.settings)
    pipeline._cache = cache
    loop = asyncio.get_running_loop()
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: [])

    def change_scene(topic):
        pipeline._latest_topic = pipeline.session_state.topic = topic
        pipeline._generation += 1
        pipeline._update_playback_scope(topic)
        while not pipeline.topic_queue.empty():
            pipeline.topic_queue.get_nowait()
        pipeline.topic_queue.put_nowait((pipeline._generation, topic))

    def receive(kind, payload):
        events.append((kind, payload))
        if kind == "images":
            pipeline.record_displayed(payload[0])
            pipeline.stop_event.set()

    pipeline.on_event = receive

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            queries.append(query)
            return [image(query)]

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            rankings.append(topic.query)
            if len(rankings) < 3:
                loop.call_soon_threadsafe(change_scene, topics[len(rankings)])
            return [replace(candidate, score=.9) for candidate in images]

    try:
        await asyncio.wait_for(pipeline._search([Provider()], Downloader(tmp_path / "pixels"), Ranker(), cache), 3)
        assert queries == ["camp", "coast"] and rankings == ["camp", "cave", "coast"]
        published = [candidate for kind, payload in events if kind == "images" for candidate in payload]
        assert [candidate.title for candidate in published] == ["coast"]
        assert pipeline._latest_topic is topics[-1] and pipeline.settings.relevance_threshold == .4
        assert "https://images.example/old.jpg" in pipeline._displayed_urls
        assert cache.seen_urls() == {image("coast").image_url}
    finally:
        pipeline.stop_event.set()
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_stop_winning_a_slow_rank_cannot_publish_late_images(tmp_path):
    entered, release = threading.Event(), threading.Event()
    events = []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache")), lambda *args: events.append(args))
    topic = Topic("camp", "A camp", scene="camp")
    pipeline._latest_topic = topic
    cache = Cache(pipeline.settings)

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            entered.set()
            assert release.wait(2)
            return [replace(candidate, score=.9) for candidate in images]

    task = asyncio.create_task(pipeline._evaluate_candidates(topic, [image("camp")], None, None,
                              Downloader(tmp_path / "pixels"), Ranker(), cache, {}, Counter(),
                              {"found": 0, "downloaded": 0, "best_score": None}))
    try:
        while not entered.is_set():
            await asyncio.sleep(.005)
        pipeline.stop_event.set()
        release.set()
        assert await asyncio.wait_for(task, 2) == (False, False)
        assert not any(kind == "images" for kind, _ in events) and not pipeline._approved_images
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cache.close()


@pytest.mark.asyncio
async def test_later_grounded_queries_are_tried_after_exhausted_guided_alternatives(tmp_path, monkeypatch):
    queries, published = [], []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache"), search_interval=.001,
                                 prefetch_count=1, min_width=32, min_height=24, relevance_threshold=.4,
                                 book_context="Arctic landscapes, tundra, rocky coastline."), lambda *args: None)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    topic = Topic("door", "A door stands near a cabin in the Arctic.", scene="Arctic door cabin",
                  visual_subjects=("door", "cabin"), queries=("door", "cabin"))
    pipeline._latest_topic = topic
    pipeline.topic_queue.put_nowait((0, topic))
    cache = Cache(pipeline.settings)
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: ["cabin"])

    async def retry(generation, topic):
        assert len(queries) <= 5, "A later grounded query must not remain starved"
        pipeline.topic_queue.put_nowait((generation, topic))

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)

    def receive(kind, payload):
        if kind == "images":
            published.extend(payload)
            pipeline.stop_event.set()

    pipeline.on_event = receive

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return await self.search_page(query, limit, 0)

        async def search_page(self, query, limit, page):
            queries.append(query)
            return [image(query)] if query == "cabin Arctic landscapes" else []

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            return [replace(candidate, score=.9) for candidate in images]

    try:
        await asyncio.wait_for(pipeline._search([Provider()], Downloader(tmp_path / "pixels"), Ranker(), cache), 3)
        assert queries == ["door Arctic landscapes", "door tundra", "Arctic landscapes",
                           "door Arctic landscapes", "cabin Arctic landscapes"]
        assert [candidate.title for candidate in published] == ["cabin Arctic landscapes"]
        assert pipeline._latest_topic is topic and pipeline.settings.relevance_threshold == .4
        assert all(len(query.split()) <= 8 for query in queries)
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_retained_scene_recovers_after_six_hours_of_no_unseen_source_results(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("semantic_image_stream.engine.time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr("semantic_image_stream.engine.fallback_queries", lambda topic: [])
    state = SessionState(segments=[TranscriptSegment(0, 6, "The camp remains beside the coast.")],
                         topic=Topic("camp", "The camp remains beside the coast.", scene="coastal camp"))
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path / "cache"), search_interval=.001, prefetch_count=1,
                                 min_width=32, min_height=24), lambda *args: None, state)
    pipeline.topic_queue = asyncio.Queue(maxsize=1)
    pipeline.topic_queue.put_nowait((0, state.topic))
    old = image("already-shown")
    old.phash = "0" * 16
    pipeline._displayed_urls.add(old.image_url)
    pipeline._recent_hashes.add(old.phash)
    cache = Cache(pipeline.settings)
    pipeline._cache = cache
    source_times, published = [], []

    async def retry(generation, topic):
        clock[0] += 300  # Advance elapsed wall time without waiting six real hours.
        assert clock[0] <= 6 * 3600 + 300, "Discovery must keep polling the retained scene"
        pipeline.topic_queue.put_nowait((generation, topic))

    monkeypatch.setattr(pipeline, "_wait_for_search_retry", retry)

    def receive(kind, payload):
        if kind == "images":
            published.extend(payload)
            pipeline.record_displayed(payload[0])
            pipeline.stop_event.set()

    pipeline.on_event = receive

    class Provider:
        name = "wikimedia"

        async def search(self, query, limit):
            return await self.search_page(query, limit, 0)

        async def search_page(self, query, limit, page):
            source_times.append(clock[0])
            return [image("fresh-camp")] if clock[0] >= 6 * 3600 else [old]

    class Ranker:
        name = "fixture"

        def rank(self, topic, images, recent):
            return [replace(candidate, score=.9) for candidate in images]

    try:
        await asyncio.wait_for(pipeline._search([Provider()], Downloader(tmp_path / "pixels"), Ranker(), cache), 5)
        assert len(source_times) > 60 and source_times[-1] == 6 * 3600
        assert [candidate.image_url for candidate in published] == [image("fresh-camp").image_url]
        assert pipeline.session_state is state and pipeline._latest_topic is state.topic
        assert len(state.segments) == 1 and old.image_url in pipeline._displayed_urls
        assert cache.seen_urls() == {image("fresh-camp").image_url}
    finally:
        pipeline._cache = None
        cache.close()


@pytest.mark.asyncio
async def test_context_publishes_each_accepted_batch_and_continues_without_new_speech(tmp_path):
    recorded = [TranscriptSegment(index * 6, (index + 1) * 6, f"Passage {index}: camp tents beside a lake.")
                for index in range(3)]
    updates = []
    pipeline = Pipeline(Settings(cache_dir=str(tmp_path), semantic_provider="ollama", use_story_examples=False,
                                 semantic_interval=.001), lambda *args: None, SessionState(segments=recorded))
    pipeline.topic_queue = asyncio.Queue(maxsize=1)

    class Analyzer:
        settings = pipeline.settings
        cursor = 0

        @property
        def has_pending_segments(self):
            return self.cursor < len(recorded)

        @property
        def context_used_segments(self):
            return self.cursor

        def analyze(self, *args):
            raise AssertionError("Monitoring must not await an entire backlog")

        def analyze_incremental(self, segments, stop_event):
            assert segments == recorded and not stop_event.is_set()
            self.cursor += 1
            return Topic("camp tents", "Camp tents beside a lake.", scene="camp tents lake",
                         visual_subjects=("camp tents",), context_used_segments=self.cursor)

    analyzer = Analyzer()
    pipeline.session_state.analyzer = analyzer

    def receive(kind, payload):
        if kind == "context" and payload["phase"] == "ready":
            updates.append((payload["segments_used"], payload["segments_recorded"], pipeline._latest_topic))
            if payload["segments_used"] == len(recorded):
                pipeline.stop_event.set()

    pipeline.on_event = receive
    await asyncio.wait_for(pipeline._interpret(analyzer), 2)
    assert [used for used, _, _ in updates] == [1, 2, 3]
    assert all(count == 3 and topic.transcript == tuple(segment.text for segment in recorded)
               for _, count, topic in updates)
    assert pipeline.session_state.segments is recorded and not analyzer.has_pending_segments

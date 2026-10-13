from dataclasses import replace
import json
import time

import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.engine import export_image
from semantic_image_stream.image_ranking import MetadataRanker, is_duplicate
from semantic_image_stream.models import ImageCandidate, Topic, TranscriptSegment
from semantic_image_stream.semantic_analysis import ContextualAnalyzer, TopicGate, sanitize_query


@pytest.mark.parametrize("text,expected,kind", [
    ("Black holes bend spacetime according to general relativity. The event horizon prevents light from escaping.", "black hole", "science"),
    ("During the French Revolution, Napoleon emerged as a military leader and transformed European politics.", "french revolution", "history"),
    ("The orchid flowers depend on pollination by insects. Plant roots exchange nutrients with mycorrhizal fungi.", "pollination", "science"),
    ("In this scientific interview we discuss how quantum experiments measure the entanglement between particles.", "quantum", "science"),
])
def test_context_topics(text, expected, kind):
    topic = ContextualAnalyzer().analyze([TranscriptSegment(0, 9, text)])
    assert expected in topic.query.casefold()
    assert topic.kind == kind
    assert len(topic.query.split()) <= 8
    assert topic.context == text


def test_named_relationships_and_transition():
    analyzer = ContextualAnalyzer()
    old = TranscriptSegment(0, 9, "The Apollo 11 mission carried Neil Armstrong and Buzz Aldrin to the moon landing.")
    astronomy = analyzer.analyze([old])
    botanical = analyzer.analyze([old, TranscriptSegment(30, 38, "Plant roots exchange nutrients with mycorrhizal fungi while orchids depend on pollination.")])
    assert "apollo" in astronomy.query.casefold()
    assert "pollination" in botanical.query
    assert "apollo" not in botanical.query.casefold()
    gate = TopicGate(0.45)
    assert gate.changed(astronomy)
    gate.accept(astronomy)
    assert not gate.changed(astronomy)
    assert gate.changed(botanical)


def test_isolated_words_do_not_trigger_search():
    assert ContextualAnalyzer().analyze([TranscriptSegment(0, 2, "apple")]) is None


def test_queries_remove_identifiers_and_operators():
    query = sanitize_query("email me@private.test https://private.test/secret C:\\secret\\file 415-555-9876 black hole site:example.com")
    assert "private" not in query
    assert "secret" not in query
    assert "9876" not in query
    assert ":" not in query
    assert len(query.split()) <= 8


def test_settings_roundtrip_and_validation(tmp_path):
    original = Settings(cache_dir=str(tmp_path / "cache"), providers=["wikimedia", "openverse"])
    path = tmp_path / "config.toml"
    original.save(path)
    assert Settings.load(path) == original
    for field, value in [("chunk_seconds", 100), ("audio_queue_size", -1), ("ranking", "invented"),
                         ("relevance_threshold", 1.1), ("providers", []), ("min_aspect_ratio", -1),
                         ("semantic_interval", float("nan")), ("retain_context", "true"),
                         ("session_dir", "invalid\x00folder")]:
        with pytest.raises(ValueError):
            replace(original, **{field: value}).validate()


def test_local_context_preferences_roundtrip_without_changing_audio_or_sources(tmp_path):
    original = Settings(retain_context=False, session_dir=str(tmp_path / "saved-story"), audio_device=151,
                        relevance_threshold=0.05, slideshow_interval=6, providers=["wikimedia", "duckduckgo", "archive"])
    path = tmp_path / "context-settings.toml"
    original.save(path)
    assert Settings.load(path) == original


def test_metadata_ranker_rejects_unrelated_and_duplicates():
    topic = Topic("black hole event horizon", "black holes curve spacetime", kind="science")
    relevant = ImageCandidate("https://img.example/1", "https://commons.wikimedia.org/wiki/File:test", "Black hole event horizon diagram", "wikimedia", width=1600, height=1000, phash="ffffffffffffffff")
    irrelevant = ImageCandidate("https://img.example/2", "https://commons.wikimedia.org/wiki/File:other", "Botanical garden trees", "wikimedia", width=4000, height=3000, phash="0000000000000000")
    assert MetadataRanker().rank(topic, [irrelevant, relevant], []) == [relevant]
    assert relevant.score >= 0.25
    assert MetadataRanker().rank(topic, [relevant], [relevant.phash]) == []
    assert is_duplicate("0000000000000001", ["0000000000000000"])
    assert not is_duplicate("ffffffffffffffff", ["0000000000000000"])


def test_cache_expiry_history_deletion_and_safe_paths(tmp_path):
    settings = Settings(cache_dir=str(tmp_path / "cache"), history_limit=2)
    cache = Cache(settings)
    outside = tmp_path / "leave-me.jpg"
    outside.write_bytes(b"outside")
    image_dir = cache.root / "images"
    image_dir.mkdir()
    local = image_dir / ("a" * 64 + ".jpg")
    local.write_bytes(b"cached")
    image = ImageCandidate("https://img.example/1", "https://source.example/1", "A title", "wikimedia", local_path=str(local))
    cache.put_query("key", [image])
    assert cache.get_query("key")[0].title == image.title
    with cache.db:
        cache.db.execute("UPDATE queries SET ts=?", (time.time() - 2 * 86400,))
    assert cache.get_query("key") is None
    cache.put_image(image)
    assert cache.get_image(image.image_url) is not None
    unsafe = replace(image, local_path=str(outside))
    cache.put_image(unsafe)
    assert cache.get_image(image.image_url) is None
    for _ in range(5):
        cache.record(image)
    assert len(cache.history()) == 2
    cache.clear()
    assert cache.history() == []
    assert not local.exists()
    assert outside.exists()
    cache.close()


def test_export_keeps_source_attribution(tmp_path):
    source = tmp_path / "cache.jpg"
    source.write_bytes(b"existing-image")
    candidate = ImageCandidate("https://img.example/a", "https://source.example/a", "A galaxy", "wikimedia",
                               creator="Astronomer", license="CC BY 4.0", local_path=str(source))
    saved, metadata = export_image(candidate, tmp_path / "saved.jpg")
    assert saved.read_bytes() == source.read_bytes()
    info = json.loads(metadata.read_text(encoding="utf-8"))
    assert info["source_url"] == candidate.source_url
    assert info["creator"] == candidate.creator
    assert info["license"] == candidate.license
    assert "local_path" not in info


def test_explicit_transition_drops_old_subject():
    text = "We discussed black holes, but now consider orchid pollination and how flowers attract insects in tropical forests."
    topic = ContextualAnalyzer().analyze([TranscriptSegment(0, 10, text)])
    assert "orchid" in topic.query
    assert "black" not in topic.query


def test_cache_expands_environment_path_at_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("SIS_TEST_CACHE", str(tmp_path))
    settings = Settings(cache_dir="$SIS_TEST_CACHE/cache").validate()
    assert settings.cache_dir == str((tmp_path / "cache").resolve())


def test_discovery_keyword_tags_do_not_make_unrelated_stock_relevant():
    topic = Topic("plant roots botany", "Plant roots exchange nutrients with fungi", kind="science")
    stock = ImageCandidate("https://img.example/w", "https://source.example/w", "Forest waterfall", "openverse",
                           description="plant roots botany photosynthesis science", width=1600, height=900)
    micrograph = ImageCandidate("https://img.example/r", "https://source.example/r", "Mature Buttercup Root", "openverse",
                                description="plant roots botany microscopy", width=1024, height=1024)
    ranked = MetadataRanker().rank(topic, [stock, micrograph], [])
    assert ranked == [micrograph]

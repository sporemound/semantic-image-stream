"""Regression coverage for stricter AI-generation metadata screening."""
from __future__ import annotations

import json
import time

import httpx
import pytest

from semantic_image_stream.cache import Cache
from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_download import DownloadError, ImageDownloader
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider,
    _generated_status,
    is_known_generated,
)


@pytest.mark.parametrize(
    "value",
    [
        "AI art portrait",
        "AIGenerated fantasy illustration",
        "Tags: midjourney; fantasy; portrait",
        "Tags: DreamUpAI; digital art",
        "Prompt: moonlit garden, CFG scale: 7",
        "FLUX.1 fantasy portrait",
        "NovelAI character art",
        "https://civitai.com/images/12345",
        "https://artist.example/art/ai-generated-garden-123",
    ],
)
def test_positive_generation_disclosures_are_detected(value):
    assert is_known_generated(value)


@pytest.mark.parametrize(
    "value",
    [
        "Stable Diffusion architecture",
        "Midjourney research results",
        "Human-drawn diagram of Stable Diffusion architecture",
        "Hand-drawn diagram explaining AI-generated image workflows",
        "Not AI-generated photograph",
        "AI-generated: false",
        "Adobe Firefly tutorial screenshot",
        "Traditional watercolor botanical study",
    ],
)
def test_discussion_negation_and_human_work_are_not_mislabeled(value):
    assert not is_known_generated(value)


@pytest.mark.parametrize(
    "row",
    [
        {"url": "https://civitai.com/images/123"},
        {"image": "https://artist.example/ai-generated-forest.jpg"},
        {"tags": [{"name": "midjourney"}, {"name": "fantasy"}]},
        {"source": "Tags: stable diffusion; landscape"},
        {"description": "Rendered with ComfyUI"},
    ],
)
def test_row_metadata_is_screened(row):
    assert _generated_status(row) is True


def test_explicit_negative_flag_remains_distinct():
    assert _generated_status({"is_ai_generated": False}, "ordinary photograph") is False
    assert _generated_status({}, "ordinary photograph") is None


@pytest.mark.asyncio
async def test_web_provider_excludes_generation_host(tmp_path):
    class FakeDDGS:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            return [{
                "title": "Moon garden",
                "image": "https://images.example.org/moon.jpg",
                "url": "https://civitai.com/images/123",
                "source": "Example",
                "width": 1200,
                "height": 800,
            }]

    settings = Settings(cache_dir=str(tmp_path), exclude_ai_generated=True)
    provider = DuckDuckGoProvider(settings, ddgs_factory=FakeDDGS)
    try:
        assert await provider.search("moon garden", 1) == []
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_download_boundary_rejects_generation_url_before_network(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    settings = Settings(
        cache_dir=str(tmp_path),
        min_width=32,
        min_height=24,
        exclude_ai_generated=True,
    )
    candidate = ImageCandidate(
        "https://images.example.org/moon.jpg",
        "https://civitai.com/images/123",
        "Moon garden",
        "duckduckgo",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        downloader = ImageDownloader(settings, client=client)
        with pytest.raises(DownloadError, match="Source metadata"):
            await downloader.download(candidate)
    assert calls == []


def test_query_cache_namespace_ignores_old_detector_results(tmp_path):
    settings = Settings(cache_dir=str(tmp_path))
    cache = Cache(settings)
    image = ImageCandidate(
        "https://images.example/photo.jpg",
        "https://source.example/photo",
        "Photo",
        "wikimedia",
    )
    try:
        old_payload = json.dumps([image.to_dict()])
        with cache.lock, cache.db:
            cache.db.execute(
                "INSERT OR REPLACE INTO queries VALUES (?,?,?)",
                ("old-key", time.time(), old_payload),
            )
        assert cache.get_query("old-key") is None
        cache.put_query("new-key", [image])
        restored = cache.get_query("new-key")
        assert restored is not None
        assert restored[0].image_url == image.image_url
    finally:
        cache.close()

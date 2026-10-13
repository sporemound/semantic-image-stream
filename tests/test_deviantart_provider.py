"""DeviantArt metadata and resilient DDGS image discovery."""
import asyncio

import httpx
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.deviantart_provider import DeviantArtProvider
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider,
    ProviderError,
    _duckduckgo_client_factory,
    build_providers,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path), min_width=32, min_height=24)


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_deviantart_preserves_artist_source_rights_and_largest_image(settings):
    body = b"""<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"
         xmlns:media="http://search.yahoo.com/mrss/"
         xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel><item>
        <title>Ghost Pipe at Midnight</title>
        <link>https://www.deviantart.com/moonlitmoss/art/Ghost-Pipe-at-Midnight-123</link>
        <dc:creator>MoonlitMoss</dc:creator>
        <description><![CDATA[<p>A moonlit botanical study.</p>]]></description>
        <category>botanical art</category>
        <media:copyright>Copyright 2026 MoonlitMoss</media:copyright>
        <media:thumbnail url="https://images.example.org/small.jpg" width="300" height="200" />
        <media:content url="https://images.example.org/full.jpg"
                       type="image/jpeg" medium="image" width="1600" height="1200" />
      </item></channel>
    </rss>"""

    def handler(request):
        assert request.url.path == "/rss.xml"
        assert request.url.params["type"] == "deviation"
        assert request.url.params["q"] == "ghost pipe sort:time meta:all"
        assert "application/rss+xml" in request.headers["accept"]
        return httpx.Response(200, content=body)

    async with client(handler) as http:
        result = (await DeviantArtProvider(settings, client=http).search("ghost pipe", 2))[0]
    assert result.image_url.endswith("full.jpg")
    assert result.source_url.endswith("Ghost-Pipe-at-Midnight-123")
    assert result.title == "Ghost Pipe at Midnight"
    assert result.creator == "MoonlitMoss"
    assert result.provider == "deviantart"
    assert result.license == "Copyright 2026 MoonlitMoss"
    assert result.description == "A moonlit botanical study."
    assert (result.width, result.height) == (1600, 1200)


@pytest.mark.asyncio
async def test_deviantart_rejects_malformed_or_entity_xml(settings):
    for body in (b"<rss><broken>", b'<!DOCTYPE rss [<!ENTITY x "bad">]><rss />'):
        async with client(lambda _: httpx.Response(200, content=body)) as http:
            with pytest.raises(ProviderError):
                await DeviantArtProvider(settings, client=http).search("botanical", 1)


@pytest.mark.asyncio
async def test_deviantart_is_built_in_selected_priority(settings):
    settings.providers = ["deviantart", "wikimedia"]
    providers = build_providers(settings)
    try:
        assert [provider.name for provider in providers] == ["deviantart", "wikimedia"]
    finally:
        await asyncio.gather(*(provider.close() for provider in providers))


@pytest.mark.asyncio
async def test_duckduckgo_failure_falls_back_to_bing_with_credit(settings, monkeypatch):
    ddgs = pytest.importorskip("ddgs.ddgs")
    calls = []

    def images(self, query, **kwargs):
        calls.append(kwargs["backend"])
        if kwargs["backend"] == "duckduckgo":
            raise RuntimeError("controlled DuckDuckGo failure")
        return [{
            "title": "Coastal cave",
            "image": "https://images.example.org/cave.jpg",
            "url": "https://source.example.org/cave",
            "source": "Bing",
            "width": 1200,
            "height": 800,
        }]

    monkeypatch.setattr(ddgs.DDGS, "images", images)
    provider = DuckDuckGoProvider(settings, ddgs_factory=_duckduckgo_client_factory())
    result = (await provider.search("coastal cave", 1))[0]
    assert calls == ["duckduckgo", "bing"]
    assert result.provider == "Bing Images"
    assert result.title == "Coastal cave"


@pytest.mark.asyncio
async def test_empty_duckduckgo_batch_also_uses_bing(settings, monkeypatch):
    ddgs = pytest.importorskip("ddgs.ddgs")
    calls = []

    def images(self, query, **kwargs):
        calls.append(kwargs["backend"])
        if kwargs["backend"] == "duckduckgo":
            return []
        return [{
            "title": "Botanical plate",
            "image": "https://images.example.org/plate.jpg",
            "url": "https://source.example.org/plate",
        }]

    monkeypatch.setattr(ddgs.DDGS, "images", images)
    provider = DuckDuckGoProvider(settings, ddgs_factory=_duckduckgo_client_factory())
    result = (await provider.search("botanical plate", 1))[0]
    assert calls == ["duckduckgo", "bing"]
    assert result.provider == "Bing Images"

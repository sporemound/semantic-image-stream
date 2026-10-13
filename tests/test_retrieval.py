"""Offline provider/API fixtures and hostile-image boundary tests."""
import asyncio
import io
from pathlib import Path
import threading

import httpx
from PIL import Image, PngImagePlugin
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_download import DownloadError, ImageDownloader, validate_public_https
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider, InternetArchiveProvider, OpenverseProvider, ProviderError,
    WikimediaProvider, build_providers, plain_text, sanitize_query,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path), min_width=32, min_height=24)


async def public_dns(host, port):
    return ["93.184.216.34"]


def png(width=120, height=80, text=None):
    output = io.BytesIO()
    image = Image.new("RGB", (width, height), (30, 100, 60))
    info = PngImagePlugin.PngInfo()
    if text is not None:
        info.add_text("Description", text)
    image.save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def candidate(url="https://images.example.org/photo.png"):
    return ImageCandidate(image_url=url, source_url="https://example.org/source",
                          title="Botanical plate", provider="wikimedia")


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_attribution_strips_executable_markup_and_query_disclosures():
    assert plain_text('<b>Alice</b> &amp; Bob<script>secret()</script><style>bad</style>') == "Alice & Bob"
    assert plain_text('&amp;lt;b&amp;gt;Alice&amp;lt;/b&amp;gt;') == "Alice"
    assert plain_text({"unexpected": "schema"}, "Unknown") == "Unknown"
    assert sanitize_query("<b>Hubble</b> alice@example.org https://private.example.org?q=1 stars\x00") == "Hubble stars"
    assert len(sanitize_query("astronomy " * 100)) <= 180


@pytest.mark.asyncio
async def test_commons_real_api_params_preserve_source_license_and_skip_generated(settings):
    def handler(request):
        assert request.url.path == "/w/api.php"
        assert request.url.params["generator"] == "search"
        assert request.url.params["gsrnamespace"] == "6"
        assert request.url.params["iiprop"] == "url|size|mime|thumbmime|extmetadata"
        row = {"title": "File:Hubble galaxy.jpg", "imageinfo": [{
            "url": "https://upload.wikimedia.org/original.jpg", "mime": "image/jpeg",
            "thumburl": "https://upload.wikimedia.org/thumb.jpg", "thumbmime": "image/jpeg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Hubble_galaxy.jpg",
            "width": 4000, "height": 3000, "thumbwidth": 1600, "thumbheight": 1200,
            "extmetadata": {"Artist": {"value": "<a>NASA</a>"},
                            "LicenseShortName": {"value": "Public domain"},
                            "LicenseUrl": {"value": "https://creativecommons.org/publicdomain/mark/1.0/"},
                            "ImageDescription": {"value": "<p>Galaxy &amp; stars</p>"}},
        }]}
        ai = {"title": "File:AI-generated galaxy.jpg", "imageinfo": row["imageinfo"]}
        return httpx.Response(200, json={"query": {"pages": [None, "bad", {}, row, ai]}})
    async with client(handler) as http:
        results = await WikimediaProvider(settings, client=http).search("Hubble galaxy", 5)
    assert len(results) == 1
    result = results[0]
    assert result.image_url.endswith("thumb.jpg")
    assert result.creator == "NASA"
    assert result.license == "Public domain"
    assert result.description == "Galaxy & stars"
    assert (result.width, result.height) == (1600, 1200)
    assert "File:Hubble_galaxy" in result.source_url
    assert result.authenticity == "Unverified source imagery"


@pytest.mark.asyncio
async def test_commons_legacy_pages_and_rasterized_diagram(settings):
    async with client(lambda r: httpx.Response(200, json={"query": {"pages": {"9": {
        "title": "File:Orbital diagram.svg", "imageinfo": [{
            "url": "https://upload.wikimedia.org/diagram.svg", "mime": "image/svg+xml",
            "thumburl": "https://upload.wikimedia.org/diagram.png", "thumbmime": "image/png",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Orbital_diagram.svg",
            "extmetadata": {"Artist": [], "LicenseShortName": {"value": "CC BY 4.0"}},
        }],
    }}}})) as http:
        result = (await WikimediaProvider(settings, client=http).search("orbit", 1))[0]
    assert result.image_url.endswith("png")
    assert result.creator == "Unknown"


@pytest.mark.asyncio
async def test_commons_excludes_document_and_video_previews(settings):
    pages = [{"title": f"File:Botany.{extension}", "imageinfo": [{
        "url": f"https://upload.wikimedia.org/botany.{extension}", "mime": mime,
        "thumburl": "https://upload.wikimedia.org/preview.png", "thumbmime": "image/png",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:Botany.{extension}",
    }]} for extension, mime in [("pdf", "application/pdf"), ("djvu", "image/vnd.djvu"), ("webm", "video/webm")]]
    # DjVu identifies as an image MIME despite representing scanned documents.
    async with client(lambda r: httpx.Response(200, json={"query": {"pages": pages}})) as http:
        assert await WikimediaProvider(settings, client=http).search("botany", 3) == []


@pytest.mark.asyncio
async def test_openverse_handles_bad_fields_flag_and_attribution(settings):
    good = {"title": "<b>Oak tree</b>", "url": "https://photos.example.org/oak.jpg",
            "foreign_landing_url": "https://museum.example.org/oak", "creator": "<p>Jane Doe</p>",
            "license": "by-sa", "license_version": "4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "width": "1024", "height": {"bad": "value"}, "tags": [{"name": "botany"}, None]}
    async with client(lambda r: httpx.Response(200, json={"results": [None, {}, good,
        dict(good, is_ai_generated=True), dict(good, tags=[{"name": "AI-generated"}]),
        dict(good, url="file:///secret.png"), dict(good, foreign_landing_url="javascript:evil()")]})) as http:
        results = await OpenverseProvider(settings, client=http).search("oak tree", 10)
    assert len(results) == 1
    assert results[0].creator == "Jane Doe"
    assert results[0].license == "BY-SA 4.0"
    assert (results[0].width, results[0].height) == (1024, 0)
    assert results[0].description == "botany"


@pytest.mark.asyncio
async def test_archive_searches_items_and_downloads_original_image_files(settings):
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path == "/advancedsearch.php":
            assert request.url.params["q"] == '("Apollo" "11") AND mediatype:image'
            return httpx.Response(200, json={"response": {"docs": [None, {"identifier": "../bad"},
                                                {"identifier": "apollo-11"}, {"identifier": "broken"}]}})
        if request.url.path.endswith("broken"):
            return httpx.Response(503)
        return httpx.Response(200, json={
            "metadata": {"title": "Apollo 11", "creator": ["NASA", "National Archives"],
                         "description": "<p>Moon landing</p>", "rights": "Check collection rights"},
            "files": [None, {"name": "original plate.jpg", "source": "original", "width": "2000", "height": "1600"},
                      {"name": "__ia_thumb.jpg", "source": "original"}, {"name": "thumbnail.png", "source": "derivative"},
                      {"name": "text.pdf", "source": "original"}, {"name": {}, "source": "original"}],
        })
    async with client(handler) as http:
        results = await InternetArchiveProvider(settings, client=http).search("Apollo 11", 5)
    assert len(results) == 1
    result = results[0]
    assert result.image_url == "https://archive.org/download/apollo-11/original%20plate.jpg"
    assert result.source_url == "https://archive.org/details/apollo-11"
    assert result.creator == "NASA; National Archives"
    assert result.license == "Check collection rights"
    assert result.authenticity == "Unverified source imagery"
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_archive_rejects_generated_item_metadata(settings):
    provider = InternetArchiveProvider(settings)
    try:
        data = {"metadata": {"title": "World War I photo", "subject": ["AI-generated"]},
                "files": [{"name": "photo.jpg", "source": "original"}]}
        assert provider._parse_item("wwi", data) == []
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls", [WikimediaProvider, OpenverseProvider, InternetArchiveProvider])
@pytest.mark.parametrize("status", [403, 429, 503])
async def test_provider_failures_surface_without_bypass_or_retries(settings, provider_cls, status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status)
    async with client(handler) as http:
        with pytest.raises(ProviderError):
            await provider_cls(settings, client=http).search("astronomy", 2)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls", [WikimediaProvider, OpenverseProvider, InternetArchiveProvider])
async def test_invalid_json_metadata_is_a_provider_error(settings, provider_cls):
    async with client(lambda r: httpx.Response(200, content=b"not JSON")) as http:
        with pytest.raises(ProviderError):
            await provider_cls(settings, client=http).search("history", 3)


@pytest.mark.asyncio
async def test_ddgs_uses_only_explicit_duckduckgo_backend(settings):
    seen = []
    class FakeDDGS:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == settings.download_timeout
        def images(self, query, **kwargs):
            seen.append((query, kwargs))
            return [None, {"title": "Earth", "image": "https://photos.example.org/earth.jpg",
                           "url": "https://museum.example.org/earth", "height": 600, "width": 800},
                    {"title": "AI generated Earth", "image": "https://photos.example.org/ai.jpg",
                     "url": "https://museum.example.org/ai"}]
    results = await DuckDuckGoProvider(settings, ddgs_factory=FakeDDGS).search("Earth", 3)
    assert len(results) == 1
    assert seen[0][1]["backend"] == "duckduckgo"
    assert results[0].license.startswith("Unknown")
    assert results[0].creator == "Unknown"


@pytest.mark.asyncio
async def test_ddgs_deadline_prevents_accumulated_uncancellable_workers(settings):
    settings.download_timeout = 0.02
    release = threading.Event()
    finished = threading.Event()
    starts = []

    class SlowDDGS:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            starts.append(threading.current_thread())
            try:
                release.wait(timeout=2)
                return []
            finally:
                finished.set()

    provider = DuckDuckGoProvider(settings, ddgs_factory=SlowDDGS)
    try:
        with pytest.raises(ProviderError, match="TimeoutError"):
            await provider.search("black hole", 1)
        with pytest.raises(ProviderError, match="still finishing"):
            await provider.search("orchid", 1)
        assert len(starts) == 1
        assert starts[0].daemon is True
    finally:
        release.set()
        assert finished.wait(timeout=1)


@pytest.mark.asyncio
async def test_build_providers_preserves_selected_priority(settings):
    settings.providers = ["openverse", "wikimedia", "archive", "duckduckgo", "wikimedia"]
    providers = build_providers(settings)
    assert [provider.name for provider in providers] == ["openverse", "wikimedia", "archive", "duckduckgo"]
    await asyncio.gather(*(provider.close() for provider in providers))


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://images.example.org/image.jpg", "file:///secret.png", "https://localhost/image.png",
    "https://127.0.0.1/image.png", "https://10.0.0.1/image.png", "https://169.254.169.254/metadata",
    "https://[::1]/image.png", "https://[::ffff:127.0.0.1]/image.png", "https://[fc00::1]/image.png",
    "https://user:password@images.example.org/image.png", "https://images.example.org:8443/image.png",
    "https://photos.local/image.png", "https://printer/image.png", "https://images.example.org\\@127.0.0.1/image.png",
    "https://images.example.org/\x00.jpg",
])
async def test_public_https_destination_rejects_unsafe_authorities(url):
    with pytest.raises(DownloadError):
        await validate_public_https(url, public_dns)


@pytest.mark.asyncio
@pytest.mark.parametrize("addresses", [["127.0.0.1"], ["93.184.216.34", "10.0.0.1"], [], ["fe80::1"], ["not-ip"]])
async def test_dns_rejects_any_non_public_answer(addresses):
    async def resolver(host, port):
        return addresses
    with pytest.raises(DownloadError):
        await validate_public_https("https://images.example.org/test.jpg", resolver)


@pytest.mark.asyncio
async def test_redirect_checked_before_request_to_private_destination(settings):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/secret.jpg"})
    async with client(handler) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError):
            await downloader.download(candidate())
    assert len(seen) == 1
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
async def test_verified_download_measures_pixels_hashes_and_strips_metadata(settings):
    async with client(lambda r: httpx.Response(200, content=png(text="NASA source plate"),
                                             headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        item = candidate()
        item.width, item.height = 9000, 9000  # Do not trust provider dimensions.
        result = await downloader.download(item)
    assert result is item
    assert (result.width, result.height) == (120, 80)
    assert len(result.phash) == 16
    assert Path(result.local_path).parent == downloader.directory
    with Image.open(result.local_path) as image:
        image.load()
        assert "Description" not in image.info
    assert len(list(downloader.directory.iterdir())) == 1
    assert result.source_url == "https://example.org/source"


@pytest.mark.asyncio
@pytest.mark.parametrize("content,mime", [(b"<html>not an image</html>", "image/png"),
                                         (png(), "text/html"), (png(), "image/svg+xml"),
                                         (b"", "image/jpeg")])
async def test_non_images_never_enter_cache(settings, content, mime):
    async with client(lambda r: httpx.Response(200, content=content, headers={"content-type": mime})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("max_image_pixels", 100), ("min_width", 500),
                                         ("min_height", 500), ("max_aspect_ratio", 1.1),
                                         ("min_aspect_ratio", 2)])
async def test_actual_resolution_aspect_and_pixel_limits(settings, field, value):
    setattr(settings, field, value)
    async with client(lambda r: httpx.Response(200, content=png(), headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks, self.delay = chunks, delay
    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [None, "1", "100000", "not-integer", "-1"])
async def test_size_limit_including_missing_and_dishonest_length(settings, length):
    settings.max_download_bytes = 100
    headers = {"content-type": "image/png"}
    if length is not None:
        headers["content-length"] = length
    async with client(lambda r: httpx.Response(200, stream=ChunkStream([b"x" * 80, b"y" * 80]), headers=headers)) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
async def test_compressed_http_body_rejected_before_decompression(settings):
    async with client(lambda r: httpx.Response(200, stream=ChunkStream([b"compressed bomb"]),
                                             headers={"content-type": "image/png", "content-encoding": "gzip"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError, match="Compressed HTTP"):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
async def test_download_entire_deadline_stops_slow_body(settings):
    settings.download_timeout = 0.03
    async with client(lambda r: httpx.Response(200, stream=ChunkStream([b"x", b"y"], delay=0.02),
                                             headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError, match="TimeoutError"):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
async def test_download_deadline_includes_dns(settings):
    settings.download_timeout = 0.01
    async def slow_dns(host, port):
        await asyncio.sleep(0.05)
        return ["93.184.216.34"]
    calls = []
    async with client(lambda r: calls.append(r) or httpx.Response(200)) as http:
        downloader = ImageDownloader(settings, client=http, resolver=slow_dns)
        with pytest.raises(DownloadError, match="TimeoutError"):
            await downloader.download(candidate())
    assert calls == []


@pytest.mark.asyncio
async def test_redirect_limit_and_revalidation(settings):
    seen, resolved = [], []
    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://images.example.org/another.jpg"})
    async def resolver(host, port):
        resolved.append(host)
        return ["93.184.216.34"]
    async with client(handler) as http:
        downloader = ImageDownloader(settings, client=http, resolver=resolver)
        with pytest.raises(DownloadError, match="redirected"):
            await downloader.download(candidate())
    assert len(seen) == 4
    assert len(resolved) == 4


@pytest.mark.asyncio
async def test_generated_embedded_metadata_rejected(settings):
    async with client(lambda r: httpx.Response(200, content=png(text="AI-generated with Stable Diffusion"),
                                             headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError, match="Embedded metadata"):
            await downloader.download(candidate())
    assert list(downloader.directory.iterdir()) == []

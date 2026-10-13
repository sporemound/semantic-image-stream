"""Offline provider/API fixtures and hostile-image boundary tests."""
import asyncio
from datetime import datetime, timezone
from email.utils import format_datetime
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import httpx
from PIL import Image, PngImagePlugin
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.image_download import DownloadError, ImageDownloader, validate_public_https
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider, InternetArchiveProvider, OpenverseProvider, ProviderError,
    WikimediaProvider, _duckduckgo_client_factory, build_providers, is_known_generated, plain_text, sanitize_query,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path), min_width=32, min_height=24)


async def public_dns(host, port):
    return ["93.184.216.34"]


def png(width=120, height=80, text=None, text_key="Description"):
    output = io.BytesIO()
    image = Image.new("RGB", (width, height), (30, 100, 60))
    info = PngImagePlugin.PngInfo()
    if text is not None:
        info.add_text(text_key, text)
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
async def test_installed_ddgs_cleans_only_owned_image_request_headers(settings, monkeypatch):
    ddgs_module = pytest.importorskip("ddgs.ddgs")
    http_module = pytest.importorskip("ddgs.http_client")
    images_module = pytest.importorskip("ddgs.engines.duckduckgo_images")
    primp = pytest.importorskip("primp")
    native_class = primp.Client
    monkeypatch.delenv("DDGS_PROXY", raising=False)
    clients, requests = [], []

    class NativeBoundary:
        """Real native header handling; replace only the outbound HTTP boundary."""
        def __init__(self, **kwargs):
            self.options = kwargs
            self.native = native_class(**kwargs)
            self.identity = (self.native.impersonate, self.native.impersonate_os)
            clients.append(self)

        @property
        def headers(self):
            return self.native.headers

        @headers.setter
        def headers(self, value):
            self.native.headers = value

        def headers_update(self, value):
            self.native.headers_update(value)
            self.original_headers = self.native.headers

        def request(self, method, url, **kwargs):
            headers = self.native.headers
            assert not any(name.casefold() == "connection" for name in headers)
            assert headers == {name: value for name, value in self.original_headers.items()
                               if name.casefold() != "connection"}
            assert (self.native.impersonate, self.native.impersonate_os) == self.identity
            requests.append((self, method, url, kwargs))
            if url == "https://duckduckgo.com":
                content = b'<script>vqd="offline-test-token";</script>'
            elif url == "https://duckduckgo.com/i.js":
                content = json.dumps({"results": [{
                    "title": "Geologists investigate anomalies",
                    "image": "https://photos.example.org/field.jpg",
                    "url": "https://museum.example.org/field",
                    "width": 800, "height": 600, "source": "Example",
                }]}).encode()
            else:
                pytest.fail(f"Unexpected search backend: {url}")
            return SimpleNamespace(status_code=200, content=content, text=content.decode())

    monkeypatch.setattr(http_module.primp, "Client", NativeBoundary)
    default_headers = dict(images_module.DuckduckgoImages.headers_update)
    ordinary = ddgs_module.DDGS(timeout=settings.download_timeout)
    ordinary_client = ordinary._get_engines("images", "duckduckgo")[0].http_client.client
    ordinary_headers = ordinary_client.headers
    assert any(name.casefold() == "connection" for name in ordinary_headers)

    results = await DuckDuckGoProvider(settings).search("geologists anomalies", 1)

    assert len(results) == 1
    assert results[0].title == "Geologists investigate anomalies"
    assert results[0].image_url == "https://photos.example.org/field.jpg"
    assert results[0].source_url == "https://museum.example.org/field"
    assert results[0].provider == "duckduckgo"
    assert len(clients) == 2  # One untouched ordinary client, one owned adapter client.
    adapted = clients[1]
    assert adapted.options == {"proxy": None, "timeout": settings.download_timeout,
                               "impersonate": "random", "impersonate_os": "random",
                               "verify": True, "ca_cert_file": None}
    assert [(method, url) for _, method, url, _ in requests] == [
        ("GET", "https://duckduckgo.com"), ("GET", "https://duckduckgo.com/i.js"),
    ]
    assert all(owner is adapted for owner, *_ in requests)
    assert requests[0][3]["params"] == {"q": "geologists anomalies"}
    assert requests[1][3]["params"]["vqd"] == "offline-test-token"
    assert requests[1][3]["params"]["p"] == "1"
    assert ordinary_client.headers == ordinary_headers
    assert images_module.DuckduckgoImages.headers_update == default_headers


def test_ddgs_adapter_refuses_unrequested_backend(monkeypatch):
    ddgs_module = pytest.importorskip("ddgs.ddgs")
    monkeypatch.setattr(ddgs_module.DDGS, "_get_engines",
                        lambda *args: [SimpleNamespace(name="another-provider")])
    with pytest.raises(ProviderError, match="requested DuckDuckGo image backend"):
        _duckduckgo_client_factory()()._get_engines("images", "duckduckgo")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["wikimedia", "openverse", "archive", "duckduckgo"])
@pytest.mark.parametrize("exclude", [True, False])
async def test_ai_source_filter_is_optional_and_preserves_explicit_marker(settings, source, exclude):
    settings.exclude_ai_generated = exclude
    if source == "wikimedia":
        data = {"query": {"pages": [{"title": "File:Arctic camp.jpg", "imageinfo": [{
            "url": "https://images.example.org/camp.jpg", "mime": "image/jpeg",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Arctic_camp.jpg",
            "extmetadata": {"Categories": {"value": "AI-generated images"}},
        }]}]}}
        async with client(lambda _: httpx.Response(200, json=data)) as http:
            results = await WikimediaProvider(settings, client=http).search("Arctic camp", 2)
    elif source == "openverse":
        data = {"results": [{"title": "Arctic camp", "url": "https://images.example.org/camp.jpg",
                             "foreign_landing_url": "https://source.example.org/camp", "is_ai_generated": "TRUE"}]}
        async with client(lambda _: httpx.Response(200, json=data)) as http:
            results = await OpenverseProvider(settings, client=http).search("Arctic camp", 2)
    elif source == "archive":
        data = {"metadata": {"title": "Arctic camp", "subject": ["AI-generated"]},
                "files": [{"name": "original.jpg", "source": "original"}]}
        async with client(lambda _: httpx.Response(200, json={})) as http:
            results = InternetArchiveProvider(settings, client=http)._parse_item("camp", data)
    else:
        class DDGS:
            def __init__(self, **kwargs):
                pass

            def images(self, query, **kwargs):
                assert kwargs["backend"] == "duckduckgo"
                return [{"title": "AI generated Arctic camp", "image": "https://images.example.org/camp.jpg",
                         "url": "https://source.example.org/camp"}]

        results = await DuckDuckGoProvider(settings, ddgs_factory=DDGS).search("Arctic camp", 2)
    if exclude:
        assert results == []
    else:
        assert len(results) == 1 and results[0].ai_generated is True
        assert ImageCandidate.from_dict(results[0].to_dict()).ai_generated is True


@pytest.mark.asyncio
async def test_unmarked_and_explicit_negative_ai_metadata_remain_distinct(settings):
    base = {"title": "Arctic camp", "url": "https://images.example.org/camp.jpg",
            "foreign_landing_url": "https://source.example.org/camp"}
    data = {"results": [base, dict(base, is_ai_generated=False), dict(base, is_ai_generated="false"),
                        dict(base, is_ai_generated="unknown")]}
    async with client(lambda _: httpx.Response(200, json=data)) as http:
        results = await OpenverseProvider(settings, client=http).search("Arctic camp", 4)
    assert [image.ai_generated for image in results] == [None, False, False, None]


@pytest.mark.asyncio
async def test_archive_file_ai_markers_are_optional_without_admitting_derivatives(settings):
    data = {"metadata": {"title": "Arctic camp", "ai_generated": False}, "files": [
        {"name": "AI-generated camp.jpg", "source": "original"},
        {"name": "regular.jpg", "source": "original"},
        {"name": "__ia_thumb.jpg", "source": "original"},
        {"name": "AI-generated-thumbnail.jpg", "source": "derivative"},
    ]}
    async with client(lambda _: httpx.Response(200, json={})) as http:
        provider = InternetArchiveProvider(settings, client=http)
        assert [image.ai_generated for image in provider._parse_item("camp", data)] == [False]
        settings.exclude_ai_generated = False
        allowed = provider._parse_item("camp", data)
    assert len(allowed) == 2 and [image.ai_generated for image in allowed] == [True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["source_text", "source_flag", "embedded"])
async def test_allowed_generated_downloads_keep_marker_after_clean_reencoding_and_cache(settings, marker):
    from semantic_image_stream.cache import Cache

    settings.exclude_ai_generated = False
    image = candidate()
    if marker == "source_text":
        image.title = "AI-generated botanical plate"
    if marker == "source_flag":
        image.ai_generated = True
    body = png(text="AI-generated with Stable Diffusion" if marker == "embedded" else None)
    async with client(lambda _: httpx.Response(200, content=body, headers={"content-type": "image/png"})) as http:
        downloaded = await ImageDownloader(settings, client=http, resolver=public_dns).download(image)
    assert downloaded.ai_generated is True and downloaded.path.is_file()
    with Image.open(downloaded.path) as decoded:
        assert not decoded.info, "Re-encoded pixels omit source textual metadata"
    cache = Cache(settings)
    try:
        cache.put_image(downloaded)
        cache.put_query("ai-filter-off", [downloaded])
        assert cache.get_image(downloaded.image_url).ai_generated is True
        assert cache.get_query("ai-filter-off")[0].ai_generated is True
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_ai_source_flag_is_rejected_before_network_with_default_filter(settings):
    image = candidate()
    image.ai_generated = True
    calls = []
    async with client(lambda request: calls.append(request) or httpx.Response(200)) as http:
        with pytest.raises(DownloadError, match="Source metadata"):
            await ImageDownloader(settings, client=http, resolver=public_dns).download(image)
    assert calls == []


@pytest.mark.parametrize("text,generated", [
    ("AI-generated photograph of a camp", True),
    ("Created using Stable Diffusion", True),
    ("Generated by Midjourney", True),
    ("AI-generated diagram of diffusion architecture", True),
    ("Not AI-generated", False),
    ("This image is not an AI generated picture", False),
    ("Non-AI-generated photographs", False),
    ("AI-generated: false", False),
    ("Human-drawn diagram of Stable Diffusion architecture", False),
    ("Hand-drawn diagram explaining AI-generated image workflows", False),
    ("Stable Diffusion architecture", False),
    ("Midjourney research results", False),
])
def test_ai_labels_distinguish_creation_evidence_from_negation_and_discussion(text, generated):
    assert is_known_generated(text) is generated


@pytest.mark.asyncio
async def test_human_diagrams_about_ai_and_negative_labels_are_not_blocked(settings):
    base = {"url": "https://images.example.org/diagram.png", "foreign_landing_url": "https://source.example.org/diagram"}
    data = {"results": [dict(base, title="Human-drawn diagram of Stable Diffusion architecture",
                             description="Hand-drawn diagram explaining AI-generated image workflows",
                             tags=[{"name": "Stable Diffusion"}]),
                        dict(base, title="Scientific schematic", description="Not AI-generated",
                             tags=[{"name": "Non-AI-generated diagrams"}]),
                        dict(base, title="Not AI-generated", is_ai_generated=True)]}
    async with client(lambda _: httpx.Response(200, json=data)) as http:
        images = await OpenverseProvider(settings, client=http).search("diffusion architecture", 3)
    assert len(images) == 2 and all(image.ai_generated is None for image in images)


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [("Software", "Stable Diffusion 2.1"),
                                       ("workflow", '{"software":"ComfyUI","nodes":[]}')])
async def test_embedded_ai_software_and_workflow_markers_respect_optional_filter(settings, key, value):
    body = png(text=value, text_key=key)
    async with client(lambda _: httpx.Response(200, content=body, headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError, match="Embedded metadata"):
            await downloader.download(candidate())
        settings.exclude_ai_generated = False
        allowed = await downloader.download(candidate())
    assert allowed.ai_generated is True


@pytest.mark.asyncio
async def test_embedded_description_of_human_ai_diagram_is_not_generation_software(settings):
    body = png(text="Human-drawn diagram of Stable Diffusion architecture")
    async with client(lambda _: httpx.Response(200, content=body, headers={"content-type": "image/png"})) as http:
        image = await ImageDownloader(settings, client=http, resolver=public_dns).download(candidate())
    assert image.path.is_file() and image.ai_generated is None


@pytest.mark.asyncio
async def test_commons_separates_fossil_period_from_original_image_creation_date(settings):
    from semantic_image_stream.temporal_filter import filter_match

    data = {"query": {"pages": [{"title": "File:Jurassic ammonite fossil.jpg", "imageinfo": [{
        "url": "https://images.example.org/fossil.jpg", "mime": "image/jpeg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Fossil.jpg",
        "timestamp": "2025-04-03T00:00:00Z",
        "extmetadata": {"DateTimeOriginal": {"value": "2021-06-12 10:30:00"},
                        "ImageDescription": {"value": "Jurassic ammonite fossil."},
                        "Categories": {"value": "Jurassic fossils|Photographs taken in 2021"}},
    }]}]}}

    def handler(request):
        assert "DateTimeOriginal" in request.url.params["iiextmetadatafilter"].split("|")
        assert "timestamp" not in request.url.params["iiprop"].split("|")
        return httpx.Response(200, json=data)

    async with client(handler) as http:
        image = (await WikimediaProvider(settings, client=http).search("ammonite fossil", 1))[0]
    created = [row for row in image.temporal_evidence if row["target"] == "created"]
    depicted = [row for row in image.temporal_evidence if row["target"] == "depicted"]
    assert len(created) == 1 and created[0]["start"] == created[0]["end"] == 2021
    assert created[0]["source"] == "Commons DateTimeOriginal"
    assert depicted and all(row["end"] < -100_000_000 for row in depicted)
    settings.time_filter_enabled = True
    settings.time_filter_start, settings.time_filter_end = -200_000_000, -150_000_000
    settings.time_filter_target = "depicted"
    assert filter_match(settings, image).matched
    settings.time_filter_target = "created"
    assert not filter_match(settings, image).matched
    settings.time_filter_start, settings.time_filter_end = 2021, 2021
    assert filter_match(settings, image).matched
    assert ImageCandidate.from_dict(image.to_dict()).temporal_evidence == image.temporal_evidence


@pytest.mark.asyncio
async def test_commons_upload_timestamp_is_not_creation_or_depicted_date(settings):
    data = {"query": {"pages": [{"title": "File:Hubble galaxy.jpg", "imageinfo": [{
        "url": "https://images.example.org/galaxy.jpg", "mime": "image/jpeg",
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Galaxy.jpg",
        "timestamp": "1999-02-03T00:00:00Z",
        "extmetadata": {"DateTime": {"value": "1999-02-03"},
                        "ImageDescription": {"value": "Galaxy and stars"}},
    }]}]}}
    async with client(lambda _: httpx.Response(200, json=data)) as http:
        image = (await WikimediaProvider(settings, client=http).search("galaxy", 1))[0]
    assert image.temporal_evidence == []


@pytest.mark.asyncio
async def test_archive_original_item_date_is_separate_from_subject_and_ingest_dates(settings):
    data = {"metadata": {"title": "Ancient temple, 500 BCE", "description": "A reconstruction of a temple in 500 BCE.",
                         "date": "1900", "publicdate": "2010-01-02", "addeddate": "2008-04-01"},
            "files": [{"name": "plate.jpg", "source": "original"}]}
    async with client(lambda _: httpx.Response(200, json={})) as http:
        provider = InternetArchiveProvider(settings, client=http)
        image = provider._parse_item("temple", data)[0]
        without_original = dict(data, metadata={key: value for key, value in data["metadata"].items() if key != "date"})
        unknown_created = provider._parse_item("temple", without_original)[0]
    created = [row for row in image.temporal_evidence if row["target"] == "created"]
    depicted = [row for row in image.temporal_evidence if row["target"] == "depicted"]
    assert len(created) == 1 and created[0]["start"] == created[0]["end"] == 1900
    assert depicted and all(row["start"] == row["end"] == -499 for row in depicted)
    assert not any(row["target"] == "created" for row in unknown_created.temporal_evidence)


@pytest.mark.asyncio
async def test_openverse_index_dates_do_not_become_creation_evidence_or_request_parameters(settings):
    settings.time_filter_enabled = True
    settings.time_filter_target = "created"
    settings.time_filter_start, settings.time_filter_end = 1800, 1900
    data = {"results": [{"title": "Cretaceous dinosaur fossil", "url": "https://images.example.org/fossil.jpg",
                         "foreign_landing_url": "https://source.example.org/fossil", "created_on": "1865-01-01",
                         "indexed_on": "2024-01-01", "tags": [{"name": "geology"}]}]}

    def handler(request):
        assert dict(request.url.params) == {"q": "dinosaur fossil", "page_size": "1"}
        return httpx.Response(200, json=data)

    async with client(handler) as http:
        image = (await OpenverseProvider(settings, client=http).search("dinosaur fossil", 1))[0]
    assert image.temporal_evidence and all(row["target"] == "depicted" for row in image.temporal_evidence)


@pytest.mark.asyncio
async def test_duckduckgo_explicit_subject_period_has_no_invented_creation_date(settings):
    class DDGS:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            return [{"title": "Cambrian trilobite fossil", "image": "https://images.example.org/fossil.jpg",
                     "url": "https://source.example.org/fossil", "date": "2024-01-01"}]

    image = (await DuckDuckGoProvider(settings, ddgs_factory=DDGS).search("trilobite fossil", 1))[0]
    assert image.temporal_evidence and all(row["target"] == "depicted" for row in image.temporal_evidence)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_header, expected_seconds", [
    ("600", 600), ("999999999999999999999999", 3600), ("0", 1),
    ("not a date or delay", 60), (None, 60),
    (format_datetime(datetime.fromtimestamp(1_700_000_090, timezone.utc), usegmt=True), 90),
    (format_datetime(datetime.fromtimestamp(1_699_999_900, timezone.utc), usegmt=True), 1),
])
async def test_rate_limit_respects_retry_after_without_requesting_different_image(settings, retry_header, expected_seconds):
    monotonic_now = [100.0]
    requests, resolutions = [], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            headers = {"retry-after": retry_header} if retry_header is not None else {}
            return httpx.Response(429, headers=headers)
        return httpx.Response(200, content=png(), headers={"content-type": "image/png"})

    async def resolver(host, port):
        resolutions.append(host)
        return await public_dns(host, port)

    async with client(handler) as http:
        downloader = ImageDownloader(settings, client=http, resolver=resolver,
                                     clock=lambda: monotonic_now[0], wall_clock=lambda: 1_700_000_000)
        with pytest.raises(DownloadError) as first:
            await downloader.download(candidate("https://images.example.org/first.png"))
        assert first.value.reason == "Source temporarily rate-limited"
        assert first.value.retry_after == expected_seconds
        with pytest.raises(DownloadError) as second:
            await downloader.download(candidate("https://images.example.org/another-query-result.png"))
        assert second.value.reason == "Source temporarily rate-limited"
        assert second.value.retry_after == expected_seconds
        assert len(requests) == 1
        assert len(resolutions) == 1  # The cooldown also skips repeated DNS work.
        assert list(downloader.directory.iterdir()) == []
        monotonic_now[0] += expected_seconds + 0.01
        result = await downloader.download(candidate("https://images.example.org/after-cooldown.png"))
        assert result.path.is_file()
        assert len(requests) == 2
        assert len(resolutions) == 2


@pytest.mark.asyncio
async def test_rate_limit_blocks_only_rejecting_hostname(settings):
    requested = []

    def handler(request):
        requested.append(request.url.host)
        if request.url.host == "upload.wikimedia.org":
            return httpx.Response(429, headers={"retry-after": "600"})
        return httpx.Response(200, content=png(), headers={"content-type": "image/png"})

    async with client(handler) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError) as blocked:
            await downloader.download(candidate("https://upload.wikimedia.org/original.png"))
        assert blocked.value.reason == "Source temporarily rate-limited"
        result = await downloader.download(candidate("https://thumb.wikimedia.org/1920px-image.png"))
    assert result.path.is_file()
    assert requested == ["upload.wikimedia.org", "thumb.wikimedia.org"]


@pytest.mark.asyncio
async def test_redirect_rate_limit_is_assigned_to_response_origin(settings):
    requested = []

    def handler(request):
        requested.append(request.url.host)
        if request.url.host == "images.example.org":
            return httpx.Response(302, headers={"location": "https://cdn.example.org/photo.png"})
        return httpx.Response(429, headers={"retry-after": "600"})

    async with client(handler) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        for name in ["first.png", "second.png"]:
            with pytest.raises(DownloadError) as error:
                await downloader.download(candidate("https://images.example.org/" + name))
            assert error.value.reason == "Source temporarily rate-limited"
    # The second entry-point request is allowed. The blocked CDN is not fetched
    # again; a redirect does not transfer the restriction to unrelated hosts.
    assert requested == ["images.example.org", "cdn.example.org", "images.example.org"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reason", [(403, "Source inaccessible"), (404, "Broken image link"),
                                           (410, "Broken image link"), (503, "Source unavailable")])
async def test_download_errors_have_private_query_free_categories(settings, status, reason):
    async with client(lambda request: httpx.Response(status)) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError) as error:
            await downloader.download(candidate("https://images.example.org/image.png?query=private-query"))
    assert error.value.reason == reason
    assert "private-query" not in str(error.value)
    assert "https://" not in error.value.reason
    assert list(downloader.directory.iterdir()) == []


@pytest.mark.asyncio
async def test_resolution_rejection_has_specific_user_visible_reason(settings):
    settings.min_width = 500
    async with client(lambda request: httpx.Response(200, content=png(), headers={"content-type": "image/png"})) as http:
        downloader = ImageDownloader(settings, client=http, resolver=public_dns)
        with pytest.raises(DownloadError) as error:
            await downloader.download(candidate())
    assert error.value.reason == "Image below minimum resolution"

"""Offline pagination contracts for fetching new source results."""
import httpx
import pytest

from semantic_image_stream.configuration import Settings
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider, InternetArchiveProvider, OpenverseProvider, ProviderError, WikimediaProvider,
)


HTTP_PROVIDERS = (WikimediaProvider, OpenverseProvider, InternetArchiveProvider)


def commons_page(number):
    return {"title": f"File:Coastal cave {number}.jpg", "imageinfo": [{
        "url": f"https://photos.example.org/cave-{number}.jpg", "mime": "image/jpeg",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:Cave_{number}.jpg",
        "extmetadata": {"Artist": {"value": "Museum"}, "LicenseShortName": {"value": "CC BY 4.0"}},
    }]}


def openverse_row(number):
    return {"title": f"Coastal cave {number}", "url": f"https://photos.example.org/cave-{number}.jpg",
            "foreign_landing_url": f"https://museum.example.org/cave-{number}",
            "creator": "Museum", "license": "by", "license_version": "4.0"}


def ddgs_row(number):
    return {"title": f"Coastal cave {number}", "image": f"https://photos.example.org/cave-{number}.jpg",
            "url": f"https://museum.example.org/cave-{number}"}


@pytest.mark.asyncio
async def test_commons_pages_use_bounded_offsets_and_preserve_attribution(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        offset = int(request.url.params.get("gsroffset", 0))
        return httpx.Response(200, json={"query": {"pages": [commons_page(offset)]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = WikimediaProvider(Settings(cache_dir=str(tmp_path)), client=client)
        first = await provider.search("coastal cave", 5)
        second = await provider.search_page("coastal cave", 5, 1)
        last = await provider.search_page("coastal cave", 1000, 9)
    assert first[0].image_url != second[0].image_url
    assert second[0].license == "CC BY 4.0" and second[0].creator == "Museum"
    assert "gsroffset" not in requests[0].url.params
    assert requests[1].url.params["gsroffset"] == "5"
    assert requests[2].url.params["gsroffset"] == "450"
    assert requests[2].url.params["gsrlimit"] == "20"
    assert last[0].image_url.endswith("cave-450.jpg")


@pytest.mark.asyncio
async def test_openverse_pages_are_one_based_and_keep_first_search_contract(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        page = int(request.url.params.get("page", 1))
        return httpx.Response(200, json={"results": [openverse_row(page)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = OpenverseProvider(Settings(cache_dir=str(tmp_path)), client=client)
        first = await provider.search("coastal cave", 5)
        second = await provider.search_page("coastal cave", 5, 1)
        await provider.search_page("coastal cave", 99, 9)
    assert first[0].image_url != second[0].image_url
    assert "page" not in requests[0].url.params
    assert requests[1].url.params["page"] == "2"
    assert requests[2].url.params["page"] == "10"
    assert requests[2].url.params["page_size"] == "50"


@pytest.mark.asyncio
async def test_archive_pages_inspect_new_items_and_keep_metadata_bounds(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/advancedsearch.php":
            page = int(request.url.params.get("page", 1))
            return httpx.Response(200, json={"response": {"docs": [{"identifier": f"coastal-cave-{page}"}]}})
        return httpx.Response(200, json={"metadata": {"title": "Coastal cave", "creator": "Archive"},
                                        "files": [{"name": "cave.jpg", "source": "original"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path)), client=client)
        first = await provider.search("coastal cave", 5)
        second = await provider.search_page("coastal cave", 99, 1)
        await provider.search_page("coastal cave", 5, 9)
    search = [request for request in requests if request.url.path == "/advancedsearch.php"]
    assert first[0].image_url != second[0].image_url
    assert "page" not in search[0].url.params
    assert search[1].url.params["page"] == "2" and search[1].url.params["rows"] == "10"
    assert search[2].url.params["page"] == "10"
    assert all(request.url.params["q"] == '("coastal" "cave") AND mediatype:image' for request in search)
    assert len(requests) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", HTTP_PROVIDERS)
@pytest.mark.parametrize("page", [-1, 10, True, None, "1", 1.0])
async def test_invalid_http_page_is_rejected_before_io(tmp_path, provider_type, page):
    def forbidden(request):
        raise AssertionError("Invalid pages cannot make a request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        provider = provider_type(Settings(cache_dir=str(tmp_path)), client=client)
        with pytest.raises(ProviderError, match="integer between 0 and 9"):
            await provider.search_page("coastal cave", 5, page)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", HTTP_PROVIDERS)
@pytest.mark.parametrize("status", [403, 429])
async def test_paged_http_failure_has_no_identity_change_or_retry(tmp_path, provider_type, status):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = provider_type(Settings(cache_dir=str(tmp_path)), client=client)
        with pytest.raises(ProviderError):
            await provider.search_page("coastal cave", 5, 1)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", HTTP_PROVIDERS)
async def test_empty_provider_page_returns_empty_without_extra_requests(tmp_path, provider_type):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"query": {"pages": []}, "results": [], "response": {"docs": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        assert await provider_type(Settings(cache_dir=str(tmp_path)), client=client).search_page("coastal cave", 5, 9) == []
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_ddgs_fetches_bounded_larger_result_prefix_then_slices_new_raw_page(tmp_path):
    calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            calls.append((query, kwargs))
            return [None, *[ddgs_row(index) for index in range(1, kwargs["max_results"])]]

    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path)), ddgs_factory=FakeDDGS)
    first = await provider.search("coastal cave", 2)
    second = await provider.search_page("coastal cave", 2, 1)
    last = await provider.search_page("coastal cave", 50, 3)
    assert [row.title for row in first] == ["Coastal cave 1"]
    assert [row.title for row in second] == ["Coastal cave 2", "Coastal cave 3"]
    assert len(last) == 50 and last[0].title == "Coastal cave 150"
    assert [kwargs["max_results"] for _, kwargs in calls] == [2, 4, 200]
    assert all(kwargs["backend"] == "duckduckgo" and kwargs["safesearch"] == "moderate" for _, kwargs in calls)


@pytest.mark.asyncio
async def test_ddgs_page_beyond_documented_app_result_cap_is_empty_without_worker(tmp_path):
    def forbidden(**kwargs):
        raise AssertionError("No worker is needed beyond the 200-result bound")

    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path)), ddgs_factory=forbidden)
    assert await provider.search_page("coastal cave", 50, 4) == []
    assert await provider.search_page("coastal cave", 99, 9) == []
    assert provider._inflight is False


@pytest.mark.asyncio
@pytest.mark.parametrize("page", [-1, 10, True, "1", 1.0])
async def test_invalid_ddgs_page_never_imports_or_calls_optional_backend(tmp_path, page):
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path)))
    with pytest.raises(ProviderError, match="integer between 0 and 9"):
        await provider.search_page("coastal cave", 5, page)
    assert provider._factory is None and provider._inflight is False


@pytest.mark.asyncio
async def test_ddgs_exhausted_short_prefix_and_access_error_are_preserved(tmp_path):
    calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            calls.append(kwargs)
            if query == "blocked":
                raise RuntimeError("Source denied access")
            return [ddgs_row(0), ddgs_row(1)]

    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path)), ddgs_factory=FakeDDGS)
    assert await provider.search_page("coastal cave", 2, 1) == []
    with pytest.raises(ProviderError, match="RuntimeError"):
        await provider.search_page("blocked", 2, 1)
    assert len(calls) == 2

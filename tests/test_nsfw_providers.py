"""Mature-result preferences use source controls and preserve explicit metadata."""
from pathlib import Path
import sys
import textwrap
import xml.etree.ElementTree as ET

import httpx
import pytest

from semantic_image_stream import ddgs_worker, search_providers
from semantic_image_stream.configuration import Settings
from semantic_image_stream.deviantart_provider import DeviantArtProvider
from semantic_image_stream.models import ImageCandidate
from semantic_image_stream.search_providers import (
    DuckDuckGoProvider, InternetArchiveProvider, OpenverseProvider, WikimediaProvider,
    _duckduckgo_client_factory, _explicit_mature_labels, _mature_status,
)


def image_row(name, **metadata):
    return {"title": name, "url": f"https://source.example/{name}",
            "image": f"https://images.example/{name}.jpg", **metadata}


@pytest.mark.parametrize("value", [True, False, None])
def test_image_mature_flag_survives_serialization(value):
    image = ImageCandidate("https://images.example/photo.jpg", "https://source.example/photo",
                           "Photo", "openverse", mature_content=value)
    assert ImageCandidate.from_dict(image.to_dict()).mature_content is value


@pytest.mark.parametrize("value", ["false", "true", 0, 1, [], {}])
def test_deserialized_mature_flag_requires_an_actual_boolean(value):
    image = ImageCandidate.from_dict({"image_url": "https://images.example/photo.jpg",
                                      "source_url": "https://source.example/photo",
                                      "title": "Photo", "provider": "openverse", "mature_content": value})
    assert image.mature_content is None


def test_explicit_metadata_is_distinct_from_unknown_or_narrative_words():
    assert _mature_status({}) is None
    assert _mature_status({"mature": False}) is False
    assert _mature_status({"mature": False}, {"nsfw": True}) is True
    assert _mature_status({"mature": False, "unstable__sensitivity": ["sensitive_text"]}) is True
    assert _mature_status({"unstable__sensitivity": ["a future unknown marker"]}) is None
    assert _explicit_mature_labels(["adult trees", "naked mole rats"]) is None
    assert _explicit_mature_labels("Art; Category:Nudity; Sculpture") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_openverse_controls_inclusion_and_keeps_sensitive_source_classification(tmp_path, allow):
    settings = Settings(cache_dir=str(tmp_path), allow_nsfw=allow)
    rows = [image_row("mature", mature=True), image_row("safe", mature=False), image_row("unknown"),
            image_row("sensitive", mature=False, unstable__sensitivity=["provider_supplied_sensitive"])]
    for row in rows:
        row["foreign_landing_url"] = row["url"]
        row["url"] = row["image"]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": rows})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenverseProvider(settings, client=client).search_page("figure study", 10, 1)
    assert requests[0].url.params["mature"] == ("true" if allow else "false")
    assert requests[0].url.params["page"] == "2"
    assert "unstable__include_sensitive_results" not in requests[0].url.params
    assert [item.title for item in result] == (["mature", "safe", "unknown", "sensitive"] if allow else ["safe", "unknown"])
    assert {item.title: item.mature_content for item in result} == (
        {"mature": True, "safe": False, "unknown": None, "sensitive": True} if allow else {"safe": False, "unknown": None}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_duckduckgo_controls_safesearch_and_blocks_only_known_mature_results(tmp_path, allow):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            calls.append(kwargs)
            return [image_row("mature", nsfw=True), image_row("safe", mature_content=False), image_row("unknown")]

    result = await DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), allow_nsfw=allow), ddgs_factory=Client).search(
        "figure study", 10
    )
    assert calls[0]["safesearch"] == ("off" if allow else "moderate")
    assert calls[0]["backend"] == "duckduckgo"
    assert [item.title for item in result] == (["mature", "safe", "unknown"] if allow else ["safe", "unknown"])
    assert {item.title: item.mature_content for item in result} == (
        {"mature": True, "safe": False, "unknown": None} if allow else {"safe": False, "unknown": None}
    )


@pytest.mark.parametrize("allow", [False, True, "missing"])
def test_worker_protocol_defaults_to_safe_and_preserves_mature_metadata(monkeypatch, allow):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def images(self, query, **kwargs):
            calls.append(kwargs)
            return [image_row("mature", is_mature=True, _sis_backend="bing"), image_row("safe", mature=False),
                    image_row("unknown")]

    monkeypatch.setattr(ddgs_worker, "_load_factory", lambda: Client)
    request = {"query": "figure study", "limit": 3, "timeout": 5}
    if allow != "missing":
        request["allow_nsfw"] = allow
    rows = ddgs_worker._run_request(request)
    assert calls[0]["safesearch"] == ("off" if allow is True else "moderate")
    assert rows[0]["mature_content"] is True and rows[0]["_sis_backend"] == "bing"
    assert rows[1]["mature_content"] is False
    assert "mature_content" not in rows[2]


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, [], {}])
def test_worker_rejects_non_boolean_preferences_before_loading_native_client(monkeypatch, value):
    monkeypatch.setattr(ddgs_worker, "_load_factory", lambda: pytest.fail("Invalid input cannot load a client"))
    with pytest.raises(ValueError, match="Invalid worker request"):
        ddgs_worker._run_request({"query": "figure study", "limit": 3, "timeout": 5, "allow_nsfw": value})


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_isolated_worker_receives_preference_and_parent_preserves_mature_flag(tmp_path, monkeypatch, allow):
    script = tmp_path / "nsfw_test_worker.py"
    script.write_text(textwrap.dedent(f"""\
        from semantic_image_stream import ddgs_worker as worker
        class Client:
            def __init__(self, **kwargs): pass
            def images(self, query, **kwargs):
                assert kwargs['safesearch'] == {'off' if allow else 'moderate'!r}
                return [{{'title': 'Source classified result', 'image': 'https://images.example/result.jpg',
                         'url': 'https://source.example/result', 'mature': True, '_sis_backend': 'bing'}}]
        worker._load_factory = lambda: Client
        raise SystemExit(worker.main())
    """), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(Path(search_providers.__file__).resolve().parents[1]))
    monkeypatch.setattr(search_providers, "_duckduckgo_worker_command", lambda: [sys.executable, str(script)])
    provider = DuckDuckGoProvider(Settings(cache_dir=str(tmp_path), allow_nsfw=allow, download_timeout=10))
    result = await provider.search("figure study", 1)
    assert len(result) == int(allow)
    if allow:
        assert result[0].mature_content is True and result[0].provider == "Bing Images"
    assert not provider._inflight and provider._worker_process is None


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("primary_fails", [False, True])
def test_bing_fallback_payload_receives_real_safesearch_preference_once(monkeypatch, allow, primary_fails):
    ddgs = pytest.importorskip("ddgs.ddgs")
    bing = pytest.importorskip("ddgs.engines.bing_images").BingImages(timeout=5)
    original = bing.build_payload
    calls, payloads = [], []
    monkeypatch.setattr(ddgs.DDGS, "_get_engines", lambda self, category, backend: [bing])

    def images(self, query, **kwargs):
        calls.append(kwargs["backend"])
        if kwargs["backend"] == "duckduckgo":
            if primary_fails:
                raise RuntimeError("Source unavailable")
            return []
        engine = self._get_engines("images", "bing")[0]
        payloads.append(engine.build_payload(query=query, region="us-en", safesearch=kwargs["safesearch"],
                                            timelimit=None, page=2, max_results=12))
        return [image_row("fallback")]

    monkeypatch.setattr(ddgs.DDGS, "images", images)
    client = _duckduckgo_client_factory()(timeout=5)
    mode = "off" if allow else "moderate"
    result = client.images("figure study", backend="duckduckgo", safesearch=mode)
    assert calls == ["duckduckgo", "bing"]
    assert payloads[0]["adlt"] == ("off" if allow else "strict")
    baseline = original(query="figure study", region="us-en", safesearch=mode, timelimit=None, page=2, max_results=12)
    assert {key: value for key, value in payloads[0].items() if key != "adlt"} == baseline
    assert result[0]["_sis_backend"] == "bing"
    adapter = bing.build_payload
    client._get_engines("images", "bing")
    assert bing.build_payload is adapter
    changed = bing.build_payload("figure study", "us-en", "moderate" if allow else "off", None)
    assert changed["adlt"] == ("strict" if allow else "off")


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_deviantart_uses_documented_rss_ratings_and_keeps_unknown_distinct(tmp_path, monkeypatch, allow):
    provider = DeviantArtProvider(Settings(cache_dir=str(tmp_path), allow_nsfw=allow))
    rss = ET.fromstring('''<rss xmlns:media="http://search.yahoo.com/mrss/"><channel>
      <item><title>Mature</title><link>https://www.deviantart.com/artist/art/one</link>
        <media:rating scheme="urn:simple">adult</media:rating>
        <media:content url="https://images.example/mature.jpg" type="image/jpeg" /></item>
      <item><title>Safe</title><link>https://www.deviantart.com/artist/art/two</link>
        <media:rating>nonadult</media:rating>
        <media:content url="https://images.example/safe.jpg" type="image/jpeg" /></item>
      <item><title>Unknown</title><link>https://www.deviantart.com/artist/art/three</link>
        <media:content url="https://images.example/unknown.jpg" type="image/jpeg" /></item>
      <item><title>Deprecated rating</title><link>https://www.deviantart.com/artist/art/four</link>
        <media:adult>true</media:adult>
        <media:content url="https://images.example/deprecated.jpg" type="image/jpeg" /></item>
    </channel></rss>''')
    requests = []

    async def xml(url, params):
        requests.append(params)
        return rss

    monkeypatch.setattr(provider, "_xml", xml)
    try:
        result = await provider.search("figure study", 10)
    finally:
        await provider.close()
    assert "mature_content" not in requests[0]
    assert [item.title for item in result] == (["Mature", "Safe", "Unknown", "Deprecated rating"] if allow else ["Safe", "Unknown"])
    assert {item.title: item.mature_content for item in result} == (
        {"Mature": True, "Safe": False, "Unknown": None, "Deprecated rating": True} if allow else {"Safe": False, "Unknown": None}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("allow", [False, True])
async def test_commons_definite_category_classification_respects_preference(tmp_path, allow):
    pages = [{"title": "File:Figure Study", "imageinfo": [{"url": "https://images.example/figure.jpg",
              "descriptionurl": "https://commons.wikimedia.org/wiki/File:Figure_Study", "mime": "image/jpeg",
              "extmetadata": {"Categories": {"value": "Sculpture; Nudity"}}}]},
             {"title": "File:Naked Mole Rat", "imageinfo": [{"url": "https://images.example/animal.jpg",
              "descriptionurl": "https://commons.wikimedia.org/wiki/File:Animal", "mime": "image/jpeg"}]}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"query": {"pages": pages}}
    ))) as client:
        result = await WikimediaProvider(Settings(cache_dir=str(tmp_path), allow_nsfw=allow), client=client).search(
            "figure study", 10
        )
    assert [item.title for item in result] == (["Figure Study", "Naked Mole Rat"] if allow else ["Naked Mole Rat"])
    assert result[-1].mature_content is None
    if allow:
        assert result[0].mature_content is True


@pytest.mark.parametrize("allow", [False, True])
def test_archive_definite_item_and_file_metadata_respect_preference(tmp_path, allow):
    provider = InternetArchiveProvider(Settings(cache_dir=str(tmp_path), allow_nsfw=allow), client=object())
    metadata = {"metadata": {"title": "Figure study", "subject": ["Sculpture", "Nudity"]},
                "files": [{"name": "image.jpg", "source": "original"}]}
    result = provider._parse_item("figure", metadata)
    assert len(result) == int(allow)
    if allow:
        assert result[0].mature_content is True
    metadata["metadata"]["subject"] = ["Naked mole rats"]
    metadata["files"] = [{"name": "mature.jpg", "source": "original", "mature_content": True},
                         {"name": "unknown.jpg", "source": "original"}]
    result = provider._parse_item("figure", metadata)
    assert len(result) == (2 if allow else 1)
    assert result[-1].mature_content is None

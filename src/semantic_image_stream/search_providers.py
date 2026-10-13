"""Bounded, provider-independent discovery of existing source images.

Only a short search query is sent to providers. No provider uploads audio or
transcripts. Access errors and rate limits are reported, never worked around.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import PurePosixPath
import threading
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from .configuration import Settings
from .models import ImageCandidate

log = logging.getLogger(__name__)
USER_AGENT = "SemanticImageStream/0.2.2 (https://github.com/sporemound/semantic-image-stream; https://github.com/sporemound)"
UNKNOWN_LICENSE = "Unknown — check source before reuse"
MAX_METADATA_BYTES = 2_000_000
_GENERATED = re.compile(
    r"\b(?:ai[\s_-]+generated|artificial intelligence[\s_-]+generated|"
    r"generated (?:using|by|with) (?:ai|midjourney|dall[\s_-]?e|stable diffusion)|"
    r"midjourney|dall[\s_-]?e|stable diffusion)\b", re.IGNORECASE
)


class ProviderError(RuntimeError):
    """A provider was unavailable, blocked, rate limited, or returned bad data."""


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[ImageCandidate]: ...

    async def search_page(self, query: str, limit: int, page: int) -> list[ImageCandidate]: ...

    async def close(self) -> None: ...


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden += 1
        elif tag.lower() in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif tag.lower() in {"p", "div", "li", "a"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def plain_text(value: Any, default: str = "", max_length: int = 2000) -> str:
    """Discard HTML, executable tags, controls, and malformed non-text fields."""
    if isinstance(value, list):
        value = "; ".join(v for v in value[:20] if isinstance(v, str))
    if not isinstance(value, str):
        return default
    parser = _PlainText()
    # Decode before parsing so entity-encoded markup cannot survive into a
    # widget that automatically interprets strings as rich text.
    parser.feed(unescape(unescape(value[:20_000])))
    text = "".join(parser.parts).replace("<", "").replace(">", "")
    text = "".join(c for c in text if c.isprintable() or c.isspace())
    return " ".join(text.split())[:max_length] or default


def sanitize_query(query: str) -> str:
    """Bound the disclosure and remove addresses, URLs and search syntax."""
    text = plain_text(query, max_length=1000)
    text = re.sub(r"https?://\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+", " ", text)
    text = "".join(c if c.isalnum() or c in " '-" else " " for c in text)
    return " ".join(text.split())[:180].strip()


def is_known_generated(*values: Any) -> bool:
    return any(_GENERATED.search(plain_text(value)) for value in values)


def _generated_flag(row: dict) -> bool:
    return any(row.get(key) in (True, "true", "yes", "1")
               for key in ("is_ai_generated", "ai_generated", "is_generated"))


def _https_url(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        return ""
    try:
        url = urlsplit(value)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            return ""
        if url.port not in (None, 443):
            return ""
        return value
    except ValueError:
        return ""


def _source_url(value: Any) -> str:
    """Attribution pages may use HTTP, but must never use executable schemes."""
    if not isinstance(value, str) or len(value) > 4096:
        return ""
    try:
        url = urlsplit(value)
        return value if url.scheme in {"http", "https"} and url.hostname and not url.username and not url.password else ""
    except ValueError:
        return ""


def _dimension(value: Any) -> int:
    try:
        return max(0, min(int(value), 1_000_000)) if not isinstance(value, bool) else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def _rows(value: Any) -> list[dict]:
    return [item for item in value[:200] if isinstance(item, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


class _HTTPProvider:
    name = ""

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=settings.download_timeout, follow_redirects=False, trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
        )

    async def _json(self, url: str, params: Any = None) -> dict:
        # Fixed documented API endpoints. Redirects/403/429 are failures, not
        # reasons to switch identities, scrape a different endpoint, or retry.
        async with self.client.stream("GET", url, params=params, follow_redirects=False,
                                      timeout=self.settings.download_timeout) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > MAX_METADATA_BYTES:
                    raise ProviderError(f"{self.name} returned excessive metadata")
                data.extend(chunk)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ProviderError(f"{self.name} returned invalid metadata")
        if value.get("error"):
            raise ProviderError(f"{self.name} returned an API error")
        return value

    async def search(self, query: str, limit: int = 10) -> list[ImageCandidate]:
        return await self.search_page(query, limit, 0)

    async def search_page(self, query: str, limit: int, page: int) -> list[ImageCandidate]:
        _validate_page(page)
        query = sanitize_query(query)
        limit = max(0, min(int(limit), 50))
        if not query or not limit:
            return []
        try:
            async with asyncio.timeout(self.settings.download_timeout):
                return (await self._search(query, limit, page))[:limit]
        except ProviderError:
            raise
        except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError) as error:
            raise ProviderError(f"{self.name} search failed ({type(error).__name__})") from error

    async def _search(self, query: str, limit: int, page: int = 0) -> list[ImageCandidate]:
        raise NotImplementedError

    async def close(self) -> None:
        if self._owned_client:
            await self.client.aclose()


class WikimediaProvider(_HTTPProvider):
    """Commons Action API generator search + imageinfo, including attribution."""
    name = "wikimedia"

    async def _search(self, query: str, limit: int, page: int = 0) -> list[ImageCandidate]:
        params = {
            "action": "query", "format": "json", "formatversion": "2",
            "generator": "search", "gsrsearch": query, "gsrnamespace": "6",
            "gsrlimit": min(limit, 20), "prop": "imageinfo", "iilimit": "1",
            "iiprop": "url|size|mime|thumbmime|extmetadata", "iiurlwidth": "1600",
            "iiextmetadatalanguage": "en", "maxlag": "5",
            "iiextmetadatafilter": "Artist|Credit|LicenseShortName|LicenseUrl|UsageTerms|ImageDescription|Categories|ObjectName",
        }
        if page:
            params["gsroffset"] = page * limit
        data = await self._json("https://commons.wikimedia.org/w/api.php", params)
        pages = _dict(data.get("query")).get("pages", [])
        # Accept both documented formatversion layouts for compatibility.
        if isinstance(pages, dict):
            pages = list(pages.values())
        candidates = []
        for page in _rows(pages):
            info = next(iter(_rows(page.get("imageinfo"))), {})
            metadata = _dict(info.get("extmetadata"))
            fields = {key: plain_text(_dict(value).get("value")) for key, value in metadata.items()}
            title = plain_text(page.get("title"), "Untitled image", 300).removeprefix("File:")
            if is_known_generated(title, *fields.values()):
                continue
            image_url = _https_url(info.get("thumburl")) or _https_url(info.get("url"))
            source_url = _source_url(info.get("descriptionurl"))
            original_mime = info.get("mime")
            mime = info.get("thumbmime") if info.get("thumburl") else info.get("mime")
            # Documents and videos have raster preview thumbnails too. Their
            # title pages/posters do not make them source-image search results.
            # SVG is an image; Commons provides a safe raster rendition.
            if (not isinstance(original_mime, str) or not original_mime.startswith("image/")
                    or original_mime in {"image/vnd.djvu", "image/x-djvu"}
                    or not image_url or not source_url or not isinstance(mime, str) or not mime.startswith("image/")):
                continue
            candidates.append(ImageCandidate(
                image_url=image_url, source_url=source_url, title=title, provider=self.name,
                creator=fields.get("Artist") or fields.get("Credit") or "Unknown",
                license=fields.get("LicenseShortName") or fields.get("UsageTerms") or UNKNOWN_LICENSE,
                license_url=_source_url(fields.get("LicenseUrl")),
                description=fields.get("ImageDescription", ""),
                width=_dimension(info.get("thumbwidth") or info.get("width")),
                height=_dimension(info.get("thumbheight") or info.get("height")),
            ))
        return candidates


class OpenverseProvider(_HTTPProvider):
    """Anonymous Openverse image API; rate limits are left intact."""
    name = "openverse"

    async def _search(self, query: str, limit: int, page: int = 0) -> list[ImageCandidate]:
        params = {"q": query, "page_size": limit}
        if page:
            params["page"] = page + 1
        data = await self._json("https://api.openverse.org/v1/images/", params)
        candidates = []
        for row in _rows(data.get("results")):
            title = plain_text(row.get("title"), "Untitled image", 300)
            tags = "; ".join(plain_text(tag.get("name")) for tag in _rows(row.get("tags")))
            if _generated_flag(row) or is_known_generated(title, row.get("description"), row.get("creator"), tags):
                continue
            image_url, source_url = _https_url(row.get("url")), _source_url(row.get("foreign_landing_url"))
            if not image_url or not source_url:
                continue
            license_name = plain_text(row.get("license"), max_length=100).upper()
            version = plain_text(row.get("license_version"), max_length=20)
            candidates.append(ImageCandidate(
                image_url=image_url, source_url=source_url, title=title, provider=self.name,
                creator=plain_text(row.get("creator"), "Unknown"),
                license=" ".join(filter(None, [license_name, version])) or UNKNOWN_LICENSE,
                license_url=_source_url(row.get("license_url")),
                description=plain_text(row.get("description")) or tags,
                width=_dimension(row.get("width")), height=_dimension(row.get("height")),
            ))
        return candidates


class InternetArchiveProvider(_HTTPProvider):
    """Search image items, then inspect documented item metadata for real files."""
    name = "archive"

    async def _search(self, query: str, limit: int, page: int = 0) -> list[ImageCandidate]:
        # Quote terms so topic words cannot inject Solr field filters/operators.
        terms = " ".join('"' + word.replace('"', '') + '"' for word in query.split())
        params = [("q", f"({terms}) AND mediatype:image"), ("rows", str(min(limit, 10))),
                  ("output", "json"), ("fl[]", "identifier")]
        if page:
            params.append(("page", str(page + 1)))
        data = await self._json("https://archive.org/advancedsearch.php", params)
        identifiers = []
        for row in _rows(_dict(data.get("response")).get("docs")):
            identifier = row.get("identifier")
            if isinstance(identifier, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", identifier):
                identifiers.append(identifier)
        # A failed item does not discard other results, and no more than ten
        # metadata requests or three concurrent requests are made per search.
        semaphore = asyncio.Semaphore(3)

        async def fetch(identifier: str) -> list[ImageCandidate]:
            async with semaphore:
                try:
                    metadata = await self._json(f"https://archive.org/metadata/{quote(identifier, safe='')}")
                    return self._parse_item(identifier, metadata)
                except (httpx.HTTPError, ProviderError, ValueError, TypeError):
                    log.debug("Internet Archive item metadata unavailable")
                    return []

        batches = await asyncio.gather(*(fetch(identifier) for identifier in identifiers))
        return [candidate for batch in batches for candidate in batch][:limit]

    def _parse_item(self, identifier: str, data: dict) -> list[ImageCandidate]:
        metadata = _dict(data.get("metadata"))
        title = plain_text(metadata.get("title"), identifier, 300)
        description = plain_text(metadata.get("description"))
        if _generated_flag(metadata) or is_known_generated(title, description, metadata.get("subject"), metadata.get("creator")):
            return []
        license_url = _source_url(metadata.get("licenseurl"))
        candidates = []
        for file in _rows(data.get("files")):
            name = file.get("name")
            if not isinstance(name, str) or len(name) > 1000:
                continue
            if PurePosixPath(name.lower()).suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp"}:
                continue
            if file.get("source") != "original" or name.startswith("__ia_thumb") or is_known_generated(name):
                continue
            candidates.append(ImageCandidate(
                image_url=f"https://archive.org/download/{quote(identifier, safe='')}/{quote(name, safe='')}",
                source_url=f"https://archive.org/details/{quote(identifier, safe='')}",
                title=title, provider=self.name, creator=plain_text(metadata.get("creator"), "Unknown"),
                license=license_url or plain_text(metadata.get("rights"), UNKNOWN_LICENSE),
                license_url=license_url, description=description,
                width=_dimension(file.get("width")), height=_dimension(file.get("height")),
            ))
        return candidates


def _duckduckgo_client_factory():
    # Import the real implementation lazily: DDGS's top-level lazy proxy
    # returns the base class when subclassed, bypassing this override.
    from ddgs.ddgs import DDGS

    class DuckDuckGoClient(DDGS):
        def _get_engines(self, category, backend):
            # This engine hook is internal to DDGS 9.16; the search extra pins
            # that minor release. Native headers use primp's public setter.
            engines = super()._get_engines(category, backend)
            if category == "images" and backend == "duckduckgo":
                for engine in engines:
                    if engine.name != "duckduckgo":
                        raise ProviderError("The requested DuckDuckGo image backend is unavailable")
                    native = engine.http_client.client
                    headers = native.headers
                    cleaned = {name: value for name, value in headers.items() if name.casefold() != "connection"}
                    if len(cleaned) != len(headers):
                        # Connection: keep-alive is legal HTTP/1.1, but causes
                        # native HTTP/2 clients to reject these image requests.
                        native.headers = cleaned
            return engines

    return DuckDuckGoClient


class DuckDuckGoProvider:
    """Optional DDGS adapter, pinned to the requested DuckDuckGo backend."""
    name = "duckduckgo"

    def __init__(self, settings: Settings, *, ddgs_factory: Any = None) -> None:
        self.settings = settings
        self._factory = ddgs_factory
        self._inflight = False

    async def search(self, query: str, limit: int = 10) -> list[ImageCandidate]:
        return await self.search_page(query, limit, 0)

    async def search_page(self, query: str, limit: int, page: int) -> list[ImageCandidate]:
        _validate_page(page)
        query, limit = sanitize_query(query), max(0, min(int(limit), 50))
        if not query or not limit:
            return []
        offset = page * limit
        if offset >= 200:
            return []
        requested = min(200, (page + 1) * limit)
        if self._inflight:
            raise ProviderError("duckduckgo previous request is still finishing")
        if self._factory is None:
            try:
                self._factory = _duckduckgo_client_factory()
            except ImportError as error:
                raise ProviderError("duckduckgo needs the optional 'search' dependency") from error
        # A synchronous library cannot be forcibly cancelled. Prevent repeated
        # timeout searches from accumulating workers; library timeout also bounds I/O.
        self._inflight = True

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def finish(value: Any = None, error: Exception | None = None) -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(value)

        def run() -> None:
            try:
                engine = self._factory(timeout=self.settings.download_timeout)
                value = engine.images(query, max_results=requested, safesearch="moderate", backend="duckduckgo")
                error = None
            except Exception as exception:
                value, error = None, exception
            finally:
                self._inflight = False
            if not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(finish, value, error)
                except RuntimeError:
                    pass

        # A native library call cannot be killed, but an outstanding request
        # must not hold up event-loop executor shutdown or application exit.
        threading.Thread(target=run, name="sis-duckduckgo-search", daemon=True).start()
        try:
            rows = await asyncio.wait_for(future, self.settings.download_timeout)
        except Exception as error:
            raise ProviderError(f"duckduckgo search failed ({type(error).__name__})") from error
        candidates = []
        page_rows = rows[offset:offset + limit] if isinstance(rows, list) else []
        for row in _rows(page_rows):
            title = plain_text(row.get("title"), "Untitled image", 300)
            if _generated_flag(row) or is_known_generated(title, row.get("source"), row.get("description")):
                continue
            image_url, source_url = _https_url(row.get("image")), _source_url(row.get("url"))
            if not image_url or not source_url:
                continue
            candidates.append(ImageCandidate(
                image_url=image_url, source_url=source_url, title=title, provider=self.name,
                width=_dimension(row.get("width")), height=_dimension(row.get("height")),
            ))
        return candidates

    async def close(self) -> None:
        pass


def _validate_page(page: int) -> None:
    if type(page) is not int or not 0 <= page <= 9:
        raise ProviderError("Search page must be an integer between 0 and 9")


def build_providers(settings: Settings) -> list[SearchProvider]:
    constructors = {"wikimedia": WikimediaProvider, "duckduckgo": DuckDuckGoProvider,
                    "openverse": OpenverseProvider, "archive": InternetArchiveProvider}
    return [constructors[name](settings) for name in dict.fromkeys(settings.providers)]

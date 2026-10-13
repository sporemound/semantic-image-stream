"""Download, verify and re-encode untrusted images before they enter the cache.

Every URL and redirect is HTTPS and resolved to public addresses before use.
The default httpx transport performs a separate DNS lookup when connecting;
this reduces SSRF exposure but cannot eliminate DNS rebinding without transport
IP pinning. See docs/privacy.md. Environment proxies are disabled.
"""
from __future__ import annotations

import asyncio
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import ipaddress
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Awaitable, Callable, Iterable
from urllib.parse import urljoin, urlsplit
import warnings

import httpx
import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from .configuration import Settings
from .models import ImageCandidate
from .search_providers import USER_AGENT, is_known_generated

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}
MAX_REDIRECTS = 3
DEFAULT_RATE_LIMIT_SECONDS = 60.0
MAX_RATE_LIMIT_SECONDS = 3600.0
Resolver = Callable[[str, int], Awaitable[Iterable[str]]]


class DownloadError(RuntimeError):
    """Rejected image with a URL/query-free reason suitable for UI counters."""

    def __init__(self, message: str, *, reason: str = "Invalid image", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.retry_after = retry_after


def _retry_after_seconds(value: str | None, now: float) -> float:
    """Parse both HTTP Retry-After formats without trusting unbounded delays."""
    seconds = DEFAULT_RATE_LIMIT_SECONDS
    if value:
        value = value.strip()[:512]
        try:
            # Bound integer parsing as well as the resulting cooldown.
            if value.isascii() and value.isdigit():
                digits = value.lstrip("0") or "0"
                seconds = MAX_RATE_LIMIT_SECONDS if len(digits) > 4 else float(int(digits))
            else:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = date.timestamp() - now
        except (ValueError, TypeError, OverflowError):
            seconds = DEFAULT_RATE_LIMIT_SECONDS
    return max(1.0, min(MAX_RATE_LIMIT_SECONDS, seconds))


async def _resolve(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list({result[4][0] for result in results})


def _public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_unspecified)
    except ValueError:
        return False


async def validate_public_https(url: str, resolver: Resolver = _resolve) -> str:
    """Validate literal IPs and all DNS answers. Return the canonical URL."""
    try:
        if not isinstance(url, str) or len(url) > 4096 or any(ord(c) < 32 for c in url):
            raise ValueError("invalid URL")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            raise ValueError("only anonymous HTTPS URLs are allowed")
        if parsed.port not in (None, 443) or "\\" in url or "%" in parsed.hostname:
            raise ValueError("invalid URL authority")
        normalized = httpx.URL(url)
        host = normalized.host.rstrip(".").lower()
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError("local host is forbidden")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host:
                raise ValueError("unqualified host is forbidden")
            addresses = list(await resolver(host, 443))
            if not addresses or any(not _public_address(address) for address in addresses):
                raise ValueError("hostname resolves to a non-public address")
        else:
            if not _public_address(str(literal)):
                raise ValueError("non-public address is forbidden")
        return str(normalized)
    except (ValueError, httpx.InvalidURL, OSError) as error:
        raise DownloadError("Image URL is not a public HTTPS destination", reason="Unsafe image URL") from error


class ImageDownloader:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None,
                 resolver: Resolver | None = None, clock: Callable[[], float] | None = None,
                 wall_clock: Callable[[], float] | None = None) -> None:
        self.settings = settings
        self.directory = Path(os.path.expandvars(settings.cache_dir)).expanduser().resolve() / "images"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.resolver = resolver or _resolve
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._host_cooldowns: dict[str, float] = {}
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=settings.download_timeout, follow_redirects=False, trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": ", ".join(sorted(ALLOWED_MIME)),
                     "Accept-Encoding": "identity"},
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=3),
        )

    async def download(self, candidate: ImageCandidate) -> ImageCandidate:
        if is_known_generated(candidate.title, candidate.description, candidate.creator, candidate.authenticity):
            raise DownloadError("Source metadata identifies a generated image", reason="Generated image excluded")
        try:
            # Entire deadline includes DNS, all redirects and stream reads.
            async with asyncio.timeout(self.settings.download_timeout):
                data = await self._read(candidate.image_url)
                encoded, extension, width, height, phash = await asyncio.to_thread(self._decode, data)
            # Only verified, re-encoded bytes are persisted. Temp paths are
            # unique, basename is content-derived, and replace is atomic.
            digest = hashlib.sha256(encoded).hexdigest()
            destination = self.directory / f"{digest}{extension}"
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as file:
                    temporary = file.name
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            candidate.local_path = str(destination)
            candidate.width, candidate.height, candidate.phash = width, height, phash
            return candidate
        except DownloadError:
            raise
        except httpx.HTTPStatusError as error:
            reason = {401: "Source inaccessible", 403: "Source inaccessible",
                      404: "Broken image link", 410: "Broken image link"}.get(error.response.status_code,
                                                                             "Source unavailable")
            raise DownloadError(f"Image download failed ({type(error).__name__})", reason=reason) from error
        except (httpx.TimeoutException, TimeoutError) as error:
            raise DownloadError(f"Image download failed ({type(error).__name__})",
                                reason="Image download timed out") from error
        except httpx.HTTPError as error:
            raise DownloadError(f"Image download failed ({type(error).__name__})", reason="Network error") from error
        except OSError as error:
            raise DownloadError(f"Image download failed ({type(error).__name__})",
                                reason="Image cache unavailable") from error
        except ValueError as error:
            raise DownloadError(f"Image download failed ({type(error).__name__})") from error

    def _check_cooldown(self, host: str) -> None:
        now = self._clock()
        # Expired hosts no longer need tracking; only active rate limits remain.
        self._host_cooldowns = {name: expiry for name, expiry in self._host_cooldowns.items() if expiry > now}
        remaining = self._host_cooldowns.get(host.lower().rstrip("."), 0.0) - now
        if remaining > 0:
            raise DownloadError("Source temporarily rate-limited; retry later",
                                reason="Source temporarily rate-limited", retry_after=remaining)

    async def _read(self, url: str) -> bytes:
        for redirect in range(MAX_REDIRECTS + 1):
            # A blocked host needs neither another network request nor another
            # DNS lookup. Validation is still mandatory before all HTTP I/O.
            try:
                host = httpx.URL(url).host
            except (httpx.InvalidURL, ValueError):
                host = ""
            self._check_cooldown(host)
            url = await validate_public_https(url, self.resolver)
            self._check_cooldown(httpx.URL(url).host)
            async with self.client.stream("GET", url, follow_redirects=False,
                                          timeout=self.settings.download_timeout) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location or redirect == MAX_REDIRECTS:
                        raise DownloadError("Image redirected too many times or without a destination")
                    url = urljoin(url, location)
                    continue
                if response.status_code == 429:
                    seconds = _retry_after_seconds(response.headers.get("retry-after"), self._wall_clock())
                    # Limit only the host that actually sent the rejection.
                    # upload.wikimedia.org does not suppress thumb.wikimedia.org.
                    host = response.url.host.lower().rstrip(".")
                    self._host_cooldowns[host] = max(self._host_cooldowns.get(host, 0.0), self._clock() + seconds)
                    raise DownloadError("Source temporarily rate-limited; retry later",
                                        reason="Source temporarily rate-limited", retry_after=seconds)
                response.raise_for_status()
                # Raster images already compress their data. Disable HTTP
                # decompression so a gzip bomb cannot inflate before the
                # streaming byte limit is applied.
                encoding = response.headers.get("content-encoding", "").lower().strip()
                if encoding not in {"", "identity"}:
                    raise DownloadError("Compressed HTTP image bodies are unsupported")
                mime = response.headers.get("content-type", "").split(";", 1)[0].lower().strip()
                if mime not in ALLOWED_MIME:
                    raise DownloadError("Image response has an unsupported content type")
                length = response.headers.get("content-length")
                if length is not None:
                    try:
                        declared = int(length)
                    except ValueError as error:
                        raise DownloadError("Image response has invalid content length") from error
                    if declared < 0 or declared > self.settings.max_download_bytes:
                        raise DownloadError("Image exceeds the download size limit", reason="Image exceeds safety limits")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(data) + len(chunk) > self.settings.max_download_bytes:
                        raise DownloadError("Image exceeds the download size limit", reason="Image exceeds safety limits")
                    data.extend(chunk)
                if not data:
                    raise DownloadError("Image response is empty")
                return bytes(data)
        raise DownloadError("Image redirect limit exceeded")

    def _decode(self, data: bytes) -> tuple[bytes, str, int, int, str]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in ALLOWED_FORMATS:
                        raise DownloadError("Unsupported decoded image format")
                    width, height = image.size
                    if width <= 0 or height <= 0 or width * height > self.settings.max_image_pixels:
                        raise DownloadError("Image exceeds the decoded pixel limit", reason="Image exceeds safety limits")
                    image.verify()
                with Image.open(io.BytesIO(data)) as original:
                    original.seek(0)  # Animated files are reduced to their first existing frame.
                    original.load()
                    image = ImageOps.exif_transpose(original)
                    width, height = image.size
                    if width < self.settings.min_width or height < self.settings.min_height:
                        raise DownloadError("Image resolution is below the configured minimum",
                                            reason="Image below minimum resolution")
                    ratio = width / height
                    if not self.settings.min_aspect_ratio <= ratio <= self.settings.max_aspect_ratio:
                        raise DownloadError("Image aspect ratio is outside the configured range",
                                            reason="Image aspect ratio outside allowed range")
                    # Read generated-image indicators from standard textual
                    # metadata too, but do not claim absence proves authenticity.
                    text_metadata = [v for v in original.info.values() if isinstance(v, str)]
                    if is_known_generated(*text_metadata):
                        raise DownloadError("Embedded metadata identifies a generated image", reason="Generated image excluded")
                    phash = str(imagehash.phash(image))
                    output = io.BytesIO()
                    if image.mode in {"RGBA", "LA"} or "transparency" in original.info:
                        clean = Image.new("RGBA", image.size)
                        clean.paste(image.convert("RGBA"))
                        clean.save(output, format="PNG", optimize=True)
                        extension = ".png"
                    elif original.format == "PNG":
                        clean = Image.new("RGB", image.size)
                        clean.paste(image.convert("RGB"))
                        clean.save(output, format="PNG", optimize=True)
                        extension = ".png"
                    else:
                        clean = Image.new("RGB", image.size)
                        clean.paste(image.convert("RGB"))
                        clean.save(output, format="JPEG", quality=92, optimize=True)
                        extension = ".jpg"
                    encoded = output.getvalue()
                    if len(encoded) > self.settings.max_download_bytes:
                        raise DownloadError("Validated image exceeds the cache file size limit", reason="Image exceeds safety limits")
                    return encoded, extension, width, height, phash
        except DownloadError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise DownloadError("Image could not be safely decoded") from error

    async def close(self) -> None:
        if self._owned_client:
            await self.client.aclose()

"""Anonymous DeviantArt Media RSS discovery with explicit artwork credit."""
from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

from .models import ImageCandidate
from .search_providers import (
    MAX_METADATA_BYTES,
    ProviderError,
    _HTTPProvider,
    _dimension,
    _generated_status,
    _https_url,
    _source_url,
    plain_text,
)
from .temporal_filter import extract_depicted_evidence

MEDIA = "{http://search.yahoo.com/mrss/}"
DC = "{http://purl.org/dc/elements/1.1/}"


def _text(node: ET.Element, path: str) -> str:
    child = node.find(path)
    return child.text if child is not None and isinstance(child.text, str) else ""


class DeviantArtProvider(_HTTPProvider):
    """Discover deviations through DeviantArt's public Media RSS endpoint."""

    name = "deviantart"

    async def _xml(self, url: str, params: Any = None) -> ET.Element:
        headers = {"Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8"}
        async with self.client.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            follow_redirects=False,
            timeout=self.settings.download_timeout,
        ) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > MAX_METADATA_BYTES:
                    raise ProviderError("deviantart returned excessive metadata")
                data.extend(chunk)
        prefix = bytes(data[:16_384]).lower()
        if b"<!doctype" in prefix or b"<!entity" in prefix:
            raise ProviderError("deviantart returned unsupported XML metadata")
        try:
            return ET.fromstring(data)
        except ET.ParseError as error:
            raise ProviderError("deviantart returned invalid metadata") from error

    async def _search(self, query: str, limit: int, page: int = 0) -> list[ImageCandidate]:
        params: dict[str, str | int] = {
            "type": "deviation",
            "q": f"{query} sort:time meta:all",
        }
        if page:
            params["offset"] = page * limit
        root = await self._xml("https://backend.deviantart.com/rss.xml", params)
        candidates = []
        for item in root.findall(".//item")[:max(limit * 4, limit)]:
            title = plain_text(_text(item, "title"), "Untitled artwork", 300)
            source_url = _source_url(_text(item, "link"))
            creator = plain_text(
                _text(item, f"{DC}creator") or _text(item, f"{MEDIA}credit"),
                "Unknown",
                300,
            )
            description = plain_text(_text(item, "description"))
            tags = "; ".join(
                value
                for value in [
                    *(plain_text(node.text, max_length=200) for node in item.findall("category")),
                    plain_text(_text(item, f"{MEDIA}keywords"), max_length=1000),
                ]
                if value
            )
            generated = _generated_status({}, title, description, tags)
            if generated is True and getattr(self.settings, "exclude_ai_generated", True):
                continue

            assets = []
            for node in item.findall(f".//{MEDIA}content"):
                image_url = _https_url(node.get("url"))
                mime = plain_text(node.get("type"), max_length=100).casefold()
                medium = plain_text(node.get("medium"), max_length=40).casefold()
                if (not image_url or (mime and not mime.startswith("image/"))
                        or (medium and medium != "image")):
                    continue
                width, height = _dimension(node.get("width")), _dimension(node.get("height"))
                assets.append((width * height, width, height, image_url))
            if not assets:
                for node in item.findall(f".//{MEDIA}thumbnail"):
                    image_url = _https_url(node.get("url"))
                    if image_url:
                        width, height = _dimension(node.get("width")), _dimension(node.get("height"))
                        assets.append((width * height, width, height, image_url))
            if not assets or not source_url:
                continue

            _, width, height, image_url = max(assets, key=lambda asset: asset[0])
            rights = plain_text(
                _text(item, f"{MEDIA}copyright")
                or _text(item, f"{DC}rights")
                or _text(item, "copyright")
            )
            candidates.append(ImageCandidate(
                image_url=image_url,
                source_url=source_url,
                title=title,
                provider=self.name,
                creator=creator,
                license=rights or "Artist copyright — see DeviantArt source for reuse terms",
                description=description or tags,
                width=width,
                height=height,
                ai_generated=generated,
                temporal_evidence=extract_depicted_evidence(
                    title,
                    description,
                    tags,
                    source="DeviantArt submission metadata",
                ),
            ))
        return candidates[:limit]

"""Conservative metadata signals for excluding known AI-generated imagery.

This module recognizes explicit source disclosures, generator/model tags,
dedicated generation hosts, and embedded workflow metadata. It does not claim
to infer image provenance from pixels.
"""
from __future__ import annotations

from html import unescape
import re
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit

_MAX_TEXT = 20_000
_CACHE_NAMESPACE = "ai-filter-metadata-v2:"

_EXPLICIT_GENERATED = re.compile(
    r"\b(?:"
    r"ai[\s_-]*(?:generated|created|made|rendered|assisted|art(?:work)?|"
    r"imag(?:e|es|ery)|illustration(?:s)?|photo(?:graph)?s?|painting|content)"
    r"|artificial[\s_-]+intelligence[\s_-]*(?:generated|created|made|rendered|"
    r"assisted|art(?:work)?|imag(?:e|es|ery)|illustration(?:s)?|content)"
    r"|generative[\s_-]+ai|aigc"
    r"|(?:generated|created|made|rendered|produced)[\s_-]+"
    r"(?:using|by|with|in|via)[\s_-]+(?:an?[\s_-]+)?"
    r"(?:ai|artificial[\s_-]+intelligence|midjourney|dall[\s_-]?e|"
    r"stable[\s_-]*diffusion|sdxl|comfyui|dreamup|novelai)"
    r"|images?[\s_-]+generated[\s_-]+by[\s_-]+artificial[\s_-]+intelligence"
    r"|(?:synthetic|machine[\s_-]*generated)[\s_-]+(?:image|imagery|art|media)"
    r")\b",
    re.IGNORECASE,
)

_GENERATION_TOOL = re.compile(
    r"\b(?:"
    r"stable[\s_-]*diffusion|sdxl|midjourney(?:[\s_-]*(?:ai|art))?|"
    r"niji(?:journey|[\s_-]*[56])|dall[\s_-]?e(?:[\s_-]*[23])?|comfyui|"
    r"automatic1111|a1111|fooocus|invokeai|novelai|dreamup(?:[\s_-]*ai)?|"
    r"leonardo(?:\.ai|[\s_-]*ai)|adobe[\s_-]+firefly|"
    r"ideogram(?:\.ai|[\s_-]*ai)?|nightcafe|civitai|tensor\.art|"
    r"seaart(?:\.ai)?|playground[\s_-]+ai|"
    r"flux(?:\.?1|[\s_-]+(?:dev|schnell|pro))|krea(?:\.ai)?|"
    r"mage\.space|starryai|getimg(?:\.ai)?|dreamstudio|craiyon|"
    r"artbreeder|wombo[\s_-]+dream|runway(?:ml|[\s_-]+gen(?:[\s_-]*[123])?)|"
    r"recraft(?:\.ai)?|bing[\s_-]+image[\s_-]+creator|"
    r"microsoft[\s_-]+designer|imagen[\s_-]*[234]|"
    r"ksampler|checkpointloadersimple|cliptextencode|emptylatentimage|vaedecode"
    r")\b",
    re.IGNORECASE,
)

_GENERATION_PARAMETERS = re.compile(
    r"\b(?:negative[\s_-]+prompt|prompt|cfg[\s_-]+scale|sampler|"
    r"sampling[\s_-]+method|model[\s_-]+hash|checkpoint|lora|lycoris|"
    r"txt2img|text2img|img2img|denoising[\s_-]+strength|hires(?:\s+fix)?|"
    r"seed|steps)\s*[:=]",
    re.IGNORECASE,
)

_DISCUSSION_CONTEXT = re.compile(
    r"\b(?:architecture|research|results?|paper|article|news|tutorial|"
    r"documentation|workflow|diagram|schematic|comparison|benchmark|"
    r"explainer|study|interface|screenshot|history|critique|review)\b",
    re.IGNORECASE,
)

_HUMAN_MADE_CONTEXT = re.compile(
    r"\b(?:human|hand|manually|traditionally)[\s_-]*"
    r"(?:drawn|made|created|illustrated|painted|photographed)\b",
    re.IGNORECASE,
)

_AI_SOURCE_HOSTS = {
    "artbreeder.com",
    "civitai.com",
    "craiyon.com",
    "dream.ai",
    "dreamstudio.ai",
    "getimg.ai",
    "ideogram.ai",
    "krea.ai",
    "leonardo.ai",
    "lexica.art",
    "mage.space",
    "midjourney.com",
    "nightcafe.studio",
    "openart.ai",
    "playground.com",
    "prompthero.com",
    "recraft.ai",
    "seaart.ai",
    "starryai.com",
    "tensor.art",
}

_AI_URL_MARKER = re.compile(
    r"(?:^|[/_.?&=-])(?:"
    r"ai[-_ ]?(?:generated|art|artwork|image|illustration)|aigenerated|aiart|"
    r"stable[-_ ]?diffusion[-_ ]?(?:art|image)|"
    r"midjourney[-_ ]?(?:art|image)|dreamup[-_ ]?(?:art|image)|"
    r"dall[-_ ]?e[-_ ]?(?:art|image)|sdxl[-_ ]?(?:art|image)|"
    r"txt2img|text2img|img2img"
    r")(?:$|[/_.?&=-])",
    re.IGNORECASE,
)

_TAG_VALUES = {
    "ai",
    "aiart",
    "aiartist",
    "aiartwork",
    "aiimage",
    "aiimages",
    "aigenerated",
    "aigeneratedart",
    "aigeneratedimage",
    "aigeneratedimages",
    "artificialintelligence",
    "artificialintelligenceart",
    "generativeai",
    "midjourney",
    "midjourneyai",
    "midjourneyart",
    "niji",
    "nijijourney",
    "stablediffusion",
    "stablediffusionart",
    "sdxl",
    "comfyui",
    "dreamup",
    "dreamupai",
    "novelai",
    "dalle",
    "dalle2",
    "dalle3",
    "leonardoai",
    "adobefirefly",
    "ideogramai",
    "nightcafe",
    "civitai",
    "tensorart",
    "seaartai",
    "playgroundai",
    "flux1",
    "fluxdev",
    "fluxschnell",
    "kreaai",
    "starryai",
    "dreamstudio",
    "craiyon",
    "artbreeder",
    "wombodream",
    "runwayml",
    "recraftai",
    "bingimagecreator",
}

_TAG_PREFIXES = (
    "midjourney",
    "stablediffusion",
    "aigenerated",
    "artificialintelligence",
    "dreamup",
    "novelai",
    "comfyui",
    "sdxl",
    "dalle",
    "leonardoai",
)

_ROW_FIELDS = {
    "ai_generated",
    "author",
    "categories",
    "category",
    "content",
    "creator",
    "description",
    "image",
    "image_url",
    "is_ai_generated",
    "is_generated",
    "keywords",
    "license",
    "media",
    "name",
    "prompt",
    "source",
    "source_url",
    "subject",
    "tags",
    "thumbnail",
    "title",
    "url",
}


def _normalize(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = unescape(unescape(value[:_MAX_TEXT]))
    text = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]{0,1000}>", " ", text)
    text = "".join(
        character for character in text
        if character.isprintable() or character.isspace()
    )
    return " ".join(text.split())[:_MAX_TEXT]


def _iter_row_metadata(value: object, depth: int = 0) -> Iterator[str]:
    if depth > 3:
        return
    if isinstance(value, str):
        yield value[:_MAX_TEXT]
        return
    if isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            if index >= 80:
                break
            if str(key).casefold() in _ROW_FIELDS:
                yield from _iter_row_metadata(item, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value[:80]:
            yield from _iter_row_metadata(item, depth + 1)


def is_known_ai_source_url(value: object) -> bool:
    """Recognize dedicated generation hosts or explicit AI-art URL markers."""
    if not isinstance(value, str) or len(value) > 4096:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.casefold().rstrip(".")
    if any(host == known or host.endswith("." + known) for known in _AI_SOURCE_HOSTS):
        return True
    path = unquote(f"{parsed.path}?{parsed.query}")[:8192]
    return bool(_AI_URL_MARKER.search(path))


def _tagged_generation(text: str) -> bool:
    marker = re.search(r"\btags?\s*:\s*(.*)", text, re.IGNORECASE)
    if marker is None:
        return False
    for value in re.split(r"[;,|#]+", marker.group(1)):
        compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
        if compact in _TAG_VALUES:
            return True
        if any(compact.startswith(prefix) for prefix in _TAG_PREFIXES):
            return True
        if compact in {"lora", "lycoris", "checkpoint"}:
            return True
    return False


def _negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 100):start]
    after = text[end:end + 50]
    return (
        bool(
            re.search(
                r"\b(?:not|never|no|without|non)[\s_-]*(?:an?[\s_-]+)?$",
                before,
                re.IGNORECASE,
            )
        )
        or bool(
            re.match(
                r"\s*[:=]\s*(?:no|false|0)\b",
                after,
                re.IGNORECASE,
            )
        )
    )


def is_generated_metadata(*values: object) -> bool:
    """Recognize positive AI-generation evidence in bounded source metadata."""
    for value in values:
        if is_known_ai_source_url(value):
            return True
        text = _normalize(value)
        if not text:
            continue
        if _tagged_generation(text):
            return True
        human_context = bool(_HUMAN_MADE_CONTEXT.search(text))
        discussion = bool(_DISCUSSION_CONTEXT.search(text))
        for match in _EXPLICIT_GENERATED.finditer(text):
            if _negated(text, match.start(), match.end()):
                continue
            if human_context and discussion:
                continue
            return True
        if _GENERATION_PARAMETERS.search(text):
            if not (human_context and discussion):
                return True
        if _GENERATION_TOOL.search(text):
            if human_context or discussion:
                continue
            return True
    return False


def is_generation_software_metadata(value: object) -> bool:
    """Recognize generator software in a software/workflow metadata field."""
    text = _normalize(value)
    return bool(
        _GENERATION_TOOL.search(text)
        or _GENERATION_PARAMETERS.search(text)
    )


def generated_status(row: dict, *values: Any) -> bool | None:
    """Return source-supported generation status without treating absence as human."""
    known_negative = False
    if isinstance(row, dict):
        for key in ("is_ai_generated", "ai_generated", "is_generated"):
            value = row.get(key)
            if isinstance(value, str):
                normalized = value.strip().casefold()
                if normalized in {"true", "yes", "1"}:
                    return True
                known_negative = known_negative or normalized in {"false", "no", "0"}
            elif value is True:
                return True
            elif value is False:
                known_negative = True
    evidence = [*values]
    evidence.extend(_iter_row_metadata(row))
    if is_generated_metadata(*evidence):
        return True
    return False if known_negative else None


def candidate_is_generated(candidate: object) -> bool:
    """Recheck a candidate at cache/download boundaries with all source fields."""
    if getattr(candidate, "ai_generated", None) is True:
        return True
    values = (
        getattr(candidate, "title", ""),
        getattr(candidate, "description", ""),
        getattr(candidate, "creator", ""),
        getattr(candidate, "license", ""),
        getattr(candidate, "authenticity", ""),
        getattr(candidate, "provider", ""),
        getattr(candidate, "source_url", ""),
        getattr(candidate, "image_url", ""),
    )
    return is_generated_metadata(*values)


def _install_cache_namespace() -> None:
    from .cache import Cache

    if getattr(Cache, "_sis_ai_filter_namespace", "") == _CACHE_NAMESPACE:
        return

    original_get_query = Cache.get_query
    original_put_query = Cache.put_query
    original_get_image = Cache.get_image

    def get_query(self, key: str):
        return original_get_query(self, _CACHE_NAMESPACE + key)

    def put_query(self, key: str, images):
        return original_put_query(self, _CACHE_NAMESPACE + key, images)

    def get_image(self, url: str):
        image = original_get_image(self, url)
        if (
            image is not None
            and getattr(self.settings, "exclude_ai_generated", True)
            and candidate_is_generated(image)
        ):
            return None
        return image

    Cache.get_query = get_query
    Cache.put_query = put_query
    Cache.get_image = get_image
    Cache._sis_ai_filter_namespace = _CACHE_NAMESPACE


def _install_download_boundary() -> None:
    from .image_download import DownloadError, ImageDownloader

    if getattr(ImageDownloader, "_sis_ai_filter_v2", False):
        return

    original_download = ImageDownloader.download

    async def download(self, candidate):
        generated = candidate_is_generated(candidate)
        if generated:
            candidate.ai_generated = True
        if generated and getattr(self.settings, "exclude_ai_generated", True):
            raise DownloadError(
                "Source metadata identifies a generated image",
                reason="Generated image excluded",
            )
        return await original_download(self, candidate)

    ImageDownloader.download = download
    ImageDownloader._sis_ai_filter_v2 = True


def install() -> None:
    """Install the stricter detector at provider, cache, and download boundaries."""
    from . import search_providers

    search_providers.is_known_generated = is_generated_metadata
    search_providers.is_known_generation_software = (
        is_generation_software_metadata
    )
    search_providers._generated_status = generated_status
    _install_cache_namespace()
    _install_download_boundary()

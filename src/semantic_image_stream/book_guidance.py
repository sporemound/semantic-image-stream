"""Use a supplied visual brief without turning its themes into story facts.

Only short, sanitized phrases enter discovery. The complete manual brief stays
local for ranking, separately from the full retained transcript.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re

from .semantic_analysis import _public_text, sanitize_query, tokens

GUIDANCE_VERSION = "visual-brief-v1"
_REGIONS = set("arctic antarctic polar tropical mediterranean alpine coastal desert jungle tundra forest ocean marine underwater lunar martian cosmic urban rural".split())
_DISTINCT_REGIONS = {"arctic", "antarctic", "tropical", "mediterranean", "lunar", "martian"}
_SETTINGS = set("landscape landscapes community communities village villages town towns city cities wilderness coast coastlines coastline shore shores sea ocean ice tundra forest forests woodland woods jungle rainforest desert mountain mountains valley valleys river rivers lake lakes island islands beach beaches glacier glaciers cave caves countryside street streets harbor harbour camp camps settlement settlements space planet planets laboratory museum castle castles canyon canyons savanna prairie meadow swamp marsh".split())
_SUBJECTS = set("door doors cabin cabins tent tents caribou muskox atv boat boats ship ships fossil fossils dinosaur dinosaurs rock rocks whale whales bear bears wolf wolves bird birds tree trees plant plants flower flowers orchid orchids satellite satellites capsule telescope telescopes reef coral artifact artifacts ruins ruin temple temples monument monuments building buildings architecture bridge bridges train trains machinery diagram diagrams chart charts map maps reconstruction reconstructions manuscript manuscripts pottery sculpture sculptures painting paintings portrait portraits photograph photographs".split())
_ABSTRACT = set("book novel story stories storytelling anthology genre horror supernatural psychological suspense silence silence uncanny uncertainty resilience isolation vulnerability trauma memory memories cultural traditional knowledge ancestral belief beliefs eerie unsettling mysterious atmospheric dread phenomena encounter encounters survival life independent short shifting character characters setting settings favor prefer show image images pictures only avoid exclude no not never except about of the and a an in on for with from".split())
_NEGATIVE = re.compile(r"^\s*(?:avoid|exclude|do not|don't|no\b|not\b|never|without)\s*[:=-]?\s*", re.I)
_LABEL = re.compile(r"^\s*(book|title|author|setting|location|environment|landscape|focus|subjects?|visuals?|themes?|mood|style|avoid|exclude)\s*:\s*(.*)$", re.I)


def _clauses(text):
    clean = _public_text(text[:2000])
    for part in re.split(r"[,;\n]|(?<=[.!?])\s+", clean):
        part = part.strip()
        if part:
            yield part


def _concrete_phrase(clause, vocabulary):
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", clause, re.UNICODE)
    matches = [index for index, word in enumerate(words) if word.casefold() in vocabulary]
    if not matches:
        return ""
    # Keep the stated descriptor beside a noun, not a long list of themes.
    first, last = matches[0], matches[-1]
    selected = words[max(0, first - 2):min(len(words), last + 2)]
    selected = [word for word in selected if word.casefold() not in _ABSTRACT]
    return sanitize_query(" ".join(selected), 5)


def visual_brief(text):
    """Bounded explicitly supplied places/objects/themes; no inferred geography."""
    settings, subjects, themes = [], [], []
    for order, clause in enumerate(_clauses(text if isinstance(text, str) else "")):
        label = _LABEL.match(clause)
        role, content = (label[1].casefold(), label[2]) if label else ("", clause)
        if role in {"book", "title", "author", "avoid", "exclude"} or _NEGATIVE.match(content):
            continue
        setting = _concrete_phrase(content, _SETTINGS | _REGIONS)
        subject = _concrete_phrase(content, _SUBJECTS)
        if role in {"setting", "location", "environment", "landscape"} and not setting:
            setting = sanitize_query(content, 5) if tokens(content) - _ABSTRACT else ""
        if role in {"focus", "subject", "subjects", "visual", "visuals"} and not subject:
            subject = sanitize_query(content, 5) if tokens(content) - _ABSTRACT else ""
        if setting and role not in {"theme", "themes", "mood", "style"}:
            priority = (10 if role in {"setting", "location", "environment", "landscape"} else 0)
            priority += 5 * bool(tokens(setting) & _REGIONS)
            settings.append((priority, order, setting))
        if subject and role not in {"theme", "themes", "mood", "style"}:
            subjects.append(subject)
        if not setting and not subject:
            clean = sanitize_query(content, 8)
            if clean:
                themes.append(clean)
    ordered = [value for _, _, value in sorted(settings, key=lambda item: (-item[0], item[1]))]
    def unique(values, limit):
        result, seen = [], set()
        for value in values:
            if value.casefold() not in seen:
                result.append(value)
                seen.add(value.casefold())
        return result[:limit]
    return {"setting_anchors": unique(ordered, 8), "subjects": unique(subjects, 8), "themes": unique(themes, 6)}


def _manual_pair(topic, settings):
    saved = getattr(topic, "image_guidance", None)
    text = saved if saved is not None else getattr(settings, "book_context", "")
    strength = getattr(topic, "guidance_strength", None)
    strength = getattr(settings, "book_guidance_strength", .65) if strength is None else strength
    return text.strip(), float(strength)


def _setting_conflict(topic, text):
    scene = " ".join((getattr(topic, "scene", ""), *getattr(topic, "visual_subjects", ())))
    actual = tokens(scene)
    recorded = tokens(" ".join(getattr(topic, "transcript", ()) or (getattr(topic, "context", ""),)))
    # A manual environment copied into a model's scene is not an independent
    # observation. Only recorded physical words can override the manual brief.
    physical = actual & recorded
    manual = tokens(" ".join(visual_brief(text)["setting_anchors"]))
    # Coasts, forests, towns and underwater scenes can all occur within the
    # supplied region. A new descriptive environment is not a new geography.
    current_regions = physical & _DISTINCT_REGIONS
    manual_regions = manual & _DISTINCT_REGIONS
    if current_regions and manual_regions and not current_regions & manual_regions:
        return True
    # Explicit newly named locations override an old background's region.
    named = re.findall(r"\b(?:in|near|at|to)\s+(?:the\s+)?([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})", scene)
    return any(tokens(place) <= recorded and not tokens(place) & tokens(text)
               for place in named if tokens(place) and not re.search(r"['’]s\b", place))


def guidance_for(topic, settings):
    """Immutable frame guidance, scoped to the current recorded environment."""
    text, strength = _manual_pair(topic, settings)
    if not text or strength <= 0:
        return "", 0.0
    # Frozen text embeddings do not reliably represent negation. A negative
    # image-selection directive must not reward the thing the user excluded.
    clauses, positive = list(_clauses(text)), []
    for clause in clauses:
        label = _LABEL.match(clause)
        if _NEGATIVE.match(clause) or (label and label[1].casefold() in {"avoid", "exclude"}):
            continue
        positive.append(clause)
    if len(positive) != len(clauses):
        text = ", ".join(positive)
    if _setting_conflict(topic, text):
        # Retain style/theme preferences, not a contradictory physical setting.
        text = ", ".join(clause for clause in _clauses(text)
                         if not visual_brief(clause)["setting_anchors"])
    return text, strength if text else 0.0


def snapshot_guidance(topic, settings):
    text = settings.book_context.strip()
    if ((not text and topic.image_guidance is None) or
            (topic.image_guidance == text and topic.guidance_strength == settings.book_guidance_strength)):
        return topic
    return replace(topic, image_guidance=text,
                   guidance_strength=settings.book_guidance_strength)


def guidance_seed(settings, transcript=()):
    """An explicit manual setting can guide images while story inference loads."""
    from .models import Topic
    if not settings.book_context.strip() or settings.book_guidance_strength <= 0:
        return None
    brief = visual_brief(settings.book_context)
    if not brief["setting_anchors"]:
        return None
    anchor = brief["setting_anchors"][0]
    transcript = tuple(transcript)
    return snapshot_guidance(Topic(anchor, " ".join(transcript), visual_subjects=(anchor,),
                                   transcript=transcript, guidance_only=True), settings)


def refresh_guidance(topic, settings):
    """Remove expired manual-only qualifiers without changing recorded words."""
    if topic.guidance_only:
        return guidance_seed(settings, topic.transcript)
    previous = topic.image_guidance or ""
    old = visual_brief(previous)
    current = visual_brief(settings.book_context if settings.book_guidance_strength > 0 else "")
    allowed = tokens(" ".join(topic.transcript or (topic.context,)))
    allowed |= tokens(" ".join((*current["setting_anchors"], *current["subjects"])))
    removed = tokens(" ".join((*old["setting_anchors"], *old["subjects"]))) - allowed
    if removed:
        def clean(text):
            return " ".join(word for word in text.split() if not tokens(word) & removed).strip()
        topic = replace(topic, query=sanitize_query(clean(topic.query)), scene=clean(topic.scene),
                        summary=clean(topic.summary), concepts=tuple(clean(value) for value in topic.concepts if clean(value)),
                        visual_subjects=tuple(clean(value) for value in topic.visual_subjects if clean(value)),
                        queries=tuple(sanitize_query(clean(value)) for value in topic.queries if clean(value)))
    return snapshot_guidance(topic, settings)


def guidance_identity(topic, settings):
    text, strength = guidance_for(topic, settings)
    if not text:
        return ""
    data = [GUIDANCE_VERSION, text, strength]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def guided_queries(topic, settings):
    """Keep discovery about the current subject, with a compatible setting cue."""
    base = sanitize_query(topic.query)
    if not base:
        return []
    text, strength = guidance_for(topic, settings)
    if not text or strength <= 0:
        return [base]
    brief = visual_brief(text)
    anchors = brief["setting_anchors"]
    if not anchors:
        return [base]
    # The subject comes first and reserves words; a long narrative query must
    # still leave room for the explicitly supplied visual environment.
    result = []
    for anchor in anchors[:2]:
        novel = [word for word in anchor.split() if not tokens(word) & tokens(base)]
        if not novel:
            combined = base
        else:
            count = min(3, len(novel))
            words = base.split()
            available = 8 - count
            if len(words) > available:
                preferred = tokens(topic.visual_subjects[0]) if topic.visual_subjects else set()
                retained = [index for index, word in enumerate(words) if tokens(word) & preferred]
                retained = retained[:available]
                if len(retained) < available and len(words) - 1 not in retained:
                    retained.append(len(words) - 1)
                retained += [index for index in range(len(words)) if index not in retained][:available - len(retained)]
                words = [words[index] for index in sorted(retained)]
            combined = sanitize_query(" ".join((*words, *novel[:count])))
        if combined and combined.casefold() not in {value.casefold() for value in result}:
            result.append(combined)
    for anchor in anchors[:1]:
        if anchor.casefold() not in {value.casefold() for value in result}:
            result.append(anchor)
    return result or [base]

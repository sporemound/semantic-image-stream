"""Small local, provenance-checked human story examples, never story facts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re

from .semantic_analysis import NARRATIVE, tokens
from .training_paths import training_file

MAX_PACK_BYTES = 2 * 1024 * 1024
MAX_EXAMPLES = 1024
MAX_PROMPT_CHARS = 1200
DATASET = "WorkInTheDark/FairytaleQA"
ATTRIBUTES = {"character", "setting", "action", "outcome resolution"}


def _text(value, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Invalid local story example text")
    return value


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _generic_tokens(text):
    return tokens(text) - NARRATIVE - {"story", "question", "answer", "passage", "character"}


@dataclass(frozen=True)
class StoryReferencePack:
    examples: tuple[dict, ...] = ()
    identity: str = ""
    status: str = "missing"
    _terms: tuple[frozenset[str], ...] = field(default=(), repr=False)

    def select(self, text: str) -> list[dict]:
        """At most two literal QAs selected locally, with a strict prompt bound."""
        wanted = _generic_tokens(text)
        # Matching only an incidental capitalized name is a poor illustration
        # selector. Common content words shared by multiple examples survive.
        capitals = tokens(" ".join(re.findall(r"\b[A-Z][a-z]{2,}\b", text)))
        rare_names = {word for word in capitals if sum(word in terms for terms in self._terms) < 2}
        wanted -= rare_names
        ranked = []
        for index, terms in enumerate(self._terms):
            overlap = wanted & terms
            if overlap:
                ranked.append((len(overlap), len(overlap) / max(1, len(terms)), index))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected = []
        for _, _, index in ranked:
            row = json.loads(_encoded(self.examples[index]))
            if len(_encoded([*selected, row])) > MAX_PROMPT_CHARS:
                # Keep a contiguous literal source excerpt around the human
                # answer. Never shorten the question/answer or invent context.
                answer_start = row["source"].casefold().find(row["answer"].casefold())
                if answer_start < 0:
                    continue
                source = row["source"]
                row["source"] = ""
                remaining = MAX_PROMPT_CHARS - len(_encoded([*selected, row]))
                if remaining < len(row["answer"]):
                    continue
                start = max(0, answer_start - max(0, remaining - len(row["answer"])) // 2)
                end = min(len(source), start + remaining)
                row["source"] = source[start:end]
                # JSON escapes can occupy more than one character.
                while len(_encoded([*selected, row])) > MAX_PROMPT_CHARS:
                    if end > answer_start + len(row["answer"]):
                        end -= 1
                    elif start < answer_start:
                        start += 1
                    else:
                        break
                    row["source"] = source[start:end]
            if len(_encoded([*selected, row])) <= MAX_PROMPT_CHARS:
                selected.append(row)
            if len(selected) == 2:
                break
        return selected


def load_story_references(training_dir: str | Path) -> StoryReferencePack:
    """Read one bounded JSON file; malformed/held-out data stays disabled."""
    path = Path(training_dir) / "story_examples.json"
    try:
        if not path.exists():
            return StoryReferencePack()
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if size > MAX_PACK_BYTES:
                raise ValueError("Local example pack exceeded its size bound")
            raw = source.read(size + 1)
            if len(raw) > size or source.read(1):
                raise ValueError("Local example pack changed while reading")
        if len(raw) > MAX_PACK_BYTES:
            raise ValueError("Local example pack exceeded its size bound")
        data = json.loads(raw)
        if (not isinstance(data, dict) or set(data) != {"version", "examples"}
                or type(data["version"]) is not int or data["version"] != 1
                or not isinstance(data["examples"], list) or not 0 < len(data["examples"]) <= MAX_EXAMPLES):
            raise ValueError("Invalid local example pack")
        rows, ids = [], set()
        for row in data["examples"]:
            if not isinstance(row, dict) or set(row) != {"id", "source", "question", "answer", "attribute", "provenance"}:
                raise ValueError("Invalid local example fields")
            identity = _text(row["id"], 160)
            if identity in ids or row["attribute"] not in ATTRIBUTES:
                raise ValueError("Invalid local example identity or skill")
            ids.add(identity)
            for key, maximum in (("source", 1200), ("question", 240), ("answer", 300)):
                _text(row[key], maximum)
            if row["answer"].casefold() not in row["source"].casefold():
                raise ValueError("Explicit reference answer is not present in its source")
            provenance = row["provenance"]
            required = {"dataset", "revision", "split", "license", "story_name", "row_index", "answer_type"}
            if (not isinstance(provenance, dict) or not required <= set(provenance)
                    or set(provenance) - required - {"license_url", "source_url"}
                    or provenance["dataset"] != DATASET or provenance["split"] != "train"
                    or provenance["license"] != "apache-2.0" or provenance["answer_type"] != "explicit"
                    or not isinstance(provenance["revision"], str)
                    or not re.fullmatch(r"[0-9a-f]{40}", provenance["revision"])
                    or type(provenance["row_index"]) is not int or provenance["row_index"] < 0):
                raise ValueError("Invalid local example training provenance")
            _text(provenance["story_name"], 160)
            for url in ("license_url", "source_url"):
                if url in provenance:
                    _text(provenance[url], 400)
            rows.append(row)
        identity = hashlib.sha256(_encoded(data).encode("utf-8")).hexdigest()
        terms = tuple(frozenset(_generic_tokens(row["source"] + " " + row["question"])) for row in rows)
        return StoryReferencePack(tuple(rows), identity, "ready", terms)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, MemoryError):
        return StoryReferencePack(status="invalid")


def story_references_for(settings) -> StoryReferencePack:
    if not getattr(settings, "use_story_examples", True):
        return StoryReferencePack(status="disabled")
    return load_story_references(training_file(settings, "story_examples.json").parent)


def inspect_story_examples(settings) -> dict:
    """Safe, local readiness information for the settings panel."""
    pack = story_references_for(settings)
    messages = {"ready": f"{len(pack.examples)} human story examples available locally.",
                "disabled": "Story examples are disabled; the saved pack is retained.",
                "missing": "No local story example pack installed.",
                "invalid": "The local story example pack could not be verified and was left unused."}
    return {"status": pack.status, "examples": len(pack.examples), "message": messages[pack.status]}

"""Provider-independent, bounded local context analysis.

The default is a transparent phrase/entity heuristic, not an LLM. Install the
spaCy extra and English model for dependency-based noun phrases and entities.
Only a compact sanitized query leaves the application; context remains local.
"""
from collections import Counter
import re
from typing import Protocol, Sequence

from .models import Topic, TranscriptSegment

STOP = set("a an the and or but if then as at by for from in into of on to with without is are was were be been being it its this that these those i you we they he she my our your their has have had do does did can could will would should may might about also really very just now here there more most less many much some any not so than such what which who how when where why know think say said says talking talk discuss discussion today look let's us well because during through over under between after before new another something things people time want going see get like one two first next last let's that's don't doesn't".split())
# These phrase mappings preserve scientific/historical specificity when present;
# they are vocabulary, never canned queries or simulated topic events.
CONCEPTS = {
    "black hole": "science", "event horizon": "science", "general relativity": "science",
    "dark matter": "science", "dark energy": "science", "cosmic microwave background": "science",
    "solar system": "science", "neutron star": "science", "james webb": "science",
    "hubble telescope": "science", "galaxy": "science", "galaxies": "science",
    "nebula": "science", "supernova": "science", "astronomy": "science",
    "photosynthesis": "science", "chloroplast": "science", "chloroplasts": "science",
    "pollination": "science", "orchid": "science", "orchids": "science",
    "mycorrhiza": "science", "mycorrhizal": "science", "plant roots": "science",
    "botany": "science", "evolution": "science", "dna": "science",
    "quantum": "science", "quantum experiments": "science", "quantum mechanics": "science",
    "quantum entanglement": "science", "entanglement": "science", "superconductivity": "science",
    "double slit": "science", "interference pattern": "science",
    "mitochondria": "science", "plate tectonics": "science",
    "french revolution": "history", "industrial revolution": "history",
    "world war": "history", "apollo 11": "history", "roman empire": "history",
    "ancient egypt": "history", "moon landing": "history", "napoleon": "history",
    "civil war": "history", "renaissance": "history", "berlin wall": "history",
}


def sanitize_query(text: str, max_words: int = 8) -> str:
    text = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]+", " ", text)
    text = re.sub(r"\b(?:\d[\s()+.-]*){7,}\b|[A-Za-z]:[\\/]\S+", " ", text)
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, re.UNICODE)
    # No query operators, file paths, URLs, or identifier-like long numbers.
    safe = [w for w in words if len(w) <= 40 and not (w.isdigit() and len(w) > 4)]
    return " ".join(safe[:max_words])[:160]


def tokens(text: str) -> set[str]:
    words = {x.casefold() for x in re.findall(r"[^\W_]+", text) if len(x) > 2 and x.casefold() not in STOP}
    def normalize(word):
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
    return {normalize(word) for word in words}


class SemanticAnalyzer(Protocol):
    def analyze(self, segments: Sequence[TranscriptSegment]) -> Topic | None: ...


class ContextualAnalyzer:
    def __init__(self, use_spacy: bool = False):
        self.nlp = None
        if use_spacy:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, segments: Sequence[TranscriptSegment]) -> Topic | None:
        if not segments:
            return None
        context = " ".join(s.text.strip() for s in segments)[-4000:]
        recent = " ".join(s.text for s in segments if s.end >= segments[-1].end - 16)[-1200:]
        # Explicit topic-turn markers sever obsolete query context while the
        # complete rolling transcript remains available locally.
        transitions = list(re.finditer(
            r"\b(?:but now|switching (?:topics|subjects)|moving on(?: to)?|"
            r"turning (?:now )?to|(?:let's|let us) (?:move|turn)(?: on)? to)\b", recent, re.IGNORECASE))
        if transitions:
            recent = recent[transitions[-1].end():]
        if len(re.findall(r"\w+", recent)) < 8:
            return None
        phrases: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        known = set()
        recent_lower = recent.casefold()
        for phrase, kind in CONCEPTS.items():
            pattern = r"\b" + re.escape(phrase) + r"s?\b"
            count = len(re.findall(pattern, recent_lower))
            if count:
                phrases[phrase] += 5 + count * 2
                kinds[kind] += count
                known.add(phrase)
        # Retain a compound discussion anchor through explanatory sentences,
        # and expand a lone broad term ("quantum") using established context.
        # Explicit transitions disable this carry-over.
        if not transitions and (not known or all(len(p.split()) == 1 for p in known)):
            for phrase, kind in CONCEPTS.items():
                if len(phrase.split()) < 2:
                    continue
                if re.search(r"\b" + re.escape(phrase) + r"s?\b", context.casefold()):
                    if not known or any(tokens(phrase) & tokens(p) for p in known):
                        phrases[phrase] += 6
                        kinds[kind] += 1
                        known.add(phrase)
        # Capitalized multiword names retain people, organizations and places.
        for name in re.findall(r"\b[A-Z][\w'-]+(?:\s+(?:[A-Z][\w'-]+|of|the)){1,5}", recent):
            name = " ".join(w for w in name.split() if w.casefold() not in STOP)
            if len(name.split()) >= 2:
                phrases[name] += 8
        if self.nlp:
            doc = self.nlp(recent)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "FAC"}:
                    phrases[ent.text] += 9
            for chunk in doc.noun_chunks:
                phrase = " ".join(t.text for t in chunk if not t.is_stop and not t.is_punct)
                if 1 <= len(phrase.split()) <= 5:
                    phrases[phrase] += 3
        else:
            # RAKE-style phrase grouping across clauses and function words.
            words = re.findall(r"[\w'-]+|[.,;:!?]", recent_lower)
            runs, run = [], []
            for word in words + ["."]:
                if word in STOP or not re.search(r"\w", word):
                    if run:
                        runs.append(run)
                    run = []
                else:
                    run.append(word)
            freq = Counter(w for r in runs for w in r)
            for r in runs:
                # Long clauses are split into meaningful two/three word groups.
                if len(r) <= 5:
                    phrase = " ".join(r)
                    phrases[phrase] += sum(freq[w] for w in r) / len(r) + min(len(r), 3)
                else:
                    for i in range(len(r) - 1):
                        phrases[" ".join(r[i:i + 2])] += (freq[r[i]] + freq[r[i + 1]]) / 2
        # Established compound concepts make better queries than surrounding
        # narration ("interview explains ..."). Other phrases still support
        # the general-domain fallback and local topic context.
        proper_names = {p for p in phrases if p[:1].isupper() and len(p.split()) >= 2}
        ranked = [(sanitize_query(p, 5), score) for p, score in sorted(phrases.items(),
                  key=lambda pair: (pair[1], len(pair[0].split())), reverse=True)
                  if tokens(p) and (p in known if known else p in proper_names if proper_names else True)]
        if not ranked:
            return None
        selected = []
        used: set[str] = set()
        for phrase, score in ranked:
            novel = tokens(phrase) - used
            if not novel:
                continue
            selected.append(phrase)
            used |= tokens(phrase)
            if len(selected) >= (2 if known else 3) or len(used) >= 7:
                break
        if known:
            if len(used) < 2:
                for phrase, _ in phrases.most_common():
                    if len(tokens(phrase) - used) >= 1 and len(tokens(phrase)) >= 2:
                        selected.append(phrase)
                        used |= tokens(phrase)
                        break
            # Include a nearby proper name only when it identifies the current
            # subject, rather than adding unrelated entities from earlier text.
            names = [p for p, _ in phrases.most_common() if p not in known and p[:1].isupper()
                     and not tokens(p) & used and len(p.split()) >= 2]
            if names and len(" ".join(selected).split()) <= 4:
                selected.append(names[0])
        query_words = []
        seen = set()
        for word in sanitize_query(" ".join(selected)).split():
            if word.casefold() not in seen:
                query_words.append(word)
                seen.add(word.casefold())
        query = " ".join(query_words)
        if not query or len(tokens(query)) < 2:
            return None
        kind = kinds.most_common(1)[0][0] if kinds else "general"
        return Topic(query=query, context=context, concepts=tuple(selected), kind=kind)


def build_analyzer(settings) -> SemanticAnalyzer:
    return ContextualAnalyzer(use_spacy=settings.semantic_provider == "spacy")


class TopicGate:
    """Sensitivity controls concept-set distance needed for a new search."""
    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity
        self.previous: Topic | None = None

    def changed(self, topic: Topic) -> bool:
        if self.previous is None:
            return True
        a, b = tokens(topic.query), tokens(self.previous.query)
        distance = 1 - len(a & b) / max(1, len(a | b))
        return distance >= (0.8 - self.sensitivity * 0.6)

    def accept(self, topic: Topic) -> None:
        self.previous = topic

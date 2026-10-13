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
STOP.update("didn't wasn't weren't isn't aren't haven't hasn't hadn't can't couldn't won't wouldn't shouldn't mustn't needn't i'm i've i'd i'll you're you've you'll we're we've we'll they're they've they'll he's she's it's there's here's who's what's where's how's".split())
NARRATIVE = set("reported happened happens explained asked answered looked walked spoke heard found returned went came told began seemed turned pushed pulled watched opened closed entered waited stood nodded smiled whispered shouted still suddenly simply slowly finally already again".split())
TRANSITIONS = re.compile(
    r"\b(?:but now|switching (?:topics|subjects)|moving on(?: to)?|"
    r"turning (?:now )?to|(?:let's|let us) (?:move|turn)(?: on)? to)\b", re.IGNORECASE)
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
    "spy satellite": "history", "reconnaissance satellite": "history",
    "space capsule": "science",
}


def _public_text(text: str) -> str:
    # Strip complete quoted paths before the token-level cleanup, so a folder
    # name containing spaces cannot accidentally become a search subject.
    text = re.sub(r"[\"'](?:[A-Za-z]:[\\/]|\\\\|/)[^\"'\r\n]*[\"']", " . ", text)
    text = re.sub(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]+", " . ", text)
    return re.sub(r"\b(?:\d[\s()+.-]*){7,}\b|[A-Za-z]:[\\/]\S+|\\\\\S+|/\S+/\S+", " . ", text)


def sanitize_query(text: str, max_words: int = 8) -> str:
    text = _public_text(text)
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, re.UNICODE)
    # No query operators, file paths, URLs, or identifier-like long numbers.
    safe = [w for w in words if len(w) <= 40 and not (w.isdigit() and len(w) > 4)]
    return " ".join(safe[:max_words])[:160]


def tokens(text: str) -> set[str]:
    text = text.replace("’", "'")
    words = {part.casefold() for word in re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)
             if word.casefold() not in STOP for part in re.findall(r"[^\W_]+", word)
             if len(part) > 2 and part.casefold() not in STOP}
    def normalize(word):
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
    return {normalize(word) for word in words}


def _active_context(text: str) -> str:
    transitions = list(TRANSITIONS.finditer(text.replace("’", "'")))
    return text[transitions[-1].end():] if transitions else text


def _analysis_text(text: str) -> str:
    text = _public_text(text).replace("’", "'")
    return re.sub(r"\b[^\W_]+(?:'[^\W_]+)+\b",
                  lambda match: " " if match[0].casefold() in STOP else match[0], text)


def _named_anchors(text: str) -> Counter[str]:
    names: Counter[str] = Counter()
    for name in re.findall(r"\b[A-Z][\w'-]+(?:\s+(?:[A-Z][\w'-]+|of|the)){1,5}", text):
        name = " ".join(w for w in name.split() if w.casefold() not in STOP)
        if len(name.split()) >= 2 and len(tokens(name)) >= 2:
            names[name] += 1
    for name in re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", text):
        if name.casefold() not in STOP and tokens(name):
            names[name] += 1
    # A single capitalized word is usually a sentence start. An explicit name
    # introduction provides evidence without treating every such word as one.
    for name in re.findall(r"\b(?:named|called|project|program|novel|ship|vessel)\s+([A-Z][\w'-]+)\b", text):
        if name.casefold() not in STOP and name.casefold() not in NARRATIVE:
            names[name] += 1
    return names


def _phrase_runs(text: str) -> list[list[str]]:
    words = re.findall(r"[\w'-]+|[.,;:!?]", text.casefold())
    runs, run = [], []
    for word in words + ["."]:
        if word in STOP or not re.search(r"\w", word):
            if run:
                runs.append(run)
            run = []
        else:
            run.append(word)
    return runs


def _repeated_compounds(text: str) -> Counter[str]:
    compounds: Counter[str] = Counter()
    for run in _phrase_runs(text):
        for length in (2, 3):
            for index in range(len(run) - length + 1):
                words = run[index:index + length]
                if not any(word in NARRATIVE for word in words) and len(tokens(" ".join(words))) >= 2:
                    compounds[" ".join(words)] += 1
    return Counter({phrase: count for phrase, count in compounds.items() if count >= 2})


def fallback_queries(topic: Topic) -> list[str]:
    """At most two local subject alternatives for a search with no results.

    Narration and a full transcript are never submitted as fallback queries.
    Explicit topic turns also apply when they are older than the recent window.
    """
    if topic.queries:
        # Model-derived alternatives share the complete grounded story state.
        # Do not replace them with incidental names from raw narration.
        alternatives, seen = [], {sanitize_query(topic.query).casefold()}
        for query in topic.queries:
            clean = sanitize_query(query)
            if clean and clean.casefold() not in seen:
                alternatives.append(clean)
                seen.add(clean.casefold())
                if len(alternatives) == 2:
                    break
        return alternatives
    public_context = _public_text(topic.context)
    active = _analysis_text(_active_context(public_context))
    # Start at the current subject's latest introduction. Other named entities
    # in that local passage are useful alternatives; names from a preceding
    # subject (even without an explicit turn marker) are not.
    positions = []
    for concept in topic.concepts:
        clean = _analysis_text(concept).strip()
        if not tokens(clean):
            continue
        matches = list(re.finditer(r"\b" + re.escape(clean) + r"s?\b",
                                   active, re.IGNORECASE))
        if matches:
            positions.append(matches[-1].start())
    if positions:
        introduction = min(positions)
        boundaries = list(re.finditer(r"[.!?;\n]", active[:introduction]))
        active = active[boundaries[-1].end():] if boundaries else active
    names = _named_anchors(active)
    compounds = _repeated_compounds(active)
    for phrase in CONCEPTS:
        if len(phrase.split()) >= 2 and re.search(r"\b" + re.escape(phrase) + r"s?\b", active.casefold()):
            compounds[phrase] += 3
    subjects: Counter[str] = Counter()
    for phrase, count in compounds.items():
        subjects[phrase] += 4 + count
    for phrase, count in names.items():
        subjects[phrase] += 5 + count
    transitioned = _active_context(public_context) != public_context
    for concept in topic.concepts:
        clean = _analysis_text(concept).strip()
        # A concept outside the active portion must not resurrect an old topic.
        present = bool(re.search(r"\b" + re.escape(clean) + r"s?\b", active, re.IGNORECASE))
        if transitioned and not present:
            continue
        if clean in names or clean.casefold() in compounds or (
                clean.casefold() in CONCEPTS and len(clean.split()) >= 2):
            subjects[clean] += 10
        elif not transitioned and _named_anchors(clean):
            subjects[clean] += 8
    alternatives = []
    used = {sanitize_query(topic.query).casefold()}
    for phrase, _ in subjects.most_common():
        query = sanitize_query(phrase)
        key = query.casefold()
        if not query or key in used or not tokens(query):
            continue
        alternatives.append(query)
        used.add(key)
        if len(alternatives) == 2:
            break
    return alternatives


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
        active_context = _analysis_text(_active_context(_public_text(context)))
        recent = _analysis_text(_active_context(_public_text(
            " ".join(s.text for s in segments if s.end >= segments[-1].end - 16)[-1200:])))
        # Explicit topic-turn markers sever obsolete query context while the
        # complete rolling transcript remains available locally.
        if len(re.findall(r"\w+", recent)) < 8:
            return None
        phrases: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        known = set()
        names = _named_anchors(recent)
        compounds = _repeated_compounds(recent)
        recent_lower = recent.casefold()
        for phrase, kind in CONCEPTS.items():
            pattern = r"\b" + re.escape(phrase) + r"s?\b"
            count = len(re.findall(pattern, recent_lower))
            if count:
                phrases[phrase] += 5 + count * 2
                kinds[kind] += count
                known.add(phrase)
        # Explanatory narration often omits the subject's name. Retain locally
        # established anchors, while a new explicit subject wins immediately.
        if not names and not compounds and (not known or all(len(p.split()) == 1 for p in known)):
            for phrase, kind in CONCEPTS.items():
                if len(phrase.split()) < 2:
                    continue
                if re.search(r"\b" + re.escape(phrase) + r"s?\b", active_context.casefold()):
                    if not known or any(tokens(phrase) & tokens(p) for p in known):
                        phrases[phrase] += 6
                        kinds[kind] += 1
                        known.add(phrase)
            if not known:
                names = _named_anchors(active_context)
                compounds = _repeated_compounds(active_context)
        for name, count in names.items():
            phrases[name] += 8 + count
        for phrase, count in compounds.items():
            phrases[phrase] += 6 + count
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
            runs = _phrase_runs(recent)
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
        proper_names = set(names) | {p for p in phrases if p[:1].isupper() and len(p.split()) >= 2}
        preferred = known or proper_names or set(compounds)
        ranked = [(sanitize_query(p, 5), score) for p, score in sorted(phrases.items(),
                  key=lambda pair: (pair[1], len(pair[0].split())), reverse=True)
                  if tokens(p) and (p in preferred if preferred else True)]
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
        if not query or (len(tokens(query)) < 2 and query not in proper_names):
            return None
        kind = kinds.most_common(1)[0][0] if kinds else "general"
        return Topic(query=query, context=context, concepts=tuple(selected), kind=kind)


def build_analyzer(settings, stop_event=None, on_status=None) -> SemanticAnalyzer:
    if settings.semantic_provider == "ollama":
        from .narrative_context import LocalNarrativeAnalyzer
        return LocalNarrativeAnalyzer(settings, stop_event=stop_event, on_status=on_status)
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

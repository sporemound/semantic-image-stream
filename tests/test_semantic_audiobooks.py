import pytest

from semantic_image_stream.models import Topic, TranscriptSegment
from semantic_image_stream.semantic_analysis import ContextualAnalyzer, fallback_queries, sanitize_query, tokens


def analyze_context(earlier, later):
    return ContextualAnalyzer().analyze([
        TranscriptSegment(0, 10, earlier), TranscriptSegment(30, 38, later),
    ])


def test_audiobook_subject_survives_explanatory_narration():
    topic = analyze_context(
        "The spy satellite project named Corona photographed secret installations. "
        "The KGB resident worked in Mexico City and tracked reconnaissance satellites.",
        "He reported another failed mission and didn't know what would happen. "
        "The captain turned the wheel and pushed the throttle.",
    )
    assert "satellite" in topic.query
    assert "didn't" not in topic.query
    alternatives = fallback_queries(topic)
    assert 1 <= len(alternatives) <= 2
    assert all(query in {"spy satellite", "reconnaissance satellite", "Mexico City", "KGB", "Corona"}
               for query in alternatives)


def test_recent_space_capsule_subject_wins_over_old_spy_satellites():
    topic = analyze_context(
        "Spy satellites photographed launch sites while the KGB worked in Mexico City.",
        "The space capsule landed in the sea after reentry. The captain watched the recovery boats approach.",
    )
    assert topic.query == "space capsule"
    assert "satellite" not in " ".join(fallback_queries(topic))
    assert "Mexico" not in " ".join(fallback_queries(topic))


@pytest.mark.parametrize("earlier,names", [
    ("Sherlock Holmes and John Watson studied the footprints outside the abandoned house.",
     {"Sherlock Holmes", "John Watson"}),
    ("Ada Lovelace discussed the analytical engine with Charles Babbage in his study.",
     {"Ada Lovelace", "Charles Babbage"}),
])
def test_generic_names_continue_through_audiobook_narration(earlier, names):
    topic = analyze_context(earlier,
                            "They didn't speak while examining the pages. "
                            "It wasn't clear how the stranger had found the room.")
    assert all(name in topic.query for name in names)
    assert set(fallback_queries(topic)) == names


def test_acronym_is_a_valid_subject_without_filling_query_with_narration():
    topic = analyze_context(
        "The KGB monitored the operation and recruited agents in the city.",
        "They didn't know which visitor had entered the building. The officer waited by the stairs.",
    )
    assert topic.query == "KGB"


def test_repeated_general_compound_continues_without_named_entities():
    topic = analyze_context(
        "A steam engine powered the machinery. The steam engine drove the factory pumps.",
        "The workers didn't know which valve had broken. They waited outside while the crew checked every part.",
    )
    assert "steam engine" in topic.query


def test_explicit_transition_older_than_recent_window_severs_old_subject():
    segments = [
        TranscriptSegment(0, 8, "Spy satellites recorded launch sites during the Cold War."),
        TranscriptSegment(10, 18, "Moving on to Sherlock Holmes and John Watson, "
                          "the story follows their investigation of a strange visitor."),
        TranscriptSegment(40, 48, "The detective didn't know how the stranger had entered. "
                          "He searched every room and window in the house."),
    ]
    topic = ContextualAnalyzer().analyze(segments)
    assert "Sherlock Holmes" in topic.query
    assert "satellite" not in topic.query
    assert all("satellite" not in query for query in fallback_queries(topic))


def test_new_named_subject_does_not_resurrect_previous_names():
    topic = analyze_context(
        "Sherlock Holmes and John Watson followed the visitor through the alley.",
        "Virginia Woolf wrote novels while Leonard Woolf edited the manuscript and reviewed the pages.",
    )
    assert "Virginia Woolf" in topic.query
    assert "Sherlock" not in topic.query
    assert set(fallback_queries(topic)) <= {"Virginia Woolf", "Leonard Woolf"}


def test_fallback_honors_transition_even_with_stale_supplied_concepts():
    topic = Topic("spy satellite", "Spy satellites watched launch sites. But now consider "
                  "Ada Lovelace and Charles Babbage as they study the analytical engine.",
                  concepts=("spy satellite",))
    assert set(fallback_queries(topic)) == {"Ada Lovelace", "Charles Babbage"}


@pytest.mark.parametrize("contraction", ["didn't", "wasn't", "weren't", "isn't", "can't", "don't", "didn’t"])
def test_contractions_do_not_turn_into_subject_tokens(contraction):
    assert tokens(contraction) == set()
    topic = analyze_context("Sherlock Holmes and John Watson studied the letter in the drawing room.",
                            f"He {contraction} know which visitor had called. They examined every page and waited.")
    assert contraction not in topic.query
    assert "didn" not in tokens(topic.query)


def test_fallback_does_not_submit_narration_or_identifiers():
    context = ('The officer reported what happened. Contact Alice@example.test or '
               'https://private.test/SecretProject then open "C:\\Private Notes\\Secret Folder.txt" '
               'and \\\\ServerName\\PrivateShare\\file.txt. Call 415-555-9876. '
               'Sherlock Holmes met John Watson near the house.')
    topic = Topic("officer reported", context)
    alternatives = fallback_queries(topic)
    assert set(alternatives) == {"Sherlock Holmes", "John Watson"}
    combined = " ".join(alternatives).casefold()
    assert not any(value in combined for value in ("private", "secret", "server", "415", "9876", "alice"))
    assert all(len(query.split()) <= 8 and query == sanitize_query(query) for query in alternatives)


def test_fallback_requires_subject_evidence_and_caps_distinct_queries():
    assert fallback_queries(Topic("captain crew", "He didn't know what happened. The captain turned and waited.")) == []
    context = "Ada Lovelace met Charles Babbage and Mary Somerville before visiting London Bridge."
    topic = Topic("Ada Lovelace Charles Babbage", context,
                  concepts=("Ada Lovelace", "Ada Lovelace", "Charles Babbage", "Mary Somerville"))
    queries = fallback_queries(topic)
    assert len(queries) == 2
    assert len({query.casefold() for query in queries}) == 2
    assert topic.query not in queries


def test_private_path_cannot_create_a_topic_turn_or_entity():
    context = ('Sherlock Holmes met John Watson near the house. '
               'Open "C:\\Private Notes\\moving on to Secret Person\\notes.txt" '
               'and wait while the detective examines the letter.')
    topic = Topic("Sherlock Holmes John Watson", context,
                  concepts=("Sherlock Holmes", "John Watson"))
    assert set(fallback_queries(topic)) == {"Sherlock Holmes", "John Watson"}

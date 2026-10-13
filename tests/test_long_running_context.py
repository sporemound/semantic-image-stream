"""Offline regressions for progress publication and adaptive passage throughput."""
import json

import pytest

from semantic_image_stream import narrative_context as narrative
from semantic_image_stream.models import TranscriptSegment
from test_narrative_context import install_local_model, model_state, segments, settings, stream_state


def measured_state(prompt, tokens=1000):
    return stream_state(model_state(prompt["new_transcript"]), prompt_eval_count=tokens)


def test_incremental_calls_publish_committed_progress_without_skipping_raw_text(monkeypatch):
    recording = segments(8)
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = len(recording[0].text) + 1
    prompts, _ = install_local_model(monkeypatch, analyzer)
    assert not analyzer.has_pending_segments

    for index in range(len(recording)):
        before = len(prompts)
        topic = analyzer.analyze_incremental(recording)
        assert len(prompts) == before + 1
        assert topic.query and topic.context_used_segments == index + 1
        assert topic.transcript == tuple(segment.text for segment in recording)
        assert topic.context == " ".join(topic.transcript)
        assert analyzer.has_pending_segments == (index + 1 < len(recording))
        checkpoint = analyzer.export_checkpoint()
        assert checkpoint["cursor"] == index + 1 and checkpoint["partial_offset"] == 0

    assert "".join(prompt["new_transcript"] for prompt in prompts) == "".join(
        segment.text for segment in recording)
    calls = len(prompts)
    analyzer.analyze_incremental(recording)
    assert len(prompts) == calls


def test_incremental_checkpoint_resumes_only_unprocessed_passages(monkeypatch):
    recording = segments(6)
    initial = narrative.LocalNarrativeAnalyzer(settings())
    initial._batch_chars = len(recording[0].text) + 1
    install_local_model(monkeypatch, initial)
    first = initial.analyze_incremental(recording)
    checkpoint = json.loads(json.dumps(initial.export_checkpoint()))
    assert first.context_used_segments == 1

    resumed = narrative.LocalNarrativeAnalyzer(settings())
    resumed._batch_chars = initial._batch_chars
    assert resumed.restore_checkpoint(checkpoint, recording)
    assert resumed.has_pending_segments
    prompts, _ = install_local_model(monkeypatch, resumed)
    second = resumed.analyze_incremental(recording)
    assert second.context_used_segments == 2
    assert prompts[0]["new_transcript"] == recording[1].text
    assert second.transcript == first.transcript


def test_explicit_two_batch_budget_preserves_remaining_work(monkeypatch):
    recording = segments(6)
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = len(recording[0].text) + 1
    prompts, _ = install_local_model(monkeypatch, analyzer)
    topic = analyzer.analyze(recording, max_batches=2)
    assert len(prompts) == 2 and topic.context_used_segments == 2
    assert analyzer.has_pending_segments and topic.transcript == tuple(segment.text for segment in recording)
    assert analyzer.export_checkpoint()["cursor"] == 2


def test_partial_segment_has_pending_progress_and_every_character_is_read(monkeypatch):
    text = "A wooden ship crosses pack ice. " * 55 + "Yes."
    recording = [TranscriptSegment(0, 300, text)]
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 512
    prompts, _ = install_local_model(monkeypatch, analyzer)
    first = analyzer.analyze_incremental(recording)
    checkpoint = analyzer.export_checkpoint()
    assert first.context_used_segments == 0
    assert 0 < checkpoint["partial_offset"] < len(text)
    assert first.context == text and analyzer.has_pending_segments
    for _ in range(20):
        if not analyzer.has_pending_segments:
            break
        analyzer.analyze_incremental(recording)
    assert not analyzer.has_pending_segments
    assert analyzer.context_used_segments == 1
    assert "".join(prompt["new_transcript"] for prompt in prompts) == text


def test_default_analyze_still_reads_all_pending_batches(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 512
    prompts, _ = install_local_model(monkeypatch, analyzer)
    recording = segments(45)
    topic = analyzer.analyze(recording)
    assert len(prompts) > 1
    assert topic.context_used_segments == len(recording)
    assert not analyzer.has_pending_segments
    assert "".join(prompt["new_transcript"].replace("\n", "") for prompt in prompts) == "".join(
        segment.text for segment in recording)


def test_capacity_reduction_recovers_after_four_accepted_headroom_batches(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    prompts, _ = install_local_model(
        monkeypatch, analyzer,
        lambda prompt, call: measured_state(prompt, 3500 if call == 1 else 1000))
    recording = segments(400)
    with pytest.raises(narrative.NarrativeContextError, match="capacity"):
        analyzer.analyze_incremental(recording)
    assert analyzer._batch_chars == 3000 and analyzer.context_used_segments == 0
    assert analyzer.has_pending_segments
    for success in range(4):
        topic = analyzer.analyze_incremental(recording)
        assert topic.context_used_segments > 0 and analyzer.has_pending_segments
        assert analyzer._batch_chars == (3000 if success < 3 else 3512)
    assert prompts[1]["new_transcript"] != prompts[0]["new_transcript"]
    assert "".join(prompt["new_transcript"].replace("\n", "") for prompt in prompts[1:]) == "".join(
        segment.text for segment in recording[:topic.context_used_segments])


def test_failed_batch_resets_growth_streak_and_keeps_exact_retry_cursor(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 512

    def generate(prompt, call):
        data = model_state(prompt["new_transcript"])
        if call in {3, 4}:
            data["evidence"] = ["Unsupported evidence from another story."]
        return stream_state(data, prompt_eval_count=1000)

    prompts, _ = install_local_model(monkeypatch, analyzer, generate)
    recording = segments(150)
    analyzer.analyze_incremental(recording)
    analyzer.analyze_incremental(recording)
    committed = analyzer.export_checkpoint()
    with pytest.raises(narrative.NarrativeContextError):
        analyzer.analyze_incremental(recording)
    assert analyzer.export_checkpoint() == committed
    assert analyzer._batch_successes == 0 and analyzer.has_pending_segments
    analyzer.analyze_incremental(recording)
    assert prompts[2]["new_transcript"] == prompts[3]["new_transcript"] == prompts[4]["new_transcript"]
    assert analyzer._batch_chars == 512 and analyzer._batch_successes == 1


@pytest.mark.parametrize("tokens", [0, 2900])
def test_unknown_or_insufficient_token_headroom_does_not_grow_budget(monkeypatch, tokens):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 512
    install_local_model(monkeypatch, analyzer, lambda prompt, call: measured_state(prompt, tokens))
    recording = segments(150)
    for _ in range(8):
        analyzer.analyze_incremental(recording)
    assert analyzer._batch_chars == 512 and analyzer._batch_successes == 0


def test_small_passages_do_not_trigger_budget_growth(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 512
    install_local_model(monkeypatch, analyzer, lambda prompt, call: measured_state(prompt))
    recording = segments(8)
    for length in range(1, len(recording) + 1):
        analyzer.analyze_incremental(recording[:length])
    assert analyzer._batch_chars == 512 and analyzer._batch_successes == 0


def test_regrowth_is_capped_at_original_budget(monkeypatch):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    analyzer._batch_chars = 5900
    install_local_model(monkeypatch, analyzer, lambda prompt, call: measured_state(prompt))
    recording = segments(1000)
    for _ in range(4):
        analyzer.analyze_incremental(recording)
    assert analyzer._batch_chars == narrative.MAX_BATCH_CHARS


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_batch_limit_cannot_change_progress(budget):
    analyzer = narrative.LocalNarrativeAnalyzer(settings())
    with pytest.raises(ValueError, match="max_batches"):
        analyzer.analyze(segments(1), max_batches=budget)
    assert analyzer.context_used_segments == 0 and not analyzer.has_pending_segments

"""Opt-in real prerecorded-speech validation; no mocked runtime components.

python scripts/validate_live.py --model tiny --device cpu
python scripts/validate_live.py --model base --images --providers wikimedia

The first run downloads public model weights. --images sends ONLY actual
extracted compact queries to image providers and downloads existing images.
Expected narration is used for reporting after inference, never as a prompt.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys
import threading
import time
import wave

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Intentionally run the checked-out source even before an editable install.
from semantic_image_stream.audio_capture import FileCapture  # noqa: E402
from semantic_image_stream.configuration import Settings  # noqa: E402
from semantic_image_stream.engine import export_image  # noqa: E402
from semantic_image_stream.image_download import ImageDownloader  # noqa: E402
from semantic_image_stream.image_ranking import MetadataRanker  # noqa: E402
from semantic_image_stream.search_providers import build_providers  # noqa: E402
from semantic_image_stream.semantic_analysis import ContextualAnalyzer, TopicGate  # noqa: E402
from semantic_image_stream.transcription import FasterWhisperTranscriber  # noqa: E402


def normalized_words(text):
    return re.findall(r"[^\W_]+", text.casefold())


def word_error_rate(reference, hypothesis):
    """Levenshtein WER; punctuation/case ignored, numbers are not normalized."""
    expected, actual = normalized_words(reference), normalized_words(hypothesis)
    previous = list(range(len(actual) + 1))
    for row, word in enumerate(expected, 1):
        current = [row]
        for column, recognized in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != recognized)))
        previous = current
    return previous[-1] / max(1, len(expected))


def covered_groups(text, groups):
    text = " ".join(normalized_words(text))
    return [any(" ".join(normalized_words(term)) in text for term in group) for group in groups]


async def retrieve_images(settings, topics, destination):
    providers = build_providers(settings)
    downloader = ImageDownloader(settings)
    ranker = MetadataRanker()
    reports = []
    try:
        # At most two genuine topic snapshots: first and final, to inspect shifts.
        chosen_topics = [topics[0]] if len(topics) == 1 else [topics[0], topics[-1]]
        for index, topic in enumerate(chosen_topics):
            report = {"query": topic.query, "providers": {}, "downloads": [], "displayable": []}
            candidates = []
            for provider in providers:
                try:
                    results = await provider.search(topic.query, 5)
                    report["providers"][provider.name] = {"results": len(results)}
                    candidates.extend(results)
                except Exception as error:
                    report["providers"][provider.name] = {"error": str(error), "type": type(error).__name__}
            # Rank actual metadata before bounded prefetch, then rerank verified
            # downloads and apply the same configured relevance threshold.
            ranked = ranker.rank(topic, candidates, [])
            prepared = []
            for candidate in ranked[:3]:
                try:
                    image = await downloader.download(candidate)
                    prepared.append(image)
                    report["downloads"].append({"title": image.title, "source_url": image.source_url,
                                                "width": image.width, "height": image.height,
                                                "creator": image.creator, "license": image.license})
                except Exception as error:
                    report["downloads"].append({"title": candidate.title, "error": str(error)})
            accepted = [image for image in ranker.rank(topic, prepared, [])
                        if image.score >= settings.relevance_threshold]
            for image_number, image in enumerate(accepted[:2]):
                path = destination / f"topic-{index + 1}-image-{image_number + 1}{image.path.suffix}"
                saved, attribution = export_image(image, path)
                report["displayable"].append({"image": str(saved), "attribution": str(attribution),
                                              "source_url": image.source_url, "title": image.title,
                                              "score": round(image.score, 4), "authenticity": image.authenticity})
            reports.append(report)
    finally:
        await downloader.close()
        for provider in providers:
            await provider.close()
    return reports


def run(args):
    fixtures = REPO / "tests" / "fixtures"
    manifest = json.loads((fixtures / "expected.json").read_text(encoding="utf-8"))
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings(audio_mode="file", whisper_model=args.model, whisper_device=args.device,
                        language="en", cache_dir=str(output / "cache"), chunk_seconds=args.chunk_seconds,
                        providers=args.providers, ranking="metadata").validate()
    stopped = threading.Event()
    transcriber = FasterWhisperTranscriber(settings, stop_event=stopped, on_status=lambda status: print(status, flush=True))
    analyzer = ContextualAnalyzer()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "settings": {"model": args.model, "requested_device": args.device, "language": "en",
                     "chunk_seconds": args.chunk_seconds, "ranking": "metadata", "images_requested": args.images},
        "fixtures": manifest["description"],
        "limitations": [
            "Synthetic Windows speech is a reproducible integration fixture, not a human speech accuracy benchmark.",
            "Tiny model results do not establish recognition quality for accents, background noise, podcasts, or conversations.",
            "Word error rate ignores case and punctuation; it does not normalize numbers or alternate spellings.",
            "Topic extraction and image ranking are heuristic baselines; metadata scores are not calibrated probabilities.",
            "Image retrieval/export does not validate fullscreen rendering, transitions, or human-rated image relevance.",
        ],
        "recordings": [],
    }
    for recording in manifest["recordings"]:
        path = fixtures / recording["file"]
        print(f"Validating real audio: {path.name}", flush=True)
        row = {"file": path.name, "domain": recording["domain"], "segments": [], "topics": []}
        started = time.monotonic()
        segments, window = [], deque()
        topics = []
        gate = TopicGate(settings.semantic_sensitivity)
        with wave.open(str(path), "rb") as file:
            row["duration_seconds"] = round(file.getnframes() / file.getframerate(), 3)
            row["recording_format"] = {"sample_rate": file.getframerate(), "channels": file.getnchannels(),
                                       "bits_per_sample": file.getsampwidth() * 8}

        def consume(chunk):
            recognized = transcriber.transcribe(chunk)
            segments.extend(recognized)
            window.extend(recognized)
            if recognized:
                cutoff = recognized[-1].end - settings.transcript_window_seconds
                while window and window[0].end < cutoff:
                    window.popleft()
                topic = analyzer.analyze(list(window))
                if topic and gate.changed(topic):
                    gate.accept(topic)
                    topics.append(topic)
                    row["topics"].append({"at_seconds": round(recognized[-1].end, 3),
                                          "query": topic.query, "concepts": list(topic.concepts), "kind": topic.kind})

        try:
            settings.audio_file = str(path)
            source = FileCapture(settings, realtime=False)
            source.run(consume, lambda status: None, stopped)
            reference = recording.get("text") or " ".join(recording["parts"])
            transcript = " ".join(segment.text for segment in segments)
            row["segments"] = [asdict(segment) for segment in segments]
            row["transcript"] = transcript
            row["word_error_rate"] = round(word_error_rate(reference, transcript), 4)
            groups = covered_groups(transcript, recording["keyword_groups"])
            row["recognized_keyword_groups"] = {"matched": sum(groups), "total": len(groups), "matches": groups}
            row["topic_expected_term_hits"] = [term for term in recording["expected_topics"]
                                               if any(term.casefold() in topic.query.casefold() for topic in topics)]
            if "expected_transition" in recording:
                transition = recording["expected_transition"]
                before = next((i for i, topic in enumerate(topics)
                               if any(term.casefold() in topic.query.casefold() for term in transition["from_any"])), None)
                after = next((i for i, topic in enumerate(topics)
                              if before is not None and i > before
                              and any(term.casefold() in topic.query.casefold() for term in transition["to_any"])), None)
                row["expected_topic_transition_observed"] = before is not None and after is not None
            if args.images and topics:
                row["image_retrieval"] = asyncio.run(retrieve_images(settings, topics, output / path.stem))
            row["local_pipeline_completed"] = bool(segments and topics)
        except KeyboardInterrupt:
            stopped.set()
            raise
        except Exception as error:
            row["error"] = str(error)
            row["error_type"] = type(error).__name__
            row["local_pipeline_completed"] = False
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["recordings"].append(row)
        report["actual_device"] = transcriber.device
        # Preserve incremental evidence even if a later provider/model fails.
        (output / "asr-validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"file": row["file"], "completed": row["local_pipeline_completed"],
                          "wer": row.get("word_error_rate"), "keyword_groups": row.get("recognized_keyword_groups"),
                          "topics": [topic["query"] for topic in row["topics"]], "error": row.get("error")}), flush=True)
    print(f"Evidence written to {output / 'asr-validation.json'}", flush=True)
    return 0 if all(row["local_pipeline_completed"] for row in report["recordings"]) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny", help="Actual faster-whisper model name or local model directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--chunk-seconds", type=float, default=6)
    parser.add_argument("--output", default=str(REPO / "validation-output"))
    parser.add_argument("--images", action="store_true", help="Send extracted short queries to real sources; download and export matching imagery")
    parser.add_argument("--providers", nargs="+", default=["wikimedia"], choices=["wikimedia", "openverse", "archive", "duckduckgo"])
    raise SystemExit(run(parser.parse_args()))

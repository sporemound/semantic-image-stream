# Validation record

Validated locally on Windows with Python 3.11.14. Results below distinguish automated boundary tests, actual local model inference and real provider/hardware checks. They are not a claim that every phase or every physical display has been certified.

## Offline regression checks

All 121 release regression tests pass. The suite covers audio-device lifecycle/queues, concurrent Windows device discovery, real prerecorded WAV decoding/resampling, silence and timestamps, Whisper's lazy-generator CUDA fallback, contextual query extraction, caching/history/export, all four provider adapters, provider failures, safe download boundaries, perceptual deduplication, pipeline scheduling and real Qt widget/transition rendering. Network/model boundaries are substituted only in the relevant offline tests. The app itself contains no simulated transcript, placeholder image feed or mock slideshow path.

Windows sandbox restrictions on asynchronous loop sockets and default temporary folders required running the suite outside that sandbox with a workspace `--basetemp`. Tests can run normally on a regular Windows installation. Ruff checks and `pip check` pass. Source and wheel packages build successfully. The GitHub Actions matrix is configured but remote run outcomes must be checked independently.

## Actual model inference and retrieval

Five prerecorded WAV files were created from original public narration using an installed Windows TTS voice. They are synthetic speech inputs, explicitly labelled in `tests/fixtures/expected.json`; expected narration is used only for metrics after actual inference, never as a transcription prompt. The actual `faster-whisper` tiny model ran locally on CPU/int8, producing timestamped segments and live contextual topics for all five recordings.

| Recording | Word error rate | Recognized keyword groups |
| --- | --- | --- |
| Astronomy / cosmology | 3.85% | 4 / 4 |
| French Revolution history | 3.51% | 3 / 4 |
| Botanical lecture | 0.00% | 5 / 5 |
| Quantum scientific interview | 3.33% | 4 / 4 |
| Astronomy → botany transition | 1.98% | 4 / 5 |

The astronomy-to-botany transition was observed. Tiny misrecognized “Bastille” as “best deal,” one “mycorrhizal” occurrence, and hallucinated a short “Thanks” in the last short audio tail. Word error rate is case/punctuation normalized, not number normalized. These clear synthetic recordings are not a benchmark for human accents, background noise, long podcasts or conversational speech.

A separate actual `whisper_device="auto"` check detected one CUDA device and transcribed the first astronomy fixture chunk with the cached tiny model on CUDA/float16. It returned the correct original sentence with timestamps 0.00–4.48 seconds, exited successfully, and produced no stderr. CPU fallback was unnecessary. This check used no image providers and does not establish large-model GPU capacity or concurrent Whisper/CLIP VRAM behavior. Details are preserved in `validation-output/gpu-validation.json`.

Actual compact extracted queries were sent to Wikimedia Commons and Openverse; authentic existing images were downloaded, validated, ranked and exported with creator/license/source JSON metadata. Live checks also exercised Internet Archive search and download. Wikimedia returned observed M87/Event Horizon Telescope imagery and Centaurus A; those source images were visually inspected. Initial tests exposed misleading book/document/video thumbnails, which are now excluded from Commons results. SVG diagrams remain supported through the service's rasterized thumbnail API.

DuckDuckGo's DDGS adapter failed with `RequestError` / “malformed headers” against the service during the live check. The adapter and failure behavior are tested offline, but DuckDuckGo's live discovery is not claimed working in this environment. No bypass was attempted.

Actual CPU CLIP (`openai/clip-vit-base-patch32`) inference ranked seven real downloaded images. The observed M87 image ranked first at approximately 0.378; an ESO astronomical illustration and Centaurus A scored approximately 0.348 and 0.343. An ambiguous drainage-hole photo scored 0.264 and a text-page image 0.255. The default threshold is now 0.30, which rejects these specific false positives while retaining the three appropriate examples. This small check does not establish calibrated accuracy or guarantee future relevance. An artist's scientific illustration is labelled as retrieved source imagery, never as an observed archival photograph.

## Desktop and hardware

Actual WASAPI loopback enumeration and capture succeeded, receiving real mono 16 kHz chunks during local playback and stopping cleanly. A real recording → actual local Whisper → live Wikimedia retrieval → Qt slideshow test displayed attributed imagery and entered fullscreen on the offscreen display. Screenshots were inspected at normal and fullscreen sizes. UI layout also fits 1360×900 and 980×690.

A subsequent desktop loopback test exposed a native PortAudio crash when device discovery and capture initialized concurrently. The race is fixed by serializing Windows backend initialization, enumeration, stream startup and cleanup. A concurrent-discovery regression test passes. After the fix, the actual Qt/WASAPI pipeline captured playback of the original astronomy recording, transcribed it locally, extracted “black hole galaxies,” and displayed the attributed Centaurus A source image after 13.75 seconds. It entered fullscreen on the offscreen display, stopped cleanly and exited successfully with no reported errors. The result and screenshots are preserved as `validation-output/desktop-loopback-e2e.*`; this validates real desktop loopback flow but does not certify physical fullscreen displays.

Physical device unplugging, long unattended playback, physical multi-monitor switching, human-rated semantic/image accuracy, large-model CUDA capacity and simultaneous GPU workload contention, a clean-machine signed installer and turnkey Linux system-audio support remain unvalidated. Automated fake-backend tests exercise reconnect and CUDA error recovery, but are not a substitute for those hardware/runtime checks.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe scripts\validate_live.py --model tiny --device cpu --images --providers wikimedia openverse
```

Local `validation-output/` evidence includes the actual ASR/provider reports, attributed downloaded samples, CLIP comparisons and desktop screenshots. These generated evidence files/model caches are ignored by Git. A remote CI pass or manual validation should be added only after it actually completes.

## Next milestones

1. Benchmark real recordings and noisy live speech; add multilingual semantic analyzers and better relationship/negation handling.
2. Improve retrieval query fallback and calibrated multimodal ranking across historical/scientific subjects; verify source provenance beyond reported metadata.
3. Validate long sessions, real CUDA runtimes and physical display/device changes; improve streaming chunk boundaries and model lifecycle behavior.
4. Pin validated network destinations in the download transport, complete PyInstaller/Qt/FFmpeg notice packaging and test a clean Windows installation.
5. Implement dedicated Linux monitor capture and publish a tested release after review.

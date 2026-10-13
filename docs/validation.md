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

## Blank-preview follow-up

A reported empty preview with working transcription revealed a scheduler starvation path: all downloads were awaited, then a topic-generation change discarded the whole batch. New regression coverage proves an acceptable first image appears before a blocked later download, inflight results are re-ranked for current context, unrelated old results remain rejected, and new topics wake the retry delay. Image status now persists independently of audio messages; decode failures do not masquerade as successful previews or exports. Download 429 responses expose safe reasons and respect exact-host Retry-After cooldowns.

The updated suite passed 144 tests and lint. An actual prerecorded astronomy → local Whisper → live Wikimedia → Qt preview check displayed an attributed source image after 11.92 seconds using metadata ranking at the unchanged 0.30 threshold. This is a reproduction check of the corrected pipeline; the user's existing running process must be restarted to load the updated code.

## Speech startup and fullscreen follow-up

Saved-device discovery now gates Start and retains the requested input instead of replacing it with a temporary default. Cached Whisper models are opened locally before attempting a first-use download. CPU CLIP loading/inference does not acquire the CUDA transcription lock. An image-ranking exception preserves the speech/semantic workers, keeps the selected ranker and threshold, cancels the failed image batch and retries on a bounded clock or new topic. Audio status distinguishes recognition in progress from captured audio with no recognized speech.

The user's downloaded turbo model resolved by its model name with local-only cache access and transcribed the first original astronomy fixture chunk on CUDA/float16. Recognition returned the expected sentence with timestamps 0.00–4.26 seconds. A complete desktop pipeline check then used actual turbo CUDA inference, actual CPU CLIP, the saved 0.30 threshold and original astronomy speech. Transcript, topic and a decoded attributed Milky Way image appeared after 21.14 seconds; the image scored 0.33524. A further transcript update arrived at 19.25 seconds while CPU CLIP was loading from 14.58–20.63 seconds. This confirms the CPU image model no longer holds up CUDA recognition.

That desktop check supplied previously retrieved genuine source images through a diagnostic cached-source provider and a separate cache. It made no live provider calls or model/image downloads. The actual microphone has been checked for available input and signal, but this fixture check does not confirm intelligible live speech or resolve the user's running process. Their audio/model choices were preserved. Live image-search validation was blocked by automatic approval review; earlier live-provider results remain documented above.

Fullscreen now hides all controls/status panels and removes layout margins and minimum-size constraints so the canvas fills the selected display. Regression checks verify full-window canvas geometry and pixels, Escape restoration, prior hidden attribution and maximized-window restoration on an offscreen display. Physical multi-monitor fullscreen remains unvalidated.

The complete updated local suite passed 157 tests, and lint passed.

## Audiobook pacing and presets

A reported five-second slideshow stalled with working narration, zero ready images and a zero-result query. Playback previously consumed each accepted batch once and then waited for a new topic. The controller now merges progressive batches and rotates a bounded current-topic approved playlist after fresh images run out. New subjects clear older pending/replay images; a lone accepted image stays visible. Decode-failed files are discarded from replay and displayed history. Unchanged interval values no longer restart the timer during presentation edits.

Fiction, History & Biography and Science & Technical profiles are selectable through an explicit Apply action. They retain the listener's audio/model/language/storage choices while tuning context, semantic/search/presentation clocks, sources and metadata thresholds. Tests cover applying a profile to unsaved controls, saving/reloading it, and the pipeline receiving the selected values.

Narration analysis now excludes contraction fragments and preserves compound/name/acronym subjects through explanatory passages. Empty searches try at most two shorter locally derived subjects with the same provider selection and relevance threshold. Offline integration checks verify fallback ranking at 0.50, topic changes rejecting obsolete fallback matches, bounded query attempts, successful empty-query caching and provider failures not being cached as empty success.

An actual Qt fullscreen playback check used two previously retrieved genuine astronomical source images ranked locally above 0.50. With a five-second interval, four consecutive displays occurred at 0.032, 5.047, 10.063 and 15.094 seconds: gaps of 5.015, 5.016 and 5.031 seconds, including playback after the original queue was consumed. The canvas filled the virtual display. This check used no live microphone or provider calls and is not a genre accuracy benchmark or a physical-monitor check.

All 198 local regression tests and lint passed. An earlier complete run had one timeout in an unchanged image-validation test; that isolated check and the subsequent complete run passed without altering its deadline or validation logic.

## Full-session audiobook relevance

Narrative interpretation now processes every recorded segment incrementally through a verified cached local Ollama GGUF model. The raw session survives Stop/Start in memory and supplies full-transcript image embeddings. Long text is encoded in 75-token CLIP windows with token-weighted aggregation, alongside current-scene and narrative-summary embeddings. Scene scope changes invalidate older replay images, even if the compact search query stays unchanged. Corrected guesses and unsupported visual qualifiers are checked locally before accepting model-derived searches; failure retains the unprocessed passage for retry. Exact overlapping speech repetitions and invalid decoded time intervals are filtered while genuine repetitions at different times remain.

An initial actual CPU CLIP comparison ran offline against 558 previously downloaded source images using a northern hunting-camp/isolated-door context. Highest-ranked images included an abandoned wooden house, an outdoor northern ATV scene, and a Nunavut photograph; their pixels were inspected. The final ranking-path check used actual locally interpreted context, all 14 supplied segments (295 encoded tokens), and the faithful book background. At the unchanged 0.40 minimum, the wooden-house image scored 0.411 while the misleading kitchen-window images scored at most 0.338, the shower curtain 0.230, and Anchorage satellite images at most 0.208. This small cached-source check demonstrates rejection of these specific earlier false positives; it does not establish correct source retrieval for every future scene or exact fictional-event depictions.

An offscreen Qt check retained 350 transcript lines, displayed full-session coverage and scene context, decoded a genuine cached source image, and filled the window in fullscreen. The installed Qwen2.5:7b model on CPU processed all 14 supplied segments in 24.95 seconds in the final single-batch check, without new model downloads or provider requests. Real-model interpretation initially exposed unsupported moonlight/snow inferences despite explicit sunlight. The final guard retained a grounded doorway subject, removed invented search details and excluded the corrected cabin guess from the summary. Model-selected recorded evidence is identified explicitly when it replaces embellished prose. Raw user-provided narration is excluded from repository fixtures and packages. These checks do not inspect the user's currently running app or certify live microphone/provider behavior.

All 291 local regression tests and lint passed for this update; source and wheel builds succeeded.

## Next milestones

1. Benchmark real recordings and noisy live speech; add multilingual semantic analyzers and better relationship/negation handling.
2. Improve retrieval query fallback and calibrated multimodal ranking across historical/scientific subjects; verify source provenance beyond reported metadata.
3. Validate long sessions, real CUDA runtimes and physical display/device changes; improve streaming chunk boundaries and model lifecycle behavior.
4. Pin validated network destinations in the download transport, complete PyInstaller/Qt/FFmpeg notice packaging and test a clean Windows installation.
5. Implement dedicated Linux monitor capture and publish a tested release after review.

# Architecture

`desktop_ui` owns Qt widgets. A background `engine.Pipeline` owns an asynchronous event loop, with a bounded queue from capture to transcription and a single pending topic. Each monitoring run receives a validated settings snapshot. `SessionState` retains the complete recognized transcript, latest topic and narrative analyzer across Stop/Start within the open app. Provider/model work never runs on the UI thread. Events cross into Qt through queued signals with session identifiers; late events from previous runs and cleared books are discarded.

| Module | Responsibility |
| --- | --- |
| `audio_capture` | Real WASAPI loopback or microphone devices; local recording replay; stateful resampling to mono 16 kHz; energy pre-gate and silence/max-duration chunk boundaries |
| `transcription` | Lazy faster-whisper/CTranslate2 model; Silero speech VAD; absolute timestamps; automatic language detection; CUDA failure recovery to CPU |
| `semantic_analysis` | Provider-independent `analyze(segments)` interface; optional basic English phrase/spaCy analysis and query topic distance |
| `narrative_context` | Cached local Ollama/GGUF inference; incremental full-session interpretation, established narrative summary, grounded scenes/subjects and structured search queries |
| `search_providers` | Ordered, concurrent Wikimedia, DDGS DuckDuckGo, Openverse and Internet Archive adapters, with isolated failure handling |
| `image_download` | Three concurrent prefetches; public HTTPS destination checks, redirect/byte/pixel/time limits, decode/verify/re-encode, actual geometry, perceptual hashes |
| `image_ranking` | Full scene/summary/transcript CLIP embeddings, older query-only CLIP and basic metadata alternatives; recent-image novelty |
| `cache` | SQLite query results, validated image metadata, attribution and displayed-image session history; TTL, bounded row counts and disk pruning |
| `slideshow` | Independent presentation timer, bounded current-topic playlist/history/queue, pause/manual advance and custom Qt transitions |
| `presets` | Explicit audiobook profiles applied to copied settings while preserving capture, speech-model and storage choices |
| `configuration` / `logging_setup` | Strict TOML fields, validation and console operational logs |

## Independent clocks

Capture runs continuously. Audio chunks default to six seconds, ending sooner after 0.8 seconds of silence. Every recognized segment is retained in `SessionState.segments`; the legacy `transcript_window_seconds` parameter does not truncate full-session context. Semantics checks for fresh recorded text every four seconds by default. Local-model processing can take longer than that interval, so recorded and processed segment counts are reported separately. Searches use a 12-second default cooldown that coalesces rapid transitions. Up to six ranked images are prefetched by default, with nine seconds per slide. Presets adjust those clocks and prefetch eight. If providers fail or no image passes filtering, the current image stays visible and a bounded-delay retry is scheduled.

Validated images are relevance-ranked and published progressively, without waiting for every slow or broken URL. If discussion context changes during retrieval, downloaded candidates are re-ranked against the latest topic before display; unrelated old results still fail the relevance check. New topics wake a failed search's retry delay immediately. There is no raw-audio history, no whole-session retranscription and no runtime simulation. The native capture queue and inference queue drop oldest work when overloaded and report this. Long uninterrupted speech uses non-overlapping chunks: a word split at a boundary can be mistranscribed; future streaming confirmation/short overlaps can improve it.

Progressive batches for the same playback scene merge into a bounded approved playlist rather than replacing queued images. Novel pending images play first; once exhausted, a playlist with at least two usable images cycles at the presentation interval without immediate repeats. The engine emits a playback key that changes for significant visual-subject/scene changes, grounded/unvisualizable transitions and interpreter changes. It can invalidate older playback even when the compact query is unchanged. Summary wording and processed-count changes do not change that key. Invalidation clears pending/replay while holding the current frame until a new image arrives. Qt decode failures are removed from playback and displayed history; the actual previous frame remains available for export. Applying an unchanged interval does not restart its timer.

A lookup providing fewer accepted images than the prefetch target can try up to two compact alternatives. Matches accumulate in a per-scene approved pool; existing images stay available while alternatives run. Narrative mode uses grounded scene queries; basic mode derives phrase/entity alternatives locally. New visual details or alternatives can refresh an undersized pool without a changed primary query. Successful empty lookups are cached; provider failures are not cached as empty success. Temporary download failures retain a delayed retry even when a partial set is ready. Alternative results keep the full local context and the same relevance threshold. Completed rankings survive compatible same-scene transcript updates; changed scenes still recheck candidates. Progressive arrivals rank new images once per scene. If no grounded visual subject is available, older playback and pending search are invalidated and no replacement image is approved. No additional provider is selected and no complete narration is sent to image providers.

The UI keeps image progress/error status separate from audio status: result, download, acceptance and rejection counts distinguish no results, failed downloads, minimum-score rejection, pause and decode errors. An HTTP 429 download response creates an exact-host cooldown using its bounded Retry-After value, leaving other selected image hosts available. Decode failures do not enter persistent displayed-image history or enable export.

## Full-session narrative context

The example configuration and audiobook presets choose `semantic_provider = "ollama"`. `LocalNarrativeAnalyzer` processes all newly recorded text incrementally with the established narrative state and optional `book_context`. Passages have a bounded character budget, including partial processing of an unusually long segment; the cursor advances only after a valid grounded response. Large prompts retry with smaller passages. This bounds each model request rather than deleting older transcript segments. The full raw transcript is also attached to the latest `Topic` for image ranking. Summaries can lose detail or misinterpret narration even when every passage has been processed.

`context_model` defaults to `qwen2.5:7b`. The app verifies complete cached GGUF weights and the corresponding local-service digest before inference. Only the fixed loopback service endpoint is used, with no environment proxies or redirects. Missing weights produce an explicit Context error instead of an automatic model download or a silent phrase-analysis fallback. If absent, the local Ollama service can be started without a visible terminal; a shared running service is never killed on Stop or window closure.

Context-model calls use CPU inference (`num_gpu = 0`), six threads by default, a bounded 4,096-token model context and a configurable request timeout. This separates narrative processing from Whisper's GPU workload. Cancellation closes the active local request. A session analysis lock serializes workers that may briefly outlive cancellation before the analyzer is rebound to a resumed run.

The UI appends new full-session transcript lines without rebuilding the whole document. Its Context tab shows the narrative summary, current scene, visual subjects, processed/recorded counts and errors. Stop freezes the accepted context against delayed callbacks; empty/short transcript snapshots cannot erase recorded text. Stop/Start reuses `SessionState`; changing interpreter/model/book background rebuilds analysis over retained text. Only New book and Delete session history create a fresh in-memory session. Closing the app discards it. Earlier transcript text already dropped by an older version cannot be reconstructed.

The optional `contextual` analyzer is a transparent English phrase heuristic. It emphasizes the last 16 seconds and bounded recent text; it does not reliably resolve pronouns, negation, chronology or complex multilingual relationships. spaCy improves English names and phrase boundaries. These modes remain explicitly labelled basic alternatives.

## Image relevance

`context_clip` compares normalized image embeddings with three text vectors: current scene/visual subjects (50%), narrative summary and optional book background (30%) and all retained transcript text (20%). Each text input is split into windows of at most 75 tokens. Transcript window embeddings are combined with token-count weights, and newly recorded additions update the cached full-session sum; a changed prefix resets that sum. No older transcript is silently truncated to CLIP's single-input limit. A fresh pipeline builds a fresh ranker, so a new book cannot reuse an old full-session embedding cache.

The combined cosine is rescaled with `score = clamp((cosine - 0.14) / 0.30, 0, 1)`. This is a relevance score, not a probability. There is no title, image-size or source-credibility bonus in this mode, and the user's numeric relevance minimum is preserved. Whole-session averaging can dilute a brief detail despite its contribution. Missing scene/summary vectors fall back to available text vectors with their weights renormalized.

Basic metadata ranking requires query-token overlap and includes small source/quality bonuses. The older `clip` mode compares pixels with only the short query and includes specificity/quality terms. Their scores are not interchangeable with `context_clip`. No mode establishes factual or historical identity.

## GPU and shutdown

Whisper uses CUDA if available, otherwise CPU int8; runtime CUDA/library/memory failures retry the current chunk on CPU. Cached Whisper weights load without remote metadata requests; only a cache miss enables first-use downloading. CLIP defaults to CPU and does not take the GPU lock in that mode. A shared lock serializes CUDA Whisper and CUDA CLIP inference, but resident models still consume VRAM. Prefer base/small Whisper plus CPU CLIP on an 8 GB GPU. Native model constructors and decoder calls cannot be forcibly cancelled; daemon model workers avoid blocking window closure, check cancellation around native calls, and do not publish after stop.

Audio-device discovery gates monitoring startup and preserves the saved input while enumeration is pending. A missing input requires a new selection rather than silently changing capture devices. Ranking failures are reported by the image worker and retried without stopping the speech and semantic workers or changing the chosen ranking method.

The initial Linux path supports microphone capture and selecting an available PulseAudio/PipeWire monitor input. WASAPI is Windows-only. The UI and provider interfaces are portable; Linux system-audio support is not yet a turnkey feature.

## Extension points

`SearchProvider` implements `name`, `async search(query, limit)`, and `async close()`. `SemanticAnalyzer` implements `analyze(segments)` returning `Topic` or `None`. Rankers implement `rank(topic, candidates, recent_hashes)`. A capture source implements `run(on_chunk, on_status, stop_event)` and `stop()`. These interfaces allow independent source/local-model changes without replacing the desktop UI.

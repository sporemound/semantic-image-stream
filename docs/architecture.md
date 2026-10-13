# Architecture

`desktop_ui` owns Qt widgets. A background `engine.Pipeline` owns an asynchronous event loop, with a bounded queue from capture to transcription and a single pending topic. Each monitoring run receives a validated settings snapshot. Provider/model work never runs on the UI thread. Events cross into Qt through queued signals with session identifiers; late events from previous runs are discarded.

| Module | Responsibility |
| --- | --- |
| `audio_capture` | Real WASAPI loopback or microphone devices; local recording replay; stateful resampling to mono 16 kHz; energy pre-gate and silence/max-duration chunk boundaries |
| `transcription` | Lazy faster-whisper/CTranslate2 model; Silero speech VAD; absolute timestamps; automatic language detection; CUDA failure recovery to CPU |
| `semantic_analysis` | Provider-independent `analyze(segments)` interface; bounded context, weighted phrases, compound names, domain concepts and topic distance; optional local spaCy entities/noun chunks |
| `search_providers` | Ordered, concurrent Wikimedia, DDGS DuckDuckGo, Openverse and Internet Archive adapters, with isolated failure handling |
| `image_download` | Three concurrent prefetches; public HTTPS destination checks, redirect/byte/pixel/time limits, decode/verify/re-encode, actual geometry, perceptual hashes |
| `image_ranking` | Local metadata overlap and contextual/source/quality scores; optional actual CLIP image/text embeddings; recent-image novelty |
| `cache` | SQLite query results, validated image metadata, attribution and displayed-image session history; TTL, bounded row counts and disk pruning |
| `slideshow` | Independent presentation timer, bounded current-topic playlist/history/queue, pause/manual advance and custom Qt transitions |
| `presets` | Explicit audiobook profiles applied to copied settings while preserving capture, speech-model and storage choices |
| `configuration` / `logging_setup` | Strict TOML fields, validation and console operational logs |

## Independent clocks

Capture runs continuously. Audio chunks default to six seconds, ending sooner after 0.8 seconds of silence. A rolling transcript window defaults to 45 seconds. Semantics evaluates fresh text every four seconds. Search runs only on a significant concept change, with a 12-second cooldown that coalesces rapid transitions. Up to six ranked images are prefetched, with nine seconds per slide. The presentation queue retains the current image while new topics are searched. If providers fail or no image passes filtering, the current image stays visible and a bounded-delay retry is scheduled.

Validated images are relevance-ranked and published progressively, without waiting for every slow or broken URL. If discussion context changes during retrieval, downloaded candidates are re-ranked against the latest topic before display; unrelated old results still fail the relevance check. New topics wake a failed search's retry delay immediately. There is no raw-audio history, no whole-session retranscription and no runtime simulation. The native capture queue and inference queue drop oldest work when overloaded and report this. Long uninterrupted speech uses non-overlapping chunks: a word split at a boundary can be mistranscribed; future streaming confirmation/short overlaps can improve it.

Progressive batches for the same topic merge into a bounded approved playlist rather than replacing queued images. Novel pending images play first; once exhausted, a playlist with at least two usable images cycles at the presentation interval without immediate repeats. Accepting a new topic clears the old pending/replay pool while preserving its current frame until a new image is accepted. Applying the same interval value does not restart its timer, so attribution/transition edits cannot postpone advancement.

An empty exact lookup can try up to two compact locally derived subject/entity queries. Successful empty lookups are cached; provider failures are not cached as empty success. Alternative results are ranked against a topic copy containing the alternative query and original full local context/kind, with the same relevance threshold. If the actual topic changes while searching, the old alternative no longer controls ranking. The alternative is identified in image status. No additional provider is selected and no complete narration leaves the app.

The UI keeps image progress/error status separate from audio status: result, download, acceptance and rejection counts distinguish no results, failed downloads, minimum-score rejection, pause and decode errors. An HTTP 429 download response creates an exact-host cooldown using its bounded Retry-After value, leaving other selected image hosts available. Decode failures do not enter persistent displayed-image history or enable export.

The default semantic analyzer is a transparent English phrase heuristic, not an LLM. It keeps multiword subjects, names and co-mentioned relationships together, and emphasizes the last 16 seconds while retaining context locally. It does not reliably resolve pronouns, negation, fine-grained chronology or complex multilingual relationships. spaCy improves English names and phrase boundaries. Alternate local analyzers can implement the same interface.

Metadata ranking requires a meaningful query-token overlap and rejects unrelated high-resolution credible-source images. Scores are heuristics, not probabilities. CLIP uses actual image/text embeddings and approximate cosine relevance combined with small specificity/quality terms. Neither proves historical identity or accuracy. The same numeric threshold can behave differently with these two rankers.

## GPU and shutdown

Whisper uses CUDA if available, otherwise CPU int8; runtime CUDA/library/memory failures retry the current chunk on CPU. Cached Whisper weights load without remote metadata requests; only a cache miss enables first-use downloading. CLIP defaults to CPU and does not take the GPU lock in that mode. A shared lock serializes CUDA Whisper and CUDA CLIP inference, but resident models still consume VRAM. Prefer base/small Whisper plus CPU CLIP on an 8 GB GPU. Native model constructors and decoder calls cannot be forcibly cancelled; daemon model workers avoid blocking window closure, check cancellation around native calls, and do not publish after stop.

Audio-device discovery gates monitoring startup and preserves the saved input while enumeration is pending. A missing input requires a new selection rather than silently changing capture devices. Ranking failures are reported by the image worker and retried without stopping the speech and semantic workers or changing the chosen ranking method.

The initial Linux path supports microphone capture and selecting an available PulseAudio/PipeWire monitor input. WASAPI is Windows-only. The UI and provider interfaces are portable; Linux system-audio support is not yet a turnkey feature.

## Extension points

`SearchProvider` implements `name`, `async search(query, limit)`, and `async close()`. `SemanticAnalyzer` implements `analyze(segments)` returning `Topic` or `None`. Rankers implement `rank(topic, candidates, recent_hashes)`. A capture source implements `run(on_chunk, on_status, stop_event)` and `stop()`. These interfaces allow independent source/local-model changes without replacing the desktop UI.

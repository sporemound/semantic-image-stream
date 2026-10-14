# Semantic Image Stream

A free Windows desktop app that listens to spoken audio, interprets the recorded story locally, and retrieves existing images for an attributed slideshow. Full-session mode combines a local narrative model with image/text embeddings. No images are generated.

## Install on Windows 11

Install 64-bit Python 3.11, 3.12 or 3.13 and Git. Python 3.14 is not supported by this dependency envelope. For full-session interpretation, install [Ollama for Windows](https://docs.ollama.com/windows) and manually download [Qwen2.5 7B](https://ollama.com/library/qwen2.5) once with the command below. The app requires its complete local weights and does not download an absent narrative model. `ollama pull` is documented in the [Ollama CLI reference](https://docs.ollama.com/cli).

```powershell
git clone https://github.com/sporemound/semantic-image-stream.git
Set-Location semantic-image-stream
git switch main
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -WithClip
ollama pull qwen2.5:7b
.\.venv\Scripts\python.exe -m semantic_image_stream --config config.local.toml
```

The repository is public. `ExecutionPolicy Bypass` applies to this one process; the script does not change your machine policy. The setup creates an isolated `.venv`, installs dependencies, and copies example settings without replacing an existing local configuration. `-WithClip` installs the image-embedding dependencies; the setup does not install Ollama or its narrative model. A compatible existing Python can be passed with `-PythonPath 'C:\path\to\python.exe'`.

Image searches/downloads and first-use Whisper/CLIP weight downloads need an internet connection. Audio, transcription, narrative interpretation and image ranking run locally. No subscription or API key is required. The narrative model runs on CPU with six threads by default, leaving the GPU available to Whisper. Qwen2.5 7B uses roughly 5 GB of memory during local interpretation. Cold starts can take tens of seconds; one local CPU check took about 40 seconds, which is not a speed guarantee. Start with **base**, or use **tiny** for a fast speech-recognition check.

## Use

The **Images** tab includes **CSV minimum relevance**, **Export CSV**, and **Refresh images**. CSV exports contain unique image links accepted during the current book session at or above the selected score, with attribution, relevance, search query, narrative summary, scene and the matching spoken passage. They include accepted images not yet displayed and are independent of the 60-item slideshow history. The export minimum initially matches the image-search relevance minimum; raise it to export the strongest matches. Scores depend on the chosen ranking method and are not probabilities. Select **Export CSV** to choose where to save the file; export remains available after monitoring stops. Newly accepted links are saved with the book when **Keep context between launches** is enabled, and **New book / clear context** clears them with the story. Older saved sessions do not contain these links until new searches run.

**Refresh images** requests fresh results for the current spoken scene while monitoring, keeping the displayed frame, transcript and narrative context. Automatic searches rotate past consumed cached pages and retry healthy providers independently of failed providers. Provider rate-limit deadlines still apply.

1. Select **System audio · WASAPI loopback**, choose the output device whose playback you want to accompany, and click **Start monitoring**. Continue playing your podcast/lecture normally. Only an input loopback stream is opened.
2. The **Transcript** tab retains all recognized text in the current book session. The **Context** tab shows the local interpretation, current scene, visual subjects, and how many recorded segments have been processed. Compact queries go to the checked image providers. Existing images are fetched, verified, deduplicated and ranked before display. Model startup and context processing can delay the first image; a displayed backlog means more recorded text is waiting to be interpreted.
3. Adjust image interval, relevance minimum, topic sensitivity, and capture/semantic/search intervals. Input/search/model settings apply to the next monitoring run; presentation interval, transitions and attribution visibility update live.
4. **Pause images** holds a frame while monitoring continues; **Next image** advances manually. The image history can revisit earlier slides. Source links open the original attribution page. **Save image + attribution** exports the verified raster and a JSON file with creator, license and source URLs.
5. Choose a fullscreen display and press **Fullscreen** or **F11** for a clean image view. **Fullscreen with title** adds the current image's own title in large cream serif lettering over a soft gradient. **Fullscreen with context** shows the live **Narrative context**, **Current scene**, **Visual subjects**, **Image title**, and **Image credits** in a readable panel; scroll longer context or image details inside the panel. New recognized speech appears as centered white subtitles with a black outline near the bottom, independently of story interpretation. Every new passage received while this view is open is shown in order, with longer speech advancing through readable two-line pages. The complete timestamped session also stays in **Transcript**. Opening this view starts with the latest recorded passage, without replaying earlier history. Its context follows the Context tab, including while a paused or earlier image remains displayed, and the title and credits follow the displayed image. All three views hide controls and status panels. **Escape** restores the normal window and attribution; **Space** pauses/resumes and **Right arrow** advances.

Use **Fullscreen information → Choose what to show** to select **Live subtitles**, **Narrative context**, **Current scene**, **Visual subjects**, **Image title**, and **Image credits** individually for the context view. Right-click the fullscreen image to change these choices without leaving the view. **Save settings** keeps your selections for the next launch. Recognized narration is displayed and stored without profanity masking or content-based redaction.

In the **Context** tab, **Export context** saves a portable JSON file containing the full timestamped transcript, accepted story interpretation, applied book guidance, and interpretation progress. Export works while monitoring and when automatic saving is off. Stop monitoring, choose **Import context**, and select that file to replace the current story; **Start monitoring** continues it. Import preserves your audio, speech model, image sources, and fullscreen choices. A matching interpreter resumes its saved progress; a changed interpreter re-reads the retained transcript. With **Keep context between launches** enabled, the imported story is saved locally as well. Context files preserve the complete recognized text.
6. **Stop** ends monitoring while retaining the local transcript and interpretation. **Start monitoring** continues the same book session. **Keep context between launches** is enabled by default: the app automatically saves its story memory locally and restores the last session on the next launch, with audio monitoring stopped. While stopped, **New book / clear context** clears the saved story and optional book background; **Delete session history** also removes recorded context and displayed-image history.

Microphone input is an alternative. To test a real recording or the included original synthetic speech fixtures:

```powershell
.\.venv\Scripts\python.exe -m semantic_image_stream --cpu --audio-file .\tests\fixtures\astronomy.wav
.\.venv\Scripts\python.exe -m semantic_image_stream --list-devices loopback
```

Device indices can change after reconnecting devices. Use **↻** to refresh. Capture interruptions retry a limited number of times and surface failures in the status area. The app keeps searching for fresh relevant images as the current queue is consumed. Automatic playback and **Next image** use unseen results only. If none are ready, the last frame remains visible while more matches are sought. A new slide requires an available unseen image that meets your quality and relevance filters. Clicking an image in history explicitly revisits that slide.

If image discovery unexpectedly exits, it restarts automatically with error backoff while speech, story context, queued images and viewing history remain intact. Slow ranking yields obsolete batches after a bounded scene correction, so repeated scene changes cannot trap searches on old pixels. Each poll still tries at most two fallback queries, rotating through the existing grounded alternatives when earlier ones are exhausted. Full-session interpretation publishes each accepted passage as it catches up rather than waiting for a whole backlog; all recorded text remains available and no pending passage is skipped.

DuckDuckGo's native search runs in a separate short-lived local process. A stuck request is terminated and fully closed before the next one starts. Internet Archive metadata outages are reported as source failures rather than cached as empty searches; source cooldowns remain respected.

Memory use is bounded with persistent native worker threads, one-image CPU ranking batches, smaller complete-text embedding batches and a shared decode/ranking allocation lock. Source pixels and export resolution are preserved. CPU CLIP uses at most four threads while respecting a smaller existing thread setting. Stop releases owned speech/image models after their actual native work finishes; a later Start waits for that cleanup, and pending votes save their labels without reloading retired models.

On Windows, model loading and large pixel operations check available allocation headroom before starting. Low memory pauses affected work and retries automatically; a speech allocation failure retains the current audio chunk for retry. Previously recognized text and story memory stay intact. These checks are estimates and volatile snapshots, so they cannot guarantee an allocation or reclaim memory used by other programs. Close unused apps if the low-memory status remains. Capture buffering stays bounded and can drop older incoming chunks if recognition remains behind for too long. The app does not change the Windows paging file or unload a shared Ollama service.

Passages with no new visual information retain the established scene while the complete transcript and narrative summary advance. Approved pending images and the slideshow timer remain intact, so those slides continue at your chosen interval. Context status explains when the previous scene is being retained. A confirmed scene/subject change still discards older pending results, including when its short search query is unchanged. Summary wording and processed-count updates alone do not reset playback.

Displayed URLs and perceptual fingerprints are remembered across scene changes, Stop/Start, app relaunch and image-cache pruning. **New book** and **Clear cache** preserve this record. **Delete session history** clears it, allowing previously displayed images to appear again. This suppresses matching URLs and perceptually similar copies; it cannot identify every heavily altered version of an image.

Use **Good fit** or **Poor fit** below the image to judge how well that picture fits the spoken story currently shown in Context. A vote is saved locally; choosing the other button changes that vote. **Poor fit** moves to the next unseen image when one is ready. If none is ready, the current frame stays visible while retrieval continues. Voting works while paused or stopped after a story context has been accepted.

Saved votes guide future ranking of similar image/context pairs, with a small bounded score adjustment. They do not change your relevance minimum, audio settings or selected sources. Full-context and CLIP modes use the locally cached image/text features; metadata mode uses separate local word features. The status confirms whether learning was updated or only the label could be saved. Votes survive relaunch, **New book**, image-cache clearing and session-history deletion. Up to 1,000 latest votes are retained in `feedback.json` beside the saved story; this file contains context/image digests and feature vectors, with no transcript text or image URLs, and is never uploaded. This personal preference memory does not fine-tune model weights or guarantee improved accuracy.

Start becomes available when audio-device discovery finishes. A missing saved input is shown explicitly; select an available input or refresh the list. Downloaded Whisper models load from the local cache first. The bottom status distinguishes receiving audio, loading recognition, and receiving audio without recognized speech. After installing an update, close the entire app and reopen it before starting monitoring.

Images preview as soon as a validated result meets the relevance minimum, while remaining images load. The image-status line shows **Found / Loaded / Ready / Skipped** counts and explains rate limits, resolution rejection, insufficient relevance, pause, or display errors. Transcription can continue independently while an image source is unavailable. Topic changes recheck downloaded candidates against current context instead of discarding an entire batch.

An image relevance-model failure appears in image status while speech monitoring continues; the selected ranking method and relevance minimum are preserved. Narrative-model errors appear in **Context**, retain recorded text for retry, and leave transcription running. An unavailable local model is not silently replaced with basic phrase matching. CPU interpretation and image ranking do not acquire the transcription GPU's lock.

## Image filters

**Allow nudity (NSFW)** is off by default. Enable it to request mature results from Openverse and turn off SafeSearch for web image searches. When off, provider-marked mature images are excluded, including cached results and displayed history. Source account restrictions still apply, and sources with missing ratings can return unlabelled nudity; this checkbox is not a visual nudity detector. The setting applies live to new image searches, and **Save settings** keeps it for the next launch. It does not filter or redact recorded text.

Enable **Filter by time period**, then choose whether the dates describe the **Depicted subject or event** or when the image/document was **Created or published**. For example, a fossil's geological age and its photograph's creation date are different filters. Drag the two handles or enter exact **Start** and **End** values, from older to newer. Inputs include `500 BCE`, `1900 CE`, `11700 BP`, `66 Ma` and `4.6 Ga`; BP and geological age units count back from 1950. **Calendar years** and **Geological time** change the slider scale without discarding exact dates. Tab selects each handle and arrow keys move it. The fields also accept dates outside the slider's displayed range.

**Include images with unknown dates** is off by default: matching needs source-supported dates of the selected type. You can enable it to allow undated objects or scientific diagrams alongside dated results. **Exclude AI-generated images** is on by default and rejects images labeled as generated in source metadata. It does not detect generated pixels, and missing labels can let generated images through.

Dated results match when their recorded period overlaps your range. Named geological units use approximate numerical boundaries from the [ICS 2026/06 chart](https://stratigraphy.org/ICSchart/ChronostratChart2026-06.pdf); the Holocene boundary uses its stated [11,700 years before 2000](https://stratigraphy.org/news/125). Creation filtering uses Commons' original creation-date metadata and Internet Archive's publication date, while upload/indexing dates do not establish creation or a subject's age. Unsupported or ambiguous dates remain unknown.

These image filters can change while monitoring. Changes refresh pending images while retaining the displayed frame, spoken story, transcript and slide interval. Click **Save settings** to keep the choices for the next launch. Time periods and date labels do not establish an image's factual identity or authenticity.

## Audiobook presets

While monitoring is stopped, choose a profile in **Audiobooks** and click **Apply**. Each profile selects **Full session · local model** interpretation and **Full context + image embeddings** ranking, alongside pacing, sources and relevance minimum. An already selected **Full context + trained relevance** mode is retained. It preserves your audio input, speech model, language, local context model, optional book background, local training choices, cache and transition. Click **Save settings** to keep the profile for the next launch, then **Start monitoring**. You can adjust its values afterward.

| Preset | Slide interval | Recorded context | Relevance minimum | Sources |
| --- | --- | --- | --- | --- |
| Fiction | 10 seconds | Full session | 0.30 | Wikimedia Commons, Openverse |
| History & Biography | 12 seconds | Full session | 0.35 | Wikimedia Commons, Internet Archive |
| Science & Technical | 10 seconds | Full session | 0.30 | Wikimedia Commons, Openverse |

Use History & Biography for historical nonfiction. **Book or context** is a multiline editor for an optional title, setting, atmosphere, visual details or background, up to 2,000 characters. **Guidance strength** defaults to 0.65: larger values give your manual background more influence on image searches and relevance while the current spoken scene remains authoritative. Zero adds no manual image guidance. Background is guidance, not evidence that an image depicts a fact about the book.

Select **Apply book guidance** to use an edited background or strength during monitoring. Applying refreshes pending images without restarting audio or transcription; the current frame and full recorded story stay in place. Typing alone leaves active guidance unchanged, and later edits remain a draft even if an earlier Apply is still completing. **Start monitoring** uses the editor's current values. **Save settings** stores those values in your configuration; the saved local story and Stop/Close retain the guidance already applied to that story. Applying while stopped saves the accepted background locally for the next Start.

An explicitly saved draft returns to the editor on the next launch while the restored story keeps its previously applied guidance. Select **Apply book guidance** or **Start monitoring** to use the draft. If you apply different guidance afterward, the older saved draft stops taking precedence over the newer story state.

**New book / clear context** explicitly starts a fresh session. Every newly recorded segment is processed incrementally into the established narrative context, and the full retained text contributes to image relevance. The profiles retain legacy `transcript_window_seconds` values of 90/120/90 for compatibility; those values do not cut off full-session mode. Older text already discarded by an earlier app version cannot be recovered.

Saved story memory includes the full timestamped transcript, applied book background and guidance strength, accepted summary/scene/subjects, and interpretation progress. Matching local-model context resumes without sending already processed text through the model again. Changing the interpreter or model re-reads the retained text; incompatible progress data also falls back to a complete read. This stores context for the existing local model and image ranking.

On Windows, the last story is a local JSON file at `%LOCALAPPDATA%\SemanticImageStream\sessions\last-session.json`, separate from the image cache. Set `session_dir` to another folder if needed. The saving status is shown in the app. Unchecking **Keep context between launches** stops saving and automatic restoration; click **Save settings** to keep that choice. The previously saved file remains until **New book / clear context** or **Delete session history**. **Clear cache** preserves saved story memory. A failed save leaves the previous complete file intact; the app stays open if closing would lose unsaved recorded text. Saved files have a 256 MiB read/write bound, and an unreadable file is left untouched until explicitly cleared.

When a search provides fewer accepted images than the configured prefetch target, the app tries up to two related queries from the interpreted visual subjects in full-session mode, or from locally identified phrases in basic mode. Matches accumulate while the established scene remains active. New grounded visual details or alternatives can refresh an undersized set even when its primary query stays the same; transcript-count and summary-only updates do not restart retrieval or clear pending images. A healthy search with no qualifying results retries at the configured **Search cooldown** (`search_interval`), rather than imposing a fixed 30-second wait. Source `Retry-After` instructions and network/model error backoff can delay retries further. The selected providers and relevance minimum remain unchanged. Image status identifies alternative searches and explains when only a small current set is available. Successful empty searches are cached, and retry timing does not guarantee a fresh slide when no qualifying unseen image is available.

## Local relevance options

The example configuration and audiobook profiles use `semantic_provider = "ollama"` with `context_model = "qwen2.5:7b"`, plus `ranking = "context_clip"`. The interpreter processes new passages with the established narrative summary and optional background. The raw transcript remains available locally. Model summaries can still omit details or misunderstand a scene; inspect **Context** when matches are wrong.

If you installed without `-WithClip`, add the image-embedding dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[clip]"
```

CLIP weights download once into the app's model cache. **Full context + image embeddings** compares actual image pixels with the current scene/subjects (50%), narrative summary and optional book background (30%), and every retained transcript segment (20%). Text is split into windows of at most 75 tokens, and full-session windows are weighted by token count rather than truncating the transcript to one CLIP input. The combined cosine similarity is rescaled to a 0–1 relevance score; it is not a probability. Source credibility and image size do not boost this score. Your selected relevance minimum is applied unchanged.

CPU image ranking is the default. GPU CLIP can be configured with `clip_device = "cuda"` if compatible PyTorch/CUDA packages are installed. Image embeddings and local interpretation do not guarantee relevance, factual identity or historical authenticity. Whole-session averaging can dilute a brief detail even though its text contributes.

The audiobook cutoffs are starting points rather than calibrated accuracy targets. If image status reports a best score below your minimum, reduce **Relevance minimum** gradually; if unrelated images appear, raise it. An unchanged frame with **0 new images ready** means the queue has no unseen approved match. Lowering the minimum can increase variety at the cost of accepting weaker matches. Click **Save settings** to retain a value changed in the app.

For a lighter baseline, select **Basic phrase analysis** (`semantic_provider = "contextual"`) and **Source metadata (basic)** (`ranking = "metadata"`). This English heuristic focuses on recent phrases and cannot reliably resolve narrative relationships. **Local CLIP embeddings** (`ranking = "clip"`) is the older query-only image matcher; it does not use full-session narrative embeddings. Its scores differ from full-context scores, so the same threshold can behave differently across modes.

### Optional trained relevance and story examples

While stopped, select **Full context + trained relevance** to use the bundled local score calibrator. **Training** shows **Trained calibration ready**, its encoder and sample counts. It was fitted on 320 public museum image/metadata pairs from [Mitsua/art-museums-pd-440k](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k). The selected model preserves full-context CLIP ordering and changes the score scale. It did not improve held-out matching accuracy. Switching modes preserves your numeric **Relevance minimum**; the optional caption-calibration suggestion is about 0.39 and is not an audiobook accuracy target. Read the [actual training report and its limits](docs/training.md#evaluation) before selecting a cutoff.

**Use local story examples** provides 128 explicit question/answer examples from the training split of [FairytaleQA](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA) to illustrate grounded story interpretation. Examples are selected on this computer and supplied only to the local interpreter. They do not supply facts about your book. Qwen and CLIP weights remain unchanged; only the small relevance candidates and selected score calibrator were fitted.

The compact model, public examples and [separate training notices](src/semantic_image_stream/data/training/TRAINING_NOTICES.txt) are bundled with the app. Trained mode requires already cached CLIP weights; it downloads neither missing encoder weights nor training data when monitoring starts. Normal image searches still use your selected providers. Optional replacement packs live in `%LOCALAPPDATA%\SemanticImageStream\training`, or `training_dir` in your configuration. **New book**, **Delete session history** and **Clear cache** preserve training files. Setting `use_story_examples = false` disables examples while keeping the pack. See [local training, reproducibility and licenses](docs/training.md).

Optional English dependency-based phrases and named entities:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[semantic]"
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Select **spaCy phrases (optional)** or set `semantic_provider = "spacy"`. This improves phrase/entity extraction but is still a basic interpretation option.

Whisper automatic GPU detection falls back to CPU when CUDA libraries/inference fail. GPU Whisper requires the compatible CUDA 12 cuBLAS and cuDNN 9 runtime described in the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper#gpu). The PowerShell setup does not install NVIDIA drivers or CUDA system runtimes. Use `--cpu` for reliable operation without CUDA. First-use native model loading/inference cannot be forcibly interrupted, but worker cancellation prevents late slides and window closure does not wait for model downloads.

## Sources, privacy and licenses

Wikimedia Commons is the default. DuckDuckGo Images is installed by the setup's `[search]` extra; Openverse and Internet Archive use anonymous documented APIs. Google Images is intentionally not bundled without a compliant access method. **Exclude AI-generated images** rejects known generated-image metadata by default; absent labels are not proof of authenticity. Every retrieved image remains labelled **Unverified source imagery** and retains its source page. Unknown creator/license metadata stays unknown. Application MIT licensing does not cover source images.

No raw audio or transcript is sent to image providers or Hugging Face. The local narrative service receives passages only through `127.0.0.1:11434`, using verified cached GGUF weights; cloud models and automatic narrative-model downloads are rejected. Selected image providers receive up to eight sanitized story-query words, plus an optional geological period hint, which can reveal discussed names/topics. SQLite stores queries, image attribution and optional displayed-image history. When context retention is enabled, transcripts and narrative state are stored separately as local JSON in the sessions folder. **Clear cache** removes queries/images/history while retaining model weights, training files and saved story context; **Delete session history** clears stored records and the local book context. See [privacy](docs/privacy.md), [provider limitations](docs/providers.md), [architecture](docs/architecture.md), [training sources and licenses](docs/training.md), and [third-party licenses](docs/third-party.md).

## Development and validation

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,search]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m build
```

Offline unit/integration tests exercise WAV decoding/resampling, silence gates, bounded buffers, Whisper generator fallback, semantics, provider failures, attribution, network/download boundaries, cache/history, engine scheduling, and real offscreen Qt rendering/transitions. Network/model behavior is explicitly substituted only in boundary-focused offline tests, never in the actual app.

Five original prerecorded WAV fixtures cover astronomy, history, botany, a scientific interview, and a rapid subject change. Their Windows TTS narration is clearly synthetic test speech. Regenerate them with `scripts/create-validation-recordings.ps1`. Opt into actual model/provider validation:

```powershell
.\.venv\Scripts\python.exe scripts\validate_live.py --model tiny --device cpu
.\.venv\Scripts\python.exe scripts\validate_live.py --model tiny --device cpu --images --providers wikimedia openverse
```

Reports and attributed retrieved samples go to the ignored `validation-output/` directory. This command downloads real model weights and, with `--images`, sends actual extracted short queries. Read [validation results and untested cases](docs/validation.md) for the evidence and its limits. CI runs offline checks and builds on Windows/Python 3.11/3.13 and Ubuntu/Python 3.12; the configured CI matrix is not proof those remote runners have passed.

An optional one-folder Windows executable build scaffold is provided:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package.ps1
```

The source/wheel build is supported; the PyInstaller executable, clean-machine installation and dependency notice bundle still need release validation. No signed installer is provided. Linux supports microphone/available monitor input selection but has no turnkey WASAPI-equivalent loopback setup.

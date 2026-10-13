# Semantic Image Stream

A free Windows desktop app that listens to spoken audio, interprets the recorded story locally, and retrieves existing images for an attributed slideshow. Full-session mode combines a local narrative model with image/text embeddings. No images are generated.

## Install on Windows 11

Install 64-bit Python 3.11, 3.12 or 3.13 and Git. Python 3.14 is not supported by this dependency envelope. For full-session interpretation, install [Ollama for Windows](https://docs.ollama.com/windows) and manually download [Qwen2.5 7B](https://ollama.com/library/qwen2.5) once with the command below. The app requires its complete local weights and does not download an absent narrative model. `ollama pull` is documented in the [Ollama CLI reference](https://docs.ollama.com/cli).

```powershell
git clone https://github.com/sporemound/semantic-image-stream.git
Set-Location semantic-image-stream
git switch codex/windows-mvp
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -WithClip
ollama pull qwen2.5:7b
.\.venv\Scripts\python.exe -m semantic_image_stream --config config.local.toml
```

The repository may require your GitHub login because its current visibility is private. `ExecutionPolicy Bypass` applies to this one process; the script does not change your machine policy. The setup creates an isolated `.venv`, installs dependencies, and copies example settings without replacing an existing local configuration. `-WithClip` installs the image-embedding dependencies; the setup does not install Ollama or its narrative model. A compatible existing Python can be passed with `-PythonPath 'C:\path\to\python.exe'`.

Image searches/downloads and first-use Whisper/CLIP weight downloads need an internet connection. Audio, transcription, narrative interpretation and image ranking run locally. No subscription or API key is required. The narrative model runs on CPU with six threads by default, leaving the GPU available to Whisper. Qwen2.5 7B uses roughly 5 GB of memory during local interpretation. Cold starts can take tens of seconds; one local CPU check took about 40 seconds, which is not a speed guarantee. Start with **base**, or use **tiny** for a fast speech-recognition check.

## Use

1. Select **System audio · WASAPI loopback**, choose the output device whose playback you want to accompany, and click **Start monitoring**. Continue playing your podcast/lecture normally. Only an input loopback stream is opened.
2. The **Transcript** tab retains all recognized text in the current book session. The **Context** tab shows the local interpretation, current scene, visual subjects, and how many recorded segments have been processed. Compact queries go to the checked image providers. Existing images are fetched, verified, deduplicated and ranked before display. Model startup and context processing can delay the first image; a displayed backlog means more recorded text is waiting to be interpreted.
3. Adjust image interval, relevance minimum, topic sensitivity, and capture/semantic/search intervals. Input/search/model settings apply to the next monitoring run; presentation interval, transitions and attribution visibility update live.
4. **Pause images** holds a frame while monitoring continues; **Next image** advances manually. The image history can revisit earlier slides. Source links open the original attribution page. **Save image + attribution** exports the verified raster and a JSON file with creator, license and source URLs.
5. Choose a fullscreen display and press **Fullscreen** or **F11**. The slideshow fills the selected display, with controls and status panels hidden. **Escape** restores the normal window and attribution; **Space** pauses/resumes and **Right arrow** advances.
6. **Stop** ends monitoring while retaining the local transcript and interpretation. **Start monitoring** continues the same book session. While stopped, **New book / clear context** starts a fresh session and clears the optional book background; **Delete session history** also clears the recorded context. Context is held in memory and is lost when the app closes.

Microphone input is an alternative. To test a real recording or the included original synthetic speech fixtures:

```powershell
.\.venv\Scripts\python.exe -m semantic_image_stream --cpu --audio-file .\tests\fixtures\astronomy.wav
.\.venv\Scripts\python.exe -m semantic_image_stream --list-devices loopback
```

Device indices can change after reconnecting devices. Use **↻** to refresh. Capture interruptions retry a limited number of times and surface failures in the status area. Fresh approved images take priority. After the first batch, the slideshow cycles the current scene's approved images at your interval if at least two are available. A significant scene/subject change clears the older playlist, including when its short search query is unchanged, and holds its last frame until another relevant image arrives. Summary wording and processed-count updates do not reset playback. A single accepted image stays visible until there is another usable match.

Start becomes available when audio-device discovery finishes. A missing saved input is shown explicitly; select an available input or refresh the list. Downloaded Whisper models load from the local cache first. The bottom status distinguishes receiving audio, loading recognition, and receiving audio without recognized speech. After installing an update, close the entire app and reopen it before starting monitoring.

Images preview as soon as a validated result meets the relevance minimum, while remaining images load. The image-status line shows **Found / Loaded / Ready / Skipped** counts and explains rate limits, resolution rejection, insufficient relevance, pause, or display errors. Transcription can continue independently while an image source is unavailable. Topic changes recheck downloaded candidates against current context instead of discarding an entire batch.

An image relevance-model failure appears in image status while speech monitoring continues; the selected ranking method and relevance minimum are preserved. Narrative-model errors appear in **Context**, retain recorded text for retry, and leave transcription running. An unavailable local model is not silently replaced with basic phrase matching. CPU interpretation and image ranking do not acquire the transcription GPU's lock.

## Audiobook presets

While monitoring is stopped, choose a profile in **Audiobooks** and click **Apply**. Each profile selects **Full session · local model** interpretation and **Full context + image embeddings** ranking, alongside pacing, sources and relevance minimum. It preserves your audio input, speech model, language, local context model, optional book background, cache and transition. Click **Save settings** to keep the profile for the next launch, then **Start monitoring**. You can adjust its values afterward.

| Preset | Slide interval | Recorded context | Relevance minimum | Sources |
| --- | --- | --- | --- | --- |
| Fiction | 10 seconds | Full session | 0.40 | Wikimedia Commons, Openverse |
| History & Biography | 12 seconds | Full session | 0.40 | Wikimedia Commons, Internet Archive |
| Science & Technical | 10 seconds | Full session | 0.35 | Wikimedia Commons, Openverse |

Use History & Biography for historical nonfiction. **Book or context** optionally supplies a title, setting or other background to the local interpreter; changing it starts a fresh session on the next Start, with a status explanation. Every newly recorded segment is processed incrementally into the established narrative context, and the full retained text contributes to image relevance. The profiles retain legacy `transcript_window_seconds` values of 90/120/90 for compatibility; those values do not cut off full-session mode. Older text already discarded by an earlier app version cannot be recovered.

If a search has no results, bounded alternative queries come from the interpreted visual subjects in full-session mode, or from locally identified phrases in basic mode. The selected providers and relevance minimum remain unchanged. The image status identifies an alternative search. Successful empty searches are cached.

## Local relevance options

The example configuration and audiobook profiles use `semantic_provider = "ollama"` with `context_model = "qwen2.5:7b"`, plus `ranking = "context_clip"`. The interpreter processes new passages with the established narrative summary and optional background. The raw transcript remains available locally. Model summaries can still omit details or misunderstand a scene; inspect **Context** when matches are wrong.

If you installed without `-WithClip`, add the image-embedding dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[clip]"
```

CLIP weights download once into the app's model cache. **Full context + image embeddings** compares actual image pixels with the current scene/subjects (50%), narrative summary and optional book background (30%), and every retained transcript segment (20%). Text is split into windows of at most 75 tokens, and full-session windows are weighted by token count rather than truncating the transcript to one CLIP input. The combined cosine similarity is rescaled to a 0–1 relevance score; it is not a probability. Source credibility and image size do not boost this score. Your selected relevance minimum is applied unchanged.

CPU image ranking is the default. GPU CLIP can be configured with `clip_device = "cuda"` if compatible PyTorch/CUDA packages are installed. Image embeddings and local interpretation do not guarantee relevance, factual identity or historical authenticity. Whole-session averaging can dilute a brief detail even though its text contributes.

For a lighter baseline, select **Basic phrase analysis** (`semantic_provider = "contextual"`) and **Source metadata (basic)** (`ranking = "metadata"`). This English heuristic focuses on recent phrases and cannot reliably resolve narrative relationships. **Local CLIP embeddings** (`ranking = "clip"`) is the older query-only image matcher; it does not use full-session narrative embeddings. Its scores differ from full-context scores, so the same threshold can behave differently across modes.

Optional English dependency-based phrases and named entities:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[semantic]"
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Select **spaCy phrases (optional)** or set `semantic_provider = "spacy"`. This improves phrase/entity extraction but is still a basic interpretation option.

Whisper automatic GPU detection falls back to CPU when CUDA libraries/inference fail. GPU Whisper requires the compatible CUDA 12 cuBLAS and cuDNN 9 runtime described in the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper#gpu). The PowerShell setup does not install NVIDIA drivers or CUDA system runtimes. Use `--cpu` for reliable operation without CUDA. First-use native model loading/inference cannot be forcibly interrupted, but worker cancellation prevents late slides and window closure does not wait for model downloads.

## Sources, privacy and licenses

Wikimedia Commons is the default. DuckDuckGo Images is installed by the setup's `[search]` extra; Openverse and Internet Archive use anonymous documented APIs. Google Images is intentionally not bundled without a compliant access method. Known generated-image metadata is excluded, but absent labels are not proof of authenticity. Every retrieved image remains labelled **Unverified source imagery** and retains its source page. Unknown creator/license metadata stays unknown. Application MIT licensing does not cover source images.

No raw audio or transcript is sent to image providers. The local narrative service receives passages only through `127.0.0.1:11434`, using verified cached GGUF weights; cloud models and automatic narrative-model downloads are rejected. Selected image providers receive up to eight sanitized query words, which can reveal discussed names/topics. SQLite stores queries, image attribution and optional displayed-image history, not transcripts or narrative summaries. **Clear cache** removes queries/images/history while retaining model weights; **Delete session history** clears stored records and the local book context. See [privacy](docs/privacy.md), [provider limitations](docs/providers.md), [architecture](docs/architecture.md), and [third-party licenses](docs/third-party.md).

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

# Semantic Image Stream

A free, local Windows desktop app that listens to spoken audio and retrieves existing images for an attributed slideshow. No images are generated. This initial release implements the audio → local Whisper → contextual search → downloaded, ranked slideshow path.

## Install on Windows 11

Install 64-bit Python 3.11, 3.12 or 3.13 and Git. Python 3.14 is not supported by this dependency envelope. From PowerShell:

```powershell
git clone https://github.com/sporemound/semantic-image-stream.git
Set-Location semantic-image-stream
git switch codex/windows-mvp
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -m semantic_image_stream --config config.local.toml
```

The repository may require your GitHub login because its current visibility is private. `ExecutionPolicy Bypass` applies to this one process; the script does not change your machine policy. The setup creates an isolated `.venv`, installs free dependencies, and copies example settings without replacing an existing local configuration. A compatible existing Python can be passed with `-PythonPath 'C:\path\to\python.exe'`.

Core operation needs an internet connection for image searches/downloads and the initial speech-model download. Audio, transcription and topic analysis stay local. No subscription or API key is required. Start with **base**, or use **tiny** for a fast CPU check. On an 8 GB GPU, **base/small** Whisper and CPU CLIP keep model memory modest.

## Use

1. Select **System audio · WASAPI loopback**, choose the output device whose playback you want to accompany, and click **Start monitoring**. Continue playing your podcast/lecture normally. Only an input loopback stream is opened.
2. The timestamped transcript and current topic appear beside the stage. Compact queries go to the checked image providers. Existing images are fetched, verified, deduplicated and ranked before display. The first image may take several seconds, longer on first model download.
3. Adjust image interval, relevance minimum, topic sensitivity, and capture/semantic/search intervals. Input/search/model settings apply to the next monitoring run; presentation interval, transitions and attribution visibility update live.
4. **Pause images** holds a frame while monitoring continues; **Next image** advances manually. The image history can revisit earlier slides. Source links open the original attribution page. **Save image + attribution** exports the verified raster and a JSON file with creator, license and source URLs.
5. Choose a fullscreen display and press **Fullscreen** or **F11**. The slideshow fills the selected display, with controls and status panels hidden. **Escape** restores the normal window and attribution; **Space** pauses/resumes and **Right arrow** advances.
6. **Stop** ends monitoring while keeping the current slide. Completed recording replay also leaves images visible until Stop. Deletion controls are enabled while stopped.

Microphone input is an alternative. To test a real recording or the included original synthetic speech fixtures:

```powershell
.\.venv\Scripts\python.exe -m semantic_image_stream --cpu --audio-file .\tests\fixtures\astronomy.wav
.\.venv\Scripts\python.exe -m semantic_image_stream --list-devices loopback
```

Device indices can change after reconnecting devices. Use **↻** to refresh. Capture interruptions retry a limited number of times and surface failures in the status area. Fresh approved images take priority. After the first batch, the slideshow cycles the current topic's approved images at your interval if at least two are available. A topic change clears the previous playlist and holds its last frame until new relevant images arrive. A single accepted image stays visible until there is another usable match.

Start becomes available when audio-device discovery finishes. A missing saved input is shown explicitly; select an available input or refresh the list. Downloaded Whisper models load from the local cache first. The bottom status distinguishes receiving audio, loading recognition, and receiving audio without recognized speech. After installing an update, close the entire app and reopen it before starting monitoring.

Images preview as soon as a validated result meets the relevance minimum, while remaining images load. The image-status line shows **Found / Loaded / Ready / Skipped** counts and explains rate limits, resolution rejection, insufficient relevance, pause, or display errors. Transcription can continue independently while an image source is unavailable. Topic changes recheck downloaded candidates against current context instead of discarding an entire batch.

An image relevance-model failure appears in image status while speech monitoring continues; the selected ranking method and relevance minimum are preserved. CPU image ranking does not acquire the transcription GPU's lock.

## Audiobook presets

While monitoring is stopped, choose a profile in **Audiobooks** and click **Apply**. The profile changes pacing, topic persistence, source selection and metadata relevance settings; it preserves your selected audio input, speech model, language, cache and transition. Click **Save settings** to keep the profile for the next launch, then **Start monitoring**. You can adjust its values afterward.

| Preset | Slide interval | Transcript context | Relevance minimum | Sources |
| --- | --- | --- | --- | --- |
| Fiction | 10 seconds | 90 seconds | 0.30 | Wikimedia Commons, Openverse |
| History & Biography | 12 seconds | 120 seconds | 0.40 | Wikimedia Commons, Internet Archive |
| Science & Technical | 10 seconds | 90 seconds | 0.35 | Wikimedia Commons, Openverse |

These profiles use the local contextual analyzer and source-metadata ranking. Use History & Biography for historical nonfiction. Longer context keeps names and subjects available through explanatory narration. If an exact query returns no results, the app tries up to two shorter subjects or names derived from the same local context, using your selected providers and unchanged relevance minimum. The image status indicates when an alternative query is being used. Successful empty searches are cached so the same unsuccessful phrase is not repeatedly sent to the sources.

## Local relevance options

The default `contextual` semantic analyzer uses compound concepts, co-mentioned entities/phrases, recent context and topic distance. It is an English heuristic baseline, not a language model. It can misunderstand negation, ambiguous subjects and fast transitions. The default metadata ranker combines query/context overlap, source weighting, quality and novelty; it is not a calibrated probability.

For actual multimodal CLIP image/text embeddings, install the extra and set `ranking = "clip"` in your TOML settings, or choose **Local CLIP embeddings** in the UI:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[clip]"
```

CLIP weights download once into the app's model cache. CPU ranking is the default, and a 0.30 relevance threshold is a useful starting point; tune it for your material. CLIP reduces some title ambiguities but does not guarantee relevance or historical authenticity. GPU CLIP can be configured with `clip_device = "cuda"` if compatible PyTorch/CUDA packages are installed.

Optional English dependency-based phrases and named entities:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[semantic]"
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

Set `semantic_provider = "spacy"`. The interfaces support replacing the analyzer or ranker with other local models later.

Whisper automatic GPU detection falls back to CPU when CUDA libraries/inference fail. GPU Whisper requires the compatible CUDA 12 cuBLAS and cuDNN 9 runtime described in the [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper#gpu). The PowerShell setup does not install NVIDIA drivers or CUDA system runtimes. Use `--cpu` for reliable operation without CUDA. First-use native model loading/inference cannot be forcibly interrupted, but worker cancellation prevents late slides and window closure does not wait for model downloads.

## Sources, privacy and licenses

Wikimedia Commons is the default. DuckDuckGo Images is installed by the setup's `[search]` extra; Openverse and Internet Archive use anonymous documented APIs. Google Images is intentionally not bundled without a compliant access method. Known generated-image metadata is excluded, but absent labels are not proof of authenticity. Every retrieved image remains labelled **Unverified source imagery** and retains its source page. Unknown creator/license metadata stays unknown. Application MIT licensing does not cover source images.

No raw audio or complete transcript is uploaded. Selected providers receive up to eight sanitized query words. Queries can still reveal discussed names/topics. SQLite stores queries, image attribution and optional displayed-image history, not transcripts. **Clear cache** removes queries/images/history while retaining model weights; **Delete history** clears stored session records and in-memory transcript/topic panels. See [privacy](docs/privacy.md), [provider limitations](docs/providers.md), [architecture](docs/architecture.md), and [third-party licenses](docs/third-party.md).

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

# Local relevance training and story examples

## Use the prepared pack

Stop monitoring and select **Full context + trained relevance**. The **Training** label shows whether the local model is usable, its encoder, image/example counts, and story-example availability. Hover over the label for the files in use. Select **Save settings** to retain the mode. The app preserves your relevance minimum when switching modes; score scales differ, so choose a cutoff deliberately.

The bundled model is a trained score calibrator fitted using 320 public image/metadata pairs. It preserves full-context CLIP ordering and did not improve held-out matching accuracy. The optional calibration-derived minimum of **0.389** applies to this small caption-pair benchmark, not to audiobook scenes or factual correctness. The app never applies it automatically. Check accepted images and **Context**, then adjust **Relevance minimum** for the material you are listening to. Choosing a different ranking mode keeps the older scoring path intact.

**Use local story examples** independently controls the prepared narrative illustrations. Turning it off keeps the files. Book memory remains separate: **New book / clear context**, **Delete session history** and **Clear cache** do not delete a trained model or public example pack.

## What was trained

The frozen `openai/clip-vit-base-patch32` encoder produces normalized 512-dimensional image and text vectors. Training fitted two candidates: a 1,025-feature logistic relevance head using elementwise product, absolute difference and cosine similarity, and a smaller monotonic cosine-only model. The full head performed worse on calibration data and was rejected. The shipped model fits cosine scale/intercept on training pairs, followed by separate calibration. Neither CLIP nor Qwen weights change.

For the selected `cosine_only` model, runtime first combines raw cosine similarities with the ordinary full-session weights: current scene/subjects 50%, summary/background 30%, and all retained transcript text 20%. It then applies the trained scalar/sigmoid calibration once. This preserves the underlying full-context CLIP ordering instead of averaging nonlinear per-context scores.

Pair positives associate each image with its compiled metadata caption. Negatives use random and cosine-hard captions from another source image. These are weak labels: a different image's caption can still describe the same object or scene. Calibration fits on separate source groups. Trained scores estimate these pair labels; they are not calibrated probabilities that an image depicts an audiobook correctly.

The public image pack is sampled from the original dataset's **train** split, deduplicated before a local 75% training / 12.5% calibration / 12.5% test split. Source-image ownership and perceptual hashes prevent an image or near-duplicate from crossing splits. The report compares matching-caption top-1 retrieval and pair AUC with raw CLIP cosine on the untouched test groups. This evaluation uses museum metadata, not recorded audiobook scenes; broader story and photograph evaluation is still needed.

## Evaluation

The [bundled training report](../src/semantic_image_stream/data/training/training-report.json) records 240 fitting images, 40 calibration images and 40 untouched test images. Each split has one positive and two weak negative caption pairs per image. Candidate selection used calibration data only: text-to-image top-1 was 40% for the full head versus 82.5% for cosine-only; pair AUC was 0.8378 versus 0.8678. The test set did not choose the model.

| Held-out caption benchmark | Selected calibration | Raw CLIP baseline |
| --- | --- | --- |
| Text-to-image top-1, 40 images | 65% | 65% |
| Pair AUC, 120 pairs | 0.790625 | 0.790625 |
| Brier score for pair labels | 0.1701 | Not reported |

The equal ranking results follow from the selected monotonic score transformation. This is an actual fitted calibrator, with no demonstrated ranking improvement or audiobook evaluation. The 0.389 minimum was chosen by calibration F1, not test accuracy; its usefulness for full-session audiobook context remains unestablished.

## Public sources and separate licenses

The 320 image pairs come from **Mitsua/art-museums-pd-440k**, revision `fba945da78b36262eb9272067197cc28d06cffbf`. English captions are compiled from human museum metadata. The source images and original metadata are CC0; ELAN MITSUA Project / Abstract Engine license the compiled dataset and captions under CC BY 4.0. Preserve this credit, the dataset revision and item source URLs when redistributing a pack. [Museum dataset card and license](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k#license).

The 128 narrative illustrations come from **WorkInTheDark/FairytaleQA**, revision `655b20b370f8ec6ef0cae5ac08f1c27ec6e72aaf`, using only its **train** split. They cover character, setting, action and outcome resolution; selected answers occur literally in the associated passage. FairytaleQA supplies human educational annotations and identifies its underlying stories as Project Gutenberg material. The dataset/repository carries Apache 2.0; underlying story rights and individual conditions remain separate. Preserve source/story identifiers and the supplied notices rather than treating the application MIT license as covering all data. [FairytaleQA card](https://huggingface.co/datasets/WorkInTheDark/FairytaleQA), [upstream repository and license](https://github.com/uci-soe/FairytaleQAData), [Project Gutenberg permissions](https://www.gutenberg.org/policy/permission.html).

The [training notices](../src/semantic_image_stream/data/training/TRAINING_NOTICES.txt) retain dataset credits and the separate CC BY 4.0 terms for the trained calibration. The [upstream Apache 2.0 license](../src/semantic_image_stream/data/training/FAIRYTALEQA_LICENSE.txt) accompanies the public story examples.

At runtime, at most two examples are selected locally within a 1,200-character prompt bound. They illustrate how to read supplied evidence; their characters and setting cannot establish facts about the listener's current book. Raw private narration is never included in a Hugging Face request or in the training artifacts.

## Files and local operation

The app bundles compact `relevance-model.json`, `training-report.json`, `story_examples.json` and training notices under `semantic_image_stream/data/training`. It does not bundle the sampled museum images or private transcripts. These assets are independent of `sessions/last-session.json`.

An installed local file in `%LOCALAPPDATA%\SemanticImageStream\training` takes precedence over its bundled counterpart. `training_dir` can select a different folder. An explicit folder with a missing file does not silently use the bundled version. Invalid packs are left unused with a status message. A missing or incompatible trained model leaves image ranking unavailable until a usable pack or another mode is selected; speech monitoring can continue. The model loader checks its schema, finite numeric values, feature layout and encoder identity; public examples have separate size/schema/provenance checks. No pickle or remote dataset code is loaded.

Trained mode and the training command require complete CLIP weights already in the app's cache. `-WithClip` installs dependencies, not weights. Trained mode does not download missing weights or a training pack during **Start monitoring**. Existing image-provider searches still require the network. Qwen also needs its manually installed local model, as described in the [main setup instructions](../README.md#install-on-windows-11).

## Reproduce a pack

Run these explicit commands from the repository with its environment installed. Preparation downloads public data; training then uses only the prepared files and cached encoder. Replace the cache path if your configuration uses another folder.

```powershell
.\.venv\Scripts\python.exe -m semantic_image_stream.training_data --output .training-pack --images 320 --source museum
.\.venv\Scripts\python.exe -m semantic_image_stream.train_relevance --pack .training-pack --cache-dir "$env:LOCALAPPDATA\SemanticImageStream"
```

Preparation accepts 100–512 images and caps total response bodies at 160 MiB. It permits fixed HTTPS hosts, validates each redirect, bounds row/image responses and checks raster formats, geometry and hashes. Story CSV is fetched at its pinned revision. The Hugging Face `/rows` endpoint has no revision selector: the sampler checks repository revision before and after acquisition, verifies revision-bearing museum assets, and retains hashes of the accepted rows and bytes. Reproducing sampling from a changing service is therefore weaker than downloading immutable files. [Official rows API](https://huggingface.co/docs/dataset-viewer/rows).

The output includes `manifest.json` with provenance/checksums/splits, sampled `images/`, `story_examples.json`, cached public embeddings, `relevance-model.json` and `training-report.json`. Keep the manifest and applicable notices with redistributed artifacts. Read the report before selecting a model or its recommended threshold; the app never applies that recommendation automatically.

For a manually prepared replacement, set `training_dir` to the pack's absolute folder in your TOML configuration and restart the app. The trained head must match your configured `clip_model`. Keep public training files separate from saved stories; there is no private-transcript upload or automatic retraining workflow.

The optional `--source commons` sampler is a prototype for a broader outdoor/photographic pack. It honors HTTP 429 by stopping acquisition, and a too-small sample is rejected for training. It is not the source of the bundled museum model. A larger diverse pack and scene-level evaluation are future work.

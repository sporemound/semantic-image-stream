# Licensing and third-party components

Original Semantic Image Stream code is MIT licensed. The original repository contained only a GPL-3.0 license notice; this implementation replaces the default code license with MIT as requested by the repository owner. Original program source from other projects has not been copied into this repository.

Dependency/model/image licenses are independent. This table is an orientation aid; inspect the installed distributions' authoritative license notices before distributing a binary. Optional extras can add obligations and size.

| Component | Upstream license / notice |
| --- | --- |
| PySide6 / Qt | LGPLv3/GPLv3/commercial choices; the open-source distribution requires preserving notices and LGPL relinking/replacement rights. Keep Qt libraries as separate files in a one-folder build. See [Qt licensing](https://doc.qt.io/qt-6/licensing.html). |
| faster-whisper, CTranslate2, PyAudioWPatch, httpx, ImageHash, spaCy | MIT, per their upstream distribution notices |
| NumPy, SciPy, sounddevice | BSD-family, per upstream notices |
| Pillow | HPND/Pillow license |
| PyAV / bundled FFmpeg | PyAV BSD-family; bundled FFmpeg components carry their own LGPL/GPL notices depending on build |
| DDGS | MIT; remote search-service terms remain separate |
| Transformers | Apache-2.0 |
| PyTorch | BSD-family with bundled third-party notices |
| Optional Ollama local runtime | [MIT](https://github.com/ollama/ollama/blob/main/LICENSE); installed separately and not bundled |
| Default Qwen2.5-7B-Instruct narrative weights | [Apache-2.0 model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct); manually cached separately and not bundled |
| OpenAI Whisper weights; optional CLIP weights | Check the selected model card and model license. Defaults use MIT-licensed upstream OpenAI models. |
| Retrieved images | Their individual source licenses, never the application's MIT license. Attribution JSON preserves reported reuse metadata; unknown licenses remain unknown. |
| Bundled trained relevance calibration and museum training metadata | CC BY 4.0; source images/original museum metadata identified as CC0. Preserve [training notices](../src/semantic_image_stream/data/training/TRAINING_NOTICES.txt) and per-item credits in the manifest. Sampled image pixels and CLIP weights are not bundled. |
| Bundled FairytaleQA human story examples | Apache 2.0 dataset annotations, separate underlying public-domain story rights; [full license](../src/semantic_image_stream/data/training/FAIRYTALEQA_LICENSE.txt). See [provenance and handling](training.md). |

The packaging script is a build scaffold, not a completed compliance bundle. Before publishing a binary, include the actual installed dependencies' license files/notices, verify FFmpeg/Qt redistribution obligations, and provide needed source/replacement rights. No credentials or model weights are checked into the repository.

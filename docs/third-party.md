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
| OpenAI Whisper weights; optional CLIP weights | Check the selected model card and model license. Defaults use MIT-licensed upstream OpenAI models. |
| Retrieved images | Their individual source licenses, never the application's MIT license. Attribution JSON preserves reported reuse metadata; unknown licenses remain unknown. |

The packaging script is a build scaffold, not a completed compliance bundle. Before publishing a binary, include the actual installed dependencies' license files/notices, verify FFmpeg/Qt redistribution obligations, and provide needed source/replacement rights. No credentials or model weights are checked into the repository.

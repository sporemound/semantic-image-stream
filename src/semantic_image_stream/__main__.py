import argparse
import os
from pathlib import Path
import sys

from .configuration import Settings
from .logging_setup import configure_logging


def main(argv=None):
    if (sys.argv[1:] if argv is None else argv) == ["--ddgs-worker"]:
        from .ddgs_worker import main as worker_main
        return worker_main()
    # Small local feature arrays do not need a large OpenBLAS worker pool.
    # Set this before optional NumPy/SciPy imports; explicit user overrides stay.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    parser = argparse.ArgumentParser(description="Local speech-driven slideshow of existing, attributed imagery.")
    parser.add_argument("--config", type=Path, help="TOML settings file")
    parser.add_argument("--audio-file", type=Path, help="Replay a real recording instead of live audio")
    parser.add_argument("--list-devices", choices=["loopback", "microphone"])
    parser.add_argument("--cpu", action="store_true", help="Run Whisper on CPU")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        if args.list_devices:
            from .audio_capture import list_devices
            for device in list_devices(args.list_devices):
                print(f"{device.index}: {device.name} ({device.sample_rate} Hz, {device.channels} channels)")
            return 0
        settings = Settings.load(args.config)
        if args.audio_file:
            if not args.audio_file.is_file():
                parser.error(f"Recording does not exist: {args.audio_file}")
            settings.audio_mode = "file"
            settings.audio_file = str(args.audio_file.resolve())
        if args.cpu:
            settings.whisper_device = "cpu"
        from .session_export_ui import run
        return run(settings, args.config)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"Cannot launch Semantic Image Stream: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

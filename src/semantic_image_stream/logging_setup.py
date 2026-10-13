"""Console operational logging only; raw audio and transcript text are excluded."""
import logging


def configure_logging(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # HTTP debug logs may contain queries; avoid them even in verbose mode.
    for name in ("httpx", "httpcore", "faster_whisper", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

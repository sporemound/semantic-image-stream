"""Local speech-driven retrieval of existing imagery."""

import os

from .ai_filter import install as _install_ai_filter

__version__ = "0.2.7"

# The filter imports image libraries, so configure native thread pools first.
# Preserve explicit user settings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

_install_ai_filter()
del _install_ai_filter

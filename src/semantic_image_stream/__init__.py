"""Local speech-driven retrieval of existing imagery."""

from .ai_filter import install as _install_ai_filter

__version__ = "0.2.5"

_install_ai_filter()
del _install_ai_filter

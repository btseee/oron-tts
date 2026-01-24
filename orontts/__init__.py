"""OronTTS: Mongolian Cyrillic (Khalkha) VITS2 Text-to-Speech System."""

from orontts.exceptions import (
    OronTTSError,
    AudioProcessingError,
    TextNormalizationError,
    ModelError,
    ConfigurationError,
)

__version__ = "0.1.0"
__all__ = [
    "OronTTSError",
    "AudioProcessingError",
    "TextNormalizationError",
    "ModelError",
    "ConfigurationError",
]

"""Custom exceptions for OronTTS."""


class OronTTSError(Exception):
    """Base exception for all OronTTS errors."""


class AudioProcessingError(OronTTSError):
    """Raised when audio processing fails."""


class TextNormalizationError(OronTTSError):
    """Raised when text normalization fails."""


class ModelError(OronTTSError):
    """Raised when model operations fail."""


class ConfigurationError(OronTTSError):
    """Raised when configuration is invalid."""


class DatasetError(OronTTSError):
    """Raised when dataset operations fail."""


class PhonemizationError(OronTTSError):
    """Raised when phonemization fails."""


class InferenceError(OronTTSError):
    """Raised when inference fails."""

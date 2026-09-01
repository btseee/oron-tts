"""OronTTS: Mongolian (Khalkha Cyrillic) text-to-speech.

The model is a finetune of upstream F5-TTS `F5TTS_v1_Base`; training lives in
the `f5_tts` package. This repository owns the Mongolian-specific layer:
text normalization, the extended vocabulary, evaluation, and the packaged
reference voices used to select a male or female speaker at inference.
"""

__version__ = "0.2.0"

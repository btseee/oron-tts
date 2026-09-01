"""Mongolian (Khalkha Cyrillic) text normalization for oron-tts.

Single source of truth, imported by oron-cleaner as well, so that the text
published in the corpus, the text scored for CER, and the text fed to the model
are guaranteed to be the same string.
"""

from oron_tts.text.normalizer import MongolianNormalizer
from oron_tts.text.numbers import NumberNormalizer
from oron_tts.text.vocab import VocabError, charset, check, load_vocab, unsupported

__all__ = [
    "MongolianNormalizer",
    "NumberNormalizer",
    "VocabError",
    "charset",
    "check",
    "load_vocab",
    "unsupported",
]

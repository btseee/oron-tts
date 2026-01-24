"""Phonemizer for Mongolian Cyrillic using rule-based transliteration.

Note: espeak-ng does NOT support Mongolian (mn). This module implements
a direct Cyrillic-to-IPA mapping based on Khalkha Mongolian phonology.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Self

from orontts.constants import PHONEME_TO_ID
from orontts.exceptions import PhonemizationError

# Mongolian Cyrillic to IPA mapping for Khalkha dialect
# Based on standard Mongolian phonology
# Reference: https://en.wikipedia.org/wiki/Mongolian_phonology
CYRILLIC_TO_IPA: dict[str, str] = {
    # Vowels (short)
    "а": "a",
    "э": "e",
    "и": "i",
    "о": "ɔ",
    "у": "ʊ",
    "ө": "o",
    "ү": "u",
    "ы": "i",  # Rare, occurs in Russian loanwords
    # Long vowels (geminated in IPA)
    "аа": "aː",
    "ээ": "eː",
    "ий": "iː",
    "ии": "iː",
    "оо": "ɔː",
    "уу": "ʊː",
    "өө": "oː",
    "үү": "uː",
    # Diphthongs
    "ай": "ai",
    "ой": "ɔi",
    "уй": "ʊi",
    "эй": "ei",
    "үй": "ui",
    # Consonants
    "б": "p",  # Unaspirated bilabial stop
    "в": "w",  # Often /w/ word-initially
    "г": "ɡ",  # Velar stop
    "д": "t",  # Unaspirated dental stop
    "ж": "dʒ",  # Postalveolar affricate
    "з": "ts",  # Actually alveolar affricate in Mongolian
    "й": "j",  # Palatal approximant
    "к": "kʰ",  # Aspirated (mostly in loanwords)
    "л": "ɮ",  # Lateral fricative (unique to Mongolian)
    "м": "m",
    "н": "n",
    "п": "pʰ",  # Aspirated (mostly in loanwords)
    "р": "r",  # Alveolar trill
    "с": "s",
    "т": "tʰ",  # Aspirated (mostly in loanwords)
    "ф": "f",  # In loanwords
    "х": "x",  # Velar fricative
    "ц": "tsʰ",  # Aspirated affricate
    "ч": "tʃʰ",  # Aspirated postalveolar affricate
    "ш": "ʃ",  # Postalveolar fricative
    "щ": "ʃtʃ",  # In Russian loanwords
    "ъ": "",  # Hard sign (silent)
    "ь": "ʲ",  # Soft sign (palatalization marker)
    "е": "je",  # /je/ word-initially, /e/ after consonants
    "ё": "jɔ",  # /jɔ/
    "ю": "jʊ",  # /jʊ/
    "я": "ja",  # /ja/
    # Uppercase (same mappings)
    "А": "a",
    "Э": "e",
    "И": "i",
    "О": "ɔ",
    "У": "ʊ",
    "Ө": "o",
    "Ү": "u",
    "Ы": "i",
    "Б": "p",
    "В": "w",
    "Г": "ɡ",
    "Д": "t",
    "Ж": "dʒ",
    "З": "ts",
    "Й": "j",
    "К": "kʰ",
    "Л": "ɮ",
    "М": "m",
    "Н": "n",
    "П": "pʰ",
    "Р": "r",
    "С": "s",
    "Т": "tʰ",
    "Ф": "f",
    "Х": "x",
    "Ц": "tsʰ",
    "Ч": "tʃʰ",
    "Ш": "ʃ",
    "Щ": "ʃtʃ",
    "Ъ": "",
    "Ь": "ʲ",
    "Е": "je",
    "Ё": "jɔ",
    "Ю": "jʊ",
    "Я": "ja",
}

# Long vowel patterns to detect (check these first)
LONG_VOWEL_PATTERNS = ["аа", "ээ", "ий", "ии", "оо", "уу", "өө", "үү",
                        "АА", "ЭЭ", "ИЙ", "ИИ", "ОО", "УУ", "ӨӨ", "ҮҮ"]

# Diphthong patterns
DIPHTHONG_PATTERNS = ["ай", "ой", "уй", "эй", "үй",
                      "АЙ", "ОЙ", "УЙ", "ЭЙ", "ҮЙ"]


@dataclass
class PhonemizerConfig:
    """Configuration for phonemizer."""

    language: str = "mn"  # Mongolian
    backend: str = "rule"  # Rule-based (espeak-ng doesn't support Mongolian)
    with_stress: bool = False  # Mongolian has fixed initial stress
    preserve_punctuation: bool = True
    punctuation_marks: str = "!'(),-.:;? "


class MongolianPhonemizer:
    """Rule-based phonemizer for Mongolian Cyrillic text.

    Uses direct Cyrillic-to-IPA transliteration based on Khalkha phonology.
    Note: espeak-ng does NOT support Mongolian, so we use rules instead.
    """

    def __init__(self, config: PhonemizerConfig | None = None) -> None:
        """Initialize phonemizer.

        Args:
            config: Phonemizer configuration.
        """
        self.config = config or PhonemizerConfig()

    def phonemize(self, text: str) -> str:
        """Convert Mongolian Cyrillic text to IPA phonemes.

        Args:
            text: Input text in Mongolian Cyrillic.

        Returns:
            IPA phoneme string.

        Raises:
            PhonemizationError: If phonemization fails.
        """
        if not text.strip():
            return ""

        try:
            result = []
            i = 0
            text_lower = text.lower()

            while i < len(text):
                matched = False

                # Check for long vowels first (2 chars)
                if i + 1 < len(text):
                    digraph = text_lower[i:i + 2]
                    if digraph in [p.lower() for p in LONG_VOWEL_PATTERNS]:
                        ipa = CYRILLIC_TO_IPA.get(digraph, "")
                        if ipa:
                            result.append(ipa)
                            i += 2
                            matched = True
                            continue
                    # Check for diphthongs
                    if digraph in [p.lower() for p in DIPHTHONG_PATTERNS]:
                        ipa = CYRILLIC_TO_IPA.get(digraph, "")
                        if ipa:
                            result.append(ipa)
                            i += 2
                            matched = True
                            continue

                if not matched:
                    char = text[i]

                    # Handle punctuation
                    if char in self.config.punctuation_marks:
                        if self.config.preserve_punctuation:
                            result.append(char)
                        i += 1
                        continue

                    # Handle whitespace
                    if char.isspace():
                        result.append(" ")
                        i += 1
                        continue

                    # Handle digits (pass through)
                    if char.isdigit():
                        result.append(char)
                        i += 1
                        continue

                    # Map single character
                    ipa = CYRILLIC_TO_IPA.get(char, "")
                    if ipa:
                        result.append(ipa)
                    # Skip unknown characters silently

                    i += 1

            # Join and clean up
            phonemes = "".join(result)
            phonemes = self._clean_phonemes(phonemes)

            return phonemes

        except Exception as e:
            raise PhonemizationError(f"Phonemization error: {e}") from e

    def _clean_phonemes(self, phonemes: str) -> str:
        """Clean and normalize phoneme output."""
        # Remove extra whitespace
        phonemes = " ".join(phonemes.split())

        # Handle punctuation preservation
        if self.config.preserve_punctuation:
            # Keep standard punctuation
            pass
        else:
            # Remove punctuation
            for p in self.config.punctuation_marks:
                if p != " ":
                    phonemes = phonemes.replace(p, "")

        return phonemes

    def phoneme_to_ids(self, phonemes: str) -> list[int]:
        """Convert phoneme string to integer IDs.

        Args:
            phonemes: IPA phoneme string.

        Returns:
            List of phoneme IDs.
        """
        ids: list[int] = []
        for char in phonemes:
            if char in PHONEME_TO_ID:
                ids.append(PHONEME_TO_ID[char])
            elif char == " ":
                ids.append(PHONEME_TO_ID.get(" ", 0))
            # Skip unknown characters
        return ids

    def text_to_ids(self, text: str) -> list[int]:
        """Convert text directly to phoneme IDs.

        Args:
            text: Input text.

        Returns:
            List of phoneme IDs.
        """
        phonemes = self.phonemize(text)
        return self.phoneme_to_ids(phonemes)

    @classmethod
    def from_config(cls, config_dict: dict) -> Self:
        """Create phonemizer from configuration dictionary."""
        config = PhonemizerConfig(**config_dict)
        return cls(config=config)


# Module-level instance for convenience
_default_phonemizer: MongolianPhonemizer | None = None


def get_phonemizer() -> MongolianPhonemizer:
    """Get or create the default phonemizer instance."""
    global _default_phonemizer
    if _default_phonemizer is None:
        _default_phonemizer = MongolianPhonemizer()
    return _default_phonemizer


def phonemize(text: str) -> str:
    """Convenience function to phonemize text.

    Args:
        text: Input text in Mongolian Cyrillic.

    Returns:
        IPA phoneme string.
    """
    return get_phonemizer().phonemize(text)


def text_to_phoneme_ids(text: str) -> list[int]:
    """Convenience function to convert text to phoneme IDs.

    Args:
        text: Input text.

    Returns:
        List of phoneme IDs.
    """
    return get_phonemizer().text_to_ids(text)


@lru_cache(maxsize=10000)
def cached_phonemize(text: str) -> str:
    """Cached phonemization for repeated texts.

    Args:
        text: Input text.

    Returns:
        IPA phoneme string.
    """
    return phonemize(text)

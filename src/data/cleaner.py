"""Mongolian Khalkha text cleaning and normalization.

Handles Cyrillic text preprocessing and numeric-to-text expansion
following Khalkha Mongolian declension rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MongolianTextCleaner:
    """Text normalizer for Mongolian Khalkha dialect.

    Performs:
    - Unicode normalization
    - Punctuation standardization
    - Numeric expansion to Cyrillic text
    - Abbreviation expansion
    """

    # Mongolian Cyrillic alphabet (uppercase + lowercase)
    CYRILLIC_CHARS: ClassVar[str] = (
        "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоӨөПпРрСсТтУуҮүФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
    )

    # Number words in Mongolian
    ONES: ClassVar[dict[int, str]] = {
        0: "тэг",
        1: "нэг",
        2: "хоёр",
        3: "гурав",
        4: "дөрөв",
        5: "тав",
        6: "зургаа",
        7: "долоо",
        8: "найм",
        9: "ес",
    }

    TENS: ClassVar[dict[int, str]] = {
        10: "арав",
        20: "хорь",
        30: "гуч",
        40: "дөч",
        50: "тавь",
        60: "жар",
        70: "дал",
        80: "ная",
        90: "ер",
    }

    HUNDREDS: ClassVar[str] = "зуу"
    THOUSANDS: ClassVar[str] = "мянга"
    MILLIONS: ClassVar[str] = "сая"
    BILLIONS: ClassVar[str] = "тэрбум"

    # Common abbreviations
    ABBREVIATIONS: ClassVar[dict[str, str]] = {
        "км": "километр",
        "м": "метр",
        "см": "сантиметр",
        "мм": "миллиметр",
        "кг": "килограмм",
        "г": "грамм",
        "мг": "миллиграмм",
        "л": "литр",
        "мл": "миллилитр",
        "ш": "ширхэг",
        "төг": "төгрөг",
        "₮": "төгрөг",
        "он": "он",
        "сар": "сар",
        "өдөр": "өдөр",
        "цаг": "цаг",
        "мин": "минут",
        "сек": "секунд",
    }

    def __call__(self, text: str) -> str:
        """Clean and normalize Mongolian text.

        Args:
            text: Raw Mongolian text.

        Returns:
            Cleaned and normalized text.
        """
        text = self.normalize_unicode(text)
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        text = self.normalize_punctuation(text)
        text = self.clean_whitespace(text)
        return text

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC form."""
        import unicodedata

        return unicodedata.normalize("NFC", text)

    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, full in self.ABBREVIATIONS.items():
            # Match abbreviation with word boundary
            pattern = rf"\b{re.escape(abbrev)}\b"
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        return text

    def expand_numbers(self, text: str) -> str:
        """Expand numeric digits to Mongolian words.

        Handles:
        - Cardinal numbers (123 → нэг зуун хорин гурав)
        - Decimal numbers (3.14 → гурван цэг арван дөрөв)
        - Phone-style sequences (read digit by digit)
        """
        # Decimal numbers
        text = re.sub(
            r"(\d+)[.,](\d+)",
            lambda m: f"{self._number_to_words(int(m.group(1)))} цэг {self._digits_to_words(m.group(2))}",
            text,
        )

        # Cardinal numbers
        text = re.sub(
            r"\b(\d+)\b",
            lambda m: self._number_to_words(int(m.group(1))),
            text,
        )

        return text

    def _number_to_words(self, n: int) -> str:
        """Convert integer to Mongolian words."""
        if n < 0:
            return f"хасах {self._number_to_words(-n)}"

        if n == 0:
            return self.ONES[0]

        if n < 10:
            return self.ONES[n]

        if n < 20:
            return f"арван {self.ONES[n - 10]}" if n > 10 else self.TENS[10]

        if n < 100:
            tens, ones = divmod(n, 10)
            result = self.TENS[tens * 10]
            if ones:
                result += f" {self.ONES[ones]}"
            return result

        if n < 1000:
            hundreds, remainder = divmod(n, 100)
            result = f"{self.ONES[hundreds]} {self.HUNDREDS}" if hundreds > 1 else self.HUNDREDS
            if remainder:
                result += f" {self._number_to_words(remainder)}"
            return result

        if n < 1_000_000:
            thousands, remainder = divmod(n, 1000)
            result = f"{self._number_to_words(thousands)} {self.THOUSANDS}"
            if remainder:
                result += f" {self._number_to_words(remainder)}"
            return result

        if n < 1_000_000_000:
            millions, remainder = divmod(n, 1_000_000)
            result = f"{self._number_to_words(millions)} {self.MILLIONS}"
            if remainder:
                result += f" {self._number_to_words(remainder)}"
            return result

        billions, remainder = divmod(n, 1_000_000_000)
        result = f"{self._number_to_words(billions)} {self.BILLIONS}"
        if remainder:
            result += f" {self._number_to_words(remainder)}"
        return result

    def _digits_to_words(self, digits: str) -> str:
        """Convert digit string to words (digit by digit)."""
        return " ".join(self.ONES[int(d)] for d in digits)

    def normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks."""
        # Normalize quotes
        text = re.sub(r"[«»„" "]", '"', text)
        text = re.sub(r"['']", "'", text)

        # Normalize dashes
        text = re.sub(r"[—–−]", "-", text)

        # Normalize ellipsis
        text = re.sub(r"\.{2,}", "...", text)

        return text

    def clean_whitespace(self, text: str) -> str:
        """Remove excess whitespace."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class MongolianPhonemizer:
    """Phonemizer interface for Mongolian using espeak-ng.

    Requires espeak-ng compiled with Mongolian (mn) support.
    """

    def __init__(
        self,
        language: str = "mn",
        backend: str = "espeak",
        with_stress: bool = True,
    ) -> None:
        """Initialize phonemizer.

        Args:
            language: Language code for espeak-ng.
            backend: Phonemizer backend ('espeak' or 'espeak-mbrola').
            with_stress: Include stress markers.
        """
        self.language = language
        self.backend = backend
        self.with_stress = with_stress
        self._phonemizer = None
        self._cleaner = MongolianTextCleaner()

    def _get_phonemizer(self):
        """Lazy initialization of phonemizer."""
        if self._phonemizer is None:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator

            # Check if Mongolian is supported
            if self.language not in EspeakBackend.supported_languages():
                raise RuntimeError(
                    f"Language '{self.language}' not supported by espeak-ng. "
                    "Ensure espeak-ng is compiled with Mongolian data."
                )

            self._phonemizer = phonemize
            self._separator = Separator(phone=" ", word=" | ", syllable="")

        return self._phonemizer

    def __call__(self, text: str) -> str:
        """Convert text to phonemes.

        Args:
            text: Mongolian text.

        Returns:
            Phoneme sequence.
        """
        # Clean text first
        text = self._cleaner(text)

        phonemize = self._get_phonemizer()

        return phonemize(
            text,
            language=self.language,
            backend=self.backend,
            separator=self._separator,
            strip=True,
            preserve_punctuation=True,
            with_stress=self.with_stress,
        )

    def phoneme_to_ids(
        self,
        phonemes: str,
        vocab: dict[str, int] | None = None,
    ) -> list[int]:
        """Convert phoneme string to token IDs.

        Args:
            phonemes: Phoneme string from __call__.
            vocab: Phoneme-to-ID mapping. If None, uses default.

        Returns:
            List of phoneme token IDs.
        """
        if vocab is None:
            vocab = self._build_default_vocab()

        tokens = phonemes.split()
        return [vocab.get(t, vocab.get("<unk>", 0)) for t in tokens]

    def _build_default_vocab(self) -> dict[str, int]:
        """Build default phoneme vocabulary."""
        # IPA symbols common in Mongolian + special tokens
        symbols = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
            "|",  # Special
            " ",
            ".",
            ",",
            "?",
            "!",  # Punctuation
            "a",
            "ɑ",
            "æ",
            "e",
            "ɛ",
            "i",
            "ɪ",  # Vowels
            "o",
            "ɔ",
            "u",
            "ʊ",
            "ø",
            "y",
            "aː",
            "eː",
            "iː",
            "oː",
            "uː",
            "øː",  # Long vowels
            "b",
            "p",
            "d",
            "t",
            "g",
            "k",
            "q",  # Plosives
            "m",
            "n",
            "ŋ",  # Nasals
            "f",
            "v",
            "s",
            "z",
            "ʃ",
            "ʒ",
            "x",
            "h",  # Fricatives
            "ts",
            "tʃ",
            "dʒ",  # Affricates
            "l",
            "r",
            "ɾ",  # Liquids
            "j",
            "w",  # Approximants
            "ˈ",
            "ˌ",  # Stress markers
        ]
        return {s: i for i, s in enumerate(symbols)}

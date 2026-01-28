"""Text cleaning and normalization for Mongolian TTS."""

import re
import unicodedata
from typing import Final

from src.utils.number_norm import NumberNormalizer
from src.utils.phonemizer import MongolianPhonemizer

PUNCTUATION_MAP: Final[dict[str, str]] = {
    "…": "...",
    "–": "-",
    "—": "-",
    "«": '"',
    "»": '"',
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
    "„": '"',
}

ALLOWED_CHARS: Final[set[str]] = set(
    "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
    " .,!?-:;\"'()"
)


class TextCleaner:
    def __init__(self) -> None:
        self._number_normalizer = NumberNormalizer()
        self._phonemizer = MongolianPhonemizer()
        self._whitespace_re = re.compile(r"\s+")
        self._multi_punct_re = re.compile(r"([.!?,]){2,}")

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def replace_punctuation(self, text: str) -> str:
        for old, new in PUNCTUATION_MAP.items():
            text = text.replace(old, new)
        return text

    def remove_invalid_chars(self, text: str) -> str:
        return "".join(c for c in text if c in ALLOWED_CHARS)

    def normalize_whitespace(self, text: str) -> str:
        text = self._whitespace_re.sub(" ", text)
        return text.strip()

    def normalize_punctuation(self, text: str) -> str:
        return self._multi_punct_re.sub(r"\1", text)

    def expand_abbreviations(self, text: str) -> str:
        abbreviations: dict[str, str] = {
            "г.": "оны",
            "км": "километр",
            "м": "метр",
            "см": "сантиметр",
            "кг": "килограмм",
            "г": "грамм",
            "л": "литр",
            "мл": "миллилитр",
            "т.": "товч",
            "тов.": "товч",
            "ж.": "жил",
            "сар.": "сар",
            "өд.": "өдөр",
            "мин.": "минут",
            "сек.": "секунд",
            "цаг.": "цаг",
        }
        for abbr, full in abbreviations.items():
            text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
        return text

    def clean(self, text: str) -> str:
        text = self.normalize_unicode(text)
        text = self.replace_punctuation(text)
        text = self.expand_abbreviations(text)
        text = self._number_normalizer.normalize_text(text)
        text = self.remove_invalid_chars(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_punctuation(text)
        return text.lower()

    def text_to_sequence(self, text: str) -> list[int]:
        cleaned = self.clean(text)
        return self._phonemizer.text_to_ids(cleaned)

    @property
    def vocab_size(self) -> int:
        return self._phonemizer.vocab_size

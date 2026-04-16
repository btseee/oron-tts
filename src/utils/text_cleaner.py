"""Text cleaning and normalization for Mongolian + Kazakh TTS."""

import re
import unicodedata
from typing import Final

from src.utils.number_norm import NumberNormalizer
from src.utils.tokenizer import CyrillicTokenizer

PUNCTUATION_MAP: Final[dict[str, str]] = {
    "…": "...",
    "–": "-",
    "—": "-",
    "«": '"',
    "»": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u201e": '"',
}

ALLOWED_CHARS: Final[set[str]] = set(
    # Mongolian Cyrillic (Khalkha)
    "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
    "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
    # Kazakh-specific
    "әғқңұһіӘҒҚҢҰҺІ"
    # Punctuation
    " .,!?-:;\"'()"
)

# Abbreviations keyed by language. Patterns require a leading digit or
# whitespace boundary to avoid matching inside regular words.
MN_ABBREVIATIONS: Final[dict[str, str]] = {
    "г.": "оны",
    "км": "километр",
    "см": "сантиметр",
    "кг": "килограмм",
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

# Single-letter units only after a digit: "5 м" → "5 метр"
MN_UNIT_ABBREVS: Final[dict[str, str]] = {
    "м": "метр",
    "г": "грамм",
    "л": "литр",
}

KZ_ABBREVIATIONS: Final[dict[str, str]] = {
    "ж.": "жыл",
    "км": "километр",
    "см": "сантиметр",
    "кг": "килограмм",
    "мл": "миллилитр",
    "мин.": "минут",
    "сек.": "секунд",
    "сағ.": "сағат",
}

KZ_UNIT_ABBREVS: Final[dict[str, str]] = {
    "м": "метр",
    "г": "грамм",
    "л": "литр",
}


class TextCleaner:
    def __init__(self) -> None:
        self._mn_normalizer = NumberNormalizer(lang="mn")
        self._kz_normalizer = NumberNormalizer(lang="kz")
        self._tokenizer = CyrillicTokenizer()
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

    def expand_abbreviations(self, text: str, lang: str = "mn") -> str:
        if lang == "kz":
            abbrevs = KZ_ABBREVIATIONS
            units = KZ_UNIT_ABBREVS
        else:
            abbrevs = MN_ABBREVIATIONS
            units = MN_UNIT_ABBREVS

        # Multi-char abbreviations: safe word-boundary match
        for abbr, full in abbrevs.items():
            text = re.sub(rf"(?<!\w){re.escape(abbr)}(?!\w)", full, text)

        # Single-letter units: only after a digit + optional space
        for abbr, full in units.items():
            text = re.sub(rf"(\d)\s*{re.escape(abbr)}(?!\w)", rf"\1 {full}", text)

        return text

    def clean(self, text: str, lang: str = "mn") -> str:
        text = self.normalize_unicode(text)
        text = self.replace_punctuation(text)
        text = self.expand_abbreviations(text, lang=lang)
        normalizer = self._kz_normalizer if lang == "kz" else self._mn_normalizer
        text = normalizer.normalize_text(text)
        text = self.remove_invalid_chars(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_punctuation(text)
        return text.lower()

    def text_to_sequence(
        self,
        text: str,
        lang: str = "mn",
        attr_tokens: list[str] | None = None,
    ) -> list[int]:
        cleaned = self.clean(text, lang=lang)
        return self._tokenizer.encode(cleaned, lang=lang, attr_tokens=attr_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

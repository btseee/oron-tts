"""Mongolian (Khalkha Cyrillic) text normalization.

One rule governs the character mapping: **preserve anything the vocabulary can
represent, map only what it cannot.** The F5-TTS vocabulary already covers
— – … « » № % all ASCII punctuation, digits and Latin letters, so conflating
them away loses information the pretrained embeddings can already carry.

Two behaviours differ deliberately from the previous TextCleaner:

* **Nothing is deleted silently.** `remove_invalid_chars` used to drop any
  character outside a hardcoded Cyrillic set, so a Latin homoglyph typed into a
  Cyrillic word (endemic in Common Voice) vanished from the text while the
  speaker still pronounced it -- manufacturing exactly the text/audio mismatch
  the CER gate exists to detect. Unrepresentable text now raises.

* **Case is preserved.** The pretrained vocabulary holds both cases of Cyrillic
  with trained embeddings; lowercasing discards 31 trained rows to save 3.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Final

from oron_tts.text.numbers import NumberNormalizer
from oron_tts.text.vocab import DEFAULT_VOCAB, check, unsupported

# Only characters the vocabulary lacks. Curly quotes and „ have no entry; NBSP
# is not a vocabulary space; the soft hyphen is an invisible line-break hint
# that carries no pronunciation and is the one character safe to drop.
CHAR_MAP: Final[dict[str, str]] = {
    "“": '"',   # “
    "”": '"',   # ”
    "‘": "'",   # ‘
    "’": "'",   # ’
    "‚": ",",   # ‚
    "„": '"',   # „
    " ": " ",   # non-breaking space
    " ": " ",   # narrow no-break space
    " ": " ",   # thin space
    "​": "",    # zero-width space
    "­": "",    # soft hyphen
    "﻿": "",    # BOM / zero-width no-break space
    "−": "-",   # minus sign
    "ʼ": "'",   # modifier apostrophe
}

# A single letter followed by a period is not an abbreviation, it is a word
# ending a sentence. Three entries were exactly that shape and all three fired
# on ordinary text:
#
#   "г." -> "оны"    shadowed the gram unit plus a full stop, so
#                    "Жин нь 5 г." became "Жин нь таван оны" -- "its weight is
#                    five of-the-year". Guaranteed rather than occasional: the
#                    abbreviation pass runs before the unit pass, so "г." was
#                    always consumed first.
#   "т." -> "товч"   fires on any sentence ending in a word ending in т
#   "ж." -> "жил"    likewise for ж
#
# The multi-letter entries below are kept: a three-character sequence ending in
# a period is unlikely to be a word boundary, and where one is ("сар.", "цаг.")
# the expansion equals the word, so a false match costs nothing.
ABBREVIATIONS: Final[dict[str, str]] = {
    "км": "километр",
    "см": "сантиметр",
    "кг": "килограмм",
    "мл": "миллилитр",
    "тов.": "товч",
    "сар.": "сар",
    "өд.": "өдөр",
    "мин.": "минут",
    "сек.": "секунд",
    "цаг.": "цаг",
}

# Single letters are only units directly after a digit: "5 м" -> "5 метр".
UNIT_ABBREVS: Final[dict[str, str]] = {
    "м": "метр",
    "г": "грамм",
    "л": "литр",
}


class MongolianNormalizer:
    """Normalize Mongolian text into exactly what the model will be fed."""

    def __init__(self, vocab_path: Path | str = DEFAULT_VOCAB) -> None:
        self._vocab_path = vocab_path
        self._numbers = NumberNormalizer()
        self._char_map = str.maketrans(CHAR_MAP)
        self._whitespace_re = re.compile(r"\s+")
        self._multi_punct_re = re.compile(r"([.!?,]){2,}")
        self._abbrev_res = [
            (re.compile(rf"(?<!\w){re.escape(a)}(?!\w)", re.IGNORECASE), full)
            for a, full in ABBREVIATIONS.items()
        ]
        self._unit_res = [
            (re.compile(rf"(\d)\s*{re.escape(a)}(?!\w)", re.IGNORECASE), rf"\1 {full}")
            for a, full in UNIT_ABBREVS.items()
        ]

    def normalize(self, text: str, strict: bool = True) -> str:
        """Return the exact string that should be published, scored and trained on.

        With `strict`, raises VocabError if the result cannot be represented.
        Callers that filter a corpus should instead use `unsupported_chars` and
        reject the row, so the reason is recorded rather than thrown away.
        """
        text = unicodedata.normalize("NFC", text)
        text = text.translate(self._char_map)

        for pattern, full in self._abbrev_res:
            text = pattern.sub(full, text)
        for pattern, repl in self._unit_res:
            text = pattern.sub(repl, text)

        text = self._numbers.normalize_text(text)

        text = self._whitespace_re.sub(" ", text).strip()
        text = self._multi_punct_re.sub(r"\1", text)

        if strict:
            check(text, self._vocab_path)
        return text

    def unsupported_chars(self, text: str) -> list[str]:
        """Characters that would silently become spaces, after normalization."""
        return unsupported(self.normalize(text, strict=False), self._vocab_path)

    def is_representable(self, text: str) -> bool:
        return not self.unsupported_chars(text)

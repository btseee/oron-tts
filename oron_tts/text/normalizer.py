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

from oron_tts.text.lexicon import all_lexicons
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

    # Cyrillic homoglyphs. і is the one non-Mongolian Cyrillic letter in
    # vocab.txt, and it is visually identical to и and pronounced the same,
    # so it passes the vocabulary gate silently and the model learns a second,
    # near-untrained embedding for a letter it already has. Folded here rather
    # than rejected: a typo should cost a character, not a clip.
    #
    # Latin look-alikes are deliberately NOT folded. Latin is in the vocabulary
    # and preserved on purpose, so mapping "c" to "с" would corrupt genuine
    # Latin text.
    "і": "и",
    "Ї": "И",
    "І": "И",
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

# Marks that carry no sound. This is the one place the module's "preserve what
# the vocabulary can represent" rule is overridden, and deliberately: the
# specification reads every one of these constructions as the bare words inside
# (sections 47-49), noting only that the *prosody* should differ -- and F5-TTS
# has no prosody token, so keeping the character asks the model to learn to say
# nothing for it, from a corpus where it is rare.
#
# The full stop, comma, question and exclamation marks are deliberately NOT
# here: they carry phrasing, and upstream's own examples keep them. Nor is the
# en dash, which is a range separator far more often than a bracket.
SILENT_MARKS: Final[str] = "«»\"()[]{}—"

# Marks that are spoken as a word when they open a token.
SPOKEN_PREFIXES: Final[dict[str, str]] = {
    "#": "хаштаг",
    "@": "эт",
}

# A bracketed number is a citation: "[12]" is "ишлэл арван хоёр".
CITATION_WORD: Final[str] = "ишлэл"


class MongolianNormalizer:
    """Normalize Mongolian text into exactly what the model will be fed."""

    def __init__(self, vocab_path: Path | str = DEFAULT_VOCAB,
                 lexicon_dir: Path | str | None = None) -> None:
        self._vocab_path = vocab_path
        self._lexicons = all_lexicons(lexicon_dir)
        self._lexicon_dir = lexicon_dir
        self._numbers = NumberNormalizer(
            reference_words=set(self._lexicons["reference_words"])
        )
        self._char_map = str.maketrans(CHAR_MAP)
        self._whitespace_re = re.compile(r"\s+")
        self._multi_punct_re = re.compile(r"([.!?,]){2,}")

        # Longest first, so "тов." wins over "т" and "МУИС" over "МУ".
        abbreviations = {**ABBREVIATIONS, **self._lexicons["abbreviations"]}
        self._abbrev_res = [
            (re.compile(rf"(?<!\w){re.escape(a)}(?!\w)"), full)
            for a, full in sorted(abbreviations.items(), key=lambda kv: -len(kv[0]))
        ]
        # Units are matched after a digit only, and longest first so "м²" beats
        # "м". Without the ordering "50 м²" became "тавин метр²".
        units = {**UNIT_ABBREVS, **self._lexicons["units"]}
        self._unit_res = [
            (re.compile(rf"(\d)\s*{re.escape(a)}(?!\w)"), rf"\1 {full}")
            for a, full in sorted(units.items(), key=lambda kv: -len(kv[0]))
        ]
        self._emoji_res = [
            (re.compile(re.escape(e)), f" {word} ")
            for e, word in self._lexicons["emoji"].items()
        ]
        self._silent_marks = str.maketrans(dict.fromkeys(SILENT_MARKS, " "))
        self._prefix_res = [
            (re.compile(rf"(?<![\w]){re.escape(mark)}(?=\w)"), word)
            for mark, word in SPOKEN_PREFIXES.items()
        ]
        self._citation_re = re.compile(r"\[\s*(\d{1,4})\s*\]")
        self._foreign_res = [
            (re.compile(rf"\b{re.escape(w)}\b"), spoken)
            for w, spoken in sorted(self._lexicons["foreign_words"].items(),
                                    key=lambda kv: -len(kv[0]))
        ]

    def normalize(self, text: str, strict: bool = True) -> str:
        """Return the exact string that should be published, scored and trained on.

        With `strict`, raises VocabError if the result cannot be represented.
        Callers that filter a corpus should instead use `unsupported_chars` and
        reject the row, so the reason is recorded rather than thrown away.
        """
        text = unicodedata.normalize("NFC", text)
        text = text.translate(self._char_map)

        # Emoji first: they are not in the vocabulary, so anything left becomes
        # a space, and an emoji sitting between two words would silently merge
        # them. An unlisted emoji is left alone and caught by the vocabulary check.
        for pattern, repl in self._emoji_res:
            text = pattern.sub(repl, text)

        # A bracketed number is a citation, so it has to be named before the
        # brackets are stripped: "[12]" is "ишлэл арван хоёр", not "арван хоёр".
        text = self._citation_re.sub(rf"{CITATION_WORD} \1", text)
        # "#Монгол" -> "хаштаг Монгол", "@bat" -> "эт bat".
        for pattern, word in self._prefix_res:
            text = pattern.sub(f"{word} ", text)

        for pattern, full in self._abbrev_res:
            text = pattern.sub(full, text)
        for pattern, repl in self._unit_res:
            text = pattern.sub(repl, text)
        # After units, so "5 kW" is a unit rather than a foreign word.
        for pattern, repl in self._foreign_res:
            text = pattern.sub(repl, text)

        text = self._numbers.normalize_text(text)

        # After the numbers, not before: the em dash is also a range separator,
        # and "2020—2024" needs it intact to be read as a range.
        text = text.translate(self._silent_marks)

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

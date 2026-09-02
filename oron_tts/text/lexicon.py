"""Lookup tables the normaliser cannot derive.

Abbreviations, letter names, foreign spellings, emoji, units. All of it is
Mongolian content that has to be written by someone who speaks Mongolian, so
none of it is generated -- see `data/lexicon/README.md`.

An absent entry is never guessed at. Each caller has a documented fallback, and
the fallbacks are chosen so the failure is inert rather than wrong: a Latin word
stays Latin (the model will attempt it), an unknown emoji is dropped (it is not
in the vocabulary and would otherwise become a space), an unknown abbreviation
is spelled out.

This is the same discipline as `SUFFIXED_FORMS` in `numbers.py`, for the same
reason: in this project the corpus text, the CER reference and the training
target are one string, so a wrong expansion is published, scored against itself,
and learned.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

LEXICON_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "lexicon"

NAMES: Final[tuple[str, ...]] = (
    "abbreviations",
    "latin_letters",
    "cyrillic_letters",
    "foreign_words",
    "emoji",
    "units",
    "reference_words",
)


def parse(text: str) -> dict[str, str]:
    """Two columns, tab-separated. `#` comments and blank lines are ignored."""
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        # Only the line ending is stripped here, not the whole line: stripping
        # first would swallow a trailing tab and turn "a<TAB>" -- an entry whose
        # spoken form was left blank -- into a missing-tab error, or worse into
        # a silently accepted row.
        line = raw.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(f"line {lineno}: expected a tab, got {line.strip()!r}")
        key, _, value = line.partition("\t")
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"line {lineno}: empty column in {line!r}")
        if key in out:
            raise ValueError(f"line {lineno}: {key!r} is already defined")
        out[key] = value
    return out


@cache
def load(name: str, directory: Path | str | None = None) -> dict[str, str]:
    """One lexicon by name. Missing file means an empty table, not an error.

    A missing file is normal: these are frames, and an empty one simply means
    every lookup falls back. It is a wrong *entry* that is dangerous, not an
    absent one.
    """
    path = Path(directory or LEXICON_DIR) / f"{name}.tsv"
    if not path.exists():
        return {}
    try:
        return parse(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def all_lexicons(directory: Path | str | None = None) -> dict[str, dict[str, str]]:
    return {name: load(name, directory) for name in NAMES}


def spell_out(word: str, directory: Path | str | None = None) -> str | None:
    """Letter by letter: "ABC" -> "эй би си".

    Returns None if any letter is missing a name, so the caller can leave the
    word alone rather than emit a half-spelled one.
    """
    latin = load("latin_letters", directory)
    cyrillic = load("cyrillic_letters", directory)
    spoken: list[str] = []
    for char in word:
        if char.isdigit():
            continue  # the number pass handles digits
        name = latin.get(char.lower()) or cyrillic.get(char.lower())
        if name is None:
            return None
        spoken.append(name)
    return " ".join(spoken) if spoken else None

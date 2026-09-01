"""Loading and validating the F5-TTS vocabulary.

The vocabulary is the contract between oron-cleaner, training and inference.
It is a file, not a Python constant, because `f5_tts` reads the same file --
duplicating the list in code is how training and inference drift apart.

The thing worth knowing: `f5_tts.model.utils.list_str_to_idx` maps any unknown
character to index 0, and index 0 is SPACE, not <unk>. A coverage gap is
therefore completely silent -- training just sees spaces. Every entry point that
turns text into ids must run `check` first.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DEFAULT_VOCAB = Path(__file__).resolve().parents[2] / "data" / "oron_mn_pinyin" / "vocab.txt"

# What list_str_to_idx returns for anything absent from the vocabulary.
OOV_INDEX = 0


class VocabError(ValueError):
    """Text contains characters the vocabulary cannot represent."""

    def __init__(self, chars: list[str], text: str) -> None:
        rendered = " ".join(f"{c!r} U+{ord(c):04X}" for c in chars)
        super().__init__(
            f"{len(chars)} character(s) absent from the vocabulary would be "
            f"silently replaced by spaces: {rendered} -- in {text[:80]!r}"
        )
        self.chars = chars
        self.text = text


def load_vocab(path: Path | str = DEFAULT_VOCAB) -> list[str]:
    """Read vocab.txt exactly the way f5_tts.model.utils.get_tokenizer does.

    `get_tokenizer` uses `line[:-1]`, so the final line must be newline
    terminated or it loses its last character.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run scripts/extend_vocab.py")
    with open(path, encoding="utf-8") as f:
        vocab = [line[:-1] for line in f]
    if not vocab or vocab[OOV_INDEX] != " ":
        raise VocabError([], f"index {OOV_INDEX} of {path} must be a single space")
    return vocab


@lru_cache(maxsize=4)
def charset(path: Path | str = DEFAULT_VOCAB) -> frozenset[str]:
    """Single characters the vocabulary can represent."""
    return frozenset(t for t in load_vocab(path) if len(t) == 1)


def unsupported(text: str, path: Path | str = DEFAULT_VOCAB) -> list[str]:
    """Distinct characters in `text` that would collapse to a space."""
    known = charset(path)
    return sorted({c for c in text if c not in known})


def check(text: str, path: Path | str = DEFAULT_VOCAB) -> None:
    """Raise if `text` cannot be represented. Call before tokenizing."""
    missing = unsupported(text, path)
    if missing:
        raise VocabError(missing, text)

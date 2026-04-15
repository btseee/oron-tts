"""Cyrillic character tokenizer for Mongolian (Khalkha) + Kazakh TTS."""

from typing import Final

# ── Special tokens ────────────────────────────────────────────────────────────
_PAD = "<PAD>"
_BOS = "<BOS>"
_EOS = "<EOS>"
_UNK = "<UNK>"

# Language tags
_LANG_MN = "[LANG_MN]"
_LANG_KZ = "[LANG_KZ]"

# Speaker attribute tags for explicit style conditioning (Option B)
_FEMALE = "[FEMALE]"
_MALE = "[MALE]"
_YOUNG = "[YOUNG]"
_MIDDLE = "[MIDDLE]"
_ELDERLY = "[ELDERLY]"

SPECIAL_TOKENS: Final[list[str]] = [
    _PAD,
    _BOS,
    _EOS,
    _UNK,
    _LANG_MN,
    _LANG_KZ,
    _FEMALE,
    _MALE,
    _YOUNG,
    _MIDDLE,
    _ELDERLY,
]

# ── Character vocabulary ──────────────────────────────────────────────────────
# Mongolian Cyrillic (Khalkha dialect)
_MN_CHARS: Final[str] = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
# Kazakh-specific Cyrillic additions (ү is shared, already in MN)
_KZ_EXTRA: Final[str] = "әғқңұһі"
# Punctuation and whitespace
_PUNCT: Final[str] = " .,!?-:;\"'()"

_ALL_CHARS: Final[str] = _MN_CHARS + _KZ_EXTRA + _PUNCT
_VOCAB: Final[list[str]] = SPECIAL_TOKENS + list(_ALL_CHARS)


class CyrillicTokenizer:
    """Character-level tokenizer for Mongolian + Kazakh Cyrillic.

    Prepends a language tag token ([LANG_MN] or [LANG_KZ]) before each
    utterance. Optional speaker attribute tokens can also be prepended
    for fully programmatic gender/age control at inference time.

    Vocab size: 11 specials + 35 MN + 7 KZ + 12 punct = 65 tokens.
    """

    def __init__(self) -> None:
        self._token2id: dict[str, int] = {tok: i for i, tok in enumerate(_VOCAB)}
        self._id2token: dict[int, str] = dict(enumerate(_VOCAB))
        self.pad_id: int = self._token2id[_PAD]
        self.bos_id: int = self._token2id[_BOS]
        self.eos_id: int = self._token2id[_EOS]
        self.unk_id: int = self._token2id[_UNK]

    @property
    def vocab_size(self) -> int:
        return len(_VOCAB)

    def encode(
        self,
        text: str,
        lang: str = "mn",
        attr_tokens: list[str] | None = None,
    ) -> list[int]:
        """Encode a single utterance to a token ID sequence.

        Args:
            text: Normalised lowercase Cyrillic text.
            lang: "mn" for Mongolian Khalkha, "kz" for Kazakh.
            attr_tokens: Optional attribute tags, e.g. ["[FEMALE]", "[YOUNG]"].
        """
        ids: list[int] = []

        lang_tag = _LANG_MN if lang == "mn" else _LANG_KZ
        ids.append(self._token2id[lang_tag])

        if attr_tokens:
            for attr in attr_tokens:
                ids.append(self._token2id.get(attr, self.unk_id))

        for ch in text:
            ids.append(self._token2id.get(ch, self.unk_id))

        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self._id2token.get(i, _UNK) for i in ids]
        return "".join(t for t in tokens if t not in SPECIAL_TOKENS)

    def token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        return self._id2token.get(idx, _UNK)

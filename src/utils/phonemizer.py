"""Rule-based Mongolian Cyrillic phonemizer for Khalkha dialect."""

from typing import Final

CYRILLIC_TO_IPA: Final[dict[str, str]] = {
    "а": "a",
    "б": "p",
    "в": "w",
    "г": "ɡ",
    "д": "t",
    "е": "jɛ",
    "ё": "jɔ",
    "ж": "tʃ",
    "з": "ts",
    "и": "i",
    "й": "j",
    "к": "kʰ",
    "л": "ɮ",
    "м": "m",
    "н": "n",
    "о": "ɔ",
    "ө": "o",
    "п": "pʰ",
    "р": "r",
    "с": "s",
    "т": "tʰ",
    "у": "ʊ",
    "ү": "u",
    "ф": "f",
    "х": "x",
    "ц": "tsʰ",
    "ч": "tʃʰ",
    "ш": "ʃ",
    "щ": "ʃtʃ",
    "ъ": "",
    "ы": "i",
    "ь": "ʲ",
    "э": "ɛ",
    "ю": "ju",
    "я": "ja",
}

VOWELS: Final[set[str]] = {"а", "е", "ё", "и", "о", "ө", "у", "ү", "ы", "э", "ю", "я"}
BACK_VOWELS: Final[set[str]] = {"а", "о", "у", "ы"}
FRONT_VOWELS: Final[set[str]] = {"э", "ө", "ү", "и"}

CONSONANT_CLUSTERS: Final[dict[str, str]] = {
    "нг": "ŋ",
    "нх": "ŋx",
    "лг": "ɮɡ",
    "рг": "rɡ",
}

PHONEME_SET: Final[list[str]] = [
    "_",  # padding
    "^",  # start
    "$",  # end
    " ",  # space
    ".",  # punctuation
    ",",
    "?",
    "!",
    "-",
    "a",
    "p",
    "w",
    "ɡ",
    "t",
    "j",
    "ɛ",
    "ɔ",
    "tʃ",
    "ts",
    "i",
    "kʰ",
    "ɮ",
    "m",
    "n",
    "o",
    "pʰ",
    "r",
    "s",
    "tʰ",
    "ʊ",
    "u",
    "f",
    "x",
    "tsʰ",
    "tʃʰ",
    "ʃ",
    "ʃtʃ",
    "ʲ",
    "ŋ",
    "ŋx",
    "ɮɡ",
    "rɡ",
    "ː",  # long vowel marker
]


class MongolianPhonemizer:
    def __init__(self) -> None:
        self._phoneme_to_id: dict[str, int] = {p: i for i, p in enumerate(PHONEME_SET)}
        self._id_to_phoneme: dict[int, str] = {i: p for i, p in enumerate(PHONEME_SET)}

    @property
    def vocab_size(self) -> int:
        return len(PHONEME_SET)

    @property
    def pad_id(self) -> int:
        return self._phoneme_to_id["_"]

    def _apply_consonant_clusters(self, text: str) -> str:
        result = text
        for cluster, replacement in CONSONANT_CLUSTERS.items():
            result = result.replace(cluster, f"[{replacement}]")
        return result

    def _detect_long_vowels(self, text: str) -> str:
        result: list[str] = []
        i = 0
        while i < len(text):
            char = text[i]
            if char in VOWELS and i + 1 < len(text) and text[i + 1] == char:
                result.append(char)
                result.append("ː")
                i += 2
            else:
                result.append(char)
                i += 1
        return "".join(result)

    def _apply_vowel_harmony(self, phonemes: list[str]) -> list[str]:
        return phonemes

    def phonemize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = self._detect_long_vowels(text)
        text = self._apply_consonant_clusters(text)

        phonemes: list[str] = ["^"]
        i = 0
        while i < len(text):
            if text[i] == "[":
                end = text.find("]", i)
                if end != -1:
                    cluster = text[i + 1 : end]
                    phonemes.append(cluster)
                    i = end + 1
                    continue

            char = text[i]
            if char in CYRILLIC_TO_IPA:
                ipa = CYRILLIC_TO_IPA[char]
                if ipa:
                    phonemes.append(ipa)
            elif char == "ː":
                phonemes.append("ː")
            elif char in " .,?!-":
                phonemes.append(char)
            i += 1

        phonemes.append("$")
        return self._apply_vowel_harmony(phonemes)

    def text_to_ids(self, text: str) -> list[int]:
        phonemes = self.phonemize(text)
        return [self._phoneme_to_id.get(p, self.pad_id) for p in phonemes]

    def ids_to_text(self, ids: list[int]) -> str:
        return "".join(self._id_to_phoneme.get(i, "") for i in ids)

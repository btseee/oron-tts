"""Mongolian number-to-text transliteration for Khalkha Cyrillic."""

from typing import Final

ONES: Final[dict[int, str]] = {
    0: "",
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

TEENS: Final[dict[int, str]] = {
    10: "арав",
    11: "арван нэг",
    12: "арван хоёр",
    13: "арван гурав",
    14: "арван дөрөв",
    15: "арван тав",
    16: "арван зургаа",
    17: "арван долоо",
    18: "арван найм",
    19: "арван ес",
}

TENS: Final[dict[int, str]] = {
    2: "хорь",
    3: "гуч",
    4: "дөч",
    5: "тавь",
    6: "жар",
    7: "дал",
    8: "ная",
    9: "ер",
}

TENS_PREFIX: Final[dict[int, str]] = {
    2: "хорин",
    3: "гучин",
    4: "дөчин",
    5: "тавин",
    6: "жаран",
    7: "далан",
    8: "наян",
    9: "ерэн",
}

HUNDREDS: Final[dict[int, str]] = {
    1: "зуу",
    2: "хоёр зуу",
    3: "гурван зуу",
    4: "дөрвөн зуу",
    5: "таван зуу",
    6: "зургаан зуу",
    7: "долоон зуу",
    8: "найман зуу",
    9: "есөн зуу",
}

HUNDREDS_PREFIX: Final[dict[int, str]] = {
    1: "зуун",
    2: "хоёр зуун",
    3: "гурван зуун",
    4: "дөрвөн зуун",
    5: "таван зуун",
    6: "зургаан зуун",
    7: "долоон зуун",
    8: "найман зуун",
    9: "есөн зуун",
}

LARGE_NUMBERS: Final[dict[int, tuple[str, str]]] = {
    1_000: ("мянга", "мянган"),
    1_000_000: ("сая", "саяын"),
    1_000_000_000: ("тэрбум", "тэрбумын"),
    1_000_000_000_000: ("их наяд", "их наядын"),
}

ORDINAL_SUFFIX: Final[dict[str, str]] = {
    "а": "дугаар",
    "о": "дугаар",
    "у": "дугаар",
    "э": "дүгээр",
    "ө": "дүгээр",
    "ү": "дүгээр",
    "и": "дүгээр",
    "ь": "дугаар",
}


class NumberNormalizer:
    def __init__(self) -> None:
        self._cache: dict[int, str] = {}

    def _get_ordinal_suffix(self, word: str) -> str:
        if not word:
            return "дугаар"
        last_vowel = ""
        for char in reversed(word.lower()):
            if char in ORDINAL_SUFFIX:
                last_vowel = char
                break
        return ORDINAL_SUFFIX.get(last_vowel, "дугаар")

    def _convert_under_100(self, n: int) -> str:
        if n == 0:
            return ""
        if n < 10:
            return ONES[n]
        if n < 20:
            return TEENS[n]
        tens, ones = divmod(n, 10)
        if ones == 0:
            return TENS[tens]
        return f"{TENS_PREFIX[tens]} {ONES[ones]}"

    def _convert_under_1000(self, n: int) -> str:
        if n < 100:
            return self._convert_under_100(n)
        hundreds, remainder = divmod(n, 100)
        if remainder == 0:
            return HUNDREDS[hundreds]
        return f"{HUNDREDS_PREFIX[hundreds]} {self._convert_under_100(remainder)}"

    def _convert_large(self, n: int, scale: int) -> tuple[str, int]:
        scale_count, remainder = divmod(n, scale)
        base, prefix = LARGE_NUMBERS[scale]
        if scale_count == 1:
            word = base
        else:
            count_word = self.convert(scale_count)
            word = f"{count_word} {prefix}"
        return word, remainder

    def convert(self, n: int) -> str:
        if n in self._cache:
            return self._cache[n]

        if n == 0:
            return "тэг"
        if n < 0:
            return f"хасах {self.convert(-n)}"

        result = self._convert_number(n)
        self._cache[n] = result
        return result

    def _convert_number(self, n: int) -> str:
        if n < 1000:
            return self._convert_under_1000(n)

        parts: list[str] = []
        remaining = n

        for scale in sorted(LARGE_NUMBERS.keys(), reverse=True):
            if remaining >= scale:
                word, remaining = self._convert_large(remaining, scale)
                parts.append(word)

        if remaining > 0:
            parts.append(self._convert_under_1000(remaining))

        return " ".join(parts)

    def convert_ordinal(self, n: int) -> str:
        cardinal = self.convert(n)
        suffix = self._get_ordinal_suffix(cardinal)
        return f"{cardinal}{suffix}"

    def normalize_text(self, text: str) -> str:
        import re

        def replace_ordinal(match: re.Match[str]) -> str:
            num = int(match.group(1))
            return self.convert_ordinal(num)

        def replace_cardinal(match: re.Match[str]) -> str:
            num = int(match.group(0))
            return self.convert(num)

        text = re.sub(r"(\d+)-р", replace_ordinal, text)
        text = re.sub(r"(\d+)-д(?:угаар|үгээр)", replace_ordinal, text)
        text = re.sub(r"\d+", replace_cardinal, text)

        return text

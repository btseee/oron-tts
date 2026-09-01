"""Number-to-text expansion for Mongolian (Khalkha) Cyrillic.

Each number word has two forms: (standalone, attributive/connecting).
Standalone is used when the word is terminal ("тав" in "тав"). Attributive is
used before nouns or within compound numbers before larger-unit words ("таван"
in "таван мянга", "тавин хувь").

Reference: num2words lang_MN.py (savoirfairelinux/num2words).

This runs in oron-cleaner as well as at inference, so that the text published in
the corpus, the text scored for CER, and the text fed to the model are the same
string. Digits must never reach the tokenizer: the model would have to learn
digit pronunciation from the handful of examples that survive filtering.
"""

from __future__ import annotations

import re
from typing import Final

# Tuples: (standalone, attributive/connecting)

ONES: Final[dict[int, tuple[str, str]]] = {
    0: ("", ""),
    1: ("нэг", "нэг"),
    2: ("хоёр", "хоёр"),
    3: ("гурав", "гурван"),
    4: ("дөрөв", "дөрвөн"),
    5: ("тав", "таван"),
    6: ("зургаа", "зургаан"),
    7: ("долоо", "долоон"),
    8: ("найм", "найман"),
    9: ("ес", "есөн"),
}

TEN: Final[tuple[str, str]] = ("арав", "арван")

TENS: Final[dict[int, tuple[str, str]]] = {
    2: ("хорь", "хорин"),
    3: ("гуч", "гучин"),
    4: ("дөч", "дөчин"),
    5: ("тавь", "тавин"),
    6: ("жар", "жаран"),
    7: ("дал", "далан"),
    8: ("ная", "наян"),
    9: ("ер", "ерэн"),
}

HUNDRED: Final[tuple[str, str]] = ("зуу", "зуун")

LARGE: Final[dict[int, tuple[str, str]]] = {
    1_000: ("мянга", "мянган"),
    1_000_000: ("сая", "сая"),
    1_000_000_000: ("тэрбум", "тэрбум"),
    1_000_000_000_000: ("их наяд", "их наяд"),
}

ORDINAL_SUFFIX: Final[dict[str, str]] = {
    "а": "дугаар",
    "о": "дугаар",
    "у": "дугаар",
    "э": "дүгээр",
    "ө": "дүгээр",
    "ү": "дүгээр",
    "и": "дүгээр",
    "е": "дүгээр",
    "ь": "дугаар",
}

CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "₮": "төгрөг",
    "₸": "теңге",
    "$": "доллар",
    "€": "евро",
    "£": "фунт",
    "¥": "иен",
    "₽": "рубль",
}

CURRENCY_CODES: Final[dict[str, str]] = {
    "MNT": "төгрөг",
    "USD": "доллар",
    "EUR": "евро",
    "GBP": "фунт",
    "JPY": "иен",
    "CNY": "юань",
    "RUB": "рубль",
    "KRW": "вон",
}

MATH_SYMBOLS: Final[dict[str, str]] = {
    "+": "нэмэх",
    "×": "үржүүлэх",
    "÷": "хуваах",
    "=": "тэнцүү",
    "≠": "тэнцүү биш",
    "<": "бага",
    ">": "их",
    "≤": "бага буюу тэнцүү",
    "≥": "их буюу тэнцүү",
    "±": "нэмэх хасах",
    "~": "ойролцоогоор",
}

FRACTION_HALF: Final[str] = "хагас"

# Ablative ("from X") for range expressions. A standalone cardinal can only end
# in one of these words, so an exact table beats a vowel-harmony heuristic --
# stem-final ь is irregular (хорь -> хориос, not *хорьоос) and a general rule
# gets it wrong.
ABLATIVE: Final[dict[str, str]] = {
    "тэг": "тэгээс", "нэг": "нэгээс", "хоёр": "хоёроос", "гурав": "гурваас",
    "дөрөв": "дөрвөөс", "тав": "таваас", "зургаа": "зургаагаас",
    "долоо": "долоогоос", "найм": "наймаас", "ес": "есөөс", "арав": "араваас",
    "хорь": "хориос", "гуч": "гучаас", "дөч": "дөчөөс", "тавь": "тавиас",
    "жар": "жараас", "дал": "далаас", "ная": "наяас", "ер": "ерээс",
    "зуу": "зуугаас", "мянга": "мянгаас", "сая": "саяас",
    "тэрбум": "тэрбумаас", "наяд": "наядаас",
}

# Every Mongolian Cyrillic letter, both cases. Note that a naive "[а-яА-ЯёЁ]"
# range is U+0410-U+044F and therefore EXCLUDES ө U+04E9 and ү U+04AF, two of
# the most common Mongolian vowels -- so it must be spelled out.
MN_LETTERS: Final[str] = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
_MN_CLASS: Final[str] = f"[{re.escape(MN_LETTERS)}]"

_ROMAN_VALUES: Final[list[tuple[str, int]]] = [
    ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
    ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
    ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
]

# Roman numerals are only expanded before one of these nouns.
#
# An unrestricted match rewrites ordinary Latin words: "MIX" parses as M(1000)
# + IX(9) and became "нэг мянга есдүгээр". Since Latin characters are in the
# F5-TTS vocab and are now preserved rather than deleted, leaving an ambiguous
# token alone is safe -- the model reads it as letters.
ROMAN_CONTEXT_WORDS: Final[tuple[str, ...]] = (
    "зуун", "зууны", "зуунд",      # century
    "анги", "ангийн",              # class/grade
    "бүлэг", "бүлгийн",            # chapter
    "хэсэг", "хэсгийн",            # section
    "сар", "сарын",                # month
    "хурал", "хурлын",             # congress/assembly
)
_ROMAN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b"
    r"(?=\s+(?:" + "|".join(ROMAN_CONTEXT_WORDS) + r")\b)"
)


class NumberNormalizer:
    """Expand digits, symbols and numeric idioms into Mongolian words."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, bool], str] = {}
        self._zero_word = "тэг"
        self._minus_word = "хасах"
        self._point_word = "цэг"
        self._percent_word = "хувь"
        self._year_suffix = "оны"
        self._month_suffix = "сарын"
        self._hour_word = "цаг"
        self._minute_word = "минут"
        self._second_word = "секунд"
        self._degree_word = "градус"

    # ── Internal conversion ───────────────────────────────────────────────

    def _get_ordinal_suffix(self, word: str) -> str:
        """Pick дугаар/дүгээр by vowel harmony, from the last harmonic vowel."""
        for char in reversed(word.lower()):
            if char in ORDINAL_SUFFIX:
                return ORDINAL_SUFFIX[char]
        return "дугаар"

    def _convert_under_100(self, n: int, attr: bool = False) -> str:
        idx = 1 if attr else 0
        if n == 0:
            return ""
        if n < 10:
            return ONES[n][idx]
        if n == 10:
            return TEN[idx]
        if n < 20:
            return f"{TEN[1]} {ONES[n - 10][idx]}"
        tens_digit, ones_digit = divmod(n, 10)
        if ones_digit == 0:
            return TENS[tens_digit][idx]
        return f"{TENS[tens_digit][1]} {ONES[ones_digit][idx]}"

    def _convert_under_1000(self, n: int, attr: bool = False) -> str:
        if n < 100:
            return self._convert_under_100(n, attr)
        hundreds_digit, remainder = divmod(n, 100)
        if remainder == 0:
            idx = 1 if attr else 0
            if hundreds_digit == 1:
                return HUNDRED[idx]
            return f"{ONES[hundreds_digit][1]} {HUNDRED[idx]}"
        h_str = (
            HUNDRED[1] if hundreds_digit == 1 else f"{ONES[hundreds_digit][1]} {HUNDRED[1]}"
        )
        return f"{h_str} {self._convert_under_100(remainder, attr)}"

    def _convert_large(self, n: int, scale: int, attr: bool = False) -> tuple[str, int]:
        scale_count, remainder = divmod(n, scale)
        base, attr_form = LARGE[scale]
        is_terminal = remainder == 0
        form = attr_form if (attr and is_terminal) else base
        if scale_count == 1:
            return form, remainder
        return f"{self._convert_number(scale_count, attr=True)} {form}", remainder

    def _convert_number(self, n: int, attr: bool = False) -> str:
        if n < 1000:
            return self._convert_under_1000(n, attr)
        parts: list[str] = []
        remaining = n
        for scale in sorted(LARGE.keys(), reverse=True):
            if remaining >= scale:
                word, remaining = self._convert_large(remaining, scale, attr=attr)
                parts.append(word)
        if remaining > 0:
            parts.append(self._convert_under_1000(remaining, attr))
        return " ".join(parts)

    # ── Public API ────────────────────────────────────────────────────────

    def convert(self, n: int) -> str:
        """Cardinal in standalone form (тав, хорь, зуу)."""
        key = (n, False)
        if key in self._cache:
            return self._cache[key]
        if n == 0:
            return self._zero_word
        if n < 0:
            return f"{self._minus_word} {self.convert(-n)}"
        result = self._convert_number(n, attr=False)
        self._cache[key] = result
        return result

    def convert_attributive(self, n: int) -> str:
        """Cardinal in attributive form, used before nouns.

        таван (мянга), тавин (хувь), зуун (төгрөг).
        """
        key = (n, True)
        if key in self._cache:
            return self._cache[key]
        if n == 0:
            return self._zero_word
        if n < 0:
            return f"{self._minus_word} {self.convert_attributive(-n)}"
        result = self._convert_number(n, attr=True)
        self._cache[key] = result
        return result

    def convert_ordinal(self, n: int) -> str:
        """Ordinal: standalone cardinal with the harmonic suffix attached."""
        cardinal = self.convert(n)
        return f"{cardinal}{self._get_ordinal_suffix(cardinal)}"

    # ── Helpers ───────────────────────────────────────────────────────────

    def _roman_to_int(self, s: str) -> int | None:
        if not s:
            return None
        result = 0
        i = 0
        for prefix, val in _ROMAN_VALUES:
            while s[i:i + len(prefix)] == prefix:
                result += val
                i += len(prefix)
        return result if i == len(s) and result > 0 else None

    def _digit_by_digit(self, s: str) -> str:
        return " ".join(
            self._zero_word if d == "0" else self.convert(int(d)) for d in s
        )

    def _currency_name(self, symbol: str) -> str:
        if symbol in CURRENCY_SYMBOLS:
            return CURRENCY_SYMBOLS[symbol]
        return CURRENCY_CODES.get(symbol.upper(), symbol)

    # ── Text normalization ────────────────────────────────────────────────

    def normalize_text(self, text: str) -> str:  # noqa: C901
        # Thousands separators: 1,234,567 -> 1234567
        text = re.sub(
            r"(\d{1,3})(?:[ ,](\d{3}))+",
            lambda m: m.group(0).replace(",", "").replace(" ", ""),
            text,
        )

        # Identifier-style hyphens (COVID-19, MP-3) are not case suffixes. Split
        # them first, or the number expands and strands the hyphen as a spoken
        # token: "COVID-арван ес".
        text = re.sub(rf"(?<=[A-Za-z{re.escape(MN_LETTERS)}])-(?=\d)", " ", text)

        def _date_ymd(m: re.Match[str]) -> str:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"{self.convert_attributive(y)} {self._year_suffix} "
                f"{self.convert_ordinal(mo)} {self._month_suffix} "
                f"{self.convert(d)}"
            )

        def _date_dmy(m: re.Match[str]) -> str:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"{self.convert_attributive(y)} {self._year_suffix} "
                f"{self.convert_ordinal(mo)} {self._month_suffix} "
                f"{self.convert(d)}"
            )

        text = re.sub(r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})", _date_ymd, text)
        text = re.sub(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", _date_dmy, text)

        def _time(m: re.Match[str]) -> str:
            h, mi, sec = int(m.group(1)), int(m.group(2)), m.group(3)
            parts = [
                f"{self.convert_attributive(h)} {self._hour_word}",
                f"{self.convert_attributive(mi)} {self._minute_word}",
            ]
            if sec is not None:
                parts.append(f"{self.convert_attributive(int(sec))} {self._second_word}")
            return " ".join(parts)

        text = re.sub(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", _time, text)

        def _temp(m: re.Match[str]) -> str:
            sign, num, unit = m.group(1), int(m.group(2)), m.group(3)
            parts: list[str] = []
            if sign == "-":
                parts.append(self._minus_word)
            parts.append(f"{self.convert_attributive(num)} {self._degree_word}")
            if unit and unit.upper() == "C":
                parts.append("цельсий")
            elif unit and unit.upper() == "F":
                parts.append("фаренгейт")
            return " ".join(parts)

        text = re.sub(r"(-?)(\d+)°\s*([CcFf])?", _temp, text)

        _sym_pattern = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)
        _code_pattern = "|".join(CURRENCY_CODES)

        def _currency_after(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(1)))} {self._currency_name(m.group(2))}"

        text = re.sub(
            rf"(\d+)\s*({_sym_pattern}|(?:{_code_pattern})(?!\w))", _currency_after, text
        )

        def _currency_before(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(2)))} {self._currency_name(m.group(1))}"

        text = re.sub(rf"({_sym_pattern})\s*(\d+)", _currency_before, text)

        def _percent(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(1)))} {self._percent_word}"

        text = re.sub(r"(\d+)%", _percent, text)

        def _decimal(m: re.Match[str]) -> str:
            frac = " ".join(self.convert(int(d)) for d in m.group(2))
            return f"{self.convert(int(m.group(1)))} {self._point_word} {frac}"

        text = re.sub(r"(\d+)\.(\d+)", _decimal, text)

        def _fraction(m: re.Match[str]) -> str:
            num, den = int(m.group(1)), int(m.group(2))
            if num == 1 and den == 2:
                return FRACTION_HALF
            # Ordinal-genitive of the denominator, harmony already chosen by
            # convert_ordinal (дугаар back-vowel / дүгээр front-vowel).
            ordinal = self.convert_ordinal(den)
            gen = ordinal + ("ийн" if ordinal.endswith("дүгээр") else "ын")
            return f"{gen} {self.convert(num)}"

        text = re.sub(r"(\d{1,2})/(\d{1,2})", _fraction, text)

        def _phone(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0)[1:])
            return f"{MATH_SYMBOLS['+']} " + self._digit_by_digit(digits)

        text = re.sub(r"\+\d[\d\s\-]{6,15}\d", _phone, text)

        def _range(m: re.Match[str]) -> str:
            lo, hi = self.convert(int(m.group(1))), self.convert(int(m.group(2)))
            head, _, last = lo.rpartition(" ")
            abl = ABLATIVE.get(last)
            lo_abl = f"{head} {abl}".strip() if abl else lo
            return f"{lo_abl} {hi} хүртэл"

        text = re.sub(r"(\d+)\s*[-–—]\s*(\d+)", _range, text)

        def _ordinal(m: re.Match[str]) -> str:
            return self.convert_ordinal(int(m.group(1)))

        text = re.sub(r"(\d+)-р\b", _ordinal, text)
        text = re.sub(r"(\d+)-д(?:угаар|үгээр|ахь)", _ordinal, text)

        # Genitive/attributive markers before nouns: 2024-ны, 1-ний, 5-ын, 3-ийн
        def _genitive(m: re.Match[str]) -> str:
            return self.convert_attributive(int(m.group(1)))

        text = re.sub(r"(\d+)-(?:ны|ний|ын|ийн)\b", _genitive, text)

        # Any remaining case suffix attaches directly to the standalone cardinal:
        # 15-нд -> арван тавнд, 20-аас -> хорьаас, 5-ыг -> тавыг. The written
        # suffix already carries the author's vowel harmony, so it is appended
        # verbatim. Without this the hyphen survives into the text as a spoken
        # token ("арван тав-нд").
        def _suffixed(m: re.Match[str]) -> str:
            word, suffix = self.convert(int(m.group(1))), m.group(2)
            head, _, last = word.rpartition(" ")
            # The ablative is irregular after stem-final ь (хорь -> хориос), so
            # prefer the table over blind concatenation.
            if suffix in ("аас", "ээс", "оос", "өөс") and last in ABLATIVE:
                return f"{head} {ABLATIVE[last]}".strip()
            return f"{word}{suffix}"

        text = re.sub(rf"(\d+)-({_MN_CLASS}+)", _suffixed, text)

        def _roman(m: re.Match[str]) -> str:
            val = self._roman_to_int(m.group(1))
            return m.group(0) if val is None else self.convert_ordinal(val)

        text = _ROMAN_RE.sub(_roman, text)

        for sym, word in MATH_SYMBOLS.items():
            if sym in text:
                text = text.replace(sym, f" {word} ")

        # A number directly before a Mongolian word takes the attributive form.
        text = re.sub(
            rf"(\d+)(?=\s+{_MN_CLASS})",
            lambda m: self.convert_attributive(int(m.group(1))),
            text,
        )

        text = re.sub(r"\d+", lambda m: self.convert(int(m.group(0))), text)
        return text

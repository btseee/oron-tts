"""Number-to-text transliteration for Mongolian (Khalkha) and Kazakh Cyrillic.

Each number word has two forms: (standalone, attributive/connecting).
Standalone is used when the word is terminal (e.g. "тав" in "тав").
Attributive is used before nouns or within compound numbers before
larger-unit words (e.g. "таван" in "таван мянга", "тавин хувь").

Reference: num2words lang_MN.py (savoirfairelinux/num2words).
"""

from __future__ import annotations

import re
from typing import Final

# ── Mongolian (Khalkha) ──────────────────────────────────────────────────────
# Tuples: (standalone, attributive/connecting)

MN_ONES: Final[dict[int, tuple[str, str]]] = {
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

MN_TEN: Final[tuple[str, str]] = ("арав", "арван")

MN_TENS: Final[dict[int, tuple[str, str]]] = {
    2: ("хорь", "хорин"),
    3: ("гуч", "гучин"),
    4: ("дөч", "дөчин"),
    5: ("тавь", "тавин"),
    6: ("жар", "жаран"),
    7: ("дал", "далан"),
    8: ("ная", "наян"),
    9: ("ер", "ерэн"),
}

MN_HUNDRED: Final[tuple[str, str]] = ("зуу", "зуун")

# (base form within compounds, attributive when terminal before noun)
MN_LARGE: Final[dict[int, tuple[str, str]]] = {
    1_000: ("мянга", "мянган"),
    1_000_000: ("сая", "сая"),
    1_000_000_000: ("тэрбум", "тэрбум"),
    1_000_000_000_000: ("их наяд", "их наяд"),
}

MN_ORDINAL_SUFFIX: Final[dict[str, str]] = {
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

# ── Kazakh ────────────────────────────────────────────────────────────────────
# Kazakh numbers do not change form before nouns — both tuple slots are equal.

KZ_ONES: Final[dict[int, tuple[str, str]]] = {
    0: ("", ""),
    1: ("бір", "бір"),
    2: ("екі", "екі"),
    3: ("үш", "үш"),
    4: ("төрт", "төрт"),
    5: ("бес", "бес"),
    6: ("алты", "алты"),
    7: ("жеті", "жеті"),
    8: ("сегіз", "сегіз"),
    9: ("тоғыз", "тоғыз"),
}

KZ_TEN: Final[tuple[str, str]] = ("он", "он")

KZ_TENS: Final[dict[int, tuple[str, str]]] = {
    2: ("жиырма", "жиырма"),
    3: ("отыз", "отыз"),
    4: ("қырық", "қырық"),
    5: ("елу", "елу"),
    6: ("алпыс", "алпыс"),
    7: ("жетпіс", "жетпіс"),
    8: ("сексен", "сексен"),
    9: ("тоқсан", "тоқсан"),
}

KZ_HUNDRED: Final[tuple[str, str]] = ("жүз", "жүз")

KZ_LARGE: Final[dict[int, tuple[str, str]]] = {
    1_000: ("мың", "мың"),
    1_000_000: ("миллион", "миллион"),
    1_000_000_000: ("миллиард", "миллиард"),
}

KZ_ORDINAL_SUFFIX: Final[dict[str, str]] = {
    "а": "нші",
    "е": "нші",
    "ы": "нші",
    "і": "нші",
    "о": "нші",
    "ө": "нші",
    "ұ": "нші",
    "ү": "нші",
}

# ── Currency symbols (both languages) ────────────────────────────────────────
# Maps symbol → (MN word, KZ word)
CURRENCY_SYMBOLS: Final[dict[str, tuple[str, str]]] = {
    "₮": ("төгрөг", "төгрөг"),
    "₸": ("теңге", "теңге"),
    "$": ("доллар", "доллар"),
    "€": ("евро", "евро"),
    "£": ("фунт", "фунт"),
    "¥": ("иен", "иен"),
    "₽": ("рубль", "рубль"),
}

# ISO 4217 codes → (MN word, KZ word)
CURRENCY_CODES: Final[dict[str, tuple[str, str]]] = {
    "MNT": ("төгрөг", "төгрөг"),
    "KZT": ("теңге", "теңге"),
    "USD": ("доллар", "доллар"),
    "EUR": ("евро", "евро"),
    "GBP": ("фунт", "фунт"),
    "JPY": ("иен", "иен"),
    "CNY": ("юань", "юань"),
    "RUB": ("рубль", "рубль"),
    "KRW": ("вон", "вон"),
}

# ── Math/special symbols ─────────────────────────────────────────────────────
MATH_SYMBOLS: Final[dict[str, tuple[str, str]]] = {
    "+": ("нэмэх", "қосу"),
    "×": ("үржүүлэх", "көбейту"),
    "÷": ("хуваах", "бөлу"),
    "=": ("тэнцүү", "тең"),
    "≠": ("тэнцүү биш", "тең емес"),
    "<": ("бага", "кіші"),
    ">": ("их", "үлкен"),
    "≤": ("бага буюу тэнцүү", "кіші немесе тең"),
    "≥": ("их буюу тэнцүү", "үлкен немесе тең"),
    "±": ("нэмэх хасах", "плюс минус"),
    "~": ("ойролцоогоор", "шамамен"),
}

# ── Roman numerals ────────────────────────────────────────────────────────────
_ROMAN_VALUES: Final[list[tuple[str, int]]] = [
    ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
    ("C", 100), ("XC", 90), ("L", 50), ("XL", 40),
    ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
]
_ROMAN_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b",
)

# ── Fraction words ────────────────────────────────────────────────────────────
# MN: "3/4" → "дөрөвний гурав" (denominator-genitive + numerator)
# KZ: "3/4" → "төрттен үш" (denominator + -тен/тан + numerator)
MN_FRACTION_HALF: Final[str] = "хагас"
KZ_FRACTION_HALF: Final[str] = "жарты"


class NumberNormalizer:
    def __init__(self, lang: str = "mn") -> None:
        self._lang = lang
        self._cache: dict[tuple[str, int, bool], str] = {}
        self._setup_tables(lang)

    def _setup_tables(self, lang: str) -> None:
        if lang == "kz":
            self._ones = KZ_ONES
            self._ten = KZ_TEN
            self._tens = KZ_TENS
            self._hundred = KZ_HUNDRED
            self._large = KZ_LARGE
            self._ordinal_suffix_map = KZ_ORDINAL_SUFFIX
            self._zero_word = "нөл"
            self._minus_word = "минус"
            self._point_word = "бүтін"
            self._percent_word = "пайыз"
            self._currency_word = "теңге"
            self._year_suffix = "жылдың"
            self._month_suffix = "айдың"
            self._hour_word = "сағат"
            self._minute_word = "минут"
            self._second_word = "секунд"
            self._degree_word = "градус"
            self._lang_idx = 1
        else:
            self._ones = MN_ONES
            self._ten = MN_TEN
            self._tens = MN_TENS
            self._hundred = MN_HUNDRED
            self._large = MN_LARGE
            self._ordinal_suffix_map = MN_ORDINAL_SUFFIX
            self._zero_word = "тэг"
            self._minus_word = "хасах"
            self._point_word = "цэг"
            self._percent_word = "хувь"
            self._currency_word = "төгрөг"
            self._year_suffix = "оны"
            self._month_suffix = "сарын"
            self._hour_word = "цаг"
            self._minute_word = "минут"
            self._second_word = "секунд"
            self._degree_word = "градус"
            self._lang_idx = 0

    @property
    def lang(self) -> str:
        return self._lang

    @lang.setter
    def lang(self, value: str) -> None:
        if value != self._lang:
            self._lang = value
            self._setup_tables(value)

    # ── Internal conversion ───────────────────────────────────────────────

    def _get_ordinal_suffix(self, word: str) -> str:
        default = "дугаар" if self._lang == "mn" else "нші"
        if not word:
            return default
        for char in reversed(word.lower()):
            if char in self._ordinal_suffix_map:
                return self._ordinal_suffix_map[char]
        return default

    def _convert_under_100(self, n: int, attr: bool = False) -> str:
        idx = 1 if attr else 0
        if n == 0:
            return ""
        if n < 10:
            return self._ones[n][idx]
        if n == 10:
            return self._ten[idx]
        if n < 20:
            ones_word = self._ones[n - 10][idx]
            return f"{self._ten[1]} {ones_word}"
        tens_digit, ones_digit = divmod(n, 10)
        if ones_digit == 0:
            return self._tens[tens_digit][idx]
        ones_word = self._ones[ones_digit][idx]
        return f"{self._tens[tens_digit][1]} {ones_word}"

    def _convert_under_1000(self, n: int, attr: bool = False) -> str:
        if n < 100:
            return self._convert_under_100(n, attr)
        hundreds_digit, remainder = divmod(n, 100)
        if remainder == 0:
            idx = 1 if attr else 0
            if hundreds_digit == 1:
                return self._hundred[idx]
            return f"{self._ones[hundreds_digit][1]} {self._hundred[idx]}"
        if hundreds_digit == 1:
            h_str = self._hundred[1]
        else:
            h_str = f"{self._ones[hundreds_digit][1]} {self._hundred[1]}"
        return f"{h_str} {self._convert_under_100(remainder, attr)}"

    def _convert_large(
        self, n: int, scale: int, attr: bool = False,
    ) -> tuple[str, int]:
        scale_count, remainder = divmod(n, scale)
        base, attr_form = self._large[scale]
        is_terminal = remainder == 0
        form = attr_form if (attr and is_terminal) else base
        if scale_count == 1:
            word = form
        else:
            count_word = self._convert_number(scale_count, attr=True)
            word = f"{count_word} {form}"
        return word, remainder

    def _convert_number(self, n: int, attr: bool = False) -> str:
        if n < 1000:
            return self._convert_under_1000(n, attr)

        parts: list[str] = []
        remaining = n

        for scale in sorted(self._large.keys(), reverse=True):
            if remaining >= scale:
                word, remaining = self._convert_large(
                    remaining, scale, attr=attr,
                )
                parts.append(word)

        if remaining > 0:
            parts.append(self._convert_under_1000(remaining, attr))

        return " ".join(parts)

    # ── Public API ────────────────────────────────────────────────────────

    def convert(self, n: int) -> str:
        """Cardinal number in standalone form (e.g. тав, хорь, зуу)."""
        cache_key = (self._lang, n, False)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if n == 0:
            return self._zero_word
        if n < 0:
            return f"{self._minus_word} {self.convert(-n)}"
        result = self._convert_number(n, attr=False)
        self._cache[cache_key] = result
        return result

    def convert_attributive(self, n: int) -> str:
        """Cardinal in attributive form (before nouns).

        E.g. таван (мянга), тавин (хувь), зуун (төгрөг).
        """
        cache_key = (self._lang, n, True)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if n == 0:
            return self._zero_word
        if n < 0:
            return f"{self._minus_word} {self.convert_attributive(-n)}"
        result = self._convert_number(n, attr=True)
        self._cache[cache_key] = result
        return result

    def convert_ordinal(self, n: int) -> str:
        """Ordinal: standalone cardinal + suffix (attached)."""
        cardinal = self.convert(n)
        suffix = self._get_ordinal_suffix(cardinal)
        return f"{cardinal}{suffix}"

    # ── Helpers ───────────────────────────────────────────────────────────

    def _roman_to_int(self, s: str) -> int | None:
        """Convert Roman numeral string to int, or None if invalid."""
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
        """Read a digit string one digit at a time."""
        return " ".join(
            self._zero_word if d == "0" else self.convert(int(d))
            for d in s
        )

    def _currency_name(self, symbol: str) -> str:
        """Get currency word for a symbol or ISO code."""
        if symbol in CURRENCY_SYMBOLS:
            return CURRENCY_SYMBOLS[symbol][self._lang_idx]
        upper = symbol.upper()
        if upper in CURRENCY_CODES:
            return CURRENCY_CODES[upper][self._lang_idx]
        return symbol

    # ── Text normalization ────────────────────────────────────────────────

    def normalize_text(self, text: str) -> str:  # noqa: C901
        # --- Strip comma/space thousands separators: 1,234,567 → 1234567 ---
        text = re.sub(
            r"(\d{1,3})(?:[ ,](\d{3}))+",
            lambda m: m.group(0).replace(",", "").replace(" ", ""),
            text,
        )
        # --- Dates ---
        # YYYY/MM/DD or YYYY.MM.DD or YYYY-MM-DD
        def _date_ymd(m: re.Match[str]) -> str:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"{self.convert_attributive(y)} {self._year_suffix} "
                f"{self.convert_ordinal(mo)} {self._month_suffix} "
                f"{self.convert(d)}"
            )

        # DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
        def _date_dmy(m: re.Match[str]) -> str:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"{self.convert_attributive(y)} {self._year_suffix} "
                f"{self.convert_ordinal(mo)} {self._month_suffix} "
                f"{self.convert(d)}"
            )

        text = re.sub(r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})", _date_ymd, text)
        text = re.sub(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", _date_dmy, text)

        # --- Time: HH:MM or HH:MM:SS ---
        def _time(m: re.Match[str]) -> str:
            h, mi = int(m.group(1)), int(m.group(2))
            sec = m.group(3)
            parts = [
                f"{self.convert_attributive(h)} {self._hour_word}",
                f"{self.convert_attributive(mi)} {self._minute_word}",
            ]
            if sec is not None:
                parts.append(
                    f"{self.convert_attributive(int(sec))} {self._second_word}"
                )
            return " ".join(parts)

        text = re.sub(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", _time, text)

        # --- Temperature: 25°C, -15°, 25° ---
        def _temp(m: re.Match[str]) -> str:
            sign = m.group(1)
            num = int(m.group(2))
            unit = m.group(3)
            parts: list[str] = []
            if sign == "-":
                parts.append(self._minus_word)
            parts.append(f"{self.convert_attributive(num)} {self._degree_word}")
            if unit and unit.upper() == "C":
                parts.append("цельсий" if self._lang == "mn" else "цельсий")
            elif unit and unit.upper() == "F":
                parts.append("фаренгейт")
            return " ".join(parts)

        text = re.sub(r"(-?)(\d+)°\s*([CcFf])?", _temp, text)

        # --- Currency: symbol after number (100₮, 100$, 100 EUR) ---
        _sym_pattern = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)
        _code_pattern = "|".join(CURRENCY_CODES)

        def _currency_after(m: re.Match[str]) -> str:
            num = int(m.group(1))
            sym = m.group(2)
            return f"{self.convert_attributive(num)} {self._currency_name(sym)}"

        text = re.sub(
            rf"(\d+)\s*({_sym_pattern}|{_code_pattern})",
            _currency_after,
            text,
        )

        # --- Currency: symbol before number ($100, €50) ---
        def _currency_before(m: re.Match[str]) -> str:
            sym = m.group(1)
            num = int(m.group(2))
            return f"{self.convert_attributive(num)} {self._currency_name(sym)}"

        text = re.sub(
            rf"({_sym_pattern})\s*(\d+)",
            _currency_before,
            text,
        )

        # --- Percent ---
        def _percent(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(1)))} {self._percent_word}"

        text = re.sub(r"(\d+)%", _percent, text)

        # --- Decimal numbers ---
        def _decimal(m: re.Match[str]) -> str:
            integer_part = int(m.group(1))
            frac_digits = " ".join(
                self.convert(int(d)) for d in m.group(2)
            )
            return f"{self.convert(integer_part)} {self._point_word} {frac_digits}"

        text = re.sub(r"(\d+)\.(\d+)", _decimal, text)

        # --- Fractions: 1/2, 3/4 (only small numerator/denominator) ---
        def _fraction(m: re.Match[str]) -> str:
            num, den = int(m.group(1)), int(m.group(2))
            if num == 1 and den == 2:
                return MN_FRACTION_HALF if self._lang == "mn" else KZ_FRACTION_HALF
            if self._lang == "mn":
                return f"{self.convert(den)} дугаарын {self.convert(num)}"
            return f"{self.convert(den)} ден {self.convert(num)}"

        text = re.sub(r"(\d{1,2})/(\d{1,2})", _fraction, text)

        # --- Phone numbers: +XXXXXXXXXXX or +XXX XXXX XXXX ---
        def _phone(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0)[1:])  # strip + and spaces
            return "нэмэх " + self._digit_by_digit(digits)

        text = re.sub(r"\+\d[\d\s\-]{6,15}\d", _phone, text)

        # --- Ranges: 10-20 (digit-dash-digit, NOT ordinals) ---
        def _range(m: re.Match[str]) -> str:
            a, b = int(m.group(1)), int(m.group(2))
            sep = "аас" if self._lang == "mn" else "ден"
            to = "хүртэл" if self._lang == "mn" else "дейін"
            return f"{self.convert(a)} {sep} {self.convert(b)} {to}"

        text = re.sub(r"(\d+)\s*[-–—]\s*(\d+)", _range, text)

        # --- Ordinals: 20-р, 3-дугаар, 5-ші ---
        def _ordinal(m: re.Match[str]) -> str:
            return self.convert_ordinal(int(m.group(1)))

        text = re.sub(r"(\d+)-р\b", _ordinal, text)
        text = re.sub(r"(\d+)-д(?:угаар|үгээр|ахь)", _ordinal, text)
        text = re.sub(r"(\d+)-(?:ші|шы)", _ordinal, text)

        # --- Roman numerals → ordinal (XV зуун = арван тавдугаар зуун) ---
        def _roman(m: re.Match[str]) -> str:
            val = self._roman_to_int(m.group(1))
            if val is None:
                return m.group(0)
            return self.convert_ordinal(val)

        text = _ROMAN_RE.sub(_roman, text)

        # --- Math/special symbols ---
        for sym, words in MATH_SYMBOLS.items():
            if sym in text:
                text = text.replace(sym, f" {words[self._lang_idx]} ")

        # --- Number followed by a Cyrillic word → attributive ---
        def _cardinal_before_noun(m: re.Match[str]) -> str:
            return self.convert_attributive(int(m.group(1)))

        text = re.sub(
            r"(\d+)(?=\s+[а-яёәғқңұһі])",
            _cardinal_before_noun,
            text,
        )

        # --- Bare cardinal numbers ---
        def _cardinal(m: re.Match[str]) -> str:
            return self.convert(int(m.group(0)))

        text = re.sub(r"\d+", _cardinal, text)

        return text

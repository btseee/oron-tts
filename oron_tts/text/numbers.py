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
    # No ₸. That currency is out of scope for this corpus, and its Mongolian
    # spelling "теңге" contains ң U+04A3, which vocab.txt does not have -- so
    # expanding it would put a space in the training text with nothing logged.
    # Left unexpanded, the symbol reaches the vocabulary check and is rejected
    # loudly instead. Every entry here is guarded by
    # test_every_emittable_constant_is_representable.
    "$": "америк доллар",
    "€": "евро",
    "£": "фунт",
    "¥": "иен",
    "₽": "рубль",
}

CURRENCY_CODES: Final[dict[str, str]] = {
    "MNT": "төгрөг",
    "USD": "америк доллар",
    "EUR": "евро",
    "GBP": "фунт",
    "JPY": "иен",
    "CNY": "юань",
    "RUB": "рубль",
    "KRW": "вон",
}

# Spoken forms from the specification, section 36: "2+2" is "хоёр дээр хоёр"
# and "5x8" is "тав үржих нь найм". Note the operands are standalone rather than
# attributive -- "тав", not "таван".
MATH_SYMBOLS: Final[dict[str, str]] = {
    "+": "дээр",
    "×": "үржих нь",
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

# Genitive of a cardinal, for reading "3/4" as "of four, three". From the
# specification, verbatim -- not generated. Mongolian numeral genitives are not
# derivable from the citation/attributive pairs in this file (see
# docs/normaliser-review.md), and the spec supplies exactly these.
#
# There are two of them, and that is deliberate rather than a duplication. The
# spec gives the n-stem genitive in fractions (3/4 -> "дөрөвний гурав") and the
# reduced one in section references (3.4.1.7 -> "гурвын дөрвийн ..."), which
# was confirmed as context-dependent rather than a typo. Entries not listed are
# absent, not guessed.
FRACTION_GENITIVE: Final[dict[int, str]] = {
    1: "нэгний",
    2: "хоёрны",
    4: "дөрөвний",
}

# Reduced genitive, for dotted references: 5.1.2 -> "тавын нэгийн хоёр".
REFERENCE_GENITIVE: Final[dict[int, str]] = {
    1: "нэгийн",
    3: "гурвын",
    4: "дөрвийн",
    5: "тавын",
    10: "аравын",
}

# A decimal is read as a fraction, not as a whole part plus a tail: "12.5%" is
# "арван хоёр арваны таван хувь" -- twelve, then five-tenths, with the place
# named first exactly as in "хоёрны нэг". There is no "бүхэл".
#
# An earlier version followed the specification's section 4, which does use
# "бүхэл", and then needed a rule for when the place word appears that no
# reading of the spec could make consistent. Confirmed as superseded: the
# fraction reading applies everywhere.
DECIMAL_PLACE: Final[dict[int, str]] = {
    1: "арваны",
    2: "зууны",
}

# The mixed fraction keeps its own linking word, which is not "бүхэл" either:
# "2 1/2" is "хоёр бүтэн хоёрны нэг".
MIXED_WHOLE_WORD: Final[str] = "бүтэн"

FRACTION_HALF: Final[str] = "хагас"


class NumeralSuffixError(ValueError):
    """A numeral suffix this module cannot expand without guessing.

    Raised rather than approximated. The corpus text, the CER reference and the
    training target are one string here, so a wrong expansion is not a cosmetic
    defect -- it is published, scored against itself, and learned. Under
    `MongolianNormalizer.normalize(strict=True)` this drops the clip instead.
    """

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

# Written case suffix -> the full word it makes, per numeral stem.
#
#     "20-аас"  ->  SUFFIXED_FORMS["хорь"]["аас"]  ->  "хориос"
#
# Tabulated, not generated. See `NumberNormalizer.attach_suffix` for why: every
# rule tried against this produced a non-word somewhere, and a wrong expansion
# here is published in the corpus, used as the CER reference, and trained on.
#
# Seeded from ABLATIVE, which is verified data. Everything else is a hole, and
# an unlisted combination raises rather than being approximated -- so the corpus
# loses a clip instead of gaining a non-word.
#
# **To extend this you need a native Khalkha speaker.** The table to fill is in
# docs/normaliser-review.md, one row per (numeral, written suffix). The four
# suffix families that matter by frequency are the genitive (-ны/-ний/-ын/-ийн),
# the dative-locative (-д/-т/-нд), the accusative (-ыг/-ийг) and the ablative
# (already covered here).
SUFFIXED_FORMS: Final[dict[str, dict[str, str]]] = {
    stem: {ablative[len(stem):] if ablative.startswith(stem) else "": ablative}
    for stem, ablative in ABLATIVE.items()
}
# The ablative is written several ways for the same word -- "20-аас" and
# "20-иос" are both read as "хориос" -- so every spelling that reaches the
# regex maps to the one verified form.
for _stem, _ablative in ABLATIVE.items():
    _forms = SUFFIXED_FORMS[_stem]
    for _written in ("аас", "ээс", "оос", "өөс", "иас", "иос", "ийс", "гаас", "гоос"):
        _forms.setdefault(_written, _ablative)
    _forms.pop("", None)
del _stem, _ablative, _forms, _written

# Forms the normalization specification supplies outright, beyond the ablative.
# Sections 8 and 9: "5-аар" is "таваар" (distributive, "five each") and "5-уул"
# is "тавуул" (collective, "the five of them"). Both are derived forms rather
# than cases, which is why they are here and not in a case table.
SUFFIXED_FORMS["тав"].update({"аар": "таваар", "уул": "тавуул"})

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

# Nouns that take the number as an *ordinal in front of them*, however the
# source writes it: "Байр 12" is "арван хоёрдугаар байр", "Бүлэг 12" is "арван
# хоёрдугаар бүлэг". Reading the digits in place gave "байр арван хоёр", which
# is the wrong construction and the wrong word order.
ORDINAL_NOUNS: Final[tuple[str, ...]] = (
    "байр", "бүлэг", "анги", "хороо", "хэсэг", "дүүрэг", "баг", "тойрог",
    "сургууль", "хороолол",
)

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

    def __init__(self, reference_words: set[str] | None = None) -> None:
        self._cache: dict[tuple[int, bool], str] = {}
        # Words that make a following "N:M" a verse reference rather than a
        # clock time. Supplied by the normaliser from data/lexicon.
        self._reference_words = reference_words or set()
        self._zero_word = "тэг"
        self._minus_word = "хасах"
        self._point_word = "цэг"
        self._percent_word = "хувь"
        self._year_suffix = "оны"
        self._month_suffix = "сарын"
        self._hour_word = "цаг"
        self._minute_word = "минут"
        self._second_word = "секунд"
        # Spec section 22: "-20 C" is "хасах хорин хэм цельс".
        self._degree_word = "хэм"

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
            # One of a scale word is never counted aloud: 1990 is "мянга есөн
            # зуун ер", not "нэг мянга ...". Two of them is, which is why this
            # is a special case for 1 rather than a rule about scale words --
            # 2990 keeps its "хоёр".
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

    def _phone_words(self, digits: str) -> str:
        """A phone number, read the way one is dictated.

        Eight digits is the Mongolian mobile length and is read as four pairs --
        "99112233" is "ерэн ес арван нэг хорин хоёр гучин гурав". Any other
        length is read digit by digit, which is what the specification gives for
        an unstructured run and what an international number needs.
        """
        if len(digits) == 8:
            return " ".join(self.convert(int(digits[i:i + 2])) for i in range(0, 8, 2))
        return self._digit_by_digit(digits)

    def _decimal_words(self, whole: str, frac: str, attr: bool = False) -> str:
        """"12.5%" -> "арван хоёр арваны таван хувь".

        A decimal is a fraction: the whole part, then the place named as a
        genitive, then the digits -- the same order as "хоёрны нэг". There is
        no linking word.

        `attr` puts the final digits in the attributive form, which is what a
        following noun needs: "таван хувь", not "тав хувь".

        A fractional part longer than the tabulated places is read digit by
        digit rather than given an invented place name.
        """
        place = DECIMAL_PLACE.get(len(frac))
        head = self.convert(int(whole))
        if place is None:
            digits = " ".join(self.convert(int(d)) for d in frac)
            return f"{head} {digits}"
        tail = self.convert_attributive(int(frac)) if attr else self.convert(int(frac))
        return f"{head} {place} {tail}"

    def _fraction_words(self, num: int, den: int) -> str:
        """"3/4" -> "дөрөвний гурав": genitive of the denominator, then the
        numerator.

        The genitive comes from `FRACTION_GENITIVE`, which is tabulated from the
        specification rather than generated. An earlier version built an
        *ordinal* and bolted a genitive onto it -- "дөрөвдүгээрийн гурав",
        roughly "of the fourth, three" -- which says the wrong thing however the
        genitive is chosen, because an ordinal names a position and not a part.
        """
        genitive = FRACTION_GENITIVE.get(den)
        if genitive is None:
            raise NumeralSuffixError(
                f"No verified genitive for {den}, so the fraction {num}/{den} "
                f"cannot be expanded. Add it to FRACTION_GENITIVE -- see "
                f"docs/normaliser-review.md."
            )
        return f"{genitive} {self.convert(num)}"

    def _reference_genitive(self, n: int) -> str:
        """The genitive used in a dotted reference: "5.1.2" -> "тавын нэгийн ...".

        A different table from `FRACTION_GENITIVE` on purpose. The spec gives
        the n-stem form in fractions (`дөрөвний`) and the reduced form here
        (`дөрвийн`), and that was confirmed as context-dependent rather than a
        typo.
        """
        genitive = REFERENCE_GENITIVE.get(n)
        if genitive is None:
            raise NumeralSuffixError(
                f"No verified genitive for {n} in a dotted reference. Add it to "
                f"REFERENCE_GENITIVE -- see docs/normaliser-review.md."
            )
        return genitive

    def attach_suffix(self, n: int, suffix: str) -> str:
        """Attach a written case suffix to a numeral: "20-аас" -> "хориос".

        Only from `SUFFIXED_FORMS`. Nothing here is generated, because every
        generation rule tried against this problem produced a non-word
        somewhere:

            concatenate onto the citation form   20-иос -> *хорьиос
            drop stem-final ь, then concatenate  20-аас -> *хораас
            use the attributive as the stem      3-ийн  -> *гуравийн

        The ABLATIVE table is why the first of those was only *mostly* wrong:
        it is a hand-written table of verified forms, and it is right wherever
        it applies. That is the shape the answer has to take. Mongolian numeral
        morphology here is not derivable from the citation/attributive pairs in
        this file -- the stems interact with the written suffix
        orthographically (хори + ийг contracts to хорийг), the tens behave
        differently from the ones (гуч -> гучаас, not *гучиас), and stems with
        an unstable vowel reduce (гурав -> гурваас). A rule that gets four of
        those right and the fifth wrong is worse than no rule, because the
        wrong one is silent.

        And silent is expensive here specifically: this string is published in
        the corpus, is the reference CER is scored against, and is the training
        target. A non-word is learned as a word, and the CER gate cannot catch
        it because the reference *is* the corrupted string.

        So an unlisted combination raises, and
        `MongolianNormalizer.normalize(strict=True)` -- the corpus path -- turns
        that into a dropped clip. A clip fewer is cheap. `docs/normaliser-review.md`
        is the table a native speaker fills in to lift this, and
        `SUFFIXED_FORMS` is where the answers go.
        """
        word = self.convert(n)
        if not suffix:
            return word
        head, _, last = word.rpartition(" ")

        known = SUFFIXED_FORMS.get(last)
        if known and suffix in known:
            return f"{head} {known[suffix]}".strip()

        raise NumeralSuffixError(
            f"No verified form for {last!r} + -{suffix} (from {n}-{suffix}). "
            f"Mongolian numeral suffixation is tabulated, not generated -- see "
            f"docs/normaliser-review.md and SUFFIXED_FORMS."
        )

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

        # "Иохан 3:16" is a chapter-and-verse reference, "Цаг 14:30" is a time,
        # and nothing in the shape of either string separates them -- both are a
        # capitalised word, a space, and two colon-separated numbers that are
        # valid as an hour and a minute. It takes knowing that Иохан is a book
        # and Цаг is not, so it takes a list. Seeded from the spec; extend it in
        # data/lexicon/reference_words.tsv.
        def _verse(m: re.Match[str]) -> str:
            return (f"{m.group(1)} {self.convert(int(m.group(2)))} "
                    f"{self.convert(int(m.group(3)))}")

        if self._reference_words:
            names = "|".join(re.escape(w) for w in sorted(self._reference_words))
            text = re.sub(rf"\b({names})\s+(\d{{1,3}}):(\d{{1,3}})\b", _verse, text)

        # A colon between numbers is a clock time only sometimes. Unguarded,
        # this rewrote "Иохан 3:16" as "Иохан гурван цаг арван зургаан минут" --
        # the same class of collision as an abbreviation shadowing a real word,
        # and just as silent.
        #
        # Two guards, both from the shape of the text rather than a word list:
        #
        #   * the hour and minute must be possible ones, so 3:16 stays a
        #     candidate but 25:70 does not;
        #   * a capitalised word immediately before it means a reference -- a
        #     book, chapter or verse -- not a time of day.
        #
        # What is left ambiguous is a bare "3:1": the spec reads it as a sports
        # score and "1:2" as a ratio, and nothing in the string separates them.
        # Those are handled below, and flagged in docs/normaliser-review.md.
        def _guarded_time(m: re.Match[str]) -> str:
            h, mi = int(m.group(1)), int(m.group(2))
            if h > 23 or mi > 59:
                return m.group(0)
            return _time(m)

        text = re.sub(
            rf"(?<!{_MN_CLASS})(?<![A-Za-z])"
            rf"(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?",
            _guarded_time,
            text,
        )

        # Whatever colon is left is a ratio, and a ratio is read as a fraction:
        # "1:2" is "хоёрны нэг" -- the same string as 1/2 -- and "3:1" is
        # "нэгний гурав". This is what settles the ambiguity the spec left
        # between a ratio and a sports score: both are the fraction.
        def _ratio(m: re.Match[str]) -> str:
            return self._fraction_words(int(m.group(1)), int(m.group(2)))

        text = re.sub(r"(?<!\d)(\d{1,3}):(\d{1,3})(?!\d)", _ratio, text)

        pass  # verse references are handled before the time pass

        def _temp(m: re.Match[str]) -> str:
            sign, num, unit = m.group(1), int(m.group(2)), m.group(3)
            parts: list[str] = []
            if sign == "-":
                parts.append(self._minus_word)
            parts.append(f"{self.convert_attributive(num)} {self._degree_word}")
            if unit and unit.upper() == "C":
                parts.append("цельс")
            elif unit and unit.upper() == "F":
                parts.append("фаренгейт")
            return " ".join(parts)

        text = re.sub(r"(-?)(\d+)°\s*([CcFf])?", _temp, text)

        _sym_pattern = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)
        _code_pattern = "|".join(CURRENCY_CODES)

        # "1.5 сая төгрөг" is one quantity, not "нэг төгрөг таван сая": the
        # scale word multiplies the number in front of it, so it has to be
        # resolved before either the decimal or the currency pass sees it.
        _scale_words = {name: value for value, (name, _) in LARGE.items()}

        def _scaled(m: re.Match[str]) -> str:
            symbol, amount, scale = m.group(1), m.group(2), m.group(3)
            value = int(round(float(amount) * _scale_words[scale]))
            if symbol:
                return (f"{self.convert_attributive(value)} "
                        f"{self._currency_name(symbol)}")
            return self.convert(value)

        text = re.sub(
            rf"({_sym_pattern})?\s*(\d+(?:\.\d+)?)\s*({'|'.join(_scale_words)})"
            rf"(?!{_MN_CLASS})",
            _scaled,
            text,
        )

        def _currency_after(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(1)))} {self._currency_name(m.group(2))}"

        text = re.sub(
            rf"(\d+)\s*({_sym_pattern}|(?:{_code_pattern})(?!\w))", _currency_after, text
        )

        def _currency_before(m: re.Match[str]) -> str:
            return f"{self.convert_attributive(int(m.group(2)))} {self._currency_name(m.group(1))}"

        text = re.sub(rf"({_sym_pattern})\s*(\d+)", _currency_before, text)

        # Percent must see the decimal before the decimal pass splits it:
        # "12.5%" was matching "5%" and leaving "арван хоёр.таван хувь".
        def _percent(m: re.Match[str]) -> str:
            whole, frac = m.group(1), m.group(2)
            if frac:
                return f"{self._decimal_words(whole, frac, attr=True)} {self._percent_word}"
            return f"{self.convert_attributive(int(whole))} {self._percent_word}"

        text = re.sub(r"(\d+)(?:\.(\d+))?%", _percent, text)

        # Dotted references before decimals: "5.1.2" and "3.4.1.7" are section
        # numbers, not numbers with a fractional part, and reading them as
        # decimals produced "тав цэг нэг.хоёр".
        def _reference(m: re.Match[str]) -> str:
            parts = [int(g) for g in m.group(0).split(".")]
            head = [self._reference_genitive(n) for n in parts[:-1]]
            return " ".join([*head, self.convert(parts[-1])])

        text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){2,}\b", _reference, text)

        def _decimal(m: re.Match[str]) -> str:
            return self._decimal_words(m.group(1), m.group(2))

        text = re.sub(r"(\d+)\.(\d+)", _decimal, text)

        # "2 1/2" is "хоёр бүтэн хоёрны нэг" -- a different linking word from
        # the plain fraction, so it has to be matched before it.
        def _mixed_fraction(m: re.Match[str]) -> str:
            whole, num, den = (int(m.group(i)) for i in (1, 2, 3))
            return (f"{self.convert(whole)} {MIXED_WHOLE_WORD} "
                    f"{self._fraction_words(num, den)}")

        text = re.sub(r"\b(\d+)\s+(\d{1,2})/(\d{1,2})\b", _mixed_fraction, text)

        def _fraction(m: re.Match[str]) -> str:
            return self._fraction_words(int(m.group(1)), int(m.group(2)))

        text = re.sub(r"(\d{1,2})/(\d{1,2})", _fraction, text)

        def _international(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0)[1:])
            return "нэмэх " + self._phone_words(digits)

        text = re.sub(r"\+\d[\d\s\-]{6,15}\d", _international, text)

        # A bare 8-digit run is a Mongolian phone number, and reading it as a
        # quantity gave "ерэн есөн сая зуун арван хоёр мянга ...". Eight digits
        # is the national mobile length, so the shape is the signal.
        def _local_phone(m: re.Match[str]) -> str:
            return self._phone_words(m.group(0))

        text = re.sub(r"(?<!\d)\d{8}(?!\d)", _local_phone, text)

        # A leading zero means the digits are an identifier, not a quantity:
        # "007" is "тэг тэг долоо", not "долоо". Read digit by digit.
        text = re.sub(
            r"(?<![\d\w])0\d+(?![\d\w])",
            lambda m: self._digit_by_digit(m.group(0)),
            text,
        )

        # A minus sign attached to a number, as opposed to a range separator or
        # an identifier hyphen: "-15" is "хасах арван тав". Without this the
        # hyphen survived into the text as a spoken token.
        text = re.sub(
            r"(?<![\d\w\-])-(?=\d)", f"{self._minus_word} ", text
        )

        def _range(m: re.Match[str]) -> str:
            lo_n, hi_n, following = int(m.group(1)), int(m.group(2)), m.group(3)
            lo = self.convert(lo_n)
            head, _, last = lo.rpartition(" ")
            abl = ABLATIVE.get(last)
            lo_abl = f"{head} {abl}".strip() if abl else lo
            # "5-10 км" is "таваас арван километр": the upper bound is
            # attributive before the unit it counts, and there is no "хүртэл".
            # Without the unit, "хүртэл" closes the range.
            if following:
                return f"{lo_abl} {self.convert_attributive(hi_n)}{following}"
            return f"{lo_abl} {self.convert(hi_n)} хүртэл"

        text = re.sub(
            rf"(\d+)\s*[-–—]\s*(\d+)(\s+{_MN_CLASS}+)?", _range, text
        )

        # "Байр 12" -> "арван хоёрдугаар байр": the number becomes an ordinal
        # and moves in front of the noun.
        def _ordinal_noun(m: re.Match[str]) -> str:
            return f"{self.convert_ordinal(int(m.group(2)))} {m.group(1).lower()}"

        _nouns = "|".join(ORDINAL_NOUNS)
        text = re.sub(rf"(?<!{_MN_CLASS})({_nouns})\s+(\d{{1,4}})", _ordinal_noun, text,
                      flags=re.IGNORECASE)

        def _ordinal(m: re.Match[str]) -> str:
            return self.convert_ordinal(int(m.group(1)))

        # A list marker: a number and a full stop, alone. "1." is "нэгдүгээр".
        # Anchored to the start so it cannot eat a sentence-final number.
        text = re.sub(r"\A\s*(\d{1,3})\.(?=\s|\Z)", _ordinal, text)

        text = re.sub(r"(\d+)-р\b", _ordinal, text)
        text = re.sub(r"(\d+)-д(?:угаар|үгээр|ахь)", _ordinal, text)

        # Genitive markers before nouns: 2024-ны, 1-ний, 5-ын, 3-ийн.
        #
        # These used to return convert_attributive(), which *deletes* the
        # genitive the author wrote: "5-ын хувь" became "таван хувь" rather than
        # "тавын хувь". Attributive and genitive are different morphemes, and
        # since this string is what CER is scored against, substituting one for
        # the other manufactures the very text/audio mismatch the CER gate
        # exists to detect.
        def _genitive(m: re.Match[str]) -> str:
            return self.attach_suffix(int(m.group(1)), m.group(2))

        text = re.sub(r"(\d+)-(ны|ний|ын|ийн)\b", _genitive, text)

        def _suffixed(m: re.Match[str]) -> str:
            return self.attach_suffix(int(m.group(1)), m.group(2))

        text = re.sub(rf"(\d+)-({_MN_CLASS}+)", _suffixed, text)

        def _roman(m: re.Match[str]) -> str:
            val = self._roman_to_int(m.group(1))
            return m.group(0) if val is None else self.convert_ordinal(val)

        text = _ROMAN_RE.sub(_roman, text)

        # A whole arithmetic expression at once, so both operands stay
        # standalone: "5x8" is "тав үржих нь найм", not "таван үржих нь найм".
        # Converted digit by digit afterwards, the left operand would pick up
        # the attributive form from the operator word following it.
        _ops = "|".join(re.escape(s) for s in MATH_SYMBOLS)

        def _arithmetic(m: re.Match[str]) -> str:
            return (f"{self.convert(int(m.group(1)))} "
                    f"{MATH_SYMBOLS[m.group(2)]} {self.convert(int(m.group(3)))}")

        text = re.sub(rf"(\d+)\s*({_ops})\s*(\d+)", _arithmetic, text)

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

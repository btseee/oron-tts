"""Reading things that are not words: ABC, H2O, RX9070, URLs, file names.

All of it reduces to one operation -- spell the letters, read the digits, name
the punctuation -- and all of it depends on the letter-name tables in
`data/lexicon`. The machinery is here; the Mongolian is there.

The fallback is always "leave it alone". A token whose letters are not all named
in the tables is returned unchanged, because Latin is in the vocabulary and the
model will attempt it, which beats emitting a half-spelled word. That is why
`spell_out` returns None rather than a partial answer.
"""

from __future__ import annotations

import re
from typing import Final

from oron_tts.text.lexicon import load, spell_out

# Read as a word, not spelled: "цэг" for the dot in a URL or a version string.
DOT_WORD: Final[str] = "цэг"
AT_WORD: Final[str] = "эт"
FOOTNOTE_WORD: Final[str] = "хөл тэмдэглэл"
VERSION_WORD: Final[str] = "верс"
POWER_WORD: Final[str] = "зэрэгт"

# Superscripts and subscripts carry a digit's value but not its codepoint, so a
# chemical formula or an exponent is invisible to every numeric rule until they
# are folded. H(sub 2)O and 10(sup 5) both depend on this.
SUPERSCRIPTS: Final[dict[str, str]] = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789", strict=True))
SUBSCRIPTS: Final[dict[str, str]] = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789", strict=True))

# A token worth spelling: Latin letters, optionally with digits, and no lower
# case run long enough to be a word. "ABC" and "RX9070" qualify; "Google" does
# not, and is left to the foreign-word table.
SPELLABLE = re.compile(r"(?<![\w.])([A-Z]{2,}\d*|[A-Z]+\d+[A-Z\d]*)(?![\w.])")

# "Б.Бат" -- an initial is a single capital, a dot, then a name.
INITIAL = re.compile(r"(?<![\w.])([А-ЯӨҮЁ])\.\s*([А-ЯӨҮЁ][а-яөүё]+)")

ROMAN_ONLY = re.compile(r"[IVXLCDM]+")

VERSION = re.compile(r"(?<![\w])[vv]\.?(\d+(?:\.\d+)+)(?![\w])")

# Top-level domains, so a host is told apart from a file name: "github.com" is
# a URL, "report.pdf" is not, and both match the same shape. Without this the
# URL rule ate every file name it saw.
TLDS: Final[frozenset[str]] = frozenset(
    ["com", "org", "net", "int", "edu", "gov", "mil", "mn", "ru", "cn", "us", "uk", "de", "fr", "jp", "kr", "io", "ai", "co", "info", "biz", "tv", "me", "app", "dev", "xyz", "online", "site", "tech"]
)

URL = re.compile(
    r"(?<![\w])(?:(https?)://)?((?:[\w-]+\.)+([a-z]{2,}))(?![\w])", re.ASCII
)
EMAIL = re.compile(r"(?<![\w])([\w.+-]+)@((?:[\w-]+\.)+[a-z]{2,})(?![\w])", re.ASCII)
FILENAME = re.compile(r"(?<![\w])([\w-]+)\.([a-z]{2,4})(?![\w])", re.ASCII)


def fold_subscripts(text: str) -> str:
    """Subscript digits to ordinary ones, for chemical formulas."""
    return text.translate(str.maketrans(SUBSCRIPTS))


def fold_superscripts(text: str) -> str:
    """Superscript digits to ordinary ones.

    Deliberately separate from the subscripts and run much later. A superscript
    is load-bearing until the unit table has seen it: fold it early and "50 m2"
    no longer matches the square-metre entry, which gave "тавин мхоёр".
    """
    return text.translate(str.maketrans(SUPERSCRIPTS))


def fold_scripts(text: str) -> str:
    """Both, for callers that want a plain-digit string."""
    return fold_superscripts(fold_subscripts(text))


def _word(part: str, directory=None) -> str | None:
    """A URL or file-name component: a known foreign word, or spelled out."""
    foreign = load("foreign_words", directory)
    lowered = part.lower()
    for key, spoken in foreign.items():
        if key.lower() == lowered:
            return spoken
    return spell_out(part, directory)


def read_initials(text: str, directory=None) -> str:
    """"Б.Бат" -> "бэ бат": the initial is spelled, the name is spoken."""
    letters = load("cyrillic_letters", directory)

    def repl(m: re.Match[str]) -> str:
        name = letters.get(m.group(1).lower())
        if name is None:
            return m.group(0)
        return f"{name} {m.group(2).lower()}"

    return INITIAL.sub(repl, text)


def read_spellable(text: str, directory=None) -> str:
    """"ABC" -> "эй би си", "RX9070" -> "ар икс ес тэг долоо тэг".

    Digits inside the token are left as digits for the number pass, which reads
    a leading-zero run digit by digit -- exactly what an identifier needs.
    """
    foreign = load("foreign_words", directory)

    def repl(m: re.Match[str]) -> str:
        token = m.group(1)
        if token in foreign:
            return foreign[token]
        # A token made only of Roman-numeral letters is left for the Roman rule,
        # which needs a context noun to fire and is the only thing that can tell
        # "XXI зуун" from an acronym. Spelling it here gave "икс икс ай зуун".
        # The cost is that a genuine acronym of those letters -- DVD, CD -- is
        # not spelled either; it stays Latin, which the model can still attempt.
        if ROMAN_ONLY.fullmatch(token):
            return token
        # In source order: "H2O" is "эйч хоёр оу", so the digit sits where it
        # was written. Sorting letters before digits gave "эйч оу хоёр".
        parts: list[str] = []
        for char in token:
            if char.isdigit():
                parts.append(char)   # left for the number pass
                continue
            name = spell_out(char, directory)
            if name is None:
                return token         # a letter with no name: leave it all alone
            parts.append(name)
        return " ".join(parts)

    return SPELLABLE.sub(repl, text)


def read_version(text: str, directory=None) -> str:
    """"v2.1.5" -> "верс хоёр цэг нэг цэг тав"."""
    def repl(m: re.Match[str]) -> str:
        parts = m.group(1).split(".")
        return f"{VERSION_WORD} " + f" {DOT_WORD} ".join(parts)

    return VERSION.sub(repl, text)


def read_email(text: str, directory=None) -> str:
    def repl(m: re.Match[str]) -> str:
        local = _word(m.group(1), directory)
        host = [_word(p, directory) for p in m.group(2).split(".")]
        if local is None or any(h is None for h in host):
            return m.group(0)
        return f"{local} {AT_WORD} " + f" {DOT_WORD} ".join(host)

    return EMAIL.sub(repl, text)


def read_url(text: str, directory=None) -> str:
    """"https://github.com" -> "эйч ти ти пи эс гитхаб цэг ком".

    The scheme is spelled; "://" is not spoken at all.
    """
    def repl(m: re.Match[str]) -> str:
        scheme, host, tld = m.group(1), m.group(2), m.group(3)
        if not scheme and tld not in TLDS:
            return m.group(0)   # a file name, not a host
        parts = [_word(p, directory) for p in host.split(".")]
        if any(p is None for p in parts):
            return m.group(0)
        spoken = f" {DOT_WORD} ".join(parts)
        if scheme:
            said = spell_out(scheme, directory)
            if said is None:
                return m.group(0)
            spoken = f"{said} {spoken}"
        return spoken

    return URL.sub(repl, text)


def read_footnote(text: str) -> str:
    """A superscript digit *standing alone* is a footnote marker.

    Attached to something it is an exponent or a unit: the square in "50 м²" is
    not a footnote, and reading it as one gave "тавин мхөл тэмдэглэл хоёр".
    """
    return re.sub(
        r"(?<![\w⁰¹²³⁴⁵⁶⁷⁸⁹])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)(?![\w⁰¹²³⁴⁵⁶⁷⁸⁹])",
        lambda m: f"{FOOTNOTE_WORD} {fold_scripts(m.group(1))}",
        text,
    )


def read_filename(text: str, directory=None) -> str:
    """"report.pdf" -> "репорт пи ди эф": the stem as a word, the extension
    spelled.

    Runs after `read_url`, which claims anything ending in a real top-level
    domain. Everything else with an extension-shaped tail is a file.
    """
    def repl(m: re.Match[str]) -> str:
        stem, ext = m.group(1), m.group(2)
        if ext in TLDS:
            return m.group(0)
        said_stem = _word(stem, directory)
        said_ext = spell_out(ext, directory)
        if said_stem is None or said_ext is None:
            return m.group(0)
        return f"{said_stem} {said_ext}"

    return FILENAME.sub(repl, text)


def read_long_digit_run(text: str, minimum: int = 10) -> str:
    """A run this long is an identifier, not a quantity.

    "ISBN 9781234567890" was being read as "есөн их наяд долоон зуун наян нэг
    тэрбум ...", which is arithmetically right and useless. Ten digits is past
    any number a sentence says aloud.
    """
    return re.sub(
        rf"(?<!\d)\d{{{minimum},}}(?!\d)",
        lambda m: " ".join(m.group(0)),
        text,
    )

"""Mongolian text normalization.

The output of `MongolianNormalizer.normalize` is published in the corpus, scored
for CER, and fed to the model. All three must be the same string, so anything
that changes here changes the training data.
"""

import pytest

from oron_tts.text import MongolianNormalizer, NumberNormalizer, VocabError


@pytest.fixture(scope="module")
def norm() -> MongolianNormalizer:
    return MongolianNormalizer()


@pytest.fixture(scope="module")
def num() -> NumberNormalizer:
    return NumberNormalizer()


# ── cardinals ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "тэг"), (1, "нэг"), (5, "тав"), (10, "арав"), (15, "арван тав"),
        (20, "хорь"), (25, "хорин тав"), (100, "зуу"), (101, "зуун нэг"),
        (1000, "мянга"), (2024, "хоёр мянга хорин дөрөв"),
        (-5, "хасах тав"),
    ],
)
def test_cardinal(num, n, expected):
    assert num.convert(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [(3, "гурван"), (4, "дөрвөн"), (5, "таван"), (9, "есөн"), (10, "арван"),
     (20, "хорин"), (100, "зуун")],
)
def test_attributive_differs_from_standalone(num, n, expected):
    """Attributive is used before nouns: таван мянга, not тав мянга."""
    assert num.convert_attributive(n) == expected


@pytest.mark.parametrize(
    "n,expected",
    [(1, "нэгдүгээр"), (2, "хоёрдугаар"), (3, "гуравдугаар"), (5, "тавдугаар"),
     (10, "аравдугаар"), (15, "арван тавдугаар")],
)
def test_ordinal_vowel_harmony(num, n, expected):
    """Suffix is дугаар after back vowels, дүгээр after front vowels."""
    assert num.convert_ordinal(n) == expected


# ── in-text expansion ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024 онд", "хоёр мянга хорин дөрвөн онд"),
        ("25 хувь", "хорин таван хувь"),
        ("1-р сар", "нэгдүгээр сар"),
        ("3-дугаар анги", "гуравдугаар анги"),
        ("2024-ны", "хоёр мянга хорин дөрвөн"),
        ("15-нд", "арван тавнд"),
        ("5 км", "таван километр"),
        ("50%", "тавин хувь"),
        ("14:30", "арван дөрвөн цаг гучин минут"),
        ("1/2", "хагас"),
        ("-15°C", "хасах арван таван градус цельсий"),
        ("100₮", "зуун төгрөг"),
    ],
)
def test_expansion(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


def test_number_before_mongolian_word_takes_attributive(norm):
    """The naive [а-я] range excludes ө U+04E9 and ү U+04AF; both must match."""
    assert norm.normalize("5 өдөр", strict=False) == "таван өдөр"
    assert norm.normalize("5 үг", strict=False) == "таван үг"


def test_range_uses_attached_ablative(norm):
    # "арав аас" with a space is not how it is spoken; ь is irregular.
    assert norm.normalize("10-20 хүн", strict=False) == "араваас хорь хүртэл хүн"
    assert norm.normalize("20-аас доош", strict=False) == "хориос доош"


# ── the two bugs this rewrite fixes ───────────────────────────────────────────

def test_roman_numerals_need_a_context_word(norm):
    """Unrestricted matching rewrote Latin words: MIX = M(1000)+IX(9)."""
    assert norm.normalize("MIX цомог", strict=False) == "MIX цомог"
    assert norm.normalize("XV зуун", strict=False) == "арван тавдугаар зуун"


def test_latin_is_preserved_not_deleted(norm):
    """All Latin in the corpus is already in the vocab with trained embeddings.

    Deleting it desynchronises text from audio: the speaker still says the word.
    """
    assert norm.normalize("Wi-Fi холболт", strict=False) == "Wi-Fi холболт"
    assert "COVID" in norm.normalize("COVID-19 тархав", strict=False)


def test_identifier_hyphen_does_not_survive_as_a_token(norm):
    # Otherwise: "COVID-арван ес", with the hyphen spoken.
    assert norm.normalize("COVID-19", strict=False) == "COVID арван ес"


# ── character mapping ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("сайн\xa0байна", "сайн байна"),      # NBSP is not a vocab space
        ("“сайн”", '"сайн"'),        # curly quotes absent from vocab
        ("сайн­байна", "сайнбайна"),      # soft hyphen is invisible
        ("﻿сайн", "сайн"),                # BOM
    ],
)
def test_char_map(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


@pytest.mark.parametrize("ch", ["—", "–", "…", "«", "»"])
def test_vocab_covered_punctuation_is_preserved(norm, ch):
    """Preserve what the vocab can represent; map only what it cannot."""
    assert ch in norm.normalize(f"сайн {ch} байна", strict=False)


def test_case_is_preserved(norm):
    """The vocab holds both cases of Cyrillic with pretrained embeddings."""
    assert norm.normalize("Сайн Байна", strict=False) == "Сайн Байна"


def test_whitespace_and_repeated_punctuation_collapse(norm):
    assert norm.normalize("сайн   байна", strict=False) == "сайн байна"
    assert norm.normalize("сайн!!!", strict=False) == "сайн!"


# ── strictness ────────────────────────────────────────────────────────────────

def test_strict_mode_raises_instead_of_deleting(norm):
    """A silent delete is what manufactured text/audio mismatch before."""
    with pytest.raises(VocabError) as exc:
        norm.normalize("сайн 你好 байна")
    assert "你" in exc.value.chars


def test_unsupported_chars_reports_without_raising(norm):
    assert norm.unsupported_chars("сайн 你 байна") == ["你"]
    assert norm.unsupported_chars("сайн байна") == []
    assert norm.is_representable("Сайн байна уу? Өнөөдөр үүлшинэ.")


def test_digits_never_survive(norm):
    """A digit reaching the tokenizer means the model must learn to say it."""
    for raw in ["2024 онд", "50%", "14:30", "COVID-19", "5 км", "1/2", "10-20"]:
        assert not any(c.isdigit() for c in norm.normalize(raw, strict=False)), raw


# ── Kazakh is gone ────────────────────────────────────────────────────────────

def test_no_kazakh_surface_remains(norm):
    import inspect

    from oron_tts.text import normalizer, numbers

    source = inspect.getsource(numbers) + inspect.getsource(normalizer)
    for marker in ["kz", "KZ", "kazakh", "Kazakh", "қазақ"]:
        assert marker not in source, f"Kazakh residue: {marker}"
    # Kazakh-only letters must not be representable as Mongolian.
    assert norm.unsupported_chars("әғқңұһі")


def test_normalizer_takes_no_lang_argument(norm):
    import inspect

    params = inspect.signature(norm.normalize).parameters
    assert "lang" not in params

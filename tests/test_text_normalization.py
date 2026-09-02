"""Mongolian text normalization.

The output of `MongolianNormalizer.normalize` is published in the corpus, scored
for CER, and fed to the model. All three must be the same string, so anything
that changes here changes the training data.
"""

import pytest

from oron_tts.text import MongolianNormalizer, NumberNormalizer, VocabError
from oron_tts.text.numbers import NumeralSuffixError


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
        # "нэг мянга", not "мянга": a bare scale word is a noun and one of them
        # still needs counting. The spec gives "1005" as "нэг мянга тав". A year
        # is the exception and drops it -- see test_a_year_drops_the_leading_neg.
        (1000, "нэг мянга"), (2024, "хоёр мянга хорин дөрөв"),
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
        ("5 км", "таван километр"),
        ("50%", "тавин хувь"),
        ("14:30", "арван дөрвөн цаг гучин минут"),
        ("1/2", "хоёрны нэг"),
        ("-15°C", "хасах арван таван градус цельсий"),
        ("100₮", "зуун төгрөг"),
    ],
)
def test_expansion(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


# ── numeral suffixes: tabulated, or refused ───────────────────────────────────
#
# Two cases used to sit in the table above:
#
#     ("2024-ны", "хоёр мянга хорин дөрвөн")   -- deletes the genitive entirely
#     ("15-нд",   "арван тавнд")               -- a non-word
#
# They asserted the defect as the specification, which is why 58 text tests
# passed while the normaliser emitted non-words on ordinary input. Deleted
# rather than corrected: neither correct form is derivable from the tables in
# numbers.py, and this project's one-string rule means a guess would be
# published, scored against itself, and trained on.


@pytest.mark.parametrize("raw,expected", [
    ("20-аас доош", "хориос доош"),     # ь is irregular: not *хорьаас
    ("20-иос доош", "хориос доош"),     # the spelling people actually write
    ("3-аас", "гурваас"),               # unstable vowel reduces
    ("4-өөс", "дөрвөөс"),
    ("1-ээс", "нэгээс"),
])
def test_a_tabulated_suffix_expands(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


@pytest.mark.parametrize("raw", ["2024-ны", "15-нд", "1-ний", "3-ийн", "20-ийг",
                                 "100-д", "5-ын хувь"])
def test_an_untabulated_suffix_is_refused_not_guessed(norm, raw):
    """The whole point of the change.

    Each of these previously produced a plausible-looking non-word. The CER gate
    cannot catch that, because the corrupted string *is* the reference it scores
    against.
    """
    with pytest.raises(NumeralSuffixError):
        norm.normalize(raw, strict=False)


def test_the_refusal_names_the_missing_entry(norm):
    """So filling the table is a lookup, not a hunt."""
    with pytest.raises(NumeralSuffixError, match="normaliser-review"):
        norm.normalize("100-д", strict=False)


@pytest.mark.parametrize("raw,expected", [("1/2", "хоёрны нэг"), ("3/4", "дөрөвний гурав")])
def test_a_tabulated_fraction_expands(norm, raw, expected):
    """Genitive of the denominator, then the numerator.

    `3/4` used to come out as `дөрөвдүгээрийн гурав` -- an ordinal plus a
    genitive, roughly "of the fourth, three". An ordinal names a position, not
    a part, so that was wrong however the genitive was chosen.
    """
    assert norm.normalize(raw, strict=False) == expected


@pytest.mark.parametrize("raw", ["2/3", "1/5", "5/7"])
def test_an_untabulated_fraction_is_refused(norm, raw):
    """FRACTION_GENITIVE holds only the denominators the spec supplies."""
    with pytest.raises(NumeralSuffixError):
        norm.normalize(raw, strict=False)


def test_ordinary_text_with_numbers_is_untouched_by_the_refusal(norm):
    assert norm.normalize("1990 онд 25 хүн ирсэн", strict=False) == (
        "мянга есөн зуун ерэн онд хорин таван хүн ирсэн"
    )


# ── abbreviations that were not abbreviations ─────────────────────────────────

def test_a_gram_unit_is_not_a_year(norm):
    """ABBREVIATIONS["г."] = "оны" shadowed the gram unit plus a full stop, and
    the abbreviation pass runs before the unit pass -- so the collision was
    guaranteed, not occasional."""
    assert norm.normalize("Жин нь 5 г.", strict=False) == "Жин нь таван грамм."


@pytest.mark.parametrize("raw", ["Энэ бол сайн зохиолч.", "Тэр ирэв.", "Би явлаа ж."])
def test_a_single_letter_before_a_full_stop_is_a_word_ending(norm, raw):
    """"т." -> "товч" and "ж." -> "жил" fired on any sentence ending that way."""
    out = norm.normalize(raw, strict=False)
    assert "товч" not in out
    assert "жил" not in out


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


# ── homoglyphs ────────────────────────────────────────────────────────────────

def test_the_ukrainian_i_is_folded_to_the_mongolian_one(norm):
    """M19. U+0456 is the one non-Mongolian Cyrillic letter in vocab.txt, so it
    passes the vocabulary gate silently -- and the model gets a second,
    near-untrained embedding for a letter it already has and a speaker
    pronounces identically."""
    assert norm.normalize("сайн \u0456 байна", strict=False) == "сайн и байна"
    assert norm.normalize("\u0406 \u0407", strict=False) == "И И"


def test_the_homoglyph_would_otherwise_pass_the_vocabulary_gate(norm):
    """Which is why folding it in normalisation is the fix, not rejecting it."""
    from oron_tts.text import charset

    assert "\u0456" in charset()          # representable, so check() says yes
    assert norm.is_representable("сайн \u0456 байна")


def test_latin_look_alikes_are_not_folded(norm):
    """Latin is in the vocabulary and preserved on purpose; folding "c" to "с"
    would corrupt genuine Latin text."""
    assert norm.normalize("BBC News", strict=False) == "BBC News"


# ── from the normalization specification ──────────────────────────────────────

def test_a_bare_scale_word_takes_neg(num):
    """The spec: "1005" is "нэг мянга тав", "1256789" starts "нэг сая"."""
    assert num.convert(1005) == "нэг мянга тав"
    assert num.convert(1_000_000) == "нэг сая"


def test_a_year_drops_the_leading_neg(norm):
    """"1990 он" is "мянга есөн зуун ерэн он" -- the same digits as a quantity
    are "нэг мянга ...". Only the following noun separates them."""
    assert norm.normalize("1990 он", strict=False) == "мянга есөн зуун ерэн он"
    # Standalone, so 90 is "ер" rather than the attributive "ерэн" -- but the
    # "нэг" is back.
    assert norm.normalize("1990", strict=False) == "нэг мянга есөн зуун ер"


def test_a_verse_reference_is_not_a_clock_time(norm):
    """Unguarded, "Иохан 3:16" became "Иохан гурван цаг арван зургаан минут".

    Nothing in the shape of the string separates it from "Цаг 14:30", so it
    takes a word list -- data/lexicon/reference_words.tsv.
    """
    assert norm.normalize("Иохан 3:16", strict=False) == "Иохан гурав арван зургаа"
    assert "цаг" in norm.normalize("Цаг 14:30 боллоо", strict=False)


def test_an_impossible_clock_time_is_left_alone(norm):
    """25:70 is not a time, so the hour/minute guard must decline it."""
    out = norm.normalize("25:70", strict=False)
    assert "цаг" not in out and "минут" not in out


@pytest.mark.parametrize("raw,expected", [
    ("3.14", "гурван бүхэл арван дөрөв"),
    ("0.05", "тэг бүхэл таван зууны"),
    ("12.5%", "арван хоёр бүхэл таван аравны хувь"),
])
def test_decimals_follow_the_spec(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


def test_a_dotted_reference_is_not_a_decimal(norm):
    """"5.1.2" is a section number. Read as a decimal it became "тав цэг нэг.хоёр"."""
    assert norm.normalize("5.1.2", strict=False) == "тавын нэгийн хоёр"
    assert norm.normalize("3.4.1.7", strict=False) == "гурвын дөрвийн нэгийн долоо"


def test_a_mixed_fraction_uses_its_own_linking_word(norm):
    """бүтэн, not бүхэл."""
    assert norm.normalize("2 1/2", strict=False) == "хоёр бүтэн хоёрны нэг"


@pytest.mark.parametrize("raw,expected", [
    ("50 м²", "тавин квадрат метр"),
    ("20 м³", "хорин шоо метр"),
    ("5 kW", "таван киловатт"),
])
def test_units_come_from_the_lexicon(norm, raw, expected):
    """Longest-first matching, or "50 м²" becomes "тавин метр²"."""
    assert norm.normalize(raw, strict=False) == expected


@pytest.mark.parametrize("raw,expected", [
    ("УИХ", "Улсын Их Хурал"),
    ("МУИС", "Монгол Улсын Их Сургууль"),
    ("д-р Бат", "доктор Бат"),
])
def test_abbreviations_come_from_the_lexicon(norm, raw, expected):
    assert norm.normalize(raw, strict=False) == expected


def test_a_foreign_word_in_the_lexicon_is_transliterated(norm):
    assert norm.normalize("Google", strict=False) == "Гүүгл"


def test_a_foreign_word_not_in_the_lexicon_stays_latin(norm):
    """Latin is in the vocabulary and the model will attempt it, which beats a
    wrong Cyrillic guess."""
    assert "Zzyzx" in norm.normalize("Zzyzx хот", strict=False)


def test_a_known_emoji_is_spoken(norm):
    assert norm.normalize("❤️", strict=False).strip() == "улаан зүрх"

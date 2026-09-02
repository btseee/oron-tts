"""The lookup tables the normaliser cannot derive.

These hold Mongolian content written by a speaker. The tests do not check that
the *content* is right -- nothing here can -- but they do check the two things
that would silently corrupt the corpus if wrong: the format, and whether every
spoken form is representable in the vocabulary. A spoken form containing a
character absent from vocab.txt becomes a space at training time, with nothing
logged.
"""

import pytest

from oron_tts.text import charset
from oron_tts.text.lexicon import LEXICON_DIR, NAMES, all_lexicons, load, parse, spell_out


def test_every_declared_lexicon_parses():
    for name in NAMES:
        load(name)  # raises with the file and line on a malformed row


def test_the_seeded_lexicons_are_not_empty():
    """A frame with nothing in it is a frame nobody filled."""
    tables = all_lexicons()
    for name in ("abbreviations", "latin_letters", "units", "foreign_words"):
        assert tables[name], f"{name}.tsv is empty"


def test_every_spoken_form_is_representable():
    """The one that matters. An unknown character maps to index 0, which is the
    SPACE token, so a lexicon entry with a stray character silently deletes
    itself from the training text."""
    vocab = charset()
    for name, table in all_lexicons().items():
        if name in {"emoji", "reference_words"}:
            continue  # keys are emoji / the value is unused
        for key, spoken in table.items():
            missing = sorted({c for c in spoken if c not in vocab})
            assert not missing, f"{name}: {key!r} -> {spoken!r} has {missing}"


def test_emoji_descriptions_are_representable():
    vocab = charset()
    for emoji, spoken in load("emoji").items():
        missing = sorted({c for c in spoken if c not in vocab})
        assert not missing, f"{emoji!r} -> {spoken!r} has {missing}"


def test_no_lexicon_key_is_blank_or_duplicated():
    """parse() raises on both; this pins that it is actually called."""
    with pytest.raises(ValueError, match="already defined"):
        parse("a\tone\na\ttwo")
    with pytest.raises(ValueError, match="empty column"):
        parse("\tone")   # no key; a trailing tab is stripped and caught below
    with pytest.raises(ValueError, match="expected a tab"):
        parse("no tab here")


def test_comments_and_blank_lines_are_ignored():
    assert parse("# note\n\na\tone\n") == {"a": "one"}


def test_a_missing_lexicon_is_empty_not_an_error(tmp_path):
    """These are frames. An empty one means every lookup falls back, which is
    the designed behaviour -- a wrong entry is dangerous, an absent one is not."""
    assert load("abbreviations", tmp_path) == {}


def test_spell_out_uses_the_letter_tables():
    assert spell_out("ABC") == "эй би си"


def test_spell_out_refuses_a_partial_answer():
    """Half a spelled-out word is worse than none: the caller leaves it alone."""
    assert spell_out("XYZ") is None


def test_the_units_lexicon_covers_the_squared_and_cubed_forms():
    """Longest-first matching depends on these existing as their own keys."""
    units = load("units")
    assert units.get("м²") and units.get("м³")
    assert units["м²"] != units["м"]


def test_the_lexicon_directory_is_where_the_readme_says():
    assert (LEXICON_DIR / "README.md").exists()
    for name in NAMES:
        assert (LEXICON_DIR / f"{name}.tsv").exists(), name


# ── everything the normaliser can emit ────────────────────────────────────────

def test_every_emittable_constant_is_representable():
    """The silent-failure class, applied to the normaliser's own vocabulary.

    An unknown character maps to index 0, which is the SPACE token. A stray
    character in a *lexicon* entry is caught above; this catches one in a
    module constant, which is the same defect with no file to review. It found
    a real one: CURRENCY_SYMBOLS mapped the tenge sign to "теңге", whose ң
    U+04A3 is absent from vocab.txt, so "100₸" would have put a space into the
    training text with nothing logged.

    SILENT_MARKS is excluded because it is an input set -- those characters are
    stripped, never written.
    """
    import oron_tts.text.identifiers as identifiers
    import oron_tts.text.normalizer as normalizer
    import oron_tts.text.numbers as numbers

    vocab = charset()
    excluded = {"normalizer.SILENT_MARKS", "normalizer.CHAR_MAP"}
    problems: list[str] = []

    def visit(label: str, value: object) -> None:
        if label in excluded:
            return
        if isinstance(value, str):
            missing = sorted({c for c in value if c not in vocab})
            if missing:
                problems.append(f"{label} = {value!r} contains {missing}")
        elif isinstance(value, dict):
            for key, inner in value.items():
                visit(f"{label}[{key!r}]", inner)

    for module in (numbers, normalizer, identifiers):
        name = module.__name__.rsplit(".", 1)[-1]
        for attribute in dir(module):
            if attribute.isupper():
                visit(f"{name}.{attribute}", getattr(module, attribute))

    assert not problems, "\n".join(problems)


def test_the_homoglyph_map_only_folds_what_the_vocabulary_accepts():
    """Folding a character the vocabulary rejects replaces a correct, loud
    rejection with a guess.

    Ї U+0407 and І U+0406 were mapped to И on the strength of looking like it.
    They are absent from vocab.txt, so they already failed the check; and on
    real Mongolian text ї is usually mojibake for ү -- "бїлэг" for "бүлэг" --
    where folding it corrupts rather than repairs.
    """
    from oron_tts.text.normalizer import CHAR_MAP

    vocab = charset()
    folded = {src for src, dst in CHAR_MAP.items() if dst and dst.isalpha()}
    for source in folded:
        assert source in vocab, (
            f"{source!r} is folded but is not in the vocabulary, so it would "
            f"have been rejected anyway -- the fold only hides that"
        )

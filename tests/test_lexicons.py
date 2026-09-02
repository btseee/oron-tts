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

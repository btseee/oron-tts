"""The normalization specification, scored as a test.

`tests/data/normalization_spec.json` holds the 85 cases from the Mongolian
normalization spec. Conformance is a number rather than an impression, and it
only moves in one direction: the guard below fails if it drops.

Two kinds of row are annotated rather than silently edited:

  corrected   the spec and the grammar notes disagreed, and the disagreement was
              resolved in favour of the grammar notes (the leading "нэг", phone
              grouping, building ordinals).
  superseded  a project-wide decision overrides the spec line -- case is
              preserved, the verified ABLATIVE table wins, phrasing punctuation
              is kept.

The cases that still fail need a lexicon entry, not code. Those are listed
explicitly so the gap is visible rather than buried in a count.
"""

import json
from pathlib import Path

import pytest

from oron_tts.text import MongolianNormalizer

SPEC = json.loads(
    (Path(__file__).parent / "data" / "normalization_spec.json").read_text(encoding="utf-8")
)

# Cases that cannot pass until data/lexicon/*.tsv is extended by a speaker:
# letter names, foreign words, chemical subscripts, an abbreviation for СБД.
NEEDS_LEXICON = {
    # an abbreviation entry for СБД
    "СБД 1-р хороо",
    # foreign words (юзер, экзампл, ком, гитхаб, репорт) and the Latin letter
    # names the tables are still missing -- g, j, k, l, m, q, u, v, w, y, z
    "https://github.com", "user@example.com", "@bat", "report.pdf",
}

# Both of the earlier open questions are closed. A ratio reads as a fraction --
# "1:2" is the same string as 1/2 -- so there is no score special case; and a
# decimal reads as a fraction too, with no "бүхэл", so the induced place-word
# rule is gone rather than patched.
AMBIGUOUS: set[str] = set()
INDUCED_RULE_CONFLICT: set[str] = set()

EXPECTED_FAILURES = NEEDS_LEXICON | AMBIGUOUS | INDUCED_RULE_CONFLICT


@pytest.fixture(scope="module")
def norm() -> MongolianNormalizer:
    return MongolianNormalizer()


def _score(norm) -> tuple[int, list[str]]:
    passing, failing = 0, []
    for case in SPEC:
        try:
            got = norm.normalize(case["in"], strict=False)
        except Exception:
            failing.append(case["in"])
            continue
        if got == case["out"]:
            passing += 1
        else:
            failing.append(case["in"])
    return passing, failing


@pytest.mark.parametrize(
    "case", [c for c in SPEC if c["in"] not in EXPECTED_FAILURES],
    ids=lambda c: f"{c['cat']}:{c['in'][:20]}",
)
def test_spec_case(norm, case):
    assert norm.normalize(case["in"], strict=False) == case["out"]


def test_conformance_does_not_regress(norm):
    """A floor, not a target. Raise it when cases start passing."""
    passing, _ = _score(norm)
    assert passing >= 80, f"conformance fell to {passing}/{len(SPEC)}"


def test_the_expected_failures_are_still_the_only_failures(norm):
    """So a newly broken case cannot hide inside the allowance."""
    _, failing = _score(norm)
    assert set(failing) <= EXPECTED_FAILURES, (
        f"newly failing: {sorted(set(failing) - EXPECTED_FAILURES)}"
    )


def test_every_allowance_is_still_needed(norm):
    """And so the allowance shrinks as the lexicons are filled."""
    _, failing = _score(norm)
    stale = EXPECTED_FAILURES - set(failing)
    assert not stale, f"these now pass and should leave the allowance: {sorted(stale)}"


def test_superseded_rows_say_why():
    """A row edited away from the spec has to carry its reason."""
    for case in SPEC:
        if "superseded" in case or "corrected" in case:
            assert case.get("superseded") or case.get("corrected")

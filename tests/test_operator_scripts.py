"""The three scripts an operator runs to decide something.

None of them ships in the model, so a defect here is not a corrupted corpus --
but each one exists to *report a number a decision rests on*, and a wrong number
is worse than no number. `measure_refusals` says how much text the normaliser
drops, `attest_forms` says which candidate spelling to ask a speaker about, and
`check_ci_imports` says whether CI will collect at all.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import attest_forms  # noqa: E402
import check_ci_imports as cci  # noqa: E402
import measure_refusals  # noqa: E402

# ── measure_refusals ──────────────────────────────────────────────────────────

def test_a_corpus_is_read_from_its_manifest(tmp_path):
    rows = [{"clip_id": "a", "text": "Сайн байна уу"},
            {"clip_id": "b", "text": "Өнөөдөр сайхан"}]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    assert measure_refusals.from_corpus(tmp_path) == [r["text"] for r in rows]


def test_a_missing_corpus_says_so_rather_than_reporting_zero(tmp_path):
    """A silent empty result would read as "nothing refuses"."""
    with pytest.raises(SystemExit):
        measure_refusals.from_corpus(tmp_path / "absent")


def test_the_sentence_filter_keeps_mongolian_and_drops_the_rest():
    """The sample has to be Mongolian, or the rate measures the wrong language."""
    assert measure_refusals.MN_RE.search("Сайн байна уу")
    assert measure_refusals.MN_RE.search("өнөөдөр")     # ө is outside [а-я]
    assert measure_refusals.MN_RE.search("үзэх")        # so is ү
    assert not measure_refusals.MN_RE.search("hello world 123")


def test_the_cause_pattern_reads_a_real_refusal_message():
    """The breakdown is only as good as this: an unparsed message would land
    every refusal in one unlabelled bucket, which is what happened the first
    time it was run."""
    from oron_tts.text import MongolianNormalizer
    from oron_tts.text.numbers import NumeralSuffixError

    try:
        MongolianNormalizer().normalize("100-д", strict=False)
    except NumeralSuffixError as exc:
        match = measure_refusals.CAUSE_RE.search(str(exc))
    assert match is not None
    assert match.group(1) == "зуу"
    assert match.group(2) == "д"


# ── attest_forms ──────────────────────────────────────────────────────────────

def test_every_ablative_entry_has_candidates_to_compare():
    """A stem with no candidate row cannot be reviewed."""
    from oron_tts.text.numbers import ABLATIVE

    covered = {stem for case in attest_forms.CANDIDATES.values() for stem in case}
    assert covered <= set(ABLATIVE), f"candidate stems not in ABLATIVE: {covered - set(ABLATIVE)}"
    assert len(covered) >= 10


def test_the_candidates_for_a_stem_are_distinct():
    for case, stems in attest_forms.CANDIDATES.items():
        for stem, forms in stems.items():
            assert len(forms) == len(set(forms)), f"{case}/{stem} repeats a candidate"


def test_the_known_homographs_are_flagged():
    """зуун is both hundred and century, and the count is useless without that."""
    assert "зуунд" in attest_forms.HOMOGRAPHS
    assert "century" in attest_forms.HOMOGRAPHS["зуунд"]


def test_counting_runs_on_a_supplied_corpus(tmp_path, capsys):
    """No network: --corpus reads a file."""
    corpus = tmp_path / "mn.txt"
    corpus.write_text("Хоёрт орсон. Гуравт орсон. Хоёрт дахин орлоо.\n", encoding="utf-8")
    sys.argv = ["attest_forms.py", "--corpus", str(corpus)]
    attest_forms.main()
    out = capsys.readouterr().out
    assert "хоёрт=2" in out
    assert "гуравт=1" in out


# ── check_ci_imports ──────────────────────────────────────────────────────────

def test_only_module_scope_imports_are_reported(tmp_path):
    """An import inside a function is deferred and must not be flagged -- that
    is the whole fix this check exists to verify."""
    f = tmp_path / "m.py"
    f.write_text("def go():\n    import torch\n    return torch\n", encoding="utf-8")
    assert cci._module_level_imports(f, "") == set()


def test_a_module_scope_import_is_seen(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import torch\n", encoding="utf-8")
    assert "torch" in cci._module_level_imports(f, "")


def test_a_relative_import_anchors_on_the_package_not_the_module(tmp_path):
    """`from .constants import X` inside pipeline/corpus.py is
    pipeline.constants. Anchoring on the module read every one of those as a
    missing third-party package."""
    f = tmp_path / "m.py"
    f.write_text("from .constants import SAMPLE_RATE\n", encoding="utf-8")
    assert cci._module_level_imports(f, "pipeline") == {"pipeline.constants"}


def test_importorskip_is_recognised_as_a_guard(tmp_path):
    f = tmp_path / "t.py"
    f.write_text('import pytest\ntorch = pytest.importorskip("torch")\n', encoding="utf-8")
    assert "torch" in cci._guarded(f)


def test_the_repository_itself_passes():
    """The check the workflow runs."""
    assert cci.main() == 0

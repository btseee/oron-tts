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


# ── the release card ──────────────────────────────────────────────────────────

def test_the_model_card_states_every_fact_the_other_docs_defer_to_it():
    """Four documents say a fact "belongs in the model card". This checks the
    card actually carries them, rather than each doc assuming another will."""
    card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    for claim, marker in [
        ("the ~8 kHz ceiling", "8 kHz"),
        ("the CER scorer's contamination", "contaminated"),
        ("the absence of a listening test", "No listening test"),
        ("the absence of watermarking", "no watermarking"),
        ("the consent basis", "consented"),
        ("the numeral refusals", "refuses"),
    ]:
        assert marker.lower() in card.lower(), f"model card omits {claim}"


def test_the_model_card_leaves_the_unmeasured_numbers_blank():
    """Publishing it with invented numbers would be worse than not publishing.
    The blanks are the point."""
    card = (ROOT / "docs" / "model-card.md").read_text(encoding="utf-8")
    assert card.count("`<>`") >= 10, "the reporting run's blanks have gone missing"


# ── the EMA trap ──────────────────────────────────────────────────────────────

def test_eval_scores_raw_weights_by_default():
    """Measured on a 30,000-update Mongolian finetune, same checkpoint and
    sentence: use_ema=True gives CER 0.921 (fluent non-words), use_ema=False
    gives 0.026. The EMA had moved 2.78% off the pretrained weights, so it was
    still essentially the base English/Chinese model. The failure is inaudible
    -- it sounds like confident speech -- so the default must be the safe one."""
    src = (ROOT / "scripts" / "eval_mn.py").read_text(encoding="utf-8")
    i = src.index('"--use-ema"')
    decl = src[i:i + 260]
    assert "default=False" in decl, "eval must score raw weights unless asked"


def test_infer_defaults_match_eval():
    """A model scored with raw weights and shipped for inference with EMA would
    sound nothing like its reported CER."""
    src = (ROOT / "oron_tts" / "infer.py").read_text(encoding="utf-8")
    assert "use_ema" in src


def test_reference_split_falls_back_when_empty():
    """A single-narrator corpus has no speaker-disjoint split, so the default
    reference split does not exist and the sweep died with
    'no rows in split validation'. It should fall back and say so."""
    src = (ROOT / "scripts" / "eval_mn.py").read_text(encoding="utf-8")
    assert "def split_sizes(" in src
    i = src.index("args.ref_split = \"validation\" if args.mode == \"select\" else \"test\"")
    block = src[i:i + 900]
    assert "split_sizes(" in block
    assert "withheld" in block, "withheld is the only holdout a 1-speaker corpus has"


def test_checkpoint_retention_survives_a_sweep():
    """keep_last_n_checkpoints was 3: on a 30,000-update run saving every 2,000
    that deleted everything before update 26,000, so the sweep could only pick
    among the last three and the early still-cloning checkpoint was gone."""
    import yaml
    for name in ("oron.yaml",):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        keep = cfg["ckpts"]["keep_last_n_checkpoints"]
        per = cfg["ckpts"]["save_per_updates"]
        assert keep >= 8, f"{name}: keeping only {keep} checkpoints starves the sweep"
        assert keep * per >= 16000, (
            f"{name}: retention spans {keep * per} updates, too narrow to sweep")


def test_tensorboard_is_declared_when_configs_ask_for_it():
    """The configs set logger: tensorboard and F5-TTS's Trainer imports
    SummaryWriter in __init__, so a missing package kills the run before its
    first update."""
    import tomllib

    import yaml
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = " ".join(deps["project"]["dependencies"]).lower()
    for name in ("oron.yaml",):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        if cfg["ckpts"].get("logger") == "tensorboard":
            assert "tensorboard" in declared, f"{name} wants tensorboard; nothing declares it"


def test_exactly_one_training_config_exists():
    """Two configs drift apart. Curriculum stages vary only epochs and learning
    rate, which the orchestrator substitutes -- so there is one file."""
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml"))
    assert configs == ["oron.yaml"], f"expected only oron.yaml, found {configs}"

"""Refusing a training run that is configured wrong.

Every check here covers something that fails *silently*: the run completes, the
loss curve looks plausible, and the model is worse with nothing in the logs to
say why. That is the failure class this whole repo is organised around, and the
config was the one place still relying on an operator remembering.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

yaml = pytest.importorskip("yaml")

from preflight import (  # noqa: E402
    APPENDED,
    check_epochs,
    check_training,
    check_vocab,
)

REAL_VOCAB = ROOT / "data" / "oron_mn_pinyin" / "vocab.txt"


def run(fn, *args):
    problems, notes = [], []
    fn(*args, problems, notes)
    return problems, notes


# ── vocabulary ────────────────────────────────────────────────────────────────

def test_the_shipped_vocabulary_passes():
    """Guards the checker against the real artifact, not a fixture of it."""
    if not REAL_VOCAB.exists():
        pytest.skip("extended vocab not built")
    problems, notes = run(check_vocab, REAL_VOCAB)
    assert not problems, problems
    assert notes


def test_the_base_vocabulary_is_refused(tmp_path):
    """The silent one: unknown ids map to 0, and 0 is the SPACE token, so 4.90%
    of Mongolian characters become spaces with nothing logged."""
    if not REAL_VOCAB.exists():
        pytest.skip("extended vocab not built")
    lines = REAL_VOCAB.read_text(encoding="utf-8").split("\n")
    short = tmp_path / "vocab.txt"
    short.write_text("\n".join(lines[: -1 - len(APPENDED)]) + "\n", encoding="utf-8")
    problems, _ = run(check_vocab, short)
    assert problems and "SPACE" in problems[0]


def test_a_sorted_vocabulary_is_refused(tmp_path):
    """Sorting misaligns all 2545 pretrained embedding rows."""
    if not REAL_VOCAB.exists():
        pytest.skip("extended vocab not built")
    lines = REAL_VOCAB.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out = tmp_path / "vocab.txt"
    out.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    problems, _ = run(check_vocab, out)
    assert problems and "misaligns" in problems[0]


def test_a_missing_vocabulary_is_refused(tmp_path):
    problems, _ = run(check_vocab, tmp_path / "nope.txt")
    assert problems


# ── epochs ────────────────────────────────────────────────────────────────────

def _data(tmp_path: Path, clips=4000, seconds=6.0) -> Path:
    d = tmp_path / "oron_mn_pinyin"
    d.mkdir(parents=True, exist_ok=True)
    (d / "duration.json").write_text(json.dumps({"duration": [seconds] * clips}),
                                     encoding="utf-8")
    if REAL_VOCAB.exists():
        shutil.copy(REAL_VOCAB, d / "vocab.txt")
    return d


def _config(**optim):
    return {"datasets": {"name": "oron_mn", "tokenizer": "pinyin",
                         "batch_size_per_gpu": 13000, "max_samples": 32},
            "optim": {"epochs": 51, "grad_accumulation_steps": 1, **optim},
            "ckpts": {"log_samples": True}}


def test_a_stale_epochs_value_is_refused(tmp_path):
    """epochs sets the LR decay length, not a stopping condition. The shipped
    value is a placeholder computed for a corpus that does not exist."""
    data = _data(tmp_path)
    problems, _ = run(check_epochs, _config(epochs=51), data)
    assert problems
    assert "LR decay length" in problems[0]
    assert "compute_epochs" in problems[0]


def test_the_computed_epochs_value_passes(tmp_path):
    """Whatever the checker says is right must satisfy the checker."""
    import math

    from compute_epochs import HOP_LENGTH, SAMPLE_RATE, pack

    data = _data(tmp_path)
    durations = json.loads((data / "duration.json").read_text(encoding="utf-8"))["duration"]
    per_epoch = pack([int(d * SAMPLE_RATE / HOP_LENGTH) for d in durations], 13000, 32)
    want = math.ceil(40_000 / per_epoch)
    problems, notes = run(check_epochs, _config(epochs=want), data)
    assert not problems, problems
    assert notes


def test_a_missing_dataset_is_refused(tmp_path):
    problems, _ = run(check_epochs, _config(), tmp_path / "absent")
    assert problems and "build_f5_dataset" in problems[0]


def test_an_unset_epochs_is_refused(tmp_path):
    problems, _ = run(check_epochs, _config(epochs=0), _data(tmp_path))
    assert problems and "unset" in problems[0]


# ── training config ───────────────────────────────────────────────────────────

def test_gradient_accumulation_above_one_is_refused(tmp_path):
    """scheduler.step() fires per batch, so >1 compresses the LR schedule."""
    problems, _ = run(check_training, _config(grad_accumulation_steps=2), _data(tmp_path))
    assert any("compressed" in p for p in problems)


def test_a_non_pinyin_tokenizer_is_refused(tmp_path):
    """load_dataset() and get_tokenizer() interpolate {name}_{tokenizer}, so
    anything else sends them to different directories."""
    cfg = _config()
    cfg["datasets"]["tokenizer"] = "custom"
    problems, _ = run(check_training, cfg, _data(tmp_path))
    assert any("different directories" in p for p in problems)


def test_a_dataset_name_that_does_not_match_the_directory_is_refused(tmp_path):
    cfg = _config()
    cfg["datasets"]["name"] = "something_else"
    problems, _ = run(check_training, cfg, _data(tmp_path))
    assert any("prepared data is in" in p for p in problems)


def test_log_samples_off_is_refused(tmp_path):
    """The previous project ran 500 epochs and logged no audio at all."""
    cfg = _config()
    cfg["ckpts"]["log_samples"] = False
    problems, _ = run(check_training, cfg, _data(tmp_path))
    assert any("log_samples" in p for p in problems)


def test_the_shipped_config_only_fails_on_epochs(tmp_path):
    """Everything except the placeholder should already be right."""
    cfg = yaml.safe_load((ROOT / "configs" / "oron.yaml").read_text(encoding="utf-8"))
    problems, _ = run(check_training, cfg, _data(tmp_path))
    assert not problems, problems


def test_epoch_check_honours_a_declared_update_target():
    """epochs sets the LR decay horizon, so it has to match the run actually
    intended. Hardcoding 40,000 made preflight refuse a deliberately shorter
    run on a small corpus: 160 epochs x 188 updates = 30,080, correct for a
    30,000-update target, was reported as 25% drift against 40,000."""
    problems: list[str] = []
    notes: list[str] = []
    config = {"datasets": {"batch_size_per_gpu": 9600, "max_samples": 32},
              "optim": {"epochs": 160}}
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        # 2,996 clips averaging 6.3 s: what the real MBSpeech corpus packs to.
        (data / "duration.json").write_text(
            json.dumps({"duration": [6.3] * 2996}), encoding="utf-8")

        check_epochs(config, data, problems, notes, target_updates=30_000)
        assert not problems, f"a matching target must pass: {problems}"

        problems.clear()
        notes.clear()
        check_epochs(config, data, problems, notes, target_updates=40_000)
        assert problems, "real drift against the declared target must still fail"
        assert "40,000" in problems[0]

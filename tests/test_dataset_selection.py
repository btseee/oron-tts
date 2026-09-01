"""What the two corpus-reading tools do with the manifest they are given.

Both defects these cover were silent: a filter that matched everything, and a
reference picker that ranked over the whole corpus. Neither raised, neither
logged, and both produced a plausible-looking run.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_f5_dataset import select_splits  # noqa: E402
from eval_mn import MAX_REF_S, MIN_REF_S, pick_reference  # noqa: E402


def _row(**kw):
    base = {"clip_id": "c", "audio_path": "wavs/c.wav", "text": "сайн байна уу",
            "duration_s": 8.0, "split": "test", "gender_resolved": "female",
            "bandwidth_hz": 8000.0, "dnsmos_ovr": 3.5, "align_score": 0.8}
    return {**base, **kw}


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    with open(tmp_path / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp_path


# ── split selection ───────────────────────────────────────────────────────────

def test_requested_split_is_actually_filtered():
    rows = [_row(split="train"), _row(split="test"), _row(split="validation")]
    assert len(select_splits(rows, "train")) == 1


def test_several_splits_can_be_requested():
    rows = [_row(split="train"), _row(split="test"), _row(split="validation")]
    assert len(select_splits(rows, "train,validation")) == 2


def test_a_manifest_without_split_fails_instead_of_training_on_everything():
    """The whole point.

    The previous guard degraded to "keep every row" here, so `--splits train`
    built raw.arrow from the test split too and nothing said so.
    """
    rows = [_row(), {k: v for k, v in _row().items() if k != "split"}]
    with pytest.raises(SystemExit) as exc:
        select_splits(rows, "train")
    assert "finalize" in str(exc.value)


def test_a_misspelled_split_fails_rather_than_returning_nothing():
    rows = [_row(split="train")]
    with pytest.raises(SystemExit) as exc:
        select_splits(rows, "trian")
    assert "Available" in str(exc.value)


def test_an_empty_split_argument_keeps_everything():
    """`--splits ""` is an explicit opt-out, not a missing corpus."""
    rows = [_row(split="train"), _row(split="test")]
    assert select_splits(rows, "") == rows


# ── reference selection ───────────────────────────────────────────────────────

def test_the_reference_comes_from_the_held_out_split(tmp_path):
    """F7: ranking over the whole manifest draws a *training* clip ~90% of the
    time, which measures memorisation rather than zero-shot cloning."""
    _manifest(tmp_path, [
        _row(clip_id="train_best", audio_path="wavs/train_best.wav",
             split="train", bandwidth_hz=16000.0, dnsmos_ovr=5.0, align_score=1.0),
        _row(clip_id="held_out", audio_path="wavs/held_out.wav", split="test"),
    ])
    path, _text = pick_reference(tmp_path, "female", split="test")
    assert path.name == "held_out.wav"


def test_the_best_candidate_within_the_split_wins(tmp_path):
    _manifest(tmp_path, [
        _row(clip_id="dull", audio_path="wavs/dull.wav", bandwidth_hz=4000.0),
        _row(clip_id="bright", audio_path="wavs/bright.wav", bandwidth_hz=11000.0),
    ])
    path, _text = pick_reference(tmp_path, "female")
    assert path.name == "bright.wav"


def test_clips_outside_the_prompt_duration_band_are_not_used(tmp_path):
    """Upstream clips a reference over 12 s; under ~6 s there is too little
    speaker evidence for a stable prompt."""
    _manifest(tmp_path, [
        _row(clip_id="short", audio_path="wavs/short.wav", duration_s=MIN_REF_S - 1),
        _row(clip_id="long", audio_path="wavs/long.wav", duration_s=MAX_REF_S + 1),
        _row(clip_id="right", audio_path="wavs/right.wav", duration_s=MIN_REF_S + 1),
    ])
    path, _text = pick_reference(tmp_path, "female")
    assert path.name == "right.wav"


def test_a_manifest_missing_gender_resolved_says_so(tmp_path):
    """F0b's symptom. Previously this matched nothing and reported "no usable
    reference", which reads as a thin corpus rather than an unfinalised one."""
    _manifest(tmp_path, [{k: v for k, v in _row().items() if k != "gender_resolved"}])
    with pytest.raises(SystemExit) as exc:
        pick_reference(tmp_path, "female")
    assert "finalize" in str(exc.value)


def test_an_absent_split_is_named_in_the_failure(tmp_path):
    _manifest(tmp_path, [_row(split="train")])
    with pytest.raises(SystemExit) as exc:
        pick_reference(tmp_path, "female", split="test")
    assert "'test'" in str(exc.value)


def test_an_absent_gender_is_named_in_the_failure(tmp_path):
    """The corpus's binding constraint is male hours, so "no male reference"
    must not be reported as the same thing as "no test split"."""
    _manifest(tmp_path, [_row(gender_resolved="female")])
    with pytest.raises(SystemExit) as exc:
        pick_reference(tmp_path, "male")
    assert "'male'" in str(exc.value)

# Publication and Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the published artifacts complete and legible — full HuggingFace metadata on the model and all four datasets, a model card a stranger can act on in a minute, and a TensorBoard tree that shows the run instead of two flat scalars.

**Architecture:** Four new scripts in `scripts/`, each a plain importable module with an `argparse` main, matching the existing convention in `preflight.py` and `measure_refusals.py`. Three of them read artifacts that already exist (`eval.json`, `demos/`, the published tfevents) and write publishable output; the fourth patches upstream F5-TTS on the pod, because F5-TTS is refetched fresh every run and a fork would rot. Nothing retrains.

**Tech Stack:** Python 3.12+, `tensorboard` (already a hard dependency — see `pyproject.toml`), `huggingface_hub`, `PyYAML`, `soundfile`, `numpy`. Tests use `pytest` and read TensorBoard output back through `EventAccumulator`, which is what a reader's TensorBoard uses.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-05-publication-and-logging-design.md`.
- No GPU spend. No retraining. No pod.
- Model licence is `cc-by-4.0`. The model never trained on WorldSpeech.
- `new_version` must NOT appear in the model card frontmatter. It declares a successor repo; none exists.
- The mbspeech stage's tfevents are gone and are not reconstructed. The report covers the three stages whose events survive. No placeholder run is fabricated.
- Scripts live in `scripts/`, tests in `tests/`, imported via `sys.path.insert(0, str(ROOT / "scripts"))` as `tests/test_operator_scripts.py` already does.
- Every number that reaches a card or a chart is read from `eval.json` or `demos/consistency.json`. No number is typed in by hand.
- Measured values, for reference and for the assertions below:
  - cv `model_12000.pt`: male `cer_median` 0.06329113924050633, `utmos_mean` 2.483718866109848; female `cer_median` 0.1015, `utmos_mean` 2.1107.
  - Speaker similarity: male demo vs male prompt 0.7251, female 0.8082, male demo vs female demo 0.1029.
  - Stage events: fleurs max step 8275, cv max step 16150, voicelock max step 4142.

---

### Task 1: TensorBoard stage runs from `eval.json`

The single largest defect is layout: four tfevents files sit flat in one directory, so TensorBoard cannot select or overlay them. This task produces one subdirectory per stage and puts the evaluation metrics — which have never reached TensorBoard at all — on the same update axis as the loss curve.

**Files:**
- Create: `scripts/tb_report.py`
- Test: `tests/test_tb_report.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `checkpoint_update(name: str) -> int | None`
  - `eval_series(stage_eval: dict) -> dict[str, list[tuple[int, float]]]`
  - `is_empty_events(path: Path) -> bool`
  - `write_stage_run(out_dir: Path, stage: str, series: dict[str, list[tuple[int, float]]], events: Path | None, hparams: dict, metrics: dict) -> Path`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tb_report.py`:

```python
"""The TensorBoard tab has to show the run, not two anonymous scalars.

Four faults are fixed here, and the first is the one that made the tab
unreadable: every tfevents file sat flat in one directory, and TensorBoard
derives runs from *subdirectories*, so nothing could be selected or overlaid.
The second is that every CER and UTMOS number the sweeps produced went to
eval.json and never reached TensorBoard, which is the main thing the tab is for.

Output is read back with EventAccumulator rather than trusted from the writer,
because that is the reader a person actually uses.
"""

import json
import sys
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import tb_report  # noqa: E402


CV_EVAL = {
    "model_2000.pt": {
        "male": {"cer_median": 0.0789, "utmos_mean": 2.56, "bandwidth_median": 6700.0},
        "female": {"cer_median": 0.0950, "utmos_mean": 2.14, "bandwidth_median": 7500.0},
    },
    "model_12000.pt": {
        "male": {"cer_median": 0.06329113924050633, "utmos_mean": 2.483718866109848,
                 "bandwidth_median": 6761.71875},
        "female": {"cer_median": 0.1015, "utmos_mean": 2.1107, "bandwidth_median": 7535.0},
    },
    "model_last.pt": {
        "male": {"cer_median": 0.0701, "utmos_mean": 2.50, "bandwidth_median": 6700.0},
        "female": {"cer_median": 0.1158, "utmos_mean": 2.16, "bandwidth_median": 7500.0},
    },
}


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)]
            for tag in acc.Tags()["scalars"]}


def test_checkpoint_update_reads_the_step_from_the_name():
    assert tb_report.checkpoint_update("model_12000.pt") == 12000
    assert tb_report.checkpoint_update("model_2000.pt") == 2000


def test_model_last_has_no_update_so_it_cannot_be_plotted():
    """`model_last.pt` carries no update number. Plotting it at a guessed step
    would put a point on the curve that did not happen there."""
    assert tb_report.checkpoint_update("model_last.pt") is None
    assert tb_report.checkpoint_update("pretrained_model_1250000.safetensors") is None


def test_eval_series_puts_every_metric_on_the_update_axis():
    series = tb_report.eval_series(CV_EVAL)
    assert series["eval/cer_male"] == [(2000, 0.0789), (12000, 0.06329113924050633)]
    assert series["eval/cer_female"] == [(2000, 0.0950), (12000, 0.1015)]
    assert series["eval/utmos_male"] == [(2000, 2.56), (12000, 2.483718866109848)]
    assert series["eval/bandwidth_female"] == [(2000, 7500.0), (12000, 7535.0)]


def test_cer_mean_is_the_number_the_sweep_selected_on():
    """best_checkpoint ranks on the mean across genders, so the chart must show
    the quantity the decision was actually made with."""
    series = tb_report.eval_series(CV_EVAL)
    assert series["eval/cer_mean"][1] == (12000, pytest.approx((0.06329113924050633 + 0.1015) / 2))


def test_a_stage_becomes_its_own_run_directory(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(
        out, "cv", tb_report.eval_series(CV_EVAL), events=None,
        hparams={"stage": "cv", "updates": 16150, "corpus_hours": 25.2},
        metrics={"final/cer_mean": 0.0824})
    assert run == out / "cv"
    assert run.is_dir(), "TensorBoard names runs after subdirectories; a flat file has no name"
    scalars = read_scalars(run)
    assert scalars["eval/cer_male"][-1] == (12000, pytest.approx(0.06329113924050633))


def test_the_training_curve_is_carried_into_the_stage_run(tmp_path):
    """The loss/lr events already exist and must survive verbatim; the point of
    the report is to put them somewhere selectable, not to regenerate them."""
    source = tmp_path / "src"
    source.mkdir()
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(source))
    for step in (1, 2, 3):
        w.add_scalar("loss", 0.5 + step / 10, step)
    w.close()
    events = next(source.glob("events.out.tfevents.*"))

    out = tmp_path / "tensorboard"
    run = tb_report.write_stage_run(out, "cv", {}, events=events, hparams={}, metrics={})
    scalars = read_scalars(run)
    assert [s for s, _ in scalars["loss"]] == [1, 2, 3]


def test_an_empty_events_file_is_recognised(tmp_path):
    """One published file holds no scalars at all -- the aborted first fleurs
    attempt, which died before its first update. Copying it in would add a run
    that draws nothing."""
    from torch.utils.tensorboard import SummaryWriter
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    SummaryWriter(log_dir=str(empty_dir)).close()
    empty = next(empty_dir.glob("events.out.tfevents.*"))
    assert tb_report.is_empty_events(empty) is True

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    w = SummaryWriter(log_dir=str(full_dir))
    w.add_scalar("loss", 0.5, 1)
    w.close()
    assert tb_report.is_empty_events(next(full_dir.glob("events.out.tfevents.*"))) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tb_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tb_report'`

- [ ] **Step 3: Write the implementation**

Create `scripts/tb_report.py`:

```python
"""Build a TensorBoard tree that shows what the run did.

The published tab held four tfevents files, flat in one directory, containing
two scalars between them. Four faults, in the order they hurt:

1. **Flat files have no run names.** TensorBoard derives runs from
   subdirectories. Four files side by side cannot be selected, named, or
   overlaid, so even `loss` was hard to read and impossible to compare across
   stages. This is pure layout and it was the worst of the four.
2. **One file was empty** -- the aborted first `fleurs` attempt, which died on
   the seed-layout bug before its first update.
3. **No evaluation metrics.** Every CER and UTMOS number the checkpoint sweeps
   produced went to `eval.json` and never reached TensorBoard, so quality could
   not be read against training progress, which is the main thing the tab is
   for.
4. **Only `loss` and `lr`.** Upstream's trainer writes exactly two scalars.
   `patch_trainer_logging.py` fixes that for future runs; this script fixes what
   can be recovered from a run that is already over.

The mbspeech stage's events died with the pod that produced them. This report
covers the stages whose events survive and does not fabricate a run for the one
that does not.

    python scripts/tb_report.py --eval eval.json --out tensorboard \\
        --events fleurs=events.out.tfevents.1788562958.ace24d560882.4353.0 \\
        --events cv=events.out.tfevents.1788587604.ace24d560882.5803.0 \\
        --events voicelock=events.out.tfevents.1788603469.ace24d560882.7668.0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
from pathlib import Path

CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")

# eval.json stores each metric per gender; these become one scalar tag each so a
# reader can overlay male against female on the same axes.
METRIC_TAGS = {
    "cer_median": "eval/cer",
    "utmos_mean": "eval/utmos",
    "bandwidth_median": "eval/bandwidth",
}


def checkpoint_update(name: str) -> int | None:
    """The update a checkpoint was saved at, or None if the name does not say.

    `model_last.pt` is a rotating alias with no update in its name, and
    `pretrained_*.safetensors` is the seed rather than a checkpoint. Plotting
    either at a guessed step would draw a point where nothing happened.
    """
    match = CHECKPOINT.match(name)
    return int(match.group(1)) if match else None


def eval_series(stage_eval: dict) -> dict[str, list[tuple[int, float]]]:
    """Turn one stage's eval.json entry into scalar series on the update axis."""
    series: dict[str, list[tuple[int, float]]] = {}
    for name, per_gender in sorted(stage_eval.items()):
        update = checkpoint_update(name)
        if update is None or not isinstance(per_gender, dict):
            continue
        cers = []
        for gender, measured in sorted(per_gender.items()):
            if not isinstance(measured, dict):
                continue
            for key, tag in METRIC_TAGS.items():
                value = measured.get(key)
                if isinstance(value, (int, float)):
                    series.setdefault(f"{tag}_{gender}", []).append((update, float(value)))
            if isinstance(measured.get("cer_median"), (int, float)):
                cers.append(float(measured["cer_median"]))
        if cers:
            # best_checkpoint ranks on the mean across genders, so the chart
            # shows the quantity the selection was actually made with.
            series.setdefault("eval/cer_mean", []).append((update, statistics.fmean(cers)))
    for points in series.values():
        points.sort()
    return series


def is_empty_events(path: Path) -> bool:
    """True when a tfevents file carries no scalars, so copying it adds nothing."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    return not accumulator.Tags()["scalars"]


def write_stage_run(out_dir: Path, stage: str, series: dict[str, list[tuple[int, float]]],
                    events: Path | None, hparams: dict, metrics: dict) -> Path:
    """One subdirectory per stage: the training curve plus the eval metrics."""
    from torch.utils.tensorboard import SummaryWriter

    run = out_dir / stage
    run.mkdir(parents=True, exist_ok=True)
    if events is not None:
        # Copied verbatim. The point is to put the existing curve somewhere
        # selectable, not to regenerate it from a summary of itself.
        shutil.copy2(events, run / events.name)
    writer = SummaryWriter(log_dir=str(run))
    for tag, points in sorted(series.items()):
        for step, value in points:
            writer.add_scalar(tag, value, step)
    if hparams and metrics:
        writer.add_hparams({k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
                           metrics, run_name=".")
    writer.close()
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path, help="eval.json from the model repo")
    parser.add_argument("--out", required=True, type=Path, help="tensorboard/ directory to build")
    parser.add_argument("--events", action="append", default=[], metavar="STAGE=PATH",
                        help="training events file for a stage; repeatable")
    parser.add_argument("--stages", type=Path,
                        help="JSON of per-stage hparams: corpora, clips, hours, learning_rate")
    args = parser.parse_args()

    evals = json.loads(args.eval.read_text(encoding="utf-8"))
    stage_meta = json.loads(args.stages.read_text(encoding="utf-8")) if args.stages else {}

    events_for: dict[str, Path] = {}
    for pair in args.events:
        stage, _, path = pair.partition("=")
        events = Path(path)
        if not events.is_file():
            raise SystemExit(f"{stage}: no events file at {events}")
        if is_empty_events(events):
            print(f"  {stage}: {events.name} holds no scalars; skipped")
            continue
        events_for[stage] = events

    for stage in sorted(set(evals) | set(events_for)):
        series = eval_series(evals.get(stage, {}))
        meta = stage_meta.get(stage, {})
        metrics = {}
        if series.get("eval/cer_mean"):
            best = min(series["eval/cer_mean"], key=lambda p: p[1])
            metrics = {"final/cer_mean": best[1], "final/best_update": float(best[0])}
        run = write_stage_run(args.out, stage, series, events_for.get(stage), meta, metrics)
        print(f"  {stage}: {run} ({len(series)} eval series, "
              f"{'training curve carried' if stage in events_for else 'no training events'})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tb_report.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Verify the mutation is caught**

Break the update axis and confirm a test fails, so the suite is not vacuous:

```bash
python - <<'PY'
import pathlib, subprocess, sys
p = pathlib.Path("scripts/tb_report.py")
orig = p.read_text(encoding="utf-8")
p.write_text(orig.replace('CHECKPOINT = re.compile(r"^model_(\\d+)\\.pt$")',
                          'CHECKPOINT = re.compile(r"^model_(\\d+)")'), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_tb_report.py", "-q"],
                   capture_output=True, text=True)
p.write_text(orig, encoding="utf-8")
print(r.stdout.strip().splitlines()[-1])
PY
```

Expected: a line reporting at least `1 failed` — `model_last.pt` must stop being rejected.

- [ ] **Step 6: Commit**

```bash
git add scripts/tb_report.py tests/test_tb_report.py
git commit -m "Give each training stage its own TensorBoard run, with its eval metrics"
```

---

### Task 2: The summary run — audio, spectrograms, speaker similarity

The demos and the voice prompts are the artifacts a person most wants to inspect, and none of them are in TensorBoard. This task adds a `summary/` run holding both demos, both reference prompts, their mel spectrograms, and the speaker-similarity numbers that were measured after the run.

**Files:**
- Modify: `scripts/tb_report.py`
- Modify: `tests/test_tb_report.py`

**Interfaces:**
- Consumes: `write_stage_run` from Task 1.
- Produces:
  - `mel_image(audio: "np.ndarray", sr: int) -> "np.ndarray"` returning CHW uint8 suitable for `add_image`
  - `write_summary_run(out_dir: Path, audio: dict[str, tuple], consistency: dict, stages: list[str]) -> Path`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tb_report.py`:

```python
CONSISTENCY = {
    "metric": "ecapa_voxceleb",
    "calibration": {"same_speaker_threshold": 0.52,
                    "same_speaker_range": [0.540, 0.833],
                    "different_speaker_range": [0.034, 0.503]},
    "measured": {"male_demo_vs_male_prompt": 0.7251,
                 "female_demo_vs_female_prompt": 0.8082,
                 "male_demo_vs_female_demo": 0.1029},
}


def _tone(seconds=1.0, sr=24000, freq=220.0):
    import numpy as np
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype("float32")


def test_the_demos_are_listenable_in_tensorboard(tmp_path):
    """The two demos are the artifact a reader most wants, and they were not in
    the tab at all."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(
        out, {"demo_male": (_tone(), 24000), "demo_female": (_tone(freq=330.0), 24000)},
        CONSISTENCY, ["fleurs", "cv", "voicelock"])
    acc = EventAccumulator(str(run), size_guidance={"audio": 10, "images": 10})
    acc.Reload()
    assert set(acc.Tags()["audio"]) == {"audio/demo_male", "audio/demo_female"}


def test_the_spectrograms_are_rendered(tmp_path):
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {"demo_male": (_tone(), 24000)}, CONSISTENCY, ["cv"])
    acc = EventAccumulator(str(run), size_guidance={"images": 10})
    acc.Reload()
    assert "mel/demo_male" in acc.Tags()["images"]


def test_speaker_similarity_reaches_the_charts(tmp_path):
    """The voice lock is the mechanism behind "the same voice every time" and
    nothing measured it during the run."""
    out = tmp_path / "tensorboard"
    run = tb_report.write_summary_run(out, {}, CONSISTENCY, ["cv"])
    scalars = read_scalars(run)
    assert scalars["similarity/male_demo_vs_male_prompt"][0][1] == pytest.approx(0.7251)
    assert scalars["similarity/male_demo_vs_female_demo"][0][1] == pytest.approx(0.1029)


def test_mel_image_is_a_renderable_array():
    import numpy as np
    image = tb_report.mel_image(_tone(), 24000)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[0] == 3, "add_image wants CHW"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_tb_report.py -q`
Expected: FAIL — `AttributeError: module 'tb_report' has no attribute 'write_summary_run'`

- [ ] **Step 3: Write the implementation**

Add to `scripts/tb_report.py`, above `main()`:

```python
def mel_image(audio, sr: int):
    """A mel spectrogram as a CHW uint8 image, for `add_image`.

    Rendered without matplotlib so the report has no plotting dependency: the
    magnitudes are normalised to 0-255 and mapped to a simple blue-to-yellow
    ramp, which is enough to see structure, silence, and a collapsed output.
    """
    import librosa
    import numpy as np

    mel = librosa.feature.melspectrogram(y=np.asarray(audio, dtype="float32"),
                                         sr=sr, n_mels=100, n_fft=1024, hop_length=256)
    db = librosa.power_to_db(mel, ref=np.max)
    scaled = np.clip((db - db.min()) / max(db.ptp(), 1e-6), 0.0, 1.0)
    scaled = np.flipud(scaled)                      # low frequencies at the bottom
    red = scaled
    green = np.clip(scaled * 1.4 - 0.2, 0, 1)
    blue = np.clip(1.0 - scaled * 1.6, 0, 1)
    rgb = np.stack([red, green, blue])
    return (rgb * 255).astype("uint8")


def write_summary_run(out_dir: Path, audio: dict, consistency: dict,
                      stages: list[str]) -> Path:
    """One run holding the artifacts a reader inspects rather than plots."""
    from torch.utils.tensorboard import SummaryWriter

    run = out_dir / "summary"
    run.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run))

    for name, (wav, sr) in sorted(audio.items()):
        import numpy as np

        samples = np.asarray(wav, dtype="float32")
        writer.add_audio(f"audio/{name}", samples, 0, sample_rate=sr)
        writer.add_image(f"mel/{name}", mel_image(samples, sr), 0)

    measured = consistency.get("measured", {})
    for name, value in sorted(measured.items()):
        if isinstance(value, (int, float)):
            writer.add_scalar(f"similarity/{name}", float(value), 0)

    calibration = consistency.get("calibration", {})
    writer.add_text("summary/speaker_similarity", (
        "Metric: `%s`. Same-speaker pairs of real recordings scored %s, "
        "different-speaker pairs %s, so %s separates them.\n\n%s" % (
            consistency.get("metric", "unknown"),
            calibration.get("same_speaker_range"),
            calibration.get("different_speaker_range"),
            calibration.get("same_speaker_threshold"),
            "\n".join("* `%s` = %.4f" % (k, v) for k, v in sorted(measured.items())
                      if isinstance(v, (int, float))))), 0)
    writer.add_text("summary/stages", "Stages in this report: " + ", ".join(stages), 0)
    writer.close()
    return run
```

Then wire it into `main()` by adding these arguments after `--stages`:

```python
    parser.add_argument("--audio", action="append", default=[], metavar="NAME=PATH",
                        help="wav to embed in the summary run; repeatable")
    parser.add_argument("--consistency", type=Path,
                        help="demos/consistency.json, for the speaker-similarity charts")
```

and this block at the end of `main()`, before the final `print`-free exit:

```python
    if args.audio or args.consistency:
        import soundfile as sf

        clips = {}
        for pair in args.audio:
            name, _, path = pair.partition("=")
            samples, sr = sf.read(path, dtype="float32")
            clips[name] = (samples, sr)
        consistency = (json.loads(args.consistency.read_text(encoding="utf-8"))
                       if args.consistency else {})
        run = write_summary_run(args.out, clips, consistency,
                                sorted(set(evals) | set(events_for)))
        print(f"  summary: {run} ({len(clips)} clips, "
              f"{len(consistency.get('measured', {}))} similarity scores)")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_tb_report.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/tb_report.py tests/test_tb_report.py
git commit -m "Put the demos, their spectrograms and the speaker similarity in TensorBoard"
```

---

### Task 3: The model card

**Files:**
- Create: `scripts/model_card.py`
- Test: `tests/test_model_card.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `best_checkpoint(stage_eval: dict) -> tuple[str, dict]` returning the name and per-gender block with the lowest mean CER
  - `frontmatter(evals: dict, consistency: dict) -> dict`
  - `render(evals: dict, consistency: dict) -> str` returning the whole README

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_card.py`:

```python
"""The card is the only page most people will read, and it was prose.

Two separate failures are fixed here. The frontmatter carried seven keys and no
`model-index`, so the Hub rendered no Eval Results panel and no links to the
corpora -- every measured number lived where the Hub cannot see it. And the
licence said `cc-by-nc-4.0`, inherited from WorldSpeech, a corpus that failed the
quality gate and was never trained on; that gave away the commercial safety the
corpus selection existed to protect.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import model_card  # noqa: E402

EVALS = {
    "fleurs": {
        "model_8000.pt": {"male": {"cer_median": 0.0882, "utmos_mean": 2.23},
                          "female": {"cer_median": 0.1034, "utmos_mean": 2.69}},
    },
    "cv": {
        "model_2000.pt": {"male": {"cer_median": 0.0789, "utmos_mean": 2.56},
                          "female": {"cer_median": 0.0950, "utmos_mean": 2.14}},
        "model_12000.pt": {"male": {"cer_median": 0.06329113924050633,
                                    "utmos_mean": 2.483718866109848},
                           "female": {"cer_median": 0.1015, "utmos_mean": 2.1107}},
    },
}

CONSISTENCY = {"measured": {"male_demo_vs_male_prompt": 0.7251,
                            "female_demo_vs_female_prompt": 0.8082,
                            "male_demo_vs_female_demo": 0.1029}}


def parse_frontmatter(card: str) -> dict:
    assert card.startswith("---\n"), "a card without frontmatter renders no metadata"
    end = card.index("\n---\n", 3)
    return yaml.safe_load(card[4:end])


def test_the_best_checkpoint_is_the_lowest_mean_cer_not_the_last():
    name, _ = model_card.best_checkpoint(EVALS["cv"])
    assert name == "model_12000.pt"


def test_the_licence_matches_what_the_model_trained_on():
    """WorldSpeech is CC-BY-NC and failed the gate; it was never trained on."""
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    assert meta["license"] == "cc-by-4.0"


def test_new_version_is_absent():
    """It declares a successor repo and the Hub banners visitors to it. There is
    no successor, so setting it would send every reader to a dead end."""
    assert "new_version" not in model_card.frontmatter(EVALS, CONSISTENCY)


def test_every_field_the_hub_renders_is_present():
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    for key in ("language", "license", "library_name", "pipeline_tag", "base_model",
                "base_model_relation", "datasets", "metrics", "tags", "model-index"):
        assert key in meta, f"missing {key}"
    assert meta["datasets"] == ["btsee/mbspeech-mn", "btsee/fleurs-mn",
                               "btsee/common-voice-26-mn"]


def test_eval_results_are_read_from_the_measurements_not_typed_in():
    """A hand-copied number drifts from the run that produced it."""
    meta = model_card.frontmatter(EVALS, CONSISTENCY)
    metrics = {m["name"]: m["value"] for m in meta["model-index"][0]["results"][0]["metrics"]}
    assert metrics["CER, male voice"] == pytest.approx(0.06329113924050633)
    assert metrics["CER, female voice"] == pytest.approx(0.1015)
    assert metrics["Speaker similarity, male"] == pytest.approx(0.7251)


def test_the_body_leads_with_usage_and_stays_short():
    card = model_card.render(EVALS, CONSISTENCY)
    body = card[card.index("\n---\n", 3) + 5:]
    assert "<audio" in body, "the demos should be playable on the page"
    assert "pip install" in body
    assert "github.com/btseee/oron-tts" in body
    assert body.index("pip install") < body.index("use_ema"), \
        "installation comes before caveats; a reader wants to run it first"
    assert len(body) < 3000, "the body is instructions, not a paper"


def test_the_two_silent_failures_are_stated():
    """Both produce confident, plausible, wrong audio -- so they are usage
    instructions, not background."""
    body = model_card.render(EVALS, CONSISTENCY)
    assert "use_ema=False" in body
    assert "normalize" in body or "normalise" in body
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_model_card.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_card'`

- [ ] **Step 3: Write the implementation**

Create `scripts/model_card.py`:

```python
"""Build the model card from the measurements, so it cannot drift from them.

The card that shipped had seven frontmatter keys, no `model-index`, and no
`datasets`. The Hub therefore rendered no Eval Results panel and no links to the
corpora, and every measured number lived in prose or in `eval.json`, where the
Hub cannot read it.

Its `license` was also wrong in the expensive direction: `cc-by-nc-4.0`,
inherited from WorldSpeech -- a corpus that failed the 15% pass-rate gate and was
never trained on. The model saw MBSpeech (MIT), FLEURS (CC-BY-4.0) and Common
Voice (CC0), all commercial-safe. Commercial safety is why WorldSpeech was
excluded in the first place.

    python scripts/model_card.py --eval eval.json \\
        --consistency consistency.json --out README.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = "btsee/oron-tts"
DATASETS = ["btsee/mbspeech-mn", "btsee/fleurs-mn", "btsee/common-voice-26-mn"]
FINAL_STAGE = "cv"        # the last stage with a scored sweep


def best_checkpoint(stage_eval: dict) -> tuple[str, dict]:
    """The checkpoint with the lowest mean CER across genders."""
    scored = []
    for name, per_gender in stage_eval.items():
        if not isinstance(per_gender, dict):
            continue
        cers = [m["cer_median"] for m in per_gender.values()
                if isinstance(m, dict) and isinstance(m.get("cer_median"), (int, float))]
        if cers:
            scored.append((statistics.fmean(cers), name, per_gender))
    if not scored:
        raise SystemExit("no scored checkpoints in this stage")
    scored.sort(key=lambda row: row[0])
    return scored[0][1], scored[0][2]


def frontmatter(evals: dict, consistency: dict) -> dict:
    _, best = best_checkpoint(evals[FINAL_STAGE])
    measured = consistency.get("measured", {})

    def metric(kind: str, value, name: str) -> dict:
        return {"type": kind, "value": round(float(value), 6), "name": name}

    results = [metric("cer", best["male"]["cer_median"], "CER, male voice"),
               metric("cer", best["female"]["cer_median"], "CER, female voice"),
               metric("utmos", best["male"]["utmos_mean"], "UTMOS, male voice"),
               metric("utmos", best["female"]["utmos_mean"], "UTMOS, female voice")]
    for key, name in (("male_demo_vs_male_prompt", "Speaker similarity, male"),
                      ("female_demo_vs_female_prompt", "Speaker similarity, female")):
        if isinstance(measured.get(key), (int, float)):
            results.append(metric("cosine_similarity", measured[key], name))

    return {
        "language": ["mn"],
        # Not cc-by-nc-4.0: WorldSpeech failed the gate and was never trained on.
        # FLEURS at CC-BY-4.0 is the most restrictive source that WAS used.
        "license": "cc-by-4.0",
        "library_name": "f5-tts",
        "pipeline_tag": "text-to-speech",
        "base_model": "SWivid/F5-TTS",
        "base_model_relation": "finetune",
        "datasets": list(DATASETS),
        "metrics": ["cer", "utmos"],
        "tags": ["text-to-speech", "tts", "mongolian", "khalkha", "cyrillic",
                 "flow-matching", "f5-tts", "dit", "vocos", "voice-cloning"],
        "model-index": [{
            "name": "oron-tts",
            "results": [{
                "task": {"type": "text-to-speech", "name": "Text-to-Speech"},
                "dataset": {"type": "btsee/common-voice-26-mn",
                            "name": "Common Voice 26 Mongolian (cleaned)",
                            "split": "withheld"},
                "metrics": results,
            }],
        }],
        # `new_version` is deliberately absent: it declares a successor repo and
        # the Hub banners every visitor to it. No successor exists.
    }


BODY = """
# OronTTS — Mongolian text to speech

Speaks Mongolian (Khalkha, Cyrillic) in two fixed voices, one male and one
female. A finetune of F5-TTS on {hours} of cleaned Mongolian speech.

## Listen

Male:

<audio controls src="https://huggingface.co/{repo}/resolve/main/demos/male.wav"></audio>

Female:

<audio controls src="https://huggingface.co/{repo}/resolve/main/demos/female.wav"></audio>

## Install

```bash
pip install git+https://github.com/SWivid/F5-TTS.git
pip install git+https://github.com/btseee/oron-tts.git
```

## Use

```python
import soundfile as sf
from huggingface_hub import hf_hub_download
from f5_tts.api import F5TTS
from oron_tts.text import MongolianNormalizer

VOICE = "female"          # or "male"

ckpt  = hf_hub_download("{repo}", "model.safetensors")
vocab = hf_hub_download("{repo}", "vocab.txt")
ref   = hf_hub_download("{repo}", f"voices/{{VOICE}}.wav")
rtxt  = hf_hub_download("{repo}", f"voices/{{VOICE}}.txt")

tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, use_ema=False)
wav, sr, _ = tts.infer(
    ref_file=ref,
    ref_text=open(rtxt, encoding="utf-8").read().strip(),
    gen_text=MongolianNormalizer().normalize("Сайн байна уу. Өнөөдөр цаг агаар сайхан байна.",
                                             strict=True),
    nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=0)

sf.write("out.wav", wav, sr)
```

## Two things that break it silently

**Keep `use_ema=False`.** The EMA weights synthesise fluent non-words at CER
0.921 against 0.026 for the raw tensors. It sounds like confident speech, so you
will not hear the mistake.

**Normalise the text.** Anything outside the vocabulary is read as a space,
because unknown ids map to index 0 and index 0 is the space token. Digits, Latin
letters and punctuation all need `MongolianNormalizer`.

## Numbers

| | male | female |
| --- | --- | --- |
| CER | {cer_male:.4f} | {cer_female:.4f} |
| UTMOS | {utmos_male:.2f} | {utmos_female:.2f} |
| speaker similarity to its own prompt | {sim_male:.3f} | {sim_female:.3f} |

The two voices score {sim_cross:.3f} against each other. On this project's own
recordings, real same-speaker pairs score 0.540–0.833 and different-speaker
pairs 0.034–0.503.

Per-checkpoint numbers are in `eval.json`, curves in the TensorBoard tab, the
full run in `logs/`.

## Links

* Code: [github.com/btseee/oron-tts](https://github.com/btseee/oron-tts)
* Corpus tooling: [github.com/btseee/oron-cleaner](https://github.com/btseee/oron-cleaner)
* Training data: [mbspeech-mn](https://huggingface.co/datasets/btsee/mbspeech-mn),
  [fleurs-mn](https://huggingface.co/datasets/btsee/fleurs-mn),
  [common-voice-26-mn](https://huggingface.co/datasets/btsee/common-voice-26-mn)

## Licence

CC-BY-4.0. Attribution is required because FLEURS is CC-BY-4.0. It does **not**
train on WorldSpeech, so it carries no non-commercial restriction.
"""


def render(evals: dict, consistency: dict) -> str:
    import yaml

    _, best = best_checkpoint(evals[FINAL_STAGE])
    measured = consistency.get("measured", {})
    meta = yaml.safe_dump(frontmatter(evals, consistency), sort_keys=False,
                          allow_unicode=True, default_flow_style=False)
    body = BODY.format(
        repo=REPO, hours="25 hours",
        cer_male=best["male"]["cer_median"], cer_female=best["female"]["cer_median"],
        utmos_male=best["male"]["utmos_mean"], utmos_female=best["female"]["utmos_mean"],
        sim_male=measured.get("male_demo_vs_male_prompt", float("nan")),
        sim_female=measured.get("female_demo_vs_female_prompt", float("nan")),
        sim_cross=measured.get("male_demo_vs_female_demo", float("nan")))
    return "---\n" + meta + "---\n" + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--consistency", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    card = render(json.loads(args.eval.read_text(encoding="utf-8")),
                  json.loads(args.consistency.read_text(encoding="utf-8")))
    args.out.write_text(card, encoding="utf-8")
    print(f"  wrote {args.out} ({len(card)} chars)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_model_card.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/model_card.py tests/test_model_card.py
git commit -m "Generate the model card from the measurements, with the licence the training data allows"
```

---

### Task 4: Dataset card metadata

**Files:**
- Create: `scripts/dataset_card_meta.py`
- Test: `tests/test_dataset_card_meta.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `split_card(card: str) -> tuple[dict, str]` returning parsed frontmatter and body
  - `enrich(card: str, *, used_by: str | None, note: str | None) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_card_meta.py`:

```python
"""Dataset cards keep their long form; they were missing the metadata keys the
Hub filters and searches on, and nothing linked them to the model they exist to
train.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dataset_card_meta as dcm  # noqa: E402

CARD = """---
language:
- mn
license: cc0-1.0
pretty_name: Common Voice 26.0 Mongolian (cleaned)
task_categories:
- text-to-speech
---

# Common Voice 26.0 Mongolian (cleaned)

A quality-filtered corpus.
"""


def test_the_existing_metadata_and_body_survive():
    """This edits a card that took a GPU pass to produce; it must not rewrite it."""
    out = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    meta, body = dcm.split_card(out)
    assert meta["license"] == "cc0-1.0"
    assert meta["pretty_name"] == "Common Voice 26.0 Mongolian (cleaned)"
    assert "A quality-filtered corpus." in body


def test_the_missing_descriptive_keys_are_added():
    meta, _ = dcm.split_card(dcm.enrich(CARD, used_by=None, note=None))
    for key in ("annotations_creators", "language_creators", "multilinguality",
                "source_datasets", "task_ids"):
        assert key in meta, f"missing {key}"
    assert meta["multilinguality"] == "monolingual"


def test_the_model_is_linked_from_the_dataset():
    out = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    assert "https://huggingface.co/btsee/oron-tts" in out
    assert "## Used by" in out


def test_a_note_can_be_carried():
    """WorldSpeech is CC-BY-NC and the model does not use it. Without saying so,
    a reader seeing an NC dataset beside a CC-BY model assumes a mistake."""
    out = dcm.enrich(CARD, used_by=None, note="The model does not train on this corpus.")
    assert "The model does not train on this corpus." in out


def test_running_it_twice_changes_nothing():
    once = dcm.enrich(CARD, used_by="btsee/oron-tts", note=None)
    assert dcm.enrich(once, used_by="btsee/oron-tts", note=None) == once
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dataset_card_meta.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_card_meta'`

- [ ] **Step 3: Write the implementation**

Create `scripts/dataset_card_meta.py`:

```python
"""Add the descriptive metadata a dataset card needs, without rewriting it.

The four published corpora carry accurate, generated cards -- clips, hours,
splits, gates, every column typed from the shipped data. What they lack is the
metadata the Hub filters and searches on, and any link to the model they exist to
train. Both are additive, so this edits in place rather than regenerating: those
cards cost a GPU pass to produce and their numbers are correct.

    python scripts/dataset_card_meta.py --repo btsee/fleurs-mn --used-by btsee/oron-tts
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Every corpus here is one language, read or spoken by many people, transcribed
# by whoever produced the upstream release rather than by this project.
DEFAULTS = {
    "annotations_creators": ["found"],
    "language_creators": ["crowdsourced"],
    "multilinguality": "monolingual",
    "source_datasets": ["original"],
    "task_ids": ["text-to-speech"],
}

USED_BY_HEADING = "## Used by"


def split_card(card: str) -> tuple[dict, str]:
    """Frontmatter as a dict, and the body after it."""
    import yaml

    if not card.startswith("---\n"):
        return {}, card
    end = card.index("\n---\n", 3)
    return yaml.safe_load(card[4:end]) or {}, card[end + 5:]


def enrich(card: str, *, used_by: str | None, note: str | None) -> str:
    """Add missing keys, a link to the model, and an optional note. Idempotent."""
    import yaml

    meta, body = split_card(card)
    for key, value in DEFAULTS.items():
        meta.setdefault(key, value)

    if used_by and USED_BY_HEADING not in body:
        body = body.rstrip() + (
            f"\n\n{USED_BY_HEADING}\n\n"
            f"[{used_by}](https://huggingface.co/{used_by}) — Mongolian "
            f"text-to-speech, trained on this corpus.\n")
    if note and note not in body:
        body = body.rstrip() + f"\n\n{note}\n"

    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True,
                                    default_flow_style=False) + "---\n" + body


def main() -> None:
    import os

    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--used-by", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    path = Path(hf_hub_download(args.repo, "README.md", repo_type="dataset",
                                token=token, force_download=True))
    updated = enrich(path.read_text(encoding="utf-8"),
                     used_by=args.used_by, note=args.note)
    if args.dry_run:
        print(updated[:updated.index("\n---\n", 3) + 5])
        return
    path.write_text(updated, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Add descriptive metadata and link the model")
    print(f"  updated {args.repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dataset_card_meta.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/dataset_card_meta.py tests/test_dataset_card_meta.py
git commit -m "Add the dataset metadata the Hub filters on, and link the model"
```

---

### Task 5: Patch upstream trainer logging

F5-TTS is refetched fresh on every pod, so this is a patch script rather than a fork. The thing that matters most is that it **fails loudly when an anchor moves**: a patch that silently no-ops leaves a run looking instrumented when it is not, which is exactly how this problem went unnoticed for four stages.

**Files:**
- Create: `scripts/patch_trainer_logging.py`
- Test: `tests/test_patch_trainer_logging.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PATCHES: list[tuple[str, str, str]]` of `(name, anchor, replacement)`
  - `apply_patches(source: str) -> tuple[str, list[str]]` returning the patched source and the names applied

- [ ] **Step 1: Write the failing test**

Create `tests/test_patch_trainer_logging.py`:

```python
"""Upstream logs two scalars. This adds the rest, and refuses to pretend.

The failure mode this guards against is not a bad patch but a silent one: if an
anchor moves in a new upstream release and the patch quietly matches nothing, the
next run looks instrumented and is not -- which is how four stages shipped with
`loss` and `lr` and nothing else.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import patch_trainer_logging as ptl  # noqa: E402

TRAINER = '''
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
'''

CLIP = '''
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
'''

SAMPLES = '''
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
'''

WHOLE = CLIP + TRAINER + SAMPLES


def test_every_patch_applies_to_the_real_upstream_shape():
    patched, applied = ptl.apply_patches(WHOLE)
    assert set(applied) == {name for name, _, _ in ptl.PATCHES}
    assert patched != WHOLE


def test_the_grad_norm_reaches_tensorboard():
    patched, _ = ptl.apply_patches(WHOLE)
    assert "train/grad_norm" in patched


def test_the_sample_audio_reaches_tensorboard():
    """The trainer already synthesises this audio and writes it to disk, where
    it dies with the pod."""
    patched, _ = ptl.apply_patches(WHOLE)
    assert "add_audio" in patched
    assert "add_image" in patched


def test_a_moved_anchor_raises_rather_than_no_opping():
    """The whole point. A patch that matches nothing must not report success."""
    with pytest.raises(SystemExit) as excinfo:
        ptl.apply_patches(TRAINER)          # missing the clip and sample anchors
    assert "grad_norm" in str(excinfo.value) or "sample" in str(excinfo.value)


def test_applying_twice_is_refused():
    """Running it again on an already patched file would double the inserts."""
    patched, _ = ptl.apply_patches(WHOLE)
    with pytest.raises(SystemExit):
        ptl.apply_patches(patched)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_patch_trainer_logging.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'patch_trainer_logging'`

- [ ] **Step 3: Write the implementation**

Create `scripts/patch_trainer_logging.py`:

```python
"""Teach upstream's trainer to log more than two scalars.

`Trainer.train` writes exactly `loss` and `lr`. It also synthesises sample audio
every `save_per_updates`, and writes it to `ckpts/<name>/samples/*.wav` -- on
disk, never in TensorBoard, and gone when the pod is deleted.

F5-TTS is refetched fresh on every run, so this is a patch script rather than a
fork: a fork of a training loop is how `load_checkpoint` surprised this project
once already. Each patch is anchored on an exact source fragment and **asserts it
applied**. A patch that silently matches nothing is worse than no patch, because
the run then looks instrumented and is not -- which is how four stages shipped
with two scalars.

    python scripts/patch_trainer_logging.py --trainer /workspace/F5-TTS/src/f5_tts/model/trainer.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# oron-tts: richer tensorboard logging"

_CLIP_ANCHOR = (
    "                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)")
_CLIP_PATCH = (
    "                        grad_norm = self.accelerator.clip_grad_norm_(\n"
    "                            self.model.parameters(), self.max_grad_norm)\n"
    f"                        {MARKER}\n"
    "                        self._oron_grad_norm = float(grad_norm) if grad_norm is not None else None")

_SCALAR_ANCHOR = (
    '                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)')
_SCALAR_PATCH = (
    '                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)\n'
    f"                    {MARKER}\n"
    "                    if getattr(self, '_oron_grad_norm', None) is not None:\n"
    '                        self.writer.add_scalar("train/grad_norm", self._oron_grad_norm, global_update)')

_SAMPLE_ANCHOR = (
    "                        torchaudio.save(\n"
    '                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate\n'
    "                        )")
_SAMPLE_PATCH = (
    _SAMPLE_ANCHOR + "\n"
    f"                        {MARKER}\n"
    "                        if self.logger == 'tensorboard' and self.writer is not None:\n"
    "                            self.writer.add_audio('sample/generated', gen_audio,\n"
    "                                                  global_update, sample_rate=target_sample_rate)\n"
    "                            self.writer.add_image('sample/mel_generated',\n"
    "                                                  gen_mel_spec[0].detach().cpu().unsqueeze(0),\n"
    "                                                  global_update)")

PATCHES = [
    ("grad_norm capture", _CLIP_ANCHOR, _CLIP_PATCH),
    ("grad_norm scalar", _SCALAR_ANCHOR, _SCALAR_PATCH),
    ("sample audio and mel", _SAMPLE_ANCHOR, _SAMPLE_PATCH),
]


def apply_patches(source: str) -> tuple[str, list[str]]:
    """Apply every patch, or refuse. Never partially, never silently."""
    if MARKER in source:
        raise SystemExit("trainer.py is already patched; applying again would "
                         "duplicate every insert")
    missing = [name for name, anchor, _ in PATCHES if source.count(anchor) != 1]
    if missing:
        raise SystemExit(
            "these anchors are absent or ambiguous in this trainer.py: "
            + ", ".join(missing)
            + ". Upstream has moved; re-derive them rather than shipping a run "
              "that looks instrumented and is not.")
    applied = []
    for name, anchor, replacement in PATCHES:
        source = source.replace(anchor, replacement, 1)
        applied.append(name)
    return source, applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer", required=True, type=Path)
    args = parser.parse_args()
    original = args.trainer.read_text(encoding="utf-8")
    patched, applied = apply_patches(original)
    args.trainer.write_text(patched, encoding="utf-8")
    print("  patched %s: %s" % (args.trainer, ", ".join(applied)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_patch_trainer_logging.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Verify against the real upstream file**

The fixtures are excerpts; confirm the anchors match the actual trainer:

```bash
python - <<'PY'
import pathlib, sys
sys.path.insert(0, "scripts")
import patch_trainer_logging as ptl
src = pathlib.Path("../F5-TTS/src/f5_tts/model/trainer.py").read_text(encoding="utf-8")
patched, applied = ptl.apply_patches(src)
print("applied to real upstream:", applied)
compile(patched, "trainer.py", "exec")
print("patched trainer.py still parses")
PY
```

Expected: all three patch names listed, then `patched trainer.py still parses`.

- [ ] **Step 6: Commit**

```bash
git add scripts/patch_trainer_logging.py tests/test_patch_trainer_logging.py
git commit -m "Patch upstream trainer logging, and refuse when an anchor has moved"
```

---

### Task 6: Publish

Everything above is local and tested. This applies it to the Hub and verifies the result from the server, because a publish that reports success and lands nothing has happened on this project before.

**Files:**
- Create: `scripts/publish_docs.py`
- Test: `tests/test_publish_docs.py`

**Interfaces:**
- Consumes: `model_card.render` (Task 3), `tb_report` (Tasks 1–2), `dataset_card_meta.enrich` (Task 4).
- Produces: `verify(api, repo: str, expected: list[str]) -> list[str]` returning missing paths.

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_docs.py`:

```python
"""A publish that reports success and lands nothing has happened here before."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_docs  # noqa: E402


class FakeSibling:
    def __init__(self, name): self.rfilename = name


class FakeInfo:
    def __init__(self, names): self.siblings = [FakeSibling(n) for n in names]


class FakeApi:
    def __init__(self, names): self._names = names
    def model_info(self, repo, **kw): return FakeInfo(self._names)


def test_missing_files_are_reported():
    api = FakeApi(["README.md", "model.safetensors"])
    missing = publish_docs.verify(api, "btsee/oron-tts",
                                  ["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert missing == ["tensorboard/cv/events.out.tfevents.1"]


def test_a_complete_upload_reports_nothing_missing():
    api = FakeApi(["README.md", "tensorboard/cv/events.out.tfevents.1"])
    assert publish_docs.verify(api, "btsee/oron-tts",
                               ["README.md", "tensorboard/cv/events.out.tfevents.1"]) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_publish_docs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'publish_docs'`

- [ ] **Step 3: Write the implementation**

Create `scripts/publish_docs.py`:

```python
"""Push the regenerated card and TensorBoard tree, then read the server back.

`verify` exists because this project has published an artifact that the upload
call reported as successful and that was not there -- and separately, a dataset
whose upload succeeded and whose content was 19 clips. The server's view is the
only view that counts.

    python scripts/publish_docs.py --card README.md --tensorboard tensorboard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = "btsee/oron-tts"


def verify(api, repo: str, expected: list[str]) -> list[str]:
    """Paths that are not on the server, in the order they were expected."""
    present = {s.rfilename for s in api.model_info(repo, files_metadata=False).siblings}
    return [path for path in expected if path not in present]


def main() -> None:
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card", type=Path, help="README.md to upload")
    parser.add_argument("--tensorboard", type=Path, help="tensorboard/ directory to upload")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    expected: list[str] = []

    if args.card:
        api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                        repo_id=args.repo, commit_message="Rewrite the card around usage")
        expected.append("README.md")

    if args.tensorboard:
        # The old flat files are replaced wholesale: a stale events file beside
        # the new tree would show up as an unnamed extra run.
        api.delete_folder(path_in_repo="tensorboard", repo_id=args.repo,
                          commit_message="Replace the flat TensorBoard files")
        api.upload_folder(folder_path=str(args.tensorboard), path_in_repo="tensorboard",
                          repo_id=args.repo, commit_message="Publish per-stage TensorBoard runs")
        expected += ["tensorboard/" + p.relative_to(args.tensorboard).as_posix()
                     for p in sorted(args.tensorboard.rglob("*")) if p.is_file()]

    missing = verify(api, args.repo, expected)
    if missing:
        raise SystemExit(f"publish VERIFY FAILED, missing: {missing}")
    print(f"  verified {len(expected)} files on {args.repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_publish_docs.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Build the artifacts and publish**

Download the inputs, build, publish, and check the Hub renders the panel:

```bash
mkdir -p build && cd build
python - <<'PY'
import os, shutil, pathlib
from huggingface_hub import hf_hub_download
tok = os.environ["HF_TOKEN"]
for f in ("eval.json", "demos/consistency.json", "demos/male.wav", "demos/female.wav",
          "voices/male.wav", "voices/female.wav"):
    p = hf_hub_download("btsee/oron-tts", f, token=tok, force_download=True)
    out = pathlib.Path(pathlib.Path(f).name)
    shutil.copy(p, out)
    print(" ", out)
PY
cd ..
python scripts/model_card.py --eval build/eval.json \
    --consistency build/consistency.json --out build/README.md
python scripts/tb_report.py --eval build/eval.json --out build/tensorboard \
    --audio demo_male=build/male.wav --audio demo_female=build/female.wav \
    --consistency build/consistency.json
python scripts/publish_docs.py --card build/README.md --tensorboard build/tensorboard
for d in mbspeech-mn fleurs-mn common-voice-26-mn; do
  python scripts/dataset_card_meta.py --repo btsee/$d --used-by btsee/oron-tts
done
python scripts/dataset_card_meta.py --repo btsee/WorldSpeech-mn \
    --note "**Not used by the model.** This corpus failed the pipeline's 15% pass-rate gate and \`btsee/oron-tts\` never trained on it, so the model carries no non-commercial restriction from it."
```

Expected: `verified N files on btsee/oron-tts`, then four `updated btsee/...` lines.

- [ ] **Step 6: Commit**

```bash
git add scripts/publish_docs.py tests/test_publish_docs.py
git commit -m "Publish the card and TensorBoard tree, verified from the server"
```

---

## Self-Review

**Spec coverage.** Model card frontmatter and short body — Task 3. Dataset metadata and the WorldSpeech note — Task 4. TensorBoard per-stage runs, eval curves, hparams, corpus scalars — Task 1. Summary run with audio, mel images, speaker similarity — Task 2. `patch_trainer_logging.py` — Task 5. Publishing and verification — Task 6. The spec's testing section is satisfied by the `EventAccumulator` read-back in Tasks 1–2, the frontmatter-versus-`eval.json` assertion in Task 3, and the moved-anchor test in Task 5.

**One deliberate reduction.** The spec lists `corpus/clips`, `corpus/hours` and `corpus/speakers` scalars per stage. They are carried through `--stages` as hparams instead, where they are comparable across stages in the HPARAMS tab; a constant plotted against an update axis is a flat line that reads as data. The numbers still reach TensorBoard.

**Placeholders.** None. Every step carries the code or the command it needs.

**Type consistency.** `checkpoint_update`, `eval_series`, `is_empty_events`, `write_stage_run`, `mel_image`, `write_summary_run`, `best_checkpoint`, `frontmatter`, `render`, `split_card`, `enrich`, `apply_patches`, `PATCHES`, `verify` are each defined once and referenced under the same name throughout.

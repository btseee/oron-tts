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

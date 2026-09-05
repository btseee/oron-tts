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


def stage_summary(stage: str, series: dict[str, list[tuple[int, float]]],
                  meta: dict) -> str:
    """What this stage trained on, and how its checkpoint was chosen.

    The `voicelock` run is why this exists: it shows `loss` and `lr` and
    nothing else, and nothing in the tab says that its checkpoint was taken by
    fallback because the sweep produced no scoreable output. A reader cannot
    tell an unscored stage from a scored one by looking at the charts, so the
    run has to say it.
    """
    corpora = meta.get("corpora") or []
    lines = [f"### {stage}", "**Trained on:** " + (", ".join(str(c) for c in corpora)
                                                   if corpora else "not recorded (no --stages metadata)")]
    points = series.get("eval/cer_mean") or []
    if points:
        update, cer = min(points, key=lambda p: p[1])
        lines.append(
            f"**Checkpoint:** `model_{update}.pt`, chosen by CER -- best mean CER "
            f"{cer:.4f} of {len(points)} evaluated checkpoint"
            f"{'' if len(points) == 1 else 's'}.")
    else:
        lines.append(
            "**Checkpoint:** chosen by **fallback**, not by CER. No checkpoint of "
            "this stage was scored -- the sweep produced no scoreable output -- so "
            "the last checkpoint was taken. The charts below show training "
            "progress only; nothing here measures quality.")
    return "\n\n".join(lines)


def corpus_scalars(meta: dict) -> dict[str, float]:
    """The stage's corpus size, as scalars, from `--stages`.

    Listed by name rather than "every number in the metadata": `learning_rate`
    and the update target describe the schedule, not the corpus, and belong in
    hparams where the HPARAMS tab can compare them.
    """
    keys = ("clips", "hours", "speakers", "male_hours", "female_hours")
    return {f"corpus/{key}": float(meta[key]) for key in keys
            if isinstance(meta.get(key), (int, float)) and not isinstance(meta[key], bool)}


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

    corpus = corpus_scalars(hparams)
    summary = stage_summary(stage, series, hparams)
    # Nothing to say, no file: an empty tfevents beside the copied curve is the
    # very "empty file" this script's docstring lists as a fault it fixes, and
    # one was written into the voicelock run. `summary` is non-empty for every
    # stage, so this is a backstop rather than a live path -- but it is the
    # condition, not a comment, that keeps it that way.
    if not (series or metrics or corpus or summary):
        return run

    writer = SummaryWriter(log_dir=str(run))
    writer.add_text("stage/summary", summary, 0)
    for tag, value in sorted(corpus.items()):
        # Step 0: a corpus size is one number for the whole stage, not a series.
        writer.add_scalar(tag, value, 0)
    for tag, points in sorted(series.items()):
        for step, value in points:
            writer.add_scalar(tag, value, step)
    if metrics:
        # hparams is optional -- the production invocation never passes
        # --stages, so hparams is {} on every call. add_hparams only requires
        # its first argument to be a dict, and an empty one is fine; gating on
        # `hparams and metrics` used to drop final/cer_mean and
        # final/best_update -- the numbers checkpoint selection actually
        # ranks on -- in the one invocation that matters.
        writer.add_hparams({k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
                           metrics, run_name=".")
    writer.close()
    return run


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
    scaled = np.clip((db - db.min()) / max(float(np.ptp(db)), 1e-6), 0.0, 1.0)
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

    import numpy as np

    for name, (wav, sr) in sorted(audio.items()):
        samples = np.asarray(wav, dtype="float32")
        writer.add_audio(f"audio/{name}", samples, 0, sample_rate=sr)
        writer.add_image(f"mel/{name}", mel_image(samples, sr), 0)

    measured = consistency.get("measured", {})
    for name, value in sorted(measured.items()):
        if isinstance(value, (int, float)):
            writer.add_scalar(f"similarity/{name}", float(value), 0)

    calibration = consistency.get("calibration", {})
    calibration_keys = ("same_speaker_range", "different_speaker_range",
                        "same_speaker_threshold")
    sentences = ["Metric: `{}`.".format(consistency.get("metric", "unknown"))]
    if all(calibration.get(key) is not None for key in calibration_keys):
        sentences.append(
            "Same-speaker pairs of real recordings scored {}, different-speaker "
            "pairs {}, so {} separates them.".format(
                *(calibration[key] for key in calibration_keys)))
    # Dropped whole rather than filled with None. Without --consistency this
    # rendered "scored None ... so None separates them", which reads as a
    # measurement; and this sentence is the only thing in the tab that says what
    # the similarity scalars mean, so a half-written one is worse than none.
    table = "\n".join(f"* `{k}` = {v:.4f}" for k, v in sorted(measured.items())
                      if isinstance(v, (int, float)))
    writer.add_text("summary/speaker_similarity",
                    " ".join(sentences) + "\n\n" + table, 0)
    writer.add_text("summary/stages", "Stages in this report: " + ", ".join(stages), 0)
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
                        help="JSON of per-stage metadata: corpora (names, for the "
                             "summary text), clips, hours, speakers, male_hours, "
                             "female_hours (corpus/* scalars), plus anything else, "
                             "which becomes hparams")
    parser.add_argument("--audio", action="append", default=[], metavar="NAME=PATH",
                        help="wav to embed in the summary run; repeatable")
    parser.add_argument("--consistency", type=Path,
                        help="demos/consistency.json, for the speaker-similarity charts")
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
              f"{len(corpus_scalars(meta))} corpus scalars, "
              f"checkpoint by {'CER' if metrics else 'FALLBACK'}, "
              f"{'training curve carried' if stage in events_for else 'no training events'})")

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


if __name__ == "__main__":
    main()

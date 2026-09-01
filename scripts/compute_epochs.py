"""Work out the `epochs` value for a target number of training updates.

In F5-TTS, `epochs` is not just a stopping condition -- it sets the length of the
learning-rate decay:

    total_updates = ceil(len(dataloader) / grad_accumulation_steps) * epochs

and the scheduler decays linearly from the peak to ~0 across it
(trainer.py:316-326). Guess it too high and the run ends while the LR is still
hot; too low and the LR reaches zero long before you stop. Neither failure is
reported anywhere.

`updates_per_epoch` is the number of batches DynamicBatchSampler produces, which
depends on how the clips pack -- not on clip count. This replicates the real
packing rather than dividing total frames by the budget, because the sampler
sorts by length first and leaves a tail of partly-filled batches.

    python scripts/compute_epochs.py --data ../F5-TTS/data/oron_mn_pinyin
    python scripts/compute_epochs.py --data <dir> --target-updates 60000
"""

import argparse
import json
import math
from pathlib import Path

# Must match configs/f5tts_mn.yaml.
DEFAULT_FRAMES = 13000
DEFAULT_MAX_SAMPLES = 32
DEFAULT_TARGET_UPDATES = 40000
HOP_LENGTH = 256
SAMPLE_RATE = 24000


def pack(frame_lengths: list[int], frames_threshold: int, max_samples: int) -> int:
    """Count batches exactly the way DynamicBatchSampler does.

    Sorts ascending, greedily fills, and drops any single clip longer than the
    whole budget -- upstream discards those silently (dataset.py:208-213).
    """
    batches = 0
    current = 0
    count = 0
    dropped = 0
    for length in sorted(frame_lengths):
        if length > frames_threshold:
            dropped += 1
            continue
        if current + length <= frames_threshold and (max_samples == 0 or count < max_samples):
            current += length
            count += 1
        else:
            batches += 1
            current, count = length, 1
    if count:
        batches += 1
    if dropped:
        print(f"[WARN] {dropped} clip(s) exceed frames_threshold and would be "
              f"silently dropped by DynamicBatchSampler")
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                    help="Prepared dataset dir containing duration.json")
    ap.add_argument("--frames-threshold", type=int, default=DEFAULT_FRAMES)
    ap.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    ap.add_argument("--target-updates", type=int, default=DEFAULT_TARGET_UPDATES)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    path = args.data / "duration.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/build_f5_dataset.py first.")
    durations = json.loads(path.read_text(encoding="utf-8"))["duration"]
    if not durations:
        raise SystemExit("duration.json is empty.")

    frames = [int(d * SAMPLE_RATE / HOP_LENGTH) for d in durations]
    hours = sum(durations) / 3600

    per_epoch = pack(frames, args.frames_threshold, args.max_samples)
    per_epoch = max(1, per_epoch // max(1, args.gpus))
    epochs = math.ceil(args.target_updates / per_epoch)

    audio_per_step = args.frames_threshold * HOP_LENGTH / SAMPLE_RATE

    print(f"clips              {len(durations):,}")
    print(f"audio              {hours:.1f} h")
    print(f"frames/step        {args.frames_threshold:,}  (~{audio_per_step:.0f} s of audio)")
    print(f"updates/epoch      {per_epoch:,}")
    print(f"target updates     {args.target_updates:,}")
    print()
    print(f"  epochs: {epochs}")
    print()
    print(f"Sets the LR decay over {per_epoch * epochs:,} updates "
          f"({per_epoch * epochs / max(args.target_updates, 1):.2f}x the target).")
    if epochs * per_epoch > args.target_updates * 1.2:
        print("[WARN] Rounding overshoots the target by >20%; the LR will still "
              "be well above zero when you stop.")
    print()
    print("The paper's small-data evidence (Tab. 9) has LJSpeech 24 h peaking at "
          "200k updates\nfrom scratch and degrading afterwards. For a finetune, "
          "sweep checkpoints and take\nthe best by evaluation -- do not assume "
          "the last one is the best.")


if __name__ == "__main__":
    main()

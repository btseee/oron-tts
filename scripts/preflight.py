"""Refuse to start a training run that is configured wrong.

Everything checked here fails *silently* at training time. The run completes,
the loss curve looks plausible, and the model is worse than it should be with
nothing in the logs to say why:

  epochs                sets the LR decay length, not a stopping condition. Too
                        high and the run ends with the LR still hot; too low and
                        it reaches zero long before you stop. The shipped value
                        is a placeholder computed for a corpus that does not
                        exist yet.
  vocab size            an unknown character maps to index 0, which is the SPACE
                        token -- 4.90% of Mongolian characters become spaces
                        with nothing logged.
  vocab order           sorting or deduplicating misaligns all 2545 pretrained
                        embedding rows.
  tokenizer directory   load_dataset() and get_tokenizer() both interpolate
                        {name}_{tokenizer}; disagreeing sends them to different
                        directories.
  grad accumulation     scheduler.step() fires per batch, so >1 compresses the
                        LR schedule by that factor.
  pretrained_ prefix    Trainer.load_checkpoint uses it to cold-start at update
                        0 and to keep the file out of checkpoint rotation.

    python scripts/preflight.py --config configs/f5tts_mn.yaml \\
        --data ../F5-TTS/data/oron_mn_pinyin
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compute_epochs import HOP_LENGTH, SAMPLE_RATE, pack  # noqa: E402

# The vocabulary scripts/extend_vocab.py produces: 2545 base entries plus the
# five Mongolian letters the base vocab lacks, appended in *discovery* order
# over MN_ALPHABET (lower case first, then upper). Read off the built file
# rather than assumed -- an earlier version of this constant guessed
# alphabetical order and this check caught it.
EXPECTED_VOCAB_ENTRIES = 2550
BASE_VOCAB_ENTRIES = 2545
APPENDED = ["ө", "ү", "Ө", "Ү", "Ъ"]  # ө ү Ө Ү Ъ

# How far `epochs` may sit from the computed value before it is worth stopping
# for. The LR schedule is linear, so 10% off is 10% of the decay in the wrong
# place -- tolerable; 50% is not.
EPOCH_TOLERANCE = 0.10


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is needed to read the config: pip install pyyaml") from None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_vocab(vocab: Path, problems: list[str], notes: list[str]) -> None:
    if not vocab.exists():
        problems.append(f"vocab not found at {vocab}")
        return
    # Line 0 is a single space and splitlines() keeps it; a trailing newline
    # must not count as an entry.
    lines = vocab.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    if len(lines) != EXPECTED_VOCAB_ENTRIES:
        problems.append(
            f"vocab has {len(lines)} entries, expected {EXPECTED_VOCAB_ENTRIES}. "
            f"An unextended vocab turns 4.90% of Mongolian characters into "
            f"spaces, silently -- unknown ids map to 0, and 0 is SPACE."
        )
        return

    if lines[BASE_VOCAB_ENTRIES:] != APPENDED:
        problems.append(
            f"the appended vocab entries are {lines[BASE_VOCAB_ENTRIES:]!r}, "
            f"expected {APPENDED!r}. They must be appended in this order: any "
            f"sort or dedup misaligns all {BASE_VOCAB_ENTRIES} pretrained rows."
        )
        return

    notes.append(f"vocab {len(lines)} entries, {APPENDED} appended in order")


def check_epochs(config: dict, data: Path, problems: list[str], notes: list[str],
                 target_updates: int = 40_000) -> None:
    duration = data / "duration.json"
    if not duration.exists():
        problems.append(
            f"{duration} not found -- run scripts/build_f5_dataset.py first. "
            "Without it `epochs` cannot be checked, and the shipped value is a "
            "placeholder for a corpus that does not exist."
        )
        return

    durations = json.loads(duration.read_text(encoding="utf-8"))["duration"]
    if not durations:
        problems.append(f"{duration} is empty.")
        return

    datasets = config.get("datasets", {})
    optim = config.get("optim", {})
    frames = int(datasets.get("batch_size_per_gpu", 13000))
    max_samples = int(datasets.get("max_samples", 32))

    per_epoch = pack([int(d * SAMPLE_RATE / HOP_LENGTH) for d in durations],
                     frames, max_samples)
    configured = int(optim.get("epochs", 0))
    if not configured:
        problems.append("optim.epochs is unset.")
        return

    total = per_epoch * configured
    # What the value should be for the target being aimed at. Defaults to
    # the runbook's 40,000, but a shorter run is a legitimate choice on a
    # small corpus -- so it is declared rather than hardcoded, and drift is
    # still caught against whatever was declared.
    want = math.ceil(target_updates / max(1, per_epoch))
    drift = abs(configured - want) / max(1, want)
    if drift > EPOCH_TOLERANCE:
        problems.append(
            f"optim.epochs is {configured}, but this dataset packs into "
            f"{per_epoch:,} updates/epoch -- {want} would give ~{target_updates:,} "
            f"updates. "
            f"epochs sets the LR decay length, so {configured} decays over "
            f"{total:,} updates and the schedule is wrong by {drift:.0%}. Run:\n"
            f"    python scripts/compute_epochs.py --data {data} "
            f"--target-updates {target_updates}"
        )
        return

    notes.append(f"epochs {configured} x {per_epoch:,} updates/epoch = {total:,} updates")


def check_training(config: dict, data: Path, problems: list[str], notes: list[str]) -> None:
    optim = config.get("optim", {})
    datasets = config.get("datasets", {})
    ckpts = config.get("ckpts", {})

    accum = int(optim.get("grad_accumulation_steps", 1))
    if accum > 1:
        problems.append(
            f"grad_accumulation_steps is {accum}. scheduler.step() fires per "
            f"batch rather than per optimiser step, so the LR schedule is "
            f"compressed by {accum}x."
        )

    tokenizer = datasets.get("tokenizer", "pinyin")
    name = datasets.get("name", "")
    if tokenizer != "pinyin":
        problems.append(
            f"tokenizer is {tokenizer!r}. load_dataset() and get_tokenizer() both "
            f"interpolate {{name}}_{{tokenizer}}, so anything but 'pinyin' sends "
            f"them to different directories. Mongolian Cyrillic passes through "
            f"convert_char_to_pinyin unchanged."
        )
    elif name and data.name != f"{name}_{tokenizer}":
        problems.append(
            f"datasets.name is {name!r}, so training reads "
            f"{name}_{tokenizer}/ -- but the prepared data is in {data.name}/."
        )

    pretrained = [p.name for p in data.parent.parent.glob("ckpts/**/pretrained_*")]
    if pretrained:
        notes.append(f"pretrained checkpoint present: {pretrained[0]}")

    if not ckpts.get("log_samples", False):
        problems.append(
            "log_samples is off. It is the only way to hear the model during "
            "training, and the previous project ran 500 epochs logging no audio."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "f5tts_mn.yaml")
    ap.add_argument("--data", type=Path, required=True,
                    help="Prepared dataset dir, e.g. ../F5-TTS/data/oron_mn_pinyin")
    ap.add_argument("--target-updates", type=int, default=40_000,
                    help="Updates this run aims for. epochs sets the LR decay "
                         "horizon, so it must match the run you intend, not a "
                         "default from a different corpus.")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="Defaults to <data>/vocab.txt, which is what training reads")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"{args.config} not found.")
    config = load_config(args.config)
    vocab = args.vocab or (args.data / "vocab.txt")

    problems: list[str] = []
    notes: list[str] = []
    check_vocab(vocab, problems, notes)
    check_epochs(config, args.data, problems, notes, args.target_updates)
    check_training(config, args.data, problems, notes)

    for note in notes:
        print(f"  ok    {note}")
    if problems:
        print()
        for p in problems:
            print(f"  FAIL  {p}")
        print(f"\n{len(problems)} problem(s). Each of these fails silently at "
              f"training time -- the run completes and the model is worse.")
        raise SystemExit(1)
    print("\nPreflight clean. Nothing here would have failed silently.")


if __name__ == "__main__":
    main()

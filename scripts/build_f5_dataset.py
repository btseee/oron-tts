"""Turn the strict corpus into the on-disk dataset F5-TTS training reads.

F5-TTS expects, under `<f5_repo>/data/<name>_<tokenizer>/`:

    raw.arrow       rows of {audio_path, text, duration}
    duration.json   {"duration": [...]} index-aligned with raw.arrow
    vocab.txt       one token per line, line 0 a single space

`prepare_csv_wavs.py` produces all three from a `metadata.csv`, so this script
filters the corpus, writes that CSV, and delegates -- then replaces the vocab it
copied.

Two things it must get right:

* **The vocabulary.** `prepare_csv_wavs.py` copies the 2545-line *base* vocab,
  which lacks Ө ө Ү ү Ъ. Training on it silently replaces 4.90% of Mongolian
  characters with spaces, because unknown ids map to 0 and 0 is the space token.

* **The directory name.** `load_dataset()` and `get_tokenizer()` both interpolate
  `{name}_{tokenizer}`, so passing `--tokenizer custom` makes them look in
  different places. Keeping `pinyin` keeps them agreed; Mongolian Cyrillic passes
  through `convert_char_to_pinyin` unchanged (verified on 300 corpus sentences).

Selecting a variant is a filter expression, not a code change:

    python scripts/build_f5_dataset.py --corpus ../oron-cleaner/output/oron_mn_strict
    python scripts/build_f5_dataset.py --corpus <dir> --filter "gender_resolved == 'female'"
    python scripts/build_f5_dataset.py --corpus <dir> --filter "bandwidth_hz >= 10000"
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO / "data" / "oron_mn_pinyin" / "vocab.txt"

# DynamicBatchSampler silently drops clips longer than batch_size_per_gpu frames,
# and CustomDataset skips out-of-range rows by advancing the index -- which
# desyncs duration.json from the sampled item. Filter here instead.
MIN_DURATION_S = 1.0
MAX_DURATION_S = 20.0


def load_manifest(corpus: Path) -> list[dict]:
    path = corpus / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"No manifest at {path}. Run oron-cleaner first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_filter(records: list[dict], expression: str | None) -> list[dict]:
    """Filter rows with a Python expression evaluated against each record."""
    if not expression:
        return records
    kept = []
    for r in records:
        try:
            if eval(expression, {"__builtins__": {}}, dict(r)):  # noqa: S307
                kept.append(r)
        except Exception as exc:
            raise SystemExit(f"Filter failed on a record: {exc}") from exc
    return kept


def select_splits(records: list[dict], splits: str) -> list[dict]:
    """Keep only the requested manifest splits, or fail saying why it cannot.

    This guard used to read `if wanted and any("split" in r for r in records)`,
    which was always False because the corpus writer never persisted `split` --
    so the filter silently did nothing and training consumed the whole corpus,
    test split included. A missing split is a corpus that was never finalised,
    not a default of "train".
    """
    wanted = {s.strip() for s in splits.split(",") if s.strip()}
    if not wanted:
        return records
    missing = [r for r in records if "split" not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(records)} manifest rows have no 'split' key.\n"
            "Run the finalize step first:\n"
            "    python clean_pipeline.py --finalize-only --corpus-dir <dir>\n"
            "Without it every split would be silently merged into training."
        )
    available = {r["split"] for r in records}
    unknown = wanted - available
    if unknown:
        raise SystemExit(
            f"Requested split(s) {sorted(unknown)} not in the manifest. "
            f"Available: {sorted(available)}."
        )
    return [r for r in records if r["split"] in wanted]


def write_metadata_csv(corpus: Path, records: list[dict], out: Path) -> int:
    """Write the `audio_file|text` CSV, with the absolute paths F5-TTS requires."""
    corpus = corpus.resolve()
    written = 0
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        for r in records:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            audio = corpus / r["audio_path"]
            if not audio.exists():
                continue
            writer.writerow([str(audio), text])
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="oron-cleaner corpus directory")
    ap.add_argument("--f5-repo", type=Path, default=REPO.parent / "F5-TTS")
    ap.add_argument("--name", default="oron_mn", help="Dataset name; directory is <name>_pinyin")
    ap.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    ap.add_argument("--filter", dest="expression", default=None,
                    help="Python expression over manifest fields, e.g. \"bandwidth_hz >= 10000\"")
    ap.add_argument("--splits", default="train",
                    help="Comma-separated manifest splits to include (default: train)")
    ap.add_argument("--workers", type=int, default=0, help="0 lets prepare_csv_wavs choose")
    args = ap.parse_args()

    records = load_manifest(args.corpus)
    print(f"manifest: {len(records)} clips")

    records = select_splits(records, args.splits)
    if args.splits.strip():
        print(f"  splits {args.splits}: {len(records)}")

    before = len(records)
    records = [r for r in records
               if MIN_DURATION_S <= float(r.get("duration_s") or 0) <= MAX_DURATION_S]
    if len(records) < before:
        print(f"  duration {MIN_DURATION_S}-{MAX_DURATION_S}s: {len(records)} "
              f"({before - len(records)} dropped)")

    records = apply_filter(records, args.expression)
    if args.expression:
        print(f"  filter {args.expression!r}: {len(records)}")
    if not records:
        raise SystemExit("Nothing left after filtering.")

    hours = sum(float(r.get("duration_s") or 0) for r in records) / 3600
    print(f"  -> {len(records)} clips, {hours:.1f} h")

    out_dir = args.f5_repo / "data" / f"{args.name}_pinyin"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    n = write_metadata_csv(args.corpus, records, csv_path)
    print(f"wrote {csv_path} ({n} rows)")

    script = args.f5_repo / "src" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py"
    if not script.exists():
        raise SystemExit(f"prepare_csv_wavs.py not found at {script}")
    cmd = [sys.executable, str(script), str(csv_path), str(out_dir)]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        detail = "\n  ".join(tail)
        hint = ""
        # prepare_csv_wavs.py imports the f5_tts package, whose __init__ pulls
        # the whole training stack. A partial install fails here rather than at
        # training time, with a traceback that does not name the cause.
        if "ModuleNotFoundError" in (proc.stderr or ""):
            missing = tail[-1].split("'")[-2] if "'" in tail[-1] else "a dependency"
            hint = (f"\n\n{missing!r} is missing. prepare_csv_wavs.py imports the "
                    "f5_tts package, whose __init__ loads the trainer, so a "
                    "partial install fails here.\n"
                    "Install the full package:  pip install -e ../F5-TTS")
        raise SystemExit(f"prepare_csv_wavs.py failed:\n  {detail}{hint}")

    # prepare_csv_wavs.py copies the 2545-line base vocab. Replacing it is not
    # optional: without Ө ө Ү ү those characters become spaces, silently.
    target_vocab = out_dir / "vocab.txt"
    base_lines = target_vocab.read_text(encoding="utf-8").count("\n") if target_vocab.exists() else 0
    target_vocab.write_bytes(args.vocab.read_bytes())
    ext_lines = target_vocab.read_text(encoding="utf-8").count("\n")
    print(f"vocab: {base_lines} -> {ext_lines} entries (replaced with {args.vocab})")

    for required in ("raw.arrow", "duration.json", "vocab.txt"):
        path = out_dir / required
        if not path.exists():
            raise SystemExit(f"Expected {path} to exist after preparation.")

    durations = json.loads((out_dir / "duration.json").read_text(encoding="utf-8"))["duration"]
    print(f"duration.json: {len(durations)} entries, {sum(durations)/3600:.1f} h")
    if len(durations) != n:
        print(f"[WARN] duration.json has {len(durations)} entries for {n} CSV rows; "
              "prepare_csv_wavs skips unreadable audio.")

    print(f"\nReady: {out_dir}")
    print(f"Train with  ++datasets.name={args.name}  and tokenizer 'pinyin'.")


if __name__ == "__main__":
    main()

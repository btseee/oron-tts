"""Choose the reference clips that become the shipped male and female voices.

F5-TTS takes voice identity from a reference clip, not from a token, so "the
model has a male and a female voice" means exactly: two curated reference clips
ship with it. There is no gender conditioning in the architecture and adding one
would be a worse answer than choosing good prompts.

What makes a clip a good prompt, in rough order of impact:

* **Bandwidth.** Output bandwidth follows the prompt. No Mongolian source is
  full-band -- Common Voice's median cutoff is 7.1 kHz and FLEURS/MBSpeech are
  hard-capped at 7.7 kHz -- so the ≥10 kHz tail, which exists only in Common
  Voice, is the difference between a dull voice and a bright one.
* **Duration 6-10 s.** Upstream clips anything over 12 s in three escalating
  stages, and wants ~1 s of trailing silence or the last word gets truncated.
* **Clean recording.** DNSMOS, and a high alignment score, which also confirms
  the transcript is right -- a wrong `ref_text` poisons every generation, since
  duration is estimated from the ref text/audio ratio.
* **Speaker spread.** Offering several candidates per gender from *different*
  speakers, so the choice is not hostage to one contributor.

Ranking is objective; the final pick should be made by listening.

    python scripts/select_voices.py --corpus ../oron-cleaner/output/oron_mn_strict
    python scripts/select_voices.py --corpus <dir> --top 5 --write voices/
"""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Upstream clips reference audio over 12 s; below ~6 s there is too little
# speaker evidence for the prompt to be stable.
MIN_REF_S = 6.0
MAX_REF_S = 10.0


def score(record: dict) -> float:
    """Rank candidate prompts. Bandwidth dominates, deliberately."""
    return (
        float(record.get("bandwidth_hz") or 0.0) / 1000.0     # ~6-12
        + float(record.get("dnsmos_ovr") or 0.0) * 1.5        # ~4-6
        + float(record.get("align_score") or 0.0) * 3.0       # ~2-3
        + float(record.get("snr_db") or 0.0) / 20.0           # ~0.5-2
    )


def candidates(records: list[dict], gender: str, top: int, per_speaker: int = 1) -> list[dict]:
    """Best prompts for a gender, spread across speakers."""
    pool = [
        r for r in records
        if r.get("gender_resolved") == gender
        and MIN_REF_S <= float(r.get("duration_s") or 0) <= MAX_REF_S
    ]
    pool.sort(key=score, reverse=True)

    picked: list[dict] = []
    seen: dict[str, int] = {}
    for r in pool:
        spk = str(r.get("client_id") or "")
        if seen.get(spk, 0) >= per_speaker:
            continue
        seen[spk] = seen.get(spk, 0) + 1
        picked.append(r)
        if len(picked) >= top:
            break
    return picked


def describe(r: dict) -> str:
    return (f"{r['clip_id']:<34} {float(r.get('duration_s') or 0):5.1f}s  "
            f"BW {float(r.get('bandwidth_hz') or 0):6.0f} Hz  "
            f"DNSMOS {float(r.get('dnsmos_ovr') or 0):4.2f}  "
            f"align {float(r.get('align_score') or 0):.3f}  "
            f"F0 {float(r.get('mean_f0_hz') or 0):5.1f} Hz")


def write_bundle(corpus: Path, chosen: dict[str, dict], out: Path) -> None:
    """Emit the voices/ bundle the inference CLI reads."""
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for gender, r in chosen.items():
        name = f"mn_{gender}_01"
        shutil.copy2(corpus / r["audio_path"], out / f"{name}.wav")
        (out / f"{name}.txt").write_text(r["text"], encoding="utf-8")
        entries[gender] = {
            "id": name,
            "gender": gender,
            "audio": f"{name}.wav",
            "text": r["text"],
            "source_clip": r["clip_id"],
            "duration_s": float(r.get("duration_s") or 0),
            "bandwidth_hz": float(r.get("bandwidth_hz") or 0),
            "dnsmos_ovr": float(r.get("dnsmos_ovr") or 0),
            "align_score": float(r.get("align_score") or 0),
            "mean_f0_hz": float(r.get("mean_f0_hz") or 0),
        }
    (out / "voices.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out}/ with {len(entries)} voice(s): {', '.join(entries)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--genders", default="male,female")
    ap.add_argument("--top", type=int, default=5, help="Candidates to list per gender")
    ap.add_argument("--write", type=Path, default=None,
                    help="Write the top pick per gender as a voices/ bundle")
    args = ap.parse_args()

    manifest = args.corpus / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found. Run oron-cleaner first.")
    with open(manifest, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"corpus: {len(records)} clips")

    chosen: dict[str, dict] = {}
    for gender in (g.strip() for g in args.genders.split(",") if g.strip()):
        picks = candidates(records, gender, args.top)
        print(f"\n=== {gender} ({len(picks)} candidates, one per speaker)")
        if not picks:
            print(f"  none between {MIN_REF_S:g}-{MAX_REF_S:g}s. "
                  "Check gender resolution and the duration gate.")
            continue
        for i, r in enumerate(picks, 1):
            print(f"  {i}. {describe(r)}")
        chosen[gender] = picks[0]
        bw = float(picks[0].get("bandwidth_hz") or 0)
        if bw < 9000:
            print(f"  [!] best available bandwidth is {bw:.0f} Hz. Output will "
                  "inherit that dullness; no Mongolian source is full-band.")

    if args.write:
        if len(chosen) < 2:
            print("\n[!] Fewer than two voices selected — the male/female "
                  "requirement is not met by this corpus.")
        write_bundle(args.corpus, chosen, args.write)
        print("Listen to each before shipping. The ranking is objective; "
              "whether a voice is pleasant is not.")


if __name__ == "__main__":
    main()

"""Does a prompt picked on real bandwidth beat the one that ships?

The model is fixed. The only thing that varies is the reference clip, which is
the whole point: swapping it moved female UTMOS from 2.11 to 2.42 without
retraining anything, and until filter policy v4 the ranking that chose it was
reading a `bandwidth_hz` column the pipeline had capped at 8 kHz.

Two stages, for the same reason the sentence list is split in half. Screening
every candidate and then reporting the winner's score from the screening run is
selection on the test set -- the winner is partly whichever clip got lucky on
those sentences. So candidates are screened on the even-index sentences, and only
the survivor is re-scored on the odd ones, which nothing was chosen on.

UTMOS decides. CER is a guardrail: both voices already sit below the 0.123
human floor, so it cannot resolve anything, but it still catches the failure that
matters -- a prompt that makes the model fluent and wrong.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path("/workspace/oron-tts")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

SEEDS = (0, 1, 2)
SCREEN_N = 40
CONFIRM_N = 120
HUMAN_CER_BASELINE = 0.123


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(ref), list(hyp)
    if not r:
        return (len(h), 0)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1], len(r)


def micro_cer(counts) -> float:
    e = sum(a for a, _ in counts)
    n = sum(b for _, b in counts)
    return e / n if n else 0.0


def bootstrap_ci(values, stat, resamples=1000, alpha=0.05, seed=0):
    items = list(values)
    if len(items) < 2:
        v = stat(items) if items else float("nan")
        return v, v, v
    rng = random.Random(seed)
    n = len(items)
    draws = sorted(stat([items[rng.randrange(n)] for _ in range(n)])
                   for _ in range(resamples))
    return (stat(items), draws[int(alpha / 2 * resamples)],
            draws[min(resamples - 1, int((1 - alpha / 2) * resamples))])


def paired(a, b, stat, resamples=2000, seed=0):
    xs, ys = list(a), list(b)
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs, ys = xs[:n], ys[:n]
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(stat([xs[i] for i in idx]) - stat([ys[i] for i in idx]))
    draws.sort()
    lo, hi = draws[int(0.025 * resamples)], draws[int(0.975 * resamples)]
    return {"diff": stat(xs) - stat(ys), "lo": lo, "hi": hi,
            "separated": lo > 0 or hi < 0, "n": n}


CORPUS = pathlib.Path("/workspace/cv")


def rank_candidates(gender: str, top: int = 5) -> list[dict]:
    """Top prompts for a gender, ranked the way select_voices ranks them.

    Read from the manifest rather than from `voices_v4/`, because
    `write_bundle` emits exactly one clip per gender (`mn_<gender>_01.wav`) --
    enough to ship, not enough to choose between. Screening needs the shortlist,
    so the same score is applied here and the top `top` are taken, at most one
    per speaker so the comparison is across voices rather than across takes of
    one voice.
    """
    rows = [json.loads(l) for l in
            (CORPUS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]

    def score(r: dict) -> float:
        return (float(r.get("bandwidth_hz") or 0) / 1000.0
                + float(r.get("dnsmos_ovr") or 0) * 1.5
                + float(r.get("align_score") or 0) * 3.0
                + float(r.get("snr_db") or 0) / 20.0)

    pool = [r for r in rows
            if r.get("gender_resolved") == gender
            and 6.0 <= float(r.get("duration_s") or 0) <= 10.0]
    pool.sort(key=score, reverse=True)

    picked, seen = [], set()
    for r in pool:
        spk = r.get("speaker_id") or r.get("client_id") or r.get("clip_id")
        if spk in seen:
            continue
        seen.add(spk)
        picked.append(r)
        if len(picked) >= top:
            break
    return picked


def candidates(gender: str) -> list[tuple[str, pathlib.Path, str]]:
    """(label, wav, ref_text) -- the incumbent first, so it is the comparison."""
    from huggingface_hub import hf_hub_download
    out = []
    wav = pathlib.Path(hf_hub_download("btsee/oron-tts", f"voices/{gender}.wav"))
    txt = pathlib.Path(hf_hub_download("btsee/oron-tts", f"voices/{gender}.txt"))
    out.append(("SHIPPED", wav, txt.read_text(encoding="utf-8").strip()))

    for i, r in enumerate(rank_candidates(gender)):
        p = CORPUS / r["audio_path"]
        if not p.is_file():
            continue
        label = f"v4_{gender}_{i}_bw{float(r.get('bandwidth_hz') or 0):.0f}"
        out.append((label, p, r["text"]))
    return out


def score(f5, sents, ref_wav, ref_text, asr, utmos, normalize):
    from eval_mn import synthesise
    counts, moses = [], []
    for text in sents:
        for seed in SEEDS:
            try:
                wav, sr = synthesise(f5, text, ref_wav, ref_text, seed)
            except Exception as exc:
                print(f"      synth failed: {str(exc)[:80]}")
                continue
            counts.append(cer_counts(normalize(text), normalize(asr.transcribe(wav, sr))))
            try:
                moses.append(utmos(wav, sr))
            except Exception:
                pass
    return counts, moses


def main() -> None:
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download

    from oron_tts.eval import MongolianASR, utmos
    from oron_tts.eval.metrics import normalize_for_scoring

    lines = [ln.strip() for ln in pathlib.Path(
        "/workspace/cv/eval_sentences.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()]
    screen, confirm = lines[0::2][:SCREEN_N], lines[1::2][:CONFIRM_N]
    print(f"screen on {len(screen)} sentences, confirm on {len(confirm)} "
          f"(disjoint, x{len(SEEDS)} seeds)")

    ckpt = hf_hub_download("btsee/oron-tts", "model.safetensors")
    vocab = hf_hub_download("btsee/oron-tts", "vocab.txt")
    asr = MongolianASR(device="cuda")
    results = {}

    for gender in ("male", "female"):
        cands = candidates(gender)
        print(f"\n{'=' * 70}\n{gender}: {len(cands)} candidates\n{'=' * 70}")
        f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab,
                   device="cuda", use_ema=False)

        screened = {}
        for label, wav, rtxt in cands:
            c, m = score(f5, screen, wav, rtxt, asr, utmos, normalize_for_scoring)
            u = statistics.fmean(m) if m else float("nan")
            screened[label] = {"utmos": u, "utmos_values": m,
                               "cer": micro_cer(c), "wav": str(wav)}
            print(f"  {label:<28} UTMOS {u:.3f}  CER {micro_cer(c):.4f}  n={len(m)}")

        ranked = sorted(screened.items(), key=lambda kv: kv[1]["utmos"], reverse=True)
        best = ranked[0][0]
        print(f"  -> screen winner: {best}")

        # Confirm the winner and the incumbent on sentences neither was chosen on.
        confirm_set = {best, "SHIPPED"}
        confirmed = {}
        for label in confirm_set:
            wav = pathlib.Path(screened[label]["wav"])
            rtxt = next(t for lb, w, t in cands if lb == label)
            c, m = score(f5, confirm, wav, rtxt, asr, utmos, normalize_for_scoring)
            cp, clo, chi = bootstrap_ci(c, micro_cer)
            up, ulo, uhi = bootstrap_ci(m, statistics.fmean)
            confirmed[label] = {"utmos": up, "utmos_lo": ulo, "utmos_hi": uhi,
                                "utmos_values": m, "cer": cp, "cer_lo": clo,
                                "cer_hi": chi, "n": len(c), "wav": str(wav)}
            print(f"  CONFIRM {label:<26} UTMOS {up:.3f} [{ulo:.3f}, {uhi:.3f}]  "
                  f"CER {cp:.4f}  n={len(m)}")

        if best != "SHIPPED":
            d = paired(confirmed[best]["utmos_values"],
                       confirmed["SHIPPED"]["utmos_values"], statistics.fmean)
            verdict = ("BETTER" if d["separated"] and d["diff"] > 0 else
                       "WORSE" if d["separated"] else "no difference")
            print(f"  {best} - SHIPPED: {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
                  f"  {verdict}")
            confirmed["_vs_shipped"] = {**d, "winner": best, "verdict": verdict}
        else:
            print("  the shipped prompt already wins its own screen")
        results[gender] = {"screen": {k: {kk: vv for kk, vv in v.items()
                                          if kk != "utmos_values"}
                                     for k, v in screened.items()},
                           "confirm": {k: ({kk: vv for kk, vv in v.items()
                                            if kk != "utmos_values"}
                                           if isinstance(v, dict) else v)
                                       for k, v in confirmed.items()}}
        del f5
        import torch
        torch.cuda.empty_cache()

    pathlib.Path("/workspace/prompt_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote /workspace/prompt_eval.json")


if __name__ == "__main__":
    main()

# Runbook

Three GPU sessions from here to a released model. Everything else is already
built and tested; what remains needs hardware.

Read [phase0-findings.md](phase0-findings.md) first — every threshold and design
choice below traces to a measurement there.

---

## Session 1 — Calibrate (cheap pod, ~1 h)

**Why first.** Every gate in `oron-cleaner/pipeline/constants.py` was set from
published figures, small samples, or reasoning about what a strict corpus needs.
**None has been checked against this corpus's actual distribution.** The full
cleaning pass is 24–48 h, so a threshold 10% too strict is an expensive thing to
discover afterwards.

Pod: **RTX A5000 24 GB**, high vCPU (audio decode is the bottleneck), 200 GB
network volume so the corpus survives pod termination and is shared with
session 2.

```bash
git clone https://github.com/btseee/oron-tts.git
git clone https://github.com/btseee/oron-cleaner.git
git clone https://github.com/SWivid/F5-TTS.git
cd oron-cleaner && pip install -e . && pip install -e ../oron-tts

printf 'API_KEY=...\nHF_TOKEN=...\n' > .env   # never commit this
pytest                                        # 93 tests, no models needed

python clean_pipeline.py --datasets cv --calibrate --limit 500 --no-upload
```

Read `output/calibration_report.txt`:

- **Independent per-gate rejection.** A clip rejected for SNR in a normal run is
  never scored for DNSMOS, so production rates are not comparable across gates.
  These are.
- **`first` column.** How often each gate would be *blamed* in a normal run. A
  gate with a high independent rate but a low first count is being masked by an
  earlier one.
- **`keeps` and `for 75%`.** What the current threshold retains on that gate
  alone, and what would retain 75%.

Then edit `constants.py` and re-run. `FILTER_POLICY_VERSION` is a hash of the
gate values, so changed thresholds invalidate cached work automatically.

**Decide before moving on:** if under ~10% of clips pass, the corpus does not
support these thresholds. Loosen the gate with the highest independent rejection
rate — usually DNSMOS or SNR — rather than everything at once.

---

## Session 2 — Build the corpus (same pod, 24–48 h)

```bash
tmux new -s clean
python clean_pipeline.py                    # cv + fleurs + mbspeech, then upload
```

Resume is keyed by clip id, so this survives interruption — rerunning re-does no
work. Progress is logged every 500 clips.

Check `output/oron_mn_strict/corpus_summary.txt` against the go/no-go criteria:

| criterion | why |
|---|---|
| ≥ 25 h total | below this a finetune has little to learn from |
| ≥ 5 h male | the binding constraint — Common Voice has 10.6 h labelled male before filtering |
| ≥ 3 male speakers | one speaker means one voice, not a male voice |

If the male floor fails, the options in order: check gender resolution in the
summary (`declared` / `propagated` / `from_f0` counts), loosen gates for
male-labelled clips only, or record 2–3 h of studio male speech — which would
move the male voice more than any other single change.

**WorldSpeech** (~221 h, 24 kHz native) is excluded as CC-BY-NC-4.0. It needs
`--datasets ws --allow-non-commercial` and makes the model non-commercial.

---

## Session 3 — Train and select (5090 or A100)

```bash
cd oron-tts
python scripts/build_f5_dataset.py --corpus ../oron-cleaner/output/oron_mn_strict
python scripts/compute_epochs.py --data ../F5-TTS/data/oron_mn_pinyin
```

Put the reported `epochs` into `configs/f5tts_mn.yaml`. It is **not** a stopping
condition — it sets the LR decay length, and being wrong is silent in both
directions.

```bash
python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt \
    --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --checkpoint-out ckpts/oron_mn/pretrained_model_1250000.safetensors

cp configs/f5tts_mn.yaml ../F5-TTS/src/f5_tts/configs/
cd ../F5-TTS && accelerate launch src/f5_tts/train/train.py --config-name f5tts_mn.yaml
```

The `pretrained_` prefix is load-bearing: `Trainer.load_checkpoint` uses it to
cold-start at update 0 and to exclude the file from checkpoint rotation.

**Smoke test first.** Stop after ~200 updates and listen to
`ckpts/.../samples/`. The audio should sound Mongolian — not silence, not
English phonotactics. `log_samples: True` is on for exactly this reason: the
previous project trained 500 epochs and logged no audio at all.

### Selecting the checkpoint

```bash
cd ../oron-tts
python scripts/eval_mn.py --sweep ../F5-TTS/ckpts/oron_mn \
    --corpus ../oron-cleaner/output/oron_mn_strict
```

**Do not ship the last checkpoint.** The paper's Tab. 9 has a 24 h model peaking
at 200k updates and degrading to twice the WER by 600k; this project's previous
run peaked at epoch 250 of 500. Training loss will not tell you which is best.

CER is reported as a ratio to the **human baseline of 0.123** — the recogniser's
own floor on correctly-transcribed human speech. Synthetic audio cannot beat it,
so a raw number against zero is meaningless.

Try `--no-ema` on early checkpoints. With decay ~0.9999 the EMA weights are
still dominated by the pretrained model for the first several thousand updates.

---

## Release

```bash
python scripts/select_voices.py --corpus ../oron-cleaner/output/oron_mn_strict \
    --top 5 --write voices/
oron-tts-infer --voice male --text "Сайн байна уу" --checkpoint <best>
oron-tts-infer --voice female --text "Сайн байна уу" --checkpoint <best>
```

**Listen before shipping.** The ranking is objective; whether a voice is pleasant
is not.

Publish to `btsee/oron-tts` as a new revision. This is a breaking change — the
existing `f5tts_best.pt` loads only via code deleted at the `v1-from-scratch`
tag — so state the old SHA in the card and remove the two stale `.pt` files
(6.85 GB each) in the same commit.

The card must say the output is **wideband ~8 kHz, not full-band**. No Mongolian
source is full-band: Common Voice's median cutoff is 7.1 kHz, FLEURS and
MBSpeech are hard-capped at 7.7 kHz. That is a property of the available data,
and a listener should not have to discover it.

---

## Things that fail silently

Each of these produces a plausible-looking model that is quietly wrong.

| | |
|---|---|
| Unextended vocab | `list_str_to_idx` maps unknown ids to **0, which is the space token** — 4.90% of Mongolian characters become spaces, with nothing logged |
| Regenerated vocab | Sorting or deduplicating misaligns all 2545 pretrained embeddings |
| `prepare_csv_wavs.py --pretrain` | Writes a vocab of only the characters in your data — 20 entries in a test run |
| Wrong `epochs` | Sets LR decay length; too high ends hot, too low reaches zero early |
| `grad_accumulation_steps > 1` | `scheduler.step()` fires per batch, compressing the LR schedule |
| v0 arch values | Every community finetune in upstream's SHARED.md predates v1 |
| Latin `ref_text` | Duration is estimated from a **UTF-8 byte-length** ratio; Cyrillic is 2 bytes/char, so output comes out ~2× too long |
| Stale Common Voice id | Each language of each release is its own dataset; the API has no search endpoint |

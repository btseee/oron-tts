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
cd oron-cleaner
pip install -e ../oron-tts                # shared text normalisation, pure stdlib
pip install -r requirements.lock          # 94 pinned packages, resolved for 3.12
pip install -e . --no-deps

printf 'API_KEY=...\nHF_TOKEN=...\n' > .env   # never commit this
pytest                                    # no models needed
python scripts/check_lockfile.py          # lock still covers pyproject
python scripts/check_ci_imports.py        # nothing needs the model stack to collect

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
| --- | --- |
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
directions. The value in the repo is a placeholder computed for a corpus that
does not exist yet, and it will be wrong for the real one.

```bash
# The pretrained file MUST land inside save_dir. Trainer.load_checkpoint
# (trainer.py:187-213) lists save_dir, and when it holds no .pt or
# .safetensors it returns early and training starts from RANDOM INIT --
# silently. That is how the previous checkpoint came to be a 336M model
# trained from scratch on 6 h of audio.
SAVE_DIR=../F5-TTS/ckpts/F5TTS_v1_Base_vocos_pinyin_oron_mn
python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt \
    --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --checkpoint-out $SAVE_DIR/pretrained_model_1250000.safetensors

python scripts/preflight.py --data ../F5-TTS/data/oron_mn_pinyin

cp configs/f5tts_mn.yaml ../F5-TTS/src/f5_tts/configs/
cd ../F5-TTS && accelerate launch src/f5_tts/train/train.py --config-name f5tts_mn.yaml
```

**Preflight is the last gate before the GPU bill starts.** It refuses a stale
`epochs`, an unextended or reordered vocabulary, a tokenizer that sends
`load_dataset` and `get_tokenizer` to different directories,
`grad_accumulation_steps > 1`, and `log_samples` off. Every one of those
completes the run and produces a worse model with nothing in the logs.

The `pretrained_` prefix is load-bearing: `Trainer.load_checkpoint` uses it to
cold-start at update 0 and to exclude the file from checkpoint rotation.

**Smoke test first.** Stop after ~200 updates and listen to
`ckpts/.../samples/`. `log_samples: True` is on for exactly this reason: the
previous project trained 500 epochs and logged no audio at all.

Three things to listen for, in order:

1. **Mongolian phonotactics** — not silence, not English.
2. **Letter-spelling.** Upstream: *"Uppercased letters (best with form like
   K.F.C.) will be uttered letter by letter"*. Every Mongolian sentence starts
   with a capital, and this corpus preserves case, so the finetune has to weaken
   that pretrained prior on essentially every utterance. If the first word comes
   out spelled rather than spoken, lowercase the corpus text — one line in
   `MongolianNormalizer` — and restart. Catching this at 200 updates costs an
   hour; catching it at the end costs the run.
3. **Digits read as digits.** The normaliser expands them; if you hear
   "тав" where the text said "5", good. If you hear nothing there, the
   corpus was built with an unextended vocabulary.

### Selecting the checkpoint

Two passes, and the order matters.

```bash
cd ../oron-tts

# 1. Choose. Validation speakers, even half of the held-out sentences.
python scripts/eval_mn.py --sweep ../F5-TTS/ckpts/oron_mn \
    --corpus ../oron-cleaner/output/oron_mn_strict

# 2. Report. Test speakers, odd half. Only the winner, only once.
python scripts/eval_mn.py --checkpoint ../F5-TTS/ckpts/oron_mn/model_<best>.pt \
    --corpus ../oron-cleaner/output/oron_mn_strict --rtf --ground-truth
```

`--mode` defaults to `select` for a sweep and `report` for a single checkpoint,
and each mode gets its own speakers *and* its own half of `eval_sentences.txt`.
Sweeping a dozen checkpoints and then publishing the winner's score on the same
sentences is selection on the test set — the winner is partly whichever
checkpoint got lucky there, and its number is optimistic by however much luck
was involved.

This is also the only thing that reads the validation split. The F5-TTS trainer
has no validation loop — zero references to `val_dataloader`, `validation` or
`val_loss` — so `metadata_validation.csv` was written and never opened.

**Do not ship the last checkpoint.** The paper's Tab. 9 has a 24 h model peaking
at 200k updates and degrading to twice the WER by 600k; this project's previous
run peaked at epoch 250 of 500. Training loss will not tell you which is best.

Four numbers per gender, each answering a different question:

| | question | how to read it |
| --- | --- | --- |
| CER | is it intelligible? | **contaminated** — see below. Ratio to the **human baseline of 0.123** — the recogniser's own floor on correctly-transcribed human speech. Synthetic audio cannot beat it, so a raw number against zero is meaningless. Scored on `eval_sentences.txt`, which no training clip contains. |
| SIM-o | is it the right voice? | cosine similarity to the prompt, WavLM-large ECAPA-TDNN. The paper reports 0.66 for F5-TTS on LibriSpeech-PC test-clean; the ground truth there is 0.69. |
| UTMOS | does it sound natural? | a proxy trained on English/Japanese MOS, never validated for Mongolian — treat it as a regression detector, not a MOS |
| bandwidth | is it dull? | follows the prompt; no Mongolian source is full-band |

Each is averaged over `--seeds` (default `0 1 2`, the paper's three-seed
protocol) and reported with a 95% CI. When two checkpoints' intervals overlap,
the harness says so instead of naming a winner.

**The CER scorer is contaminated, and cannot simply be swapped.**
`bayartsogt/wav2vec2-large-xlsr-mongolian` is, per its own model card,
fine-tuned on Common Voice Mongolian — the corpus this model trains on. It is
both oron-cleaner's gate and this harness's scorer, so it has seen the training
speakers and sentences and has not seen FLEURS' or MBSpeech's. Two consequences:

- the **gate** is lenient on Common Voice and strict on the others, biasing
  corpus composition by source rather than by quality. `corpus_summary.txt`
  now prints CER by source so the gap is visible — a markedly lower median for
  `cv` on audio of comparable quality *is* the contamination;
- the **evaluation** inherits it, in the model's favour.

It is kept as the default because it is the best Mongolian recogniser available
(CER 0.123 median against whisper-large-v3's 0.311 on the same clips), and
replacing it would move the baseline every number here is quoted against. Take a
second opinion instead:

```bash
python scripts/eval_mn.py --checkpoint <best> --corpus <dir>     --asr-model facebook/mms-1b-all --baseline <its own human floor>
```

Measure that model's own floor on the same held-out human audio first, or the
ratio means nothing. **State the contamination in the model card** — a CER
quoted without it reads as ~15% better than it is.

**SIM-o needs a manual download.** `wavlm_large_finetune.pth`, linked from
`F5-TTS/src/f5_tts/eval/README.md` under *Download Evaluation Model
Checkpoints*. Point `--sim-checkpoint` or `ORON_WAVLM_CKPT` at it. Without it
the run continues and says SIM-o is unavailable — it will not substitute another
speaker model, because the number is only comparable to the paper's if it comes
from the same one.

Try `--no-ema` on early checkpoints. With decay ~0.9999 the EMA weights are
still dominated by the pretrained model for the first several thousand updates.

`--rtf` times the winner at NFE 8/16/32 and reports real-time factor, median and
p90 latency, and peak memory. The paper reports RTF 0.15 at NFE 16 on datacentre
hardware; a 336M DiT solving an ODE is not obviously real time anywhere else.
Read it beside the CER at the same NFE — the setting that buys the speed is the
one that costs the quality.

---

## Release

```bash
python scripts/select_voices.py --corpus ../oron-cleaner/output/oron_mn_strict \
    --top 5 --write voices/
oron-tts-infer --voice male --text "Сайн байна уу" --checkpoint <best>
oron-tts-infer --voice female --text "Сайн байна уу" --checkpoint <best>
```

**Listen before shipping.** The ranking is objective; whether a voice is pleasant
is not. `--ground-truth` scores the held-out human audio with the same
instruments, which is the ceiling every other number should be read against —
the paper reports that row in every table, and without it a CER of 0.19 is
either close to the ceiling or twice it with nothing to say which.

For a naturalness claim rather than an intelligibility one, follow
[listening-test.md](listening-test.md): the paper's CMOS/SMOS protocol, 20
native Khalkha listeners, 30 randomised rounds, against ground truth and an
external baseline.

Publish to `btsee/oron-tts` as a new revision. This is a breaking change — the
existing `f5tts_best.pt` loads only via code deleted at the `v1-from-scratch`
tag — so state the old SHA in the card and remove the two stale `.pt` files
(6.85 GB each) in the same commit.

Fill in [model-card.md](model-card.md) and publish it with the weights: it is
written with every measured fact already in place and the reporting run's
numbers left as blanks, so releasing without it is a choice rather than an
oversight.

The card must say the output is **wideband ~8 kHz, not full-band**. No Mongolian
source is full-band: Common Voice's median cutoff is 7.1 kHz, FLEURS and
MBSpeech are hard-capped at 7.7 kHz. That is a property of the available data,
and a listener should not have to discover it.

---

## Things that fail silently

Each of these produces a plausible-looking model that is quietly wrong.

| | |
| --- | --- |
| Unextended vocab | `list_str_to_idx` maps unknown ids to **0, which is the space token** — 4.90% of Mongolian characters become spaces, with nothing logged |
| Regenerated vocab | Sorting or deduplicating misaligns all 2545 pretrained embeddings |
| `prepare_csv_wavs.py --pretrain` | Writes a vocab of only the characters in your data — 20 entries in a test run |
| Wrong `epochs` | Sets LR decay length; too high ends hot, too low reaches zero early |
| `grad_accumulation_steps > 1` | `scheduler.step()` fires per batch, compressing the LR schedule |
| v0 arch values | Every community finetune in upstream's SHARED.md predates v1 |
| Latin `ref_text` | Duration is estimated from a **UTF-8 byte-length** ratio; Cyrillic is 2 bytes/char, so output comes out ~2× too long |
| Stale Common Voice id | Each language of each release is its own dataset; the API has no search endpoint |

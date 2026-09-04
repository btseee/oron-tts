# One model, four corpora — design

**Date:** 2026-09-04
**Status:** approved, pending implementation

Produce a single Mongolian TTS model trained cumulatively over four corpora,
with one fixed male voice and one fixed female voice, published to
`btsee/oron-tts` with complete logs and demos and nothing else.

Everything currently in that repo is deleted first. The `v1` branch stays as
provenance for the retired from-scratch model.

---

## Why this shape

Four decisions were settled before design:

1. **Cumulative, not sequential.** Stage *n* trains on corpora 1..*n*. Training
   each corpus in isolation would end the run on WorldSpeech — the noisiest and
   most misaligned source — and let it overwrite everything learned from the
   clean ones.

2. **WorldSpeech transcripts get trimmed by forced alignment.** Measured on 800
   clips: 6.9% pass the gates, and among clips that *pass*, 80% have audio
   shorter than their transcript and 53% end mid-word. The segments are
   misaligned with their text. Training on that teaches the model to speak words
   that are not in the recording, and the CER gate cannot see it because the
   wrong string is the reference.

3. **Both fixed voices come from WorldSpeech, held out of training.** It is
   24 kHz native — the only Mongolian source with real full-band content. Output
   bandwidth follows the reference prompt, so every other source caps the shipped
   voices at ~7.7 kHz. Holding them out keeps CER and any similarity number
   honest.

4. **One model file, best-checkpoint handoff.** Each stage publishes to the same
   `model.safetensors`. The next stage resumes from the stage's *best* checkpoint
   by CER, not its last, so overfitting drift does not compound across four
   stages.

---

## Corpora

| stage | adds | clips | hours | notes |
| --- | --- | --- | --- | --- |
| 1 | MBSpeech | ~2,800 | ~5.0 | one male narrator, 16 kHz native |
| 2 | + FLEURS | ~1,900 | ~4.3 | both genders, 16 kHz native |
| 3 | + Common Voice 26 | ~15,100 | ~16.3 | 378 speakers, both genders |
| 4 | + WorldSpeech | 10,000-60,000 | 8-50 | 24 kHz native, full-band |

Stage 4's range is wide because it is the open empirical question: untrimmed,
6.9% of WorldSpeech survives the gates (~9,500 clips, ~7 h); the trimming exists
to raise that, and how far is not knowable without running it.

**Decision rule.** A 400-clip calibration runs first. If the trimmed pass rate
is below 15%, the full pass is not worth $18 and the run stops at stage 3 with
WorldSpeech used only to source the two reference voices. At or above 15% the
full pass proceeds. Either way the outcome is reported before the money is
spent.

CV26 and FLEURS corpora already exist on the volume and are reused. MBSpeech
must be re-cleaned — its corpus died with an earlier volume.

---

## Components

### `oron-cleaner`

**New: `pipeline/trimming.py`**

MMS-FA already runs as the transcript gate and produces per-token timings. This
module uses them to cut the transcript to the span the audio actually covers,
then lets the normal gates re-score the trimmed pair.

- Input: audio, transcript, alignment result.
- Output: trimmed transcript, plus `text_trimmed` and `text_discarded` recorded
  on the clip so a corpus can be audited for how much was cut.
- Runs for every source. Where alignment is already tight it is a no-op, so it
  does not need a per-source flag.
- Refuses to trim more than a configurable fraction (default 40%): a clip whose
  transcript is mostly wrong is a broken clip, not a trimming opportunity.

**Unchanged:** the gate chain, parquet packaging, card generation.

### `oron-tts`

**New: `scripts/train_curriculum.py`** — the whole run.

Per stage: build the F5 dataset from the union of corpora so far, compute
epochs, preflight, train, sweep, select best checkpoint by CER, publish, hand
that checkpoint to the next stage.

**New: `scripts/make_demos.py`** — synthesises a fixed sentence list with both
voices and writes `demos/`.

**Replaced: `configs/oron.yaml`** — one config. `f5tts_mn.yaml` and
`f5tts_mbspeech.yaml` are deleted.

**Logging.** Every stage writes to `/workspace/logs/<stage>.log` and to stdout,
so the container log and `runpodctl pod logs` both carry the full run.

**TensorBoard.** F5-TTS logs loss and learning rate. The orchestrator adds:
stage boundaries as markers, gradient norm, and the per-stage evaluation CER for
each gender as scalars — so the tab shows the curriculum, not one loss curve.
The `tfevents` directory is published with the model.

---

## Pipeline

```
re-clean MBSpeech ──┐
reuse CV26, FLEURS ─┤
clean WorldSpeech ──┴─→ publish btsee/WorldSpeech-mn
      (trimmed)          │
                         ├─→ select 2 full-band voices, exclude from training
                         │
  stage 1  MBSpeech                  → sweep → best → publish
  stage 2  + FLEURS                  → sweep → best → publish
  stage 3  + Common Voice 26         → sweep → best → publish
  stage 4  + WorldSpeech             → sweep → best → publish + demos
                         │
                         └─→ verify from the server → delete pod
```

**Termination is gated on verified publication, never on elapsed time.** A
watchdog exists only as a crash backstop, set well beyond the expected finish.

---

## Published layout

```
btsee/oron-tts
├── README.md                model card with measured numbers
├── config.json              architecture, mel, tokenizer, inference defaults
├── model.safetensors        ONE model, raw (non-EMA) weights
├── vocab.txt                2,550 entries
├── voices/
│   ├── male.wav   male.txt
│   └── female.wav female.txt
├── demos/                   both voices, several sentences
├── tensorboard/             full curriculum
├── logs/train.log
└── eval.json                per-stage CER per gender, with confidence intervals
```

Nothing else. The 39 files and 4.05 GB currently there are removed.

---

## Failure handling

- **Every stage is resumable.** Completion markers, not "output exists" checks —
  a partial manifest previously satisfied an existence check and a whole corpus
  pass was silently skipped.
- **Publication is verified from the server** before the next stage starts, and
  before the pod is deleted.
- **Refuse rather than ship a bad artifact:** no publish without a
  `.safetensors`; no shard whose parquet row groups exceed the Hub's scan limit;
  no training launch while preflight objects.
- **Raw weights, always.** The evaluator and the publisher both take
  `model_state_dict`. Measured: EMA weights sit 2.78% off the pretrained weights
  after 30,000 updates and synthesise fluent non-words at CER 0.921 against 0.026
  for the raw tensors, and the failure is inaudible.
- **On early failure the pod stays up.** Terminating on a fault destroys the log
  that explains it and costs a pod recreation per iteration.

---

## Testing

- `trimming.py`: a clip whose transcript overruns its audio is trimmed to the
  spoken span; a clip already aligned is unchanged; a clip needing >40% removal
  is rejected rather than trimmed.
- `train_curriculum.py`: stage handoff uses the best checkpoint by CER, not the
  last; a stage cannot start before the previous stage's publication verifies.
- Config: exactly one config file exists; it names `logger: tensorboard` and the
  package declares tensorboard.
- Both suites (348 + 226) stay green.

---

## Cost and time

| item | GPU | cost |
| --- | --- | --- |
| re-clean MBSpeech | 0.7 h | $0.50 |
| clean WorldSpeech, 138,529 clips with trimming | 24.6 h | $18.20 |
| 4-stage training, ~55,000 updates | 5.1 h | $3.77 |
| 4 sweeps, demos, publication | 3.0 h | $2.22 |
| 100 GB volume, ~2 days | — | $0.46 |
| retries (EU-RO-1 wedged five pods on 2026-09-04) | | ~$5.00 |
| **total** | **~34 h** | **~$30** |

Balance at design time: $16.69. **Top-up required: $15.**

Hardware: 1 × RTX 4090 24 GB, RunPod secure cloud, EU-RO-1 (the network volume
is pinned to that datacenter).

---

## Known limitations, stated up front

- **WorldSpeech is CC-BY-NC-4.0.** The published `WorldSpeech-mn` carries that
  licence, and a model trained on it inherits the non-commercial restriction.
  This was accepted deliberately.
- **Gender labels on WorldSpeech are inferred from F0**, not self-reported.
  `gender_source` records this per clip.
- **No listening test.** UTMOS is a proxy trained on English and Japanese MOS
  data and has never been validated for Mongolian.
- **SIM-o requires a manual WavLM download.** If absent, speaker similarity is
  reported as unmeasured rather than substituted.
- **The CER recogniser was fine-tuned on Common Voice Mongolian**, which is
  stage 3 of this curriculum. Read CER against the ground-truth row, not zero.

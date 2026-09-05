# Publication and logging

Design for three related pieces of work, all downstream of a finished training
run: complete the HuggingFace metadata, shorten the model card to something a
person can act on, and make TensorBoard show what the run actually did.

Written 2026-09-05, after the four-stage curriculum completed and the pod
self-terminated. No further training is in scope. Everything here operates on
artifacts that already exist, plus code changes that take effect on the next
run.

## Context

The model at `btsee/oron-tts` is one F5-TTS finetune trained cumulatively over
three corpora, with a final low-LR stage on two speakers to lock the voices.

| stage | corpora | updates | outcome |
| --- | --- | --- | --- |
| mbspeech | MBSpeech | 8,000 | published |
| fleurs | + FLEURS | 8,275 | best `model_8000`, mean CER 0.0958 |
| cv | + Common Voice 26 | 16,150 | best `model_12000`, mean CER 0.0824 |
| voicelock | the two chosen speakers | 4,142 | `model_4000`, chosen by fallback |

Measured on the withheld split, 20 sentences x 3 seeds per gender:

| | male | female |
| --- | --- | --- |
| CER | 0.0633 | 0.1015 |
| UTMOS | 2.48 | 2.11 |
| speaker similarity to own prompt | 0.725 | 0.808 |

Speaker similarity between the two voices is 0.103. Calibrated on this project's
own recordings, genuine same-speaker pairs score 0.540-0.833 and
different-speaker pairs 0.034-0.503, so both voices match their prompts as
closely as two real recordings of one person, and differ from each other as much
as two strangers.

## Problem 1 — the metadata is thin and one field is wrong

`btsee/oron-tts` carries seven frontmatter keys. It has no `datasets`, no
`metrics`, no `model-index`, so the Hub renders no Eval Results panel and no
links to the corpora it was trained on. Every number lives in prose or in
`eval.json`, where the Hub cannot see it.

The `license` is also **wrong, in the direction that matters**. It says
`cc-by-nc-4.0`, inherited from WorldSpeech — a corpus that **failed the 15%
pass-rate gate and was never trained on**. The model saw MBSpeech (MIT), FLEURS
(CC-BY-4.0) and Common Voice (CC0), all commercial-safe. Commercial safety was
the constraint that caused WorldSpeech to be excluded in the first place, so the
card currently gives away the exact thing the corpus selection was designed to
protect.

## Problem 2 — the model card is prose where it should be instructions

4,191 characters, opening with an architecture description and a stage name. A
reader who wants to make it speak has to get past the EMA analysis, the voice
selection table and the corpus table first. There are no playable demos, and no
link to the source repositories.

## Problem 3 — TensorBoard shows almost nothing

Inspected with `EventAccumulator` against the four published files:

```
events...2707.0   scalars=[]                  <- empty file
events...4353.0   scalars=['loss','lr']       fleurs     8,275 points
events...5803.0   scalars=['loss','lr']       cv        16,150 points
events...7668.0   scalars=['loss','lr']       voicelock  4,142 points
```

Four distinct faults:

1. **All four files sit flat in one directory.** TensorBoard derives runs from
   subdirectories. Flat files cannot be selected, named, or overlaid, so even
   the two scalars that exist are hard to read and impossible to compare across
   stages. This is the largest single defect and it is pure layout.
2. **One file is empty** — the aborted first `fleurs` attempt, which died on the
   seed-layout bug before its first update.
3. **The mbspeech curves are gone.** That stage trained on the first pod;
   `publish_model.py` restages the whole tensorboard directory from the current
   machine, so the second pod's publish replaced the set. Unrecoverable.
4. **Only `loss` and `lr` are logged.** Upstream's trainer writes exactly two
   scalars (`trainer.py:398-400`). No grad norm, no audio, no mel images, no
   hparams. Most importantly **no evaluation metrics**: every CER and UTMOS
   number produced by the checkpoint sweeps is written to `eval.json` and never
   reaches TensorBoard, so quality cannot be read against training progress —
   which is the main thing the tab is for.

The trainer *does* synthesise sample audio during training, but writes it to
`ckpts/oron/samples/*.wav` and never calls `add_audio`. Those files died with
the pod.

## Design

### Model card

Frontmatter carries the detail, so the body can stay short.

```yaml
language: [mn]
license: cc-by-4.0
library_name: f5-tts
pipeline_tag: text-to-speech
base_model: SWivid/F5-TTS
base_model_relation: finetune
datasets: [btsee/mbspeech-mn, btsee/fleurs-mn, btsee/common-voice-26-mn]
metrics: [cer, utmos]
tags: [text-to-speech, tts, mongolian, khalkha, cyrillic,
       flow-matching, f5-tts, dit, vocos, voice-cloning]
model-index:
- name: oron-tts
  results:
  - task: {type: text-to-speech, name: Text-to-Speech}
    dataset:
      type: btsee/common-voice-26-mn
      name: Common Voice 26 Mongolian (cleaned)
      split: withheld
    metrics:
    - {type: cer, value: 0.0633, name: "CER, male voice"}
    - {type: cer, value: 0.1015, name: "CER, female voice"}
    - {type: utmos, value: 2.48, name: "UTMOS, male voice"}
    - {type: utmos, value: 2.11, name: "UTMOS, female voice"}
    - {type: cosine_similarity, value: 0.725, name: "Speaker similarity, male"}
    - {type: cosine_similarity, value: 0.808, name: "Speaker similarity, female"}
```

`new_version` is deliberately **omitted**. It declares that a successor
repository supersedes this one, and the Hub renders a banner sending visitors
there. No successor exists, so setting it would send every visitor to a dead
end. Lineage is expressed by `base_model` plus `base_model_relation`, and the
stage is recorded in `config.json`.

Body, in this order, and nothing else:

1. One paragraph: what it is, what it speaks, the two voices.
2. Two playable demos, `<audio controls>` against
   `https://huggingface.co/btsee/oron-tts/resolve/main/demos/{male,female}.wav`.
3. Install — two `pip install` lines.
4. One copy-paste Python block that produces a wav file.
5. The two gotchas that silently ruin output, one line each: `use_ema=False`,
   and normalise the text.
6. Links: both GitHub repositories, the three datasets.
7. Licence, one line.

The gotchas stay in the short card because both fail *silently* — EMA weights
synthesise confident non-words at CER 0.921 against 0.026, and characters
outside the vocabulary are read as spaces because unknown ids map to index 0,
which is the space token. A reader who skips them gets plausible-sounding
output that is wrong, so they are usage instructions, not background.

### Dataset cards

All four keep their long form; the request there was full detail. They gain the
standard descriptive metadata they lack — `annotations_creators`,
`language_creators`, `multilinguality`, `source_datasets`, `task_ids` — and a
"Used by" section linking to `btsee/oron-tts`.

`WorldSpeech-mn` keeps `cc-by-nc-4.0` and gains an explicit note that the model
does **not** train on it. Without that, a reader who sees an NC dataset beside a
CC-BY model will reasonably assume one of the two labels is a mistake.

### TensorBoard

Two components, because the run is over and the next one has not started.

**`scripts/tb_report.py`** — rebuilds the published tree from surviving
artifacts. Writes `tensorboard/<stage>/` per stage, which is what makes runs
selectable at all, and into each:

* the existing loss/lr events, copied verbatim;
* `eval/cer_male`, `eval/cer_female`, `eval/cer_mean`, `eval/utmos_male`,
  `eval/utmos_female`, `eval/bandwidth_male`, `eval/bandwidth_female`, each
  plotted against the checkpoint's update number, read from `eval.json`;
* `corpus/clips`, `corpus/hours`, `corpus/speakers`, `corpus/male_hours`,
  `corpus/female_hours`;
* `add_hparams` — target updates, epochs, learning rate, frames per step,
  corpus hours — against final CER, so the HPARAMS tab compares stages;
* `add_text` naming the corpora in the stage and the checkpoint chosen, and
  whether it was chosen by CER or by fallback.

Plus a `summary/` run holding `add_audio` for both demos and both reference
prompts, `add_image` of their mel spectrograms, and the speaker-similarity
table as `add_text` and scalars.

The empty tfevents file is dropped. The missing mbspeech stage is stated in the
summary text rather than silently absent — a reader counting three stages in a
four-stage curriculum should be told why.

**`scripts/patch_trainer_logging.py`** — applies the in-loop logging upstream
cannot be asked for. F5-TTS is refetched fresh on every pod, so this is a patch
script rather than a fork: it locates each insertion point, applies it, and
**asserts the edit landed**, failing loudly otherwise. A patch that silently
no-ops is worse than no patch, because the run then looks instrumented and is
not. It adds:

* `train/grad_norm` from the value `clip_grad_norm_` already returns;
* `add_audio` of the sample the trainer already synthesises, at the update it
  was generated;
* `add_image` of that sample's mel spectrogram;
* `add_hparams` at the end of training.

### Testing

* `tb_report.py` is tested against a fixture `eval.json` and a synthetic events
  file: assert one subdirectory per stage, assert the eval scalars exist with
  the expected step numbers, assert the empty file is dropped, assert audio tags
  are present in the summary run. Read back with `EventAccumulator`, which is
  what a reader's TensorBoard uses, rather than trusting the writer.
* `patch_trainer_logging.py` is tested against a copy of upstream `trainer.py`:
  assert every insertion applied, and assert the script **raises** when given a
  file whose anchors have moved. Mutation check: remove one anchor, expect a
  failure.
* The card frontmatter is validated by parsing the YAML back and asserting the
  `model-index` metric values match `eval.json`, so the panel cannot drift from
  the measurements.

## Out of scope

* Retraining, or any GPU spend.
* Recovering the mbspeech curves.
* SIM-o on the paper's scale — it needs a 1.3 GB WavLM checkpoint and an s3prl
  fetch through `torch.hub` that prompts for trust and so cannot run
  unattended. `speaker_similarity_any()` already falls back and reports which
  metric produced the number.
* Fixing the voice-lock sweep that produced no scoreable output. It is recorded
  on the card as a fallback selection; diagnosing it needs a pod.

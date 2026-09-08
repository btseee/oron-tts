# Working in oron-tts

Mongolian (Khalkha Cyrillic) text-to-speech. The model is a **finetune of
upstream F5-TTS `F5TTS_v1_Base`** — training runs in the `f5_tts` package, not
here. This repository owns text normalisation, the vocabulary contract,
evaluation, and the packaged reference voices.

Published artifacts live at `btsee/oron-tts` on HuggingFace. The corpus pipeline
is a separate repository, `oron-cleaner`.

## Five things that fail silently

Each of these has cost a real run. None of them raises.

**1. `use_ema=False`, always.** The EMA weights synthesise fluent non-words —
CER 0.921 against 0.026 for the raw tensors — and they *sound* like confident
speech, so you will not hear the mistake. `oron_tts/infer.py` defaults to
`False`; do not flip it. The cause is mechanical: F5-TTS's default
`ema_decay=0.9999` has a half-life of 6,931 updates, and the final voice-lock
stage ran 4,142, so the EMA is still dominated by the pretrained weights.

**2. Out-of-vocabulary characters become spaces.**
`f5_tts.model.utils.list_str_to_idx` maps anything absent from `vocab.txt` to
index 0, and index 0 is the space token — there is no `<unk>`, and nothing logs
it. On the unextended base vocabulary this was 4.90% of all tokens, because `ө`
and `ү` are ordinary Mongolian vowels. Any path turning text into ids must call
`oron_tts.text.check` first.

**3. Vocabulary order is load-bearing.** Every pretrained embedding row is
addressed by position. New tokens are *appended*; the 2545 base entries keep
their exact indices. Never sort, deduplicate, or regenerate the vocabulary.
`tests/test_vocab_coverage.py` enforces this.

**4. CER has a floor of 0.123.** That is what
`bayartsogt/wav2vec2-large-xlsr-mongolian` scores on real human speech with
human transcripts. Both shipped voices already score below it, so CER cannot
resolve quality differences — it is a guardrail against fluent nonsense, not a
ranking metric. Rank on UTMOS with bootstrap intervals. The same recogniser was
fine-tuned on Common Voice, which is most of the training data, so it is
contaminated in the model's favour.

**5. `bandwidth_hz` was censored before filter policy v4.** The corpus pipeline
decoded every source to 16 kHz before measuring, so the column could not exceed
8 kHz for any corpus and the published audio was band-limited to match. Any
corpus built under v3 or earlier carries a bandwidth column that is the
truncation, not the source. `select_voices.py` ranks bandwidth first, so on a
pre-v4 corpus that ranking never discriminated. Check
`<corpus>/provenance.json` → `filter_policy_version` before trusting it.

## How to run things

```bash
python -m pytest tests/ -q                    # 440+ tests, all must pass
python scripts/eval_mn.py --checkpoint <ckpt> --corpus <dir>
python scripts/select_voices.py --corpus <dir> --top 5 --write voices/
python scripts/model_card.py --eval eval.json --consistency consistency.json \
    --measured eval_n100.json --out README.md
```

`scripts/build_f5_dataset.py` refuses a corpus whose
`provenance.json` filter policy does not match the installed oron-cleaner. That
gate is deliberate: pooling corpora built under different thresholds produces a
silently half-truncated dataset.

## Evaluation rules that are not optional

- **Seed everything.** `CFM.sample` draws its own seed when none is given, so
  an unseeded harness compares checkpoints under different noise.
- **Split the sentence list.** `load_test_sentences(mode="select")` takes even
  indices, `"report"` odd ones. Reporting a winner's score on the sentences that
  selected it is selection on the test set.
- **Micro-CER, not a mean of ratios.** A mean weights a four-character line the
  same as a sixty-character one. `oron_tts.eval.metrics.micro_cer`.
- **Percentile bootstrap, not `1.96 * sd / sqrt(n)`.** CER is a bounded ratio
  with a long tail; the normal approximation reports symmetric bounds for a
  distribution that is not, and can put the lower bound below zero.
- **Pair comparisons on the same sentences.** `paired_bootstrap`. An unpaired
  interval carries the between-sentence variance twice and hides real
  differences.

## The card

`scripts/model_card.py` generates what ships. `docs/model-card.md` is its
long-form companion and is **not** published. These two diverged once and the
published copy was the one missing the consent and contamination caveats, so
`tests/test_model_card.py` now asserts every disclosure survives generation, and
that the body stays under 4,200 characters. That bound has only ever moved for
measurement — intervals and sample sizes — never for prose.

Numbers in the card come from a measurement file, never typed in. The
frontmatter feeds the Hub's Eval Results panel and is asserted to agree with the
body table.

## Where the evidence lives

- `docs/phase0-findings.md` — every threshold and how it was measured. Read
  before changing anything about text or the vocabulary.
- `docs/normaliser-review.md` — the numeral-suffix table that needs a native
  Khalkha speaker. The normaliser refuses rather than guesses.
- `docs/runbook.md` — end-to-end procedure.
- `eval/` on the Hub — held-out sentence lists, per-utterance results.
  Untracked here on purpose; it is regenerated by every run.

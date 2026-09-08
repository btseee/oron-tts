# Working in oron-tts

Mongolian (Khalkha Cyrillic) text-to-speech. The model is a **finetune of
upstream F5-TTS `F5TTS_v1_Base`** — training runs in the `f5_tts` package, not
here. This repository owns text normalisation, the vocabulary contract,
evaluation, and the packaged reference voices.

Published artifacts live at `btsee/oron-tts` on HuggingFace. The corpus pipeline
is a separate repository, `oron-cleaner`.

**The published corpus text, the text CER is scored against, and the text
training reads are one string.** `oron_tts.text` is what makes that one string
possible: it has no dependencies of its own, so oron-cleaner can import the
same normaliser without pulling torch into a data pipeline. Do not add a heavy
import to that package — `scripts/check_ci_imports.py` gates it.

## Six things that fail silently

Each of these has cost a real run. None of them raises.

**1. `use_ema=False`, always.** The EMA weights synthesise fluent non-words —
CER 0.921 against 0.026 for the raw tensors — and they *sound* like confident
speech, so you will not hear the mistake. `oron_tts/infer.py` defaults to
`False`; do not flip it. Measured on a 30,000-update finetune, the EMA weights
sat 2.78% off the pretrained weights — they had not converged, and why has
never been established. Do not assume more updates fixes it.

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
pre-v4 corpus it still ranks, but the numbers are not source bandwidth. Check
`<corpus>/provenance.json` → `filter_policy_version` before trusting it.

**6. A publish call reporting success does not mean anything shipped.** This
project has published an artifact the upload call reported as successful and
that was not there, a dataset whose upload succeeded with 19 clips, and a
delete-before-upload bug that emptied the remote TensorBoard tree while
reporting success. `scripts/publish_docs.py` now reads the server back after
every publish — the server's view is the only view that counts.

## Text and vocabulary

- **Never delete characters to make text fit.** An earlier `remove_invalid_chars`
  dropped Latin homoglyphs, so a Latin character typed into a Cyrillic word
  vanished from the text while the speaker still pronounced it — manufacturing
  the exact text/audio mismatch the CER gate exists to detect. Unrepresentable
  text now raises; see `oron_tts/text/normalizer.py`.
- **Mongolian Cyrillic is not `[а-яА-Я]`.** That range is U+0410–U+044F and
  excludes `ө` U+04E9 and `ү` U+04AF, two ordinary Mongolian vowels. Use
  `oron_tts.text.numbers.MN_LETTERS`.
- **Do not revive the from-scratch model or load F5-TTS weights into it.** It
  could not load upstream weights and its text handling defeated the
  architecture; it is recoverable at the `v1-from-scratch` tag but is history,
  not a starting point (`README.md`, Status section).

Conventions: Python ≥3.12, ruff line length 100, isort with
`known-first-party = oron_tts`. Tests are hermetic — fixtures are committed, no
network.

## How to run things

```bash
ruff check .                                   # CI gate
python scripts/check_ci_imports.py             # CI gate: no test may reach torch at import time
python -m pytest tests/ -q                     # 440+ tests, all must pass
python scripts/eval_mn.py --checkpoint <ckpt> --corpus <dir>
python scripts/select_voices.py --corpus <dir> --top 5 --write voices/
python scripts/model_card.py --eval eval.json --consistency consistency.json \
    --measured eval_n100.json --out README.md
```

`scripts/build_f5_dataset.py` refuses a corpus whose
`provenance.json` filter policy does not match the installed oron-cleaner —
deliberately: pooling corpora built under different thresholds produces a
silently half-truncated dataset. But that refusal is a no-op, not a gate, when
`pipeline.constants` cannot be imported: on a training pod with `f5_tts` and
`oron-tts` but not `oron-cleaner` — the normal training environment — a v3
corpus passes with no message at all.

## Evaluation rules that are not optional

- **Seed everything.** `F5TTS.infer` draws its own seed when none is given, so
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
- `eval/` on the Hub — held-out sentence lists, preserved so "select" and
  "report" halves stay fixed across runs. Untracked here on purpose; the
  *results* under it, not the sentence lists, are what each run rewrites.
  `consistency.json` is the exception: it lives under `demos/`, not `eval/`.

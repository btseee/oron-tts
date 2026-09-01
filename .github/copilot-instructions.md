# Working in this repository

Mongolian (Khalkha Cyrillic) TTS. The model is a **finetune of upstream F5-TTS
`F5TTS_v1_Base`** — training runs in the `f5_tts` package, not here. This
repository owns text normalization, the vocabulary contract, evaluation, and the
packaged reference voices.

Read [docs/phase0-findings.md](../docs/phase0-findings.md) before changing
anything about text or the vocabulary. Every number below is measured.

## Things that are true and easy to get wrong

**Out-of-vocabulary characters become spaces, silently.**
`f5_tts.model.utils.list_str_to_idx` maps anything absent from `vocab.txt` to
index 0, and index 0 is the space token — there is no `<unk>`. Nothing logs it.
On the unextended base vocabulary this was 4.90% of all tokens, because `ө` and
`ү` are ordinary Mongolian vowels. Any code path turning text into ids must call
`oron_tts.text.check` first.

**Vocabulary order is load-bearing.** Every pretrained embedding row is addressed
by position. New tokens are *appended*; the 2545 base entries keep their exact
indices. Never sort, deduplicate or regenerate the vocabulary — a "sorted unique
characters" rebuild misaligns all of them. `tests/test_vocab_coverage.py`
enforces this.

**Never delete characters to make text fit.** The previous `remove_invalid_chars`
dropped anything outside a hardcoded set, so a Latin homoglyph typed into a
Cyrillic word vanished from the text while the speaker still pronounced it —
manufacturing exactly the text/audio mismatch the CER gate exists to detect.
Unrepresentable text must raise, or be reported via `unsupported_chars` so the
caller can reject the row *with a recorded reason*.

**Latin is in the vocabulary.** All 22 Latin characters occurring in the corpus
have pretrained embeddings, and Latin appears in only 2.7% of rows. Keep it.

**Case is preserved.** Both cases of Cyrillic have pretrained embeddings.
Lowercasing discards 31 trained rows to save 3 appended ones.

**Mongolian Cyrillic is not `[а-яА-Я]`.** That range is U+0410–U+044F and
excludes `ө` U+04E9 and `ү` U+04AF. Use `oron_tts.text.numbers.MN_LETTERS`.

**One text string, three consumers.** The corpus text published by oron-cleaner,
the text scored for CER, and the text fed to the model must be byte-identical.
That is why `oron_tts.text` has **no dependencies** — so a data pipeline can
import it without pulling torch. Do not add a heavy import to that package.

## Layout

```
oron_tts/
  text/          normalizer.py, numbers.py, vocab.py   <- pure stdlib
  audio.py       mel parameters matching charactr/vocos-mel-24khz exactly
scripts/
  extend_vocab.py
data/oron_mn_pinyin/vocab.txt                          <- 2550 entries
tests/
  test_vocab_coverage.py, test_text_normalization.py
  fixtures/mn_text_sample.jsonl                        <- real corpus text
```

## Does not exist here

The DiT, CFM, trainer, dataset, tokenizer class, checkpoint manager and training
configs were removed — they were a from-scratch reimplementation that could not
load upstream weights and whose text handling defeated the architecture. They are
recoverable at the `v1-from-scratch` tag. **Do not propose reviving them or
loading F5-TTS weights into them.**

There is also no Kazakh. It was removed deliberately; do not reintroduce a
`lang` parameter.

## Conventions

- Python ≥3.12, ruff (line length 100), isort with `known-first-party = oron_tts`.
- Tests are hermetic: fixtures are committed, no network.
- When a threshold or a decision is non-obvious, comment *why*, with the measured
  number. The reasoning is the part that is expensive to recover.

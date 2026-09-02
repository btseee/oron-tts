# OronTTS

Text-to-speech for Mongolian (Khalkha Cyrillic), built as a finetune of
[F5-TTS](https://arxiv.org/abs/2410.06885) `F5TTS_v1_Base`.

Training itself runs in upstream [F5-TTS](https://github.com/SWivid/F5-TTS).
This repository owns the Mongolian-specific layer:

| | |
| --- | --- |
| `oron_tts/text/` | Text normalization, number expansion, the vocabulary contract. Pure stdlib. |
| `oron_tts/audio.py` | Mel parameters matching `charactr/vocos-mel-24khz` exactly. |
| `oron_tts/eval/` | Objective metrics. Upstream's eval supports only zh/en. |
| `scripts/extend_vocab.py` | Builds the extended vocabulary and grows the text-embedding matrix. |
| `scripts/build_f5_dataset.py` | Corpus to the `raw.arrow` / `duration.json` tree training reads. |
| `scripts/compute_epochs.py` | Solves for `epochs`, which sets the LR decay length. |
| `scripts/eval_mn.py` | Scores a checkpoint, or sweeps a directory of them. |
| `scripts/select_voices.py` | Picks the reference clips that become the shipped voices. |
| `oron_tts/infer.py` | `oron-tts-infer --voice male\|female`. |
| `configs/f5tts_mn.yaml` | The finetune config. |
| `data/oron_mn_pinyin/vocab.txt` | 2550 entries: the 2545 pretrained ones, plus `Ө ө Ү ү Ъ`. |
| `docs/phase0-findings.md` | The measurements this design rests on. |
| `docs/runbook.md` | The three GPU sessions from here to a release. |

## Status

Rebuilt from a previous from-scratch architecture that could not work. See
[docs/phase0-findings.md](docs/phase0-findings.md) for the evidence; the old code
is recoverable at the `v1-from-scratch` tag.

- [x] Vocabulary extension, verified on the real checkpoint
- [x] Mongolian text normalization (Kazakh removed)
- [x] Evaluation ASR selected and its human baseline measured
- [x] Training config and the corpus-to-dataset bridge
- [x] Evaluation harness
- [ ] Strict corpus from [oron-cleaner](../oron-cleaner) — needs a GPU
- [ ] Finetune run
- [x] Reference voice selection and the inference CLI
- [ ] Release

## Why a finetune, not a new model

`F5TTS_v1_Base` is 336M parameters trained on ~95,000 hours. Its vocabulary
already contains **65 of the 70 Mongolian Cyrillic letters**, at lines
1628–1693 of `vocab.txt`, so extending it costs **five new embedding rows**
rather than a new vocabulary.

An earlier draft of this README said those 61 rows come with *trained*
embeddings, and that overstates it. The paper says why they exist (§5.1):

> all other language characters exist in the Emilia dataset **as there are many
> code-switched sentences**

Emilia is Chinese/English podcast audio; the Cyrillic rows are incidental to it.
Measured on the real checkpoint, the Cyrillic rows (mean ‖row‖ 14.177,
std 0.627) are indistinguishable from Hangul (14.174 / 0.627) and from the table
mean (14.108 / 0.624), while the demonstrably high-frequency ASCII-lowercase
rows sit apart at 13.576 / 0.600. That is consistent with rows carrying little
training signal — though norm statistics alone cannot prove it, since an
initialisation and a trained scale can coincide.

So adaptation costs five new rows **plus retraining 65 barely-trained ones**.
The approach still holds: the value being reused is the *acoustic* prior in the
DiT and the vocoder, which ~95,000 hours bought and which no amount of Mongolian
text changes. The vocabulary is a convenience, not the argument.

The previous approach — a hand-written reimplementation trained from scratch —
reached its best validation loss at epoch 250 of 500 and then overfitted for the
remaining half of the run, on ~7 hours of single-speaker audio. It also could not
load upstream weights: key names, `ff_mult` and vocabulary size all differ, so
`--pretrain-ckpt` silently loaded nothing while reporting success.

## The failure mode this repo is built around

`f5_tts.model.utils.list_str_to_idx` maps any character absent from `vocab.txt`
to index 0 — and **index 0 is the space token, not `<unk>`**. A vocabulary gap is
therefore completely silent: training simply sees spaces where letters should be.

On the unextended base vocabulary that is **4.90% of all tokens**, because `ө`
and `ү` are ordinary Mongolian vowels. `tests/test_vocab_coverage.py` exists to
make that regression impossible to reintroduce unnoticed.

The same principle governs text handling: nothing is deleted silently.
Unrepresentable text raises, so a corpus builder records the reason and drops the
row rather than shipping text that no longer matches its audio.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # text layer only, no torch
pip install -e ".[dev,audio,eval]" # add tensors and the evaluation ASR
```

`oron_tts.text` deliberately has **no dependencies**, so
[oron-cleaner](../oron-cleaner) can import the normalizer without pulling torch
into a data pipeline. That shared import is what guarantees the text published in
the corpus, the text scored for CER, and the text fed to the model are the same
string.

## Text normalization

```python
from oron_tts.text import MongolianNormalizer

norm = MongolianNormalizer()
norm.normalize("2024 онд 25 хувь өссөн.")
# 'хоёр мянга хорин дөрвөн онд хорин таван хувь өссөн.'

norm.unsupported_chars("сайн 你 байна")   # ['你'] — reject the row, don't edit it
```

| Input | Output |
| --- | --- |
| `2024 онд` | хоёр мянга хорин дөрвөн онд |
| `1-р сар` | нэгдүгээр сар |
| `15-нд` | арван тавнд |
| `10-20 хүн` | араваас хорь хүртэл хүн |
| `14:30` | арван дөрвөн цаг гучин минут |
| `-15°C` | хасах арван таван градус цельсий |
| `3/4` | *refused* — see [normaliser-review.md](docs/normaliser-review.md) |
| `XV зуун` | арван тавдугаар зуун |
| `MIX цомог` | MIX цомог *(unchanged — see below)* |
| `Wi-Fi холболт` | Wi-Fi холболт *(Latin is in the vocab; keep it)* |

Case is **preserved**: the base vocabulary carries both cases of Cyrillic, so
lowercasing would collapse 31 rows into 3.

That reasoning is embedding-row accounting, and it does not address what those
rows were *trained to mean*. Upstream states: *"Uppercased letters (best with
form like K.F.C.) will be uttered letter by letter"*
(`F5-TTS/src/f5_tts/infer/README.md`). Every Mongolian sentence begins with a
capital, so the finetune has to weaken that prior on essentially every
utterance. Whether ~30 h is enough is an empirical question this repo has not
answered — **listen for letter-spelling at the 200-update smoke test**
(see the runbook). If it appears, lowercasing the corpus is the fix, and it is a
one-line change in `MongolianNormalizer`.

Roman numerals expand only before a context noun (`зуун`, `анги`, `бүлэг`, …).
Unrestricted matching rewrote ordinary Latin words — `MIX` parses as `M`(1000) +
`IX`(9). Since Latin characters are in the vocabulary and are preserved rather
than deleted, leaving an ambiguous token alone is the safe outcome.

## Building the vocabulary

```bash
python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt
```

Adding the checkpoint surgery as well:

```bash
python scripts/extend_vocab.py \
    --out data/oron_mn_pinyin/vocab.txt \
    --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors \
    --checkpoint-out ckpts/oron_mn/pretrained_model_1250000.safetensors
```

Two deliberate departures from upstream's Gradio-only `expand_model_embeddings`:

- New rows are seeded from the **empirical mean and standard deviation of the
  pretrained Cyrillic rows**, not `torch.randn`. Measured on the real
  checkpoint, pretrained Cyrillic rows have element std 0.627 and mean row-norm
  14.18, so a `randn` row is ~1.6× too long; seeded rows land at 14.28.
- A `.pt` base checkpoint is **refused**. Upstream re-saves `.pt` files with
  `model_state_dict` left unexpanded, which then fails on load.

New tokens are appended, never inserted, so every pretrained index is preserved.
A regenerated "sorted unique characters" vocabulary would misalign all 2545.

## Training

```bash
# 1. corpus -> the tree F5-TTS reads
python scripts/build_f5_dataset.py --corpus ../oron-cleaner/output/oron_mn_strict

# 2. how many epochs for the LR schedule
python scripts/compute_epochs.py --data ../F5-TTS/data/oron_mn_pinyin

# 3. grow the embedding by five rows
python scripts/extend_vocab.py --out data/oron_mn_pinyin/vocab.txt     --checkpoint ckpts/F5TTS_v1_Base/model_1250000.safetensors     --checkpoint-out ckpts/oron_mn/pretrained_model_1250000.safetensors

# 4. train (from the F5-TTS repo, with configs/f5tts_mn.yaml copied in)
accelerate launch src/f5_tts/train/train.py --config-name f5tts_mn.yaml

# 5. pick the best checkpoint -- it will not be the last one
python scripts/eval_mn.py --sweep ckpts/oron_mn --corpus <corpus>
```

## Voices

F5-TTS takes voice identity from a **reference clip**, not from a token, so "a
male and a female voice" means exactly two curated reference clips shipping with
the model. There is no gender conditioning in the architecture, and adding one
would be a worse answer than choosing good prompts.

```bash
python scripts/select_voices.py --corpus <corpus> --top 5 --write voices/
oron-tts-infer --voice male --text "Сайн байна уу" --checkpoint <ckpt>
oron-tts-infer --ref-audio mine.wav --ref-text "..." --text "..." --checkpoint <ckpt>
```

Candidates are ranked by bandwidth first — output bandwidth follows the prompt,
and the ≥10 kHz tail exists only in Common Voice — then DNSMOS, alignment score
and SNR, restricted to 6–10 s and one clip per speaker. The ranking is objective;
which voice is *pleasant* is not, so listen before shipping.

`ref_text` ships with each clip and is not optional. Duration is estimated from
the **UTF-8 byte-length ratio** of reference to generated text
(`utils_infer.py:503-505`), and Cyrillic is 2 bytes per character — a Latin
reference transcript against Mongolian output yields roughly twice the intended
length.

## Evaluation

CER, not WER: Mongolian is agglutinative, so one wrong suffix makes a whole word
wrong and WER saturates.

The recogniser has a floor. `bayartsogt/wav2vec2-large-xlsr-mongolian` scores
**CER 0.123 median on real human speech with human transcripts** — synthetic
audio cannot beat that, so a raw CER means nothing against zero. `eval_mn.py`
reports the ratio to that baseline.

Checkpoint selection is the point. The paper's own small-data evidence (Tab. 9)
has a 24 h model peaking at 200k updates and degrading to twice the WER by 600k,
and this project's previous run overfitted from epoch 250 of 500. Sweep, and
listen before shipping.

## Tests

```bash
pytest
ruff check oron_tts/ scripts/
```

`tests/test_vocab_coverage.py` includes `test_base_vocab_would_fail_this`, which
asserts the coverage check is load-bearing rather than vacuous.

## License

MIT

## Citation

```bibtex
@software{oron-tts2026,
  title  = {OronTTS: Mongolian Text-to-Speech},
  author = {Badral, Battseren},
  year   = {2026},
  url    = {https://github.com/btsee/oron-tts}
}
```

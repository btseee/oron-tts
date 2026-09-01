# OronTTS

Text-to-speech for Mongolian (Khalkha Cyrillic), built as a finetune of
[F5-TTS](https://arxiv.org/abs/2410.06885) `F5TTS_v1_Base`.

Training itself runs in upstream [F5-TTS](https://github.com/SWivid/F5-TTS).
This repository owns the Mongolian-specific layer:

| | |
|---|---|
| `oron_tts/text/` | Text normalization, number expansion, the vocabulary contract. Pure stdlib. |
| `oron_tts/audio.py` | Mel parameters matching `charactr/vocos-mel-24khz` exactly. |
| `oron_tts/eval/` | Objective metrics. Upstream's eval supports only zh/en. |
| `scripts/extend_vocab.py` | Builds the extended vocabulary and grows the text-embedding matrix. |
| `scripts/build_f5_dataset.py` | Corpus to the `raw.arrow` / `duration.json` tree training reads. |
| `scripts/compute_epochs.py` | Solves for `epochs`, which sets the LR decay length. |
| `scripts/eval_mn.py` | Scores a checkpoint, or sweeps a directory of them. |
| `configs/f5tts_mn.yaml` | The finetune config. |
| `data/oron_mn_pinyin/vocab.txt` | 2550 entries: the 2545 pretrained ones, plus `Ө ө Ү ү Ъ`. |
| `docs/phase0-findings.md` | The measurements this design rests on. |

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
- [ ] Reference voice selection
- [ ] Release

## Why a finetune, not a new model

`F5TTS_v1_Base` is 336M parameters trained on ~95,000 hours. Its vocabulary
already contains **61 of the 66 Mongolian Cyrillic letters** with trained
embeddings, at lines 1628–1693 of `vocab.txt`. Adapting it to Mongolian costs
**five new embedding rows**, not a new model.

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
|---|---|
| `2024 онд` | хоёр мянга хорин дөрвөн онд |
| `1-р сар` | нэгдүгээр сар |
| `15-нд` | арван тавнд |
| `10-20 хүн` | араваас хорь хүртэл хүн |
| `14:30` | арван дөрвөн цаг гучин минут |
| `-15°C` | хасах арван таван градус цельсий |
| `3/4` | дөрөвдүгээрийн гурав |
| `XV зуун` | арван тавдугаар зуун |
| `MIX цомог` | MIX цомог *(unchanged — see below)* |
| `Wi-Fi холболт` | Wi-Fi холболт *(Latin is in the vocab; keep it)* |

Case is **preserved**: the base vocabulary carries both cases of Cyrillic with
trained embeddings, so lowercasing would discard 31 trained rows to save 3.

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

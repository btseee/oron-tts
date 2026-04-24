# OronTTS — AI Agent Reference

Quick-reference for AI agents working in this codebase. Read this before editing any file.

---

## What this project is

Non-autoregressive TTS for **Mongolian** (Khalkha Cyrillic) and **Kazakh** (Cyrillic) using
**F5-TTS** — OT-Conditional Flow Matching + Diffusion Transformer (DiT). No GAN. No duration
predictor. No VITS. All model code is native PyTorch.

---

## Architecture snapshot

```
Text → CyrillicTokenizer → TextEmbedding ─┐
                                           ├→ DiT backbone → CFM (ODE) → mel → Vocos → WAV
Ref mel (or zeros) ────────────────────────┘
```

| Component | Class | File |
|-----------|-------|------|
| Text encoder | `TextEmbedding` | `src/models/encoder.py` |
| Backbone | `DiT` | `src/models/dit.py` |
| Flow matching | `CFM` | `src/models/flow.py` |
| Top-level | `F5TTS` | `src/models/f5tts.py` |
| DiT blocks | `RMSNorm, AdaLayerNorm, Attention, DiTBlock, RotaryEmbedding` | `src/models/modules.py` |
| Vocoder | Pretrained `Vocos` (`charactr/vocos-mel-24khz`) | `src/models/f5tts.py` (`_get_vocos`) |

> `decoder.py` contains a `VocosDecoder` stub that is **not used at runtime**. Do not import it.

---

## Invariants — never change these

| Setting | Value | Why |
|---------|-------|-----|
| `sample_rate` | 24 000 Hz | Matches pretrained Vocos |
| `n_mels` | 100 | Matches pretrained Vocos |
| `n_fft` | 1 024 | Matches pretrained Vocos |
| `hop_length` | 256 | Matches pretrained Vocos |
| `vocab_size` | 65 | Hardcoded in `CyrillicTokenizer`; must match YAML `model.vocab_size` |
| Python version | 3.12 (exactly) | |
| Mel formula | `torchaudio MelSpectrogram(power=1, center=True)` + `log(clamp(x, 1e-5))` | Matches Vocos input format |

---

## File map

```
src/
  data/
    dataset.py       # TTSDataset, TTSCollator, DynamicBatchSampler
    denoiser.py      # AudioDenoiser (DeepFilterNet, optional)
    hf_wrapper.py    # HFDatasetWrapper, CommonVoiceWrapper, MBSpeechWrapper
  models/
    dit.py           # DiT backbone (gradient-checkpointed per block)
    decoder.py       # VocosDecoder — NOT USED at runtime
    encoder.py       # TextEmbedding: token embed + ConvNeXtV2 + positional
    flow.py          # CFM: OT-CFM, infilling masking, CFG, Euler ODE
    f5tts.py         # F5TTS top-level: synthesize(), _get_vocos(), from_config()
    modules.py       # RMSNorm, AdaLayerNorm, RotaryEmbedding, DiTBlock, Attention, ...
  training/
    trainer.py       # F5Trainer: train loop, AMP, EMA, DynamicBatchSampler, SequentialLR
  utils/
    audio.py         # AudioProcessor: load/mel/norm/trim/save
    checkpoint.py    # CheckpointManager: save/load/pretrained/push-pull HF
    number_norm.py   # NumberNormalizer: Mongolian + Kazakh cardinal/ordinal/fraction/percent
    text_cleaner.py  # TextCleaner: NFC → punct → abbrev → numbers → filter → lower
    tokenizer.py     # CyrillicTokenizer: vocab 65, char-level
configs/
  local.yaml         # Small: dim=512, depth=12 — RTX 5070 Ti
  runpod.yaml        # Base:  dim=1024, depth=22 — A100/L40
  colab.yaml         # Small: same as local, compile:false, fp16 — T4
scripts/
  prepare.py         # prep dataset → upload HF
  train.py           # pull HF → train → push HF
  infer.py           # CLI synthesis
  clean_local_cv.py  # process local Common Voice tar.gz
  test_pipeline.py   # smoke test (--hf for real data)
  setup/
    runpod_setup.sh
```

---

## Tokenizer

`CyrillicTokenizer` — character-level, **vocab size 65** (fixed).

| IDs | Tokens |
|-----|--------|
| 0 | `<PAD>` |
| 1 | `<BOS>` — reserved, not inserted by `encode()` |
| 2 | `<EOS>` — reserved, not inserted by `encode()` |
| 3 | `<UNK>` |
| 4 | `[LANG_MN]` |
| 5 | `[LANG_KZ]` |
| 6–10 | `[FEMALE]` `[MALE]` `[YOUNG]` `[MIDDLE]` `[ELDERLY]` |
| 11–45 | Mongolian Cyrillic lowercase (35 chars) |
| 46–52 | Kazakh-specific: `ә ғ қ ң ұ һ і` |
| 53–64 | Punctuation + space |

`encode()` output order: `[LANG_TAG] [attr_tokens...] [char ids...]`

> BOS/EOS are **not** prepended/appended by `encode()`. They are reserved vocabulary slots only.

Text cleaning pipeline (in `TextCleaner.clean()`):
1. Unicode NFC
2. Punctuation map (`–`→`-`, `«»`→`""`, etc.)
3. Abbreviation expansion (case-insensitive)
4. Number expansion via `NumberNormalizer`
5. Drop chars outside `ALLOWED_CHARS`
6. Collapse whitespace → lowercase

---

## Training

### Loss
Flow matching MSE on velocity field — computed inline in `CFM.forward()`. No separate loss file.

### Text token stretching
Text tokens are **stretched** (not padded) to match the mel sequence length. `_stretch_text_to_len(ids, T)` maps every mel frame at position `i` to text token `ids[int(i * N / T)]`. Every frame gets a real token; no frame ever receives a filler embedding. This is critical for text conditioning to work with small datasets.

### Infilling
Random span mask per sample, fraction sampled from `frac_lengths_mask` range (default `[0.7, 1.0]`; all configs set `[0.3, 0.9]` for small-dataset training). Unmasked frames = conditioning signal.

### CFG
Drops audio/cond independently during training.
- Config values: `audio_drop_prob: 0.1`, `cond_drop_prob: 0.05` (lower than CFM code defaults 0.3/0.2).
- `frac_lengths_mask: [0.3, 0.9]` — all three YAML configs use this instead of the code default `[0.7, 1.0]`.

### Optimizer & scheduler
```
AdamW(weight_decay=0.01)
SequentialLR:
  [0, warmup_steps)       → LinearLR(start_factor=1e-4)
  [warmup_steps, T_total) → CosineAnnealingLR(eta_min=1e-6)
```
`T_total` estimated at trainer init from `num_epochs × steps_per_epoch`.

### Monitoring
`F5Trainer` logs to TensorBoard (step-level: `train/loss`, `train/lr`, `train/grad_norm`; epoch-level: `epoch/train_loss`, `epoch/val_loss`). Every `audio_sample_interval` epochs (default 10), synthesises diagnostic sentences with EMA weights and logs audio waveforms (`audio/mn/...`) and mel images (`mel/mn/...`). TensorBoard logs are uploaded to `tb_logs/` in the HuggingFace repo on `--push-to-hub`.

### AMP
Auto-detected at runtime:
- SM ≥ 8.0 (Ampere+): bf16, no GradScaler
- SM < 8.0 (Turing/T4): fp16, GradScaler enabled
- CPU: disabled

### Batching
`DynamicBatchSampler` (frame-budget): sorts by duration, greedily packs until
`sum(mel_frames) ≤ frames_threshold`. Configured via `batch_size_type: frame`.

### torch.compile
Compiles **only** `model.cfm.backbone` (DiT) with `dynamic=True`.
Never compile the full model — `CFM.forward()` has Python branching that causes graph breaks.
Set `compile: false` on T4/Colab.

### Gradient checkpointing
Per-DiTBlock. Enable via `gradient_checkpointing: true` in config.

### EMA
`torch_ema.ExponentialMovingAverage`, decay=0.9999. Updated every step on rank-0.
EMA weights are used for inference, saved separately in checkpoints.

---

## Inference

`F5TTS.synthesize()` — two modes:

**Voice cloning** (pass `ref_audio_path` + `ref_text`):
```python
audio = model.synthesize(
    text="Сайн байна уу", lang="mn",
    ref_audio_path="ref.wav", ref_text="...",
    n_steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
)
```

**Attribute tokens** (no reference audio):
```python
audio = model.synthesize(
    text="Сайн байна уу", lang="mn",
    attr_tokens=["[FEMALE]", "[YOUNG]"],
    n_steps=32,
)
```

### Vocoder
`_get_vocos(device)` lazy-loads `Vocos.from_pretrained("charactr/vocos-mel-24khz")` on the first
call and caches it via `object.__setattr__` outside the `nn.Module` parameter tree.
Requires internet on first use. Never saved in checkpoints.

### cfg_strength tuning
Default 2.0. Reduce to 1.0–1.5 if output sounds over-processed or noisy (model not fully converged).

---

## Configs

All three YAMLs share the same key schema. No keys nested under `training:` — everything is
top-level except `model:`.

```yaml
# Audio (fixed)
sample_rate: 24000
n_mels: 100
n_fft: 1024
hop_length: 256
win_length: 1024
fmin: 0.0
fmax: 8000.0

# Training
batch_size: 16
batch_size_type: frame        # "frame" = DynamicBatchSampler, "sample" = fixed
frames_threshold: 3000        # 38400 for runpod, 12000 for colab
max_samples: 8
learning_rate: 1.0e-4
betas: [0.9, 0.999]
warmup_steps: 1000
num_epochs: 500
ema_decay: 0.9999
max_grad_norm: 1.0
grad_accumulation_steps: 1
log_interval: 100
save_interval: 5
use_tqdm: true
num_workers: 6
pin_memory: true

# Hardware
gradient_checkpointing: true
compile: true                 # false on T4/Colab
use_tf32: true                # false on T4
cudnn_benchmark: true         # false on T4

# Model
model:
  dim: 512            # 1024 for runpod
  depth: 12           # 22 for runpod
  heads: 8            # 16 for runpod
  ff_mult: 4
  vocab_size: 65      # MUST match CyrillicTokenizer
  text_dim: 256       # 512 for runpod
  conv_layers: 4
  p_dropout: 0.1
  audio_drop_prob: 0.1
  cond_drop_prob: 0.05
  frac_lengths_mask: [0.3, 0.9]  # default [0.7, 1.0]; lower = more context visible during training
```

### Config profiles

| Config | GPU | `model.dim` | `model.depth` | `frames_threshold` | `compile` |
|--------|-----|-------------|---------------|-------------------|-----------|
| `local.yaml` | RTX 5070 Ti | 512 | 12 | 3 000 | true |
| `runpod.yaml` | A100/L40 | 1 024 | 22 | 38 400 | true |
| `colab.yaml` | T4 (15 GB) | 512 | 12 | 12 000 | false |

---

## Checkpoints

`CheckpointManager` saves `.pt` files with keys: `step`, `model_state_dict`,
`optimizer_state_dict`, `scheduler_state_dict`, `ema_state_dict`, `loss`, `config`.
Auto-rotates (keeps `max_checkpoints` most recent + `best`).

Key methods:
- `save(...)` — standard checkpoint
- `load(...)` — returns `{step, loss, ema_state_dict}`
- `load_pretrained_f5tts(model, path, device, strict)` — loads official F5-TTS `.safetensors` with key remapping
- `push_to_hub(repo_id, token, private, log_dir=None)` / `pull_from_hub(repo_id, filename, token)` — when `log_dir` is provided, TB logs are uploaded to `tb_logs/` in the HF repo`

---

## Dataset: btsee/mbspeech_mn

- 3 846 samples, split `train` only
- Audio column: `audio` — raw bytes at 16 kHz; resampled to 24 kHz at load time
- Text columns: `sentence_norm` (preferred), `sentence_orig`
- **No `sentence` column** — the dataset card is wrong
- `TTSDataset.from_hf_dataset` auto-detects `sentence_norm` from a candidates list

---

## Coding rules

- **Python 3.12 only.** Strict type hints on every function. No `Any`. No `# type: ignore`.
- Use `Final` for module-level constants. Use `cast()` for PyTorch stub workarounds.
- Class-level `nn.Buffer` annotations required (e.g. `inv_freq: torch.Tensor`) so Pyright resolves correctly.
- Linter: Ruff (`line-length=100`, `target-version=py312`). No Black.
- Imports: isort (`profile=black`, `line_length=100`).
- `pad_id=-1` for **batch-level** padding in `TTSCollator` (batch padding to max-T; not 0 because PAD token is a valid vocab ID).
- Text tokens within a sample are **stretched** to mel length via `_stretch_text_to_len`, not padded with -1.

---

## Things that do NOT exist in this repo

| What you might look for | Reality |
|------------------------|---------|
| `VocosDecoder` at runtime | Exists in `decoder.py` but is never instantiated |
| GAN / discriminator | Deleted — see `discriminator.py` stub (unused) |
| Duration predictor | Deleted |
| `HiFi-GAN` | Never existed |
| Flat LR after warmup | Replaced with cosine decay in `trainer.py` |
| `vocos_dim` / `vocos_layers` / `vocos_intermediate` in YAMLs | Removed |
| BOS/EOS inserted by `encode()` | Not inserted; IDs 1 and 2 are reserved slots only |

---

## Common entry points

```bash
# Env setup (Windows)
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Smoke test
.venv\Scripts\python scripts/test_pipeline.py        # synthetic only
.venv\Scripts\python scripts/test_pipeline.py --hf   # + real btsee/mbspeech_mn

# Dataset prep
python scripts/prepare.py --output-dir data/processed --dataset all

# Training
python scripts/train.py --config configs/runpod.yaml --dataset btsee/mbspeech_mn \
    --push-to-hub --hf-repo btsee/oron-tts

# Fine-tune from official F5-TTS checkpoint
python scripts/train.py --config configs/runpod.yaml --dataset btsee/mbspeech_mn \
    --pretrain-ckpt F5TTS_Base.safetensors

# Inference
python scripts/infer.py --text "Сайн байна уу" --lang mn \
    --attr-tokens "[FEMALE],[YOUNG]" --output out.wav

python scripts/infer.py --text "Сайн байна уу" --lang mn \
    --ref-audio ref.wav --ref-text "..." --output out.wav
```

---

## Number normalisation (Mongolian)

`NumberNormalizer` in `src/utils/number_norm.py` handles:
- Cardinals: `2024` → `хоёр мянга хорин дөрөв`
- Ordinals: `1-р` → `нэгдүгээр`; vowel-harmony suffix (`дүгээр` for front vowels, `дугаар` for back)
- Genitive: `2024-ны` → `хоёр мянга хорин дөрвөн` (calls `convert_attributive()`)
- Fractions: `3/4` → `дөрөвдүгээрийн гурав` (vowel-harmony-correct genitive)
- Decimals, percentages, currency (MNT/USD/EUR), Roman numerals
- Kazakh cardinal: handled by same class with `lang="kz"` path

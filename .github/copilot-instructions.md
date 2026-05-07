# OronTTS Project Instructions

## General Principles

- **Language:** Python 3.12 (exactly). Strict type hints on every function.
- **Style:** Functional and modular. Avoid monolithic files.
- **Documentation:** Minimal, meaningful comments only. Self-documenting code.
- **No Coqui, no ESPnet, no Coqui-TTS.** All model code is native PyTorch.
- **No VITS.** The architecture is F5-TTS (Flow Matching + Diffusion Transformer). Do not introduce GAN, HiFi-GAN, discriminators, or duration predictors.

## Architecture: F5-TTS

OronTTS is a non-autoregressive TTS system for Mongolian (Khalkha Cyrillic) and Kazakh (Cyrillic) based on the F5-TTS paper.

### Model stack

| Component | Class | File |
| --------- | ----- | ---- |
| Text encoder | `TextEmbedding` | `src/models/encoder.py` |
| Backbone | `DiT` (Diffusion Transformer) | `src/models/dit.py` |
| Flow matching | `CFM` (OT-Conditional Flow Matching) | `src/models/flow.py` |
| Vocoder | Pretrained `Vocos` (`charactr/vocos-mel-24khz`) — lazy-loaded, not a trained parameter | `src/models/f5tts.py` (`_get_vocos`) |
| Top-level | `F5TTS` | `src/models/f5tts.py` |
| DiT blocks | `RMSNorm, AdaLayerNorm, Attention, DiTBlock, RotaryEmbedding, ...` | `src/models/modules.py` |

### Training

- Flow matching loss (MSE on velocity field) is computed inline in `CFM.forward()` — no separate loss function.
- **Infilling training**: random span masking, fraction configurable via `model.frac_lengths_mask` (paper default `[0.7, 1.0]`, also used in all three YAML configs). The unmasked portion is the conditioning signal.
- **Text token stretching**: text tokens are linearly stretched to fill the full mel sequence (`_stretch_text_to_len` in both `dataset.py` and `f5tts.py`). Every mel frame receives the text token at approximately its temporal position — **not** padding. This is essential: padding 90%+ of frames with filler embeddings makes text conditioning ineffective.
- **CFG dropout**: `audio_drop_prob=0.3`, `cond_drop_prob=0.2` (paper defaults; configs match). High `audio_drop_prob` is required so the model is well-trained on pure-noise conditioning, which is what ref-free synthesis hits at inference time.
- **Validation determinism**: `CFM.forward` switches to a fixed `t=0.5`, centred mid-fraction span, deterministic noise, and no CFG dropout when `self.training is False`. This makes `val_loss` comparable across epochs and prevents “best checkpoint” from being picked by RNG luck.
- Optimizer: AdamW (`weight_decay=0.01`). Scheduler: `SequentialLR` — LinearLR warmup (`start_factor=1e-4`) for `warmup_steps`, then CosineAnnealingLR decaying to `eta_min=1e-6` over the remaining steps. EMA maintained via `torch_ema`.
- **AMP**: Auto-detected — bf16 on SM≥8.0 (Ampere+), fp16 on SM<8.0 (Turing/T4), disabled on CPU. GradScaler used only for fp16.
- **TensorBoard**: `F5Trainer` logs step-level (`train/loss`, `train/lr`, `train/grad_norm`) and epoch-level (`epoch/train_loss`, `epoch/val_loss`) metrics. Audio samples synthesised every `audio_sample_interval` epochs (default 10) and logged as `audio/mn/{tag}` + `mel/mn/{tag}`. With `--push-to-hub`, checkpoints and TensorBoard logs are uploaded to HuggingFace `tb_logs/` at each checkpoint save and again at the end of training.
- **DynamicBatchSampler**: Frame-budget batching — sorts samples by duration, greedily packs batches where `sum(mel_frames) ≤ frames_threshold`. No samples are discarded.
- **Gradient checkpointing**: Per-DiTBlock, enabled via `gradient_checkpointing: true` in config. Essential for T4/RTX 5070 Ti.
- **torch.compile**: Compiles only `model.cfm.backbone` (DiT) with `dynamic=True`. Disabled on Colab T4 (`compile: false`). The CFM wrapper has Python branching that prevents compilation.
- Trainer class: `F5Trainer` (`src/training/trainer.py`).

### Inference modes

1. **Voice cloning** — pass `ref_audio_path` and `ref_text` to `F5TTS.synthesize()`; the reference mel + transcript are used as conditioning and to set duration via the token-count ratio.
2. **Ref-free** — omit `ref_audio_path`. Conditioning is zero, duration is estimated from char count and `speed` (default 1.0; use `--speed` on the CLI). This is the OOD regime; expect lower fidelity than ref-based.
3. **Long-text inference**: `F5TTS.synthesize()` splits long text at punctuation or word boundaries by default (`max_chars_per_chunk=120`, `pause_s=0.25`). Do not generate poems/chapters as one ref-free sequence; training clips are capped at 30 s.
4. **Inference params**: `cfg_strength` (classifier-free guidance, default 2.0), `sway_sampling_coef` (sway sampling, default -1.0), `n_steps` (ODE steps, default 32), `speed` (rate multiplier, default 1.0), `seed`, `max_chars_per_chunk`.
5. **Vocoder**: `F5TTS._get_vocos(device)` lazy-loads `Vocos.from_pretrained("charactr/vocos-mel-24khz")` on first call and caches it outside the `nn.Module` parameter tree. It is never saved in checkpoints.

## Tokenizer

`CyrillicTokenizer` (`src/utils/tokenizer.py`) — character-level, vocab size **65**.

- 35 Mongolian Cyrillic (lower) + 7 Kazakh-specific (`әғқңұһі`) + 12 punctuation/space
- Special tokens: `<PAD>=0, <BOS>=1, <EOS>=2, <UNK>=3`
- Language tags: `[LANG_MN]=4, [LANG_KZ]=5`
- Attribute tags: `[FEMALE]=6, [MALE]=7, [YOUNG]=8, [MIDDLE]=9, [ELDERLY]=10`
- Supported language codes are exactly `mn` and `kz`; unknown codes raise `ValueError` instead of silently falling through.

Usage:

```python
tokenizer = CyrillicTokenizer()
ids = tokenizer.encode("сайн байна уу", lang="mn", attr_tokens=["[FEMALE]"])
text = tokenizer.decode(ids)
```

`TextCleaner` (`src/utils/text_cleaner.py`) wraps the tokenizer and provides `clean(text, lang)` and `text_to_sequence(text, lang, attr_tokens)`.

## Audio

- **Sample rate:** 24 000 Hz (all defaults must be 24000).
- **Mel bins:** 100. **n_fft:** 1024. **hop_length:** 256. **win_length:** 1024.
- `fmin`/`fmax` are **not** configurable — torchaudio defaults `(0, sr/2)` are used to match pretrained Vocos. The YAML keys were removed in April 2026 because they were silently ignored.
- `AudioProcessor` in `src/utils/audio.py` — `load_audio()`, `mel_spectrogram()`, `normalize_audio()`, `trim_silence()`, `save_audio()`.
- `AudioProcessor` receives `n_fft` explicitly from `F5TTS.__init__`.
- Mel format matches pretrained Vocos: `torchaudio.transforms.MelSpectrogram(power=1, center=True)` + `log(clamp(x, 1e-5))`.

## Project Structure

```text
src/
  data/
    dataset.py       # TTSDataset, TTSCollator, DynamicBatchSampler
    denoiser.py      # AudioDenoiser (DeepFilterNet)
    hf_wrapper.py    # HFDatasetWrapper, CommonVoiceWrapper, MBSpeechWrapper
  models/
    dit.py           # DiT backbone
    decoder.py       # VocosDecoder (unused at runtime — pretrained Vocos is used instead)
    encoder.py       # TextEmbedding (token embed + ConvNeXtV2 blocks)
    flow.py          # CFM (OT-CFM) with infilling + CFG
    f5tts.py         # F5TTS top-level model
    modules.py       # DiT building blocks (RMSNorm, RotaryEmbedding, DiTBlock, etc.)
  training/
    trainer.py       # F5Trainer
  utils/
    audio.py         # AudioProcessor
    checkpoint.py    # CheckpointManager (save/load/pretrained/HF push-pull)
    number_norm.py   # NumberNormalizer (Mongolian + Kazakh cardinal, ordinal, decimal, percent, currency, Roman)
    text_cleaner.py  # TextCleaner
    tokenizer.py     # CyrillicTokenizer
configs/
  local.yaml         # F5-TTS Small: dim=512, depth=12, heads=8 (RTX 5070 Ti)
  runpod.yaml        # F5-TTS Base: dim=1024, depth=22, heads=16 (A100/L40)
  colab.yaml         # F5-TTS Small: same as local, tuned for T4 (compile: false, fp16)
scripts/
  prepare.py         # Clean + denoise + upload to HF
  train.py           # Pull from HF + train + push model
  infer.py           # Run synthesis (requires --checkpoint)
  clean_local_cv.py  # Process local Common Voice tar.gz without HF
  test_pipeline.py   # End-to-end smoke test (synthetic + optional --hf)
  setup/
    runpod_setup.sh  # RunPod: install deps, smoke test
notebooks/
  oron_tts_colab.ipynb  # Google Colab training notebook
```

## Configs

All three YAML configs use the same keys. Critical keys:

- `model.dim`, `model.depth`, `model.heads`, `model.ff_mult`, `model.vocab_size` (must match tokenizer = 65)
- `model.text_dim`, `model.conv_layers`, `model.p_dropout`
- `model.audio_drop_prob`, `model.cond_drop_prob` — CFG dropout (paper defaults 0.3/0.2; required for ref-free synthesis to be in-distribution)
- `model.frac_lengths_mask` — infilling mask fraction range (paper default `[0.7, 1.0]`)
- `sample_rate` (must be 24000), `n_mels` (must be 100) — top-level YAML keys, not nested
- `batch_size`, `warmup_steps`, `num_epochs`, `learning_rate` — top-level YAML keys, not nested under `training`
- `batch_size_type` (`"frame"` for DynamicBatchSampler, `"sample"` for fixed batch size)
- `frames_threshold` (max total mel frames per batch when `batch_size_type="frame"`)
- `max_samples` (max samples per batch cap when using frame-budget batching)
- `max_checkpoints` (rotating `.pt` checkpoint retention; best checkpoint is kept separately)
- `audio_sample_interval` (epochs between TensorBoard audio/mel diagnostics)
- `gradient_checkpointing` (enable for VRAM-constrained GPUs like T4/RTX 5070 Ti)
- `compile` (default `true`; set `false` on T4/Colab to avoid torch.compile overhead)
- `use_tf32`, `cudnn_benchmark` — performance flags (disabled on T4)
- `grad_accumulation_steps` — effective batch multiplier

### Config profiles

| Config | Target GPU | model.dim | model.depth | frames_threshold | max_checkpoints | compile |
| ------ | ---------- | --------- | ----------- | ---------------- | --------------- | ------- |
| `local.yaml` | RTX 5070 Ti | 512 | 12 | 3000 | 5 | true |
| `runpod.yaml` | A100/L40 | 1024 | 22 | 38400 | 5 | true |
| `colab.yaml` | T4 (15 GB) | 512 | 12 | 48000 | 2 | false |

## Checkpoints

`CheckpointManager` (`src/utils/checkpoint.py`):

- `save(step, model, optimizer, scheduler, ema_state, loss, config, is_best)` — saves `.pt` checkpoint, auto-rotates (keeps last `max_checkpoints`)
- `load(model, optimizer, scheduler, path, load_best, device)` — loads `.pt` checkpoint, returns `{step, loss, ema_state_dict}`
- `load_pretrained_f5tts(model, checkpoint_path, device, strict)` — loads official F5-TTS `.safetensors` or `.pt` with key remapping
- `push_to_hub(repo_id, token, private, log_dir=None)` — uploads checkpoint dir to HuggingFace; when `log_dir` is provided, TensorBoard logs are uploaded under `tb_logs/`
- `pull_from_hub(repo_id, filename, token)` — downloads checkpoint from HuggingFace

Training resume restores raw model weights and rebuilds the EMA tracker from `ema_state_dict`, so resumed long runs continue from the same EMA state used for inference.

## Formatting and Quality

- **Linter/Formatter:** Ruff (line-length=100, target-version=py312). No Black.
- **Imports:** isort (profile=black, line_length=100).
- **Typing:** Strict. No `Any` unless unavoidable. No `# type: ignore`. Use `Final` for module-level constants. Use `cast()` for PyTorch stub workarounds.
- **Registered buffers**: Always declare class-level type annotations (e.g. `inv_freq: torch.Tensor`) so Pyright resolves the type correctly.

## Workflow

1. **Local prep:** `python scripts/prepare.py --output-dir data/processed --dataset common-voice`
   - Loads from HF → cleans text → denoises audio → saves WAV + metadata.json
   - Or use `scripts/clean_local_cv.py --input cv_mn.tar.gz` for local archives.
2. **Cloud training:** `python scripts/train.py --config configs/runpod.yaml --dataset btsee/mbspeech_mn`
    - Pulls dataset from HF → trains F5-TTS → saves checkpoints → logs metrics to console.
    - Resume: add `--resume` flag. Fine-tune: add `--pretrain-ckpt F5TTS_Base.safetensors`.
    - Push to HF: add `--push-to-hub --hf-repo btsee/oron-tts --hub-upload-interval 1`.

Colab persistence: pass absolute Drive paths with `--log-dir` and `--checkpoint-dir`; relative paths such as `output/logs` stay on ephemeral `/content/oron-tts`.

3. **Inference:** `python scripts/infer.py --checkpoint output/checkpoints/f5tts_best.pt --text "Сайн байна уу" --lang mn --output out.wav`
   - Optionally `--ref-audio ref.wav --ref-text "..."` for voice cloning (recommended for best quality).
   - Optionally `--speed 1.2` to adjust speaking rate (>1 faster, <1 slower).

Runtime guards skip non-finite mel/loss batches and non-finite unscaled gradients before optimizer, scheduler, or EMA updates. `CFM.sample()` also validates durations, lens, `n_steps`, and `cfg_strength`, and seeded sampling uses a local `torch.Generator`.

## CLI Entry Points

```bash
oron-prepare   # → scripts.prepare:main
oron-train     # → scripts.train:main
oron-infer     # → scripts.infer:main
```

## Environment Setup

```powershell
# Use exactly Python 3.12 — the only fully supported version
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev,inference]"

# Secrets — create .env at repo root (never commit)
# HF_TOKEN=hf_...         (HuggingFace personal access token)

# Verify
.venv\Scripts\python scripts/test_pipeline.py        # synthetic only
.venv\Scripts\python scripts/test_pipeline.py --hf   # + real btsee/mbspeech_mn
```

### Dataset: btsee/mbspeech_mn

- 3,846 samples, split `train` only
- Audio column: `audio` — stored as raw bytes (16 kHz), resampled to 24 kHz at load time
- Text columns: `sentence_norm` (preferred) and `sentence_orig`
- `TTSDataset.from_hf_dataset` auto-detects `sentence_norm` via the candidates list
- `sentence` column does **not** exist — the dataset card is incorrect

## Hardware Targets

- **Local:** RTX 5070 Ti (16 GB) for dataset prep and small test runs (Small config).
- **Colab:** T4 (15 GB) free tier with fp16 AMP, no torch.compile (Colab config).
- **Cloud:** NVIDIA A100 or L40 on RunPod for full training (Base config).

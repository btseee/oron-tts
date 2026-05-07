# OronTTS

Non-autoregressive TTS for Mongolian (Khalkha Cyrillic) and Kazakh (Cyrillic) using [F5-TTS](https://arxiv.org/abs/2410.06885) — Flow Matching + Diffusion Transformer.

## Features

- **F5-TTS**: OT-CFM + DiT backbone. No GAN, no duration predictor. Flow matching loss computed inline in CFM.
- **Voice cloning**: pass a 3–10 s reference WAV at inference time (recommended for best quality).
- **Ref-free synthesis**: skip the reference; duration is estimated from char count and `--speed`.
- **Bilingual**: Mongolian + Kazakh Cyrillic, character-level tokenizer (vocab 65).
- **Number normalisation**: Mongolian + Kazakh cardinal, ordinal, fraction, percent, currency.
- **Audio denoising**: DeepFilterNet for preprocessing non-professional recordings.

## Installation

**Windows (local dev)**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[dev,inference]"
```

**Linux / RunPod**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,inference]"
```

## Project Structure

```bash
src/
  data/       # TTSDataset, AudioDenoiser, HF wrappers
  models/     # DiT, CFM, F5TTS, TextEmbedding (pretrained Vocos vocoder)
  training/   # F5Trainer
  utils/      # AudioProcessor, CyrillicTokenizer, TextCleaner, CheckpointManager, NumberNormalizer
configs/
  local.yaml    # Small (dim=512, depth=12) — RTX 5070 Ti dev
  runpod.yaml   # Base  (dim=1024, depth=22) — cloud training
  colab.yaml    # Small (dim=512, depth=12) — Colab T4 (compile: false, fp16)
scripts/
  prepare.py        # clean + denoise + upload to HF
  train.py          # train + push model
  infer.py          # synthesise speech
  clean_local_cv.py # process local Common Voice tar.gz
  test_pipeline.py  # end-to-end smoke test
  setup/
    runpod_setup.sh # RunPod one-shot setup
```

## Usage

### Dataset preparation

From Hugging Face:

```bash
python scripts/prepare.py \
    --output-dir data/processed \
    --dataset all \
    --upload \
    --hf-repo btsee/oron-tts-dataset
```

From a local Common Voice tar.gz:

```bash
python scripts/clean_local_cv.py \
    --input cv_mn.tar.gz \
    --output-dir data/processed/cv_mn \
    --max-samples 5000
```

### Training

**Local** (Small config):

```bash
python scripts/train.py \
    --config configs/local.yaml \
    --dataset btsee/mbspeech_mn
```

**RunPod** (Base config):

```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --push-to-hub \
    --hf-repo btsee/oron-tts \
    --hub-upload-interval 1
```

Fine-tune from a pretrained F5-TTS checkpoint:

```bash
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --pretrain-ckpt F5TTS_Base.safetensors
```

Resume explicitly with `--resume`. Training does not auto-resume, which keeps accidental restarts from silently continuing an old run.

**Colab** (persistent Drive logs and checkpoints):

```bash
python scripts/train.py \
    --config configs/colab.yaml \
    --dataset btsee/mbspeech_mn \
    --log-dir /content/drive/MyDrive/oron-tts/logs \
    --checkpoint-dir /content/drive/MyDrive/oron-tts/checkpoints \
    --resume
```

Relative paths such as `output/logs` live on Colab's ephemeral `/content/oron-tts`, not on Google Drive. Use absolute Drive paths when TensorBoard logs must persist.

### Inference

```bash
# Voice cloning (recommended)
python scripts/infer.py \
    --checkpoint output/checkpoints/f5tts_best.pt \
    --text "Сайн байна уу" --lang mn \
    --ref-audio ref.wav \
    --ref-text "Энэ бол жишээ өгүүлбэр" \
    --output out.wav

# Ref-free (lower fidelity; tune --speed if pacing is off)
python scripts/infer.py \
    --checkpoint output/checkpoints/f5tts_best.pt \
    --text "Сайн байна уу" --lang mn \
    --speed 1.0 \
    --cfg-strength 1.5 \
    --max-chars-per-chunk 120 \
    --output out.wav

# Kazakh
python scripts/infer.py \
    --checkpoint output/checkpoints/f5tts_best.pt \
    --text "Сәлеметсіз бе" --lang kz \
    --output out_kz.wav
```

Long inputs are split automatically at punctuation or word boundaries. This keeps each generated segment close to the 1-30 s training range and prevents long ref-free passages from turning into speech-like but unintelligible audio. Use `--max-chars-per-chunk 0` only for short texts or debugging.

## Mongolian Numbers

|Input    | Output                  |
|---------|-------------------------|
| 10      | арван                   |
| 25      | хорин тав               |
| 100     | зуун                    |
| 1-р     | нэгдүгээр               |
| 2024    | хоёр мянга хорин дөрөв  |
| 3/4     | дөрөвдүгээрийн гурав    |
| 2024-ны | хоёр мянга хорин дөрвөн |

## Environment

Create `.env` at the repo root (never commit — already in `.gitignore`):

```bash
HF_TOKEN=hf_...       # HuggingFace token — read + write scope
```

`train.py` and `prepare.py` load it automatically via `python-dotenv`.

## RunPod Training

### Pod settings

| Field              | Value                                                                  |
|--------------------|------------------------------------------------------------------------|
| GPU                | **L40S**                                                               |
| GPU count          | **1**                                                                  |
| Cloud tier         | **Secure Cloud**                                                       |
| Template           | **RunPod PyTorch** (`runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2404`) |
| Container disk     | **20 GB**                                                              |
| Volume disk        | **50 GB**                                                              |
| Volume mount       | `/workspace`                                                           |
| `HF_TOKEN` env var | your HuggingFace token (read + write)                                  |

The Base config peaks at ~13 GB VRAM; the L40S 48 GB gives a 3.5× margin. Add the env vars in the **Environment Variables** section of the pod creation form.

### Setup

Connect via **Web Terminal**, then:

```bash
tmux new-session -s setup

cd /workspace
git clone https://github.com/btseee/oron-tts.git
cd oron-tts
bash scripts/setup/runpod_setup.sh
```

The script creates `.venv` with `--system-site-packages` (inherits pre-installed PyTorch + CUDA), installs project deps, and runs a 10-step smoke test. Close the tab at any time — re-attach with `tmux attach -t setup`.

If you skipped the env vars form, create `.env` instead (auto-loaded by `train.py`):

```bash
cat > /workspace/oron-tts/.env <<'EOF'
HF_TOKEN=hf_...
EOF
```

### Train

```bash
source .venv/bin/activate
python scripts/train.py \
    --config configs/runpod.yaml \
    --dataset btsee/mbspeech_mn \
    --push-to-hub \
    --hf-repo btsee/oron-tts \
    --hub-upload-interval 1
```

Metrics are logged to console (loss, val_loss, samples/s, ETA) and TensorBoard. Checkpoints land on the 50 GB volume and survive pod restarts. When `--push-to-hub` is used, checkpoints and TensorBoard logs are uploaded under `tb_logs/` at every checkpoint save and again at the end of training. To resume after a pod restart, re-run training with `--resume`.

### Cost

| Scenario                | Cost at ~$0.86/hr |
|-------------------------|-------------------|
| Smoke test (< 5 min)    | < $0.08           |
| 1 epoch × 3 846 samples | ~$0.21            |
| 500 epochs (full run)   | ~$107             |

Terminate the pod (not just stop it) after training to end container disk billing.

## Configuration

All three configs use the same keys (all top-level, no `training:` nesting):

```yaml
sample_rate: 24000
n_mels: 100
n_fft: 1024
hop_length: 256

batch_size: 16
batch_size_type: frame      # "frame" = DynamicBatchSampler, "sample" = fixed
frames_threshold: 3000      # 38400 for runpod.yaml, 48000 for colab.yaml
max_samples: 8              # cap samples per frame-budget batch
warmup_steps: 1000
num_epochs: 500
ema_decay: 0.9999
use_tqdm: true            # set false for RunPod container logs
log_interval: 100
save_interval: 5          # save a rotating checkpoint every N epochs
max_checkpoints: 5        # keep this many rotating .pt files (+ f5tts_best.pt)
audio_sample_interval: 10 # TensorBoard audio/mel diagnostics every N epochs
gradient_checkpointing: true
compile: true             # false in colab.yaml

model:
  dim: 512               # 1024 for runpod.yaml
  depth: 12              # 22 for runpod.yaml
  heads: 8               # 16 for runpod.yaml
  vocab_size: 65         # fixed — must match CyrillicTokenizer
  audio_drop_prob: 0.3   # CFG audio dropout (paper default)
  cond_drop_prob: 0.2    # CFG conditioning dropout (paper default)
  frac_lengths_mask: [0.7, 1.0]  # infilling mask range (paper default)
```

## Development

```bash
ruff check src/ scripts/
ruff format src/ scripts/
isort src/ scripts/
```

## License

MIT

## Citation

If you use OronTTS in your research, please cite:

```bibtex
@software{oron-tts2026,
  title  = {OronTTS: Mongolian and Kazakh Text-to-Speech with F5-TTS},
  author = {Badral, Battseren},
  year   = {2026},
  url    = {https://github.com/btsee/oron-tts}
}
```

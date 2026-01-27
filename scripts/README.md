# Scripts Directory

Organized scripts for OronTTS development, training, and deployment.

## Structure

```text
scripts/
├── setup/                    # Environment setup
│   └── setup_runpod.sh       # Automated Runpod.io setup
├── training/                 # Training scripts
│   ├── train.py              # Training script (wraps F5-TTS)
│   └── train_background.sh   # Background training for RunPod
├── data/                     # Data processing
│   └── prepare_cv_dataset.py # Common Voice dataset prep
└── inference/                # Inference scripts
    └── infer.py              # TTS inference (using F5-TTS)
```

## Quick Start

### Local Development

```bash
# Activate environment
source .venv/bin/activate

# Initialize submodules (F5-TTS)
git submodule update --init --recursive

# Install F5-TTS from submodule
pip install -e third_party/F5-TTS

# Inference
python scripts/inference/infer.py \
    --ref-audio reference.wav \
    --ref-text "Энэ бол жишээ текст" \
    --gen-text "Сайн байна уу" \
    --output output.wav
```

### Training

```bash
# Background training (RTX 4090 optimized, from scratch)
./scripts/training/train_background.sh

# Or with custom dataset
./scripts/training/train_background.sh btsee/common-voices-24-mn

# Interactive training
python scripts/training/train.py \
    --dataset-name btsee/common-voices-24-mn \
    --epochs 500 \
    --batch-size 1800

# Finetune from pretrained (faster convergence)
python scripts/training/train.py \
    --dataset-name btsee/common-voices-24-mn \
    --finetune \
    --epochs 100 \
    --batch-size 1800

# View training logs
tail -f logs/train_common-voices-24-mn_*.log

# TensorBoard
tensorboard --logdir ckpts/ --bind_all
```

### Batch Synthesis

```bash
python scripts/inference/infer.py \
    --ref-audio reference.wav \
    --ref-text "Энэ бол жишээ текст" \
    --input-file texts.txt \
    --output-dir outputs/audio
```

## Environment Variables

- `HF_TOKEN` - HuggingFace API token
- `HUB_REPO` - Model repository (default: btsee/oron-tts)
- `DATASET` - Dataset name (default: btsee/common-voices-24-mn)

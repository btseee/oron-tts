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
# Interactive training (finetune from pretrained)
python scripts/training/train.py \
    --dataset-name oron_mn \
    --finetune \
    --epochs 100 \
    --batch-size 3200

# Background training on RunPod (recommended)
./scripts/training/train_background.sh oron_mn

# View training logs
tail -f logs/train_oron_mn_*.log

# Check training status
cat logs/train_oron_mn.pid | xargs ps -p
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

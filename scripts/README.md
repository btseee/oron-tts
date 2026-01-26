# Scripts Directory

Organized scripts for OronTTS development, training, and deployment.

## Structure

```
scripts/
├── setup/              # Environment setup
│   └── setup_runpod.sh    # Automated Runpod.io setup
├── training/           # Training scripts
│   ├── train.py           # Main training script
│   └── generate_samples.py # Sample generation
├── data/               # Data processing
│   └── prepare_cv_dataset.py  # Common Voice dataset prep
└── inference/          # Inference scripts
    └── infer.py           # TTS inference
```

## Quick Start

### Local Development
```bash
# Activate environment
source .venv/bin/activate

# Train locally
python scripts/training/train.py --config configs/light.yaml

# Inference
python scripts/inference/infer.py --checkpoint path/to/model --text "Сайн байна уу"
```

### Runpod Setup
```bash
# On Runpod instance
cd /workspace
wget https://raw.githubusercontent.com/btseee/oron-tts/main/scripts/setup/setup_runpod.sh
chmod +x setup_runpod.sh
export HF_TOKEN=hf_xxx
./setup_runpod.sh

# Start background training
/workspace/train_bg.sh oron configs/light.yaml
```

## Environment Variables

- `HF_TOKEN` - HuggingFace API token
- `WANDB_API_KEY` - Weights & Biases API key (optional)
- `HUB_REPO` - Model repository (default: btsee/oron-tts)
- `DATASET` - Dataset name (default: btsee/common-voices-24-mn)

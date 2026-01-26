# OronTTS: Mongolian Text-to-Speech with Flow Matching

A high-quality Mongolian (Khalkha dialect) text-to-speech system based on **F5-TTS** architecture using Conditional Flow Matching with a Diffusion Transformer (DiT) backbone.

## Features

- **F5-TTS Architecture**: State-of-the-art flow matching for TTS
- **Mongolian Support**: Full Khalkha Mongolian text normalization and phonemization
- **Multi-Speaker**: Support for multiple speakers with gender separation
- **DeepFilterNet**: Audio denoising for non-professional recordings
- **Flash Attention 2**: Optimized attention for faster training
- **HuggingFace Integration**: Dataset streaming and model checkpointing
- **Runpod Ready**: Automated cloud training setup

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/your-username/oron-tts.git
cd oron-tts

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Install espeak-ng with Mongolian support
# See scripts/setup_runpod.sh for build instructions
```

### Runpod Cloud Training

```bash
# On a fresh Runpod instance
wget https://raw.githubusercontent.com/your-username/oron-tts/main/scripts/setup_runpod.sh
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Project Structure

```
oron-tts/
├── configs/                 # Model configurations
│   ├── base.yaml           # Base configuration
│   ├── light.yaml          # Lightweight model
│   └── hq.yaml             # High-quality model
├── src/
│   ├── core/               # Core model components
│   │   ├── cfm.py          # Conditional Flow Matching
│   │   ├── dit.py          # Diffusion Transformer
│   │   ├── model.py        # F5-TTS model
│   │   └── ode.py          # ODE solvers
│   ├── data/               # Data processing
│   │   ├── audio.py        # Audio/mel processing
│   │   ├── cleaner.py      # Mongolian text normalization
│   │   └── dataset.py      # Dataset handling
│   ├── modules/            # Neural network modules
│   │   ├── attention.py    # Flash Attention + RoPE
│   │   ├── duration.py     # Duration prediction
│   │   └── embeddings.py   # Various embeddings
│   └── utils/              # Utilities
│       ├── checkpoint.py   # Checkpoint management
│       ├── hub.py          # HuggingFace Hub
│       └── logging.py      # Logging utilities
└── scripts/
    ├── train.py            # Training script
    ├── infer.py            # Inference script
    └── setup_runpod.sh     # Runpod setup
```

## Model Variants

| Variant | Parameters | Layers | Heads | Use Case |
|---------|------------|--------|-------|----------|
| Light   | ~50M       | 12     | 8     | Fast inference, edge deployment |
| Base    | ~150M      | 16     | 12    | Balanced quality/speed |
| HQ      | ~300M      | 22     | 16    | Maximum quality |

## Training

### Single GPU

```bash
python scripts/train.py \
    --config configs/light.yaml \
    --hf-dataset your-username/mongolian-speech \
    --output-dir outputs
```

### Multi-GPU with Accelerate

```bash
accelerate launch scripts/train.py \
    --config configs/hq.yaml \
    --hf-dataset your-username/mongolian-speech \
    --hub-repo your-username/oron-tts-hq
```

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML configuration |
| `--resume` | Resume from latest checkpoint |
| `--hf-dataset` | HuggingFace dataset name |
| `--hub-repo` | Sync checkpoints to Hub |
| `--output-dir` | Local output directory |

## Inference

### Single Utterance

```bash
python scripts/infer.py \
    --model outputs/final_model \
    --text "Сайн байна уу" \
    --output hello.wav
```

### Batch Synthesis

```bash
python scripts/infer.py \
    --model outputs/final_model \
    --input-file texts.txt \
    --output-dir outputs/audio \
    --speaker 0
```

### Python API

```python
from src.core import F5TTS

# Load model
model = F5TTS.from_pretrained("your-username/oron-tts-hq")

# Synthesize
mel = model.synthesize(phonemes, speaker_id=0, cfg_scale=2.0)
```

## Mongolian Text Processing

The text cleaner handles:

- **Numeric expansion**: `123` → `нэг зуун хорин гурав`
- **Abbreviation expansion**: `км` → `километр`
- **Unicode normalization**: NFC form
- **Punctuation standardization**

```python
from src.data import MongolianTextCleaner

cleaner = MongolianTextCleaner()
clean_text = cleaner("2024 онд 5 км")
# Output: "хоёр мянга хорин дөрөв онд тав километр"
```

## Audio Pipeline

```python
from src.data import AudioProcessor, AudioConfig

config = AudioConfig(
    sample_rate=24000,
    n_mels=100,
    denoise=True,  # DeepFilterNet
)

processor = AudioProcessor(config)
mel = processor.process("audio.wav")
```

## HuggingFace Hub Integration

### Automatic Checkpoint Sync

```python
from src.utils import HubManager, HubConfig

hub = HubManager(HubConfig(
    repo_id="your-username/oron-tts",
    private=True,
))

# Upload checkpoint
hub.upload_checkpoint("outputs/checkpoint_00010000.pt")

# Sync entire training state
hub.upload_training_state("outputs/")
```

## Requirements

- Python 3.11+
- PyTorch 2.4.0+
- CUDA 12.4+
- espeak-ng with Mongolian support

## Citation

```bibtex
@misc{oron-tts,
  title={OronTTS: Mongolian Text-to-Speech with Flow Matching},
  year={2025},
}
```

## License

Apache 2.0

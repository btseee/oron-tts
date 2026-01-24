# OronTTS

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Mongolian Cyrillic (Khalkha) Text-to-Speech System using VITS2 Architecture**

OronTTS is a high-quality, end-to-end neural TTS system designed specifically for the Mongolian language (Khalkha dialect). It leverages the VITS2 architecture for natural-sounding speech synthesis.

## Features

- 🎯 **Native Mongolian Support**: Full Cyrillic text normalization with proper number-to-text conversion
- 🔊 **VITS2 Architecture**: State-of-the-art variational inference with adversarial learning
- 🧹 **Audio Cleaning**: DeepFilterNet integration for noise suppression
- 👥 **Multi-Speaker**: Support for male/female voice separation
- ☁️ **Cloud Ready**: Optimized for Runpod.io GPU training
- 🤗 **HuggingFace Integration**: Seamless dataset and model management

## Installation

### Prerequisites

- Python 3.11+
- espeak-ng (build from source for Mongolian support)
- CUDA 12.x (for GPU training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/orontts/oron-tts.git
cd oron-tts

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### System Dependencies (Ubuntu/Debian)

```bash
# Install espeak-ng from source
sudo apt-get install -y git make autoconf automake libtool pkg-config
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng
./autogen.sh && ./configure && make && sudo make install

# Install audio dependencies
sudo apt-get install -y libsndfile1 ffmpeg
```

## Project Structure

```
oron-tts/
├── orontts/
│   ├── preprocessing/     # Audio cleaning, text normalization
│   ├── dataset/           # HuggingFace integration, data loading
│   ├── model/             # VITS2 architecture
│   ├── training/          # Training loop, checkpoints
│   ├── inference/         # Synthesis wrappers
│   └── scripts/           # CLI entry points
├── configs/               # Model configurations
├── scripts/               # Setup and utility scripts
└── tests/                 # Unit tests
```

## Usage

### Data Preparation

```bash
# Prepare and clean audio dataset
orontts-prepare --input-dir /path/to/raw/audio --output-dir /path/to/cleaned
```

### Training

```bash
# Train with lightweight config (faster, less VRAM)
orontts-train --config configs/vits2_light.json --data-dir /path/to/dataset

# Train with high-quality config
orontts-train --config configs/vits2_hq.json --data-dir /path/to/dataset
```

### Inference

```bash
# Synthesize speech
orontts-infer --model /path/to/checkpoint.ckpt --text "Сайн байна уу" --output output.wav
```

### Python API

```python
from orontts.inference import Synthesizer

# Load model
synth = Synthesizer.from_pretrained("orontts/mongolian-vits2")

# Generate speech
audio = synth.synthesize("Сайн байна уу", speaker_id=0)
audio.save("output.wav")
```

## Configuration

Two configuration presets are provided:

| Config | Use Case | VRAM | Quality |
|--------|----------|------|---------|
| `vits2_light.json` | Fast inference, limited resources | ~4GB | Good |
| `vits2_hq.json` | Production quality | ~8GB | Excellent |

## Runpod Deployment

```bash
# Run setup script on Runpod instance
bash scripts/setup_runpod.sh

# Start training
orontts-train --config configs/vits2_hq.json --data-dir /workspace/data
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run tests
pytest tests/

# Format code
ruff format src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [VITS2](https://github.com/p0p4k/vits2_pytorch) - Base architecture
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) - Phonemization
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Audio cleaning
- [HuggingFace](https://huggingface.co/) - Model and dataset hosting

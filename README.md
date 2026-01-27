# OronTTS: Mongolian Text-to-Speech

High-quality Mongolian (Khalkha dialect) text-to-speech using **F5-TTS** with Conditional Flow Matching.

## Features

- **F5-TTS Backend**: Finetuned from pretrained F5-TTS for high quality
- **Mongolian Khalkha**: Native support with text normalization and number expansion
- **Multi-Speaker**: Male and female voices with zero-shot voice cloning
- **Combined Dataset**: 5,800+ samples from mbspeech + Common Voice
- **Optimized Training**: Best settings for low loss and natural speech

## Quick Start

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/btseee/oron-tts.git
cd oron-tts

# Install OronTTS
pip install -e .

# Install F5-TTS from submodule
pip install -e third_party/F5-TTS
```

### Development Setup

```bash
# Install with local dependencies (data preprocessing)
pip install -e ".[local]"

# Install with dev dependencies (testing, linting)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Inference

```bash
# Generate example audio with trained model
python scripts/inference/infer.py

# Custom text
python scripts/inference/infer.py \
    --text "Сайн байна уу! Би монгол хэлээр ярьж байна."

# With specific checkpoint
python scripts/inference/infer.py \
    --checkpoint /workspace/output/ckpts/mongolian-tts/model_50000.pt \
    --text "Энэ бол жишээ текст" \
    --output custom_output.wav
```

## Project Structure

```text
oron-tts/
├── .vscode/                 # VS Code configuration
│   ├── settings.json        # Python, Ruff, MyPy settings
│   ├── launch.json          # Debug configurations
│   ├── tasks.json           # Build, lint, test tasks
│   └── extensions.json      # Recommended extensions
├── src/
│   ├── data/
│   │   ├── audio.py         # Audio preprocessing (DeepFilterNet)
│   │   ├── cleaner.py       # Mongolian text normalization
│   │   └── dataset.py       # Dataset handling utilities
│   └── utils/
│       ├── logging.py       # Rich console logging
│       ├── hub.py           # HuggingFace Hub API wrappers
│       └── checkpoint.py    # Model checkpoint utilities
├── scripts/
│   ├── data/
│   │   └── prepare_combined_dataset.py  # Dataset preparation
│   ├── training/
│   │   ├── train.py         # Optimized training script
│   │   └── train.sh         # RunPod training wrapper
│   ├── inference/
│   │   └── infer.py         # Generate speech
│   ├── setup/
│   │   └── runpod_setup.sh  # RunPod environment setup
│   └── utils/
│       └── upload_to_hub.py # HuggingFace upload
├── tests/
│   ├── conftest.py          # Pytest configuration
│   └── test_cleaner.py      # Text cleaner tests
├── third_party/
│   └── F5-TTS/              # Official F5-TTS (submodule)
└── /workspace/output/       # Training outputs (RunPod/Docker)
    ├── ckpts/               # Model checkpoints
    ├── logs/                # Training logs
    └── runs/                # TensorBoard logs
```

## Python API

```python
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, "third_party/F5-TTS/src")

from src.data import MongolianTextCleaner
# Use scripts/inference/infer.py for generation

cleaner = MongolianTextCleaner()
text = cleaner("2024 онд 5 км")
# Output: "хоёр мянга хорин дөрөв онд тав километр"
```

## Mongolian Text Processing

- **Numeric expansion**: `123` → `нэг зуун хорин гурав`
- **Unicode normalization**: NFC form
- **Punctuation standardization**

```python
from src.data import MongolianTextCleaner

cleaner = MongolianTextCleaner()
text = cleaner("2024 онд 5 км")
# Output: "хоёр мянга хорин дөрөв онд тав километр"
```

## Training

### RunPod Setup (Cloud Training)

```bash
# First-time setup on RunPod instance
bash scripts/setup/runpod_setup.sh

# This will:
# - Clone repository with F5-TTS submodule
# - Install dependencies (PyTorch, F5-TTS)
# - Download pretrained checkpoint
# - Set up environment variables
```

### Prepare Combined Dataset

```bash
# Combine mbspeech (3.8k samples) + Common Voice (best male/female voices)
python scripts/data/prepare_combined_dataset.py
```

### Train with Optimal Settings

```bash
# Finetune from pretrained F5-TTS (recommended)
python scripts/training/train.py

# Output saved to /workspace/output/
```

**Training Strategy:**
- Finetuning (NOT from scratch) - essential for small datasets
- Learning rate: 7.5e-5 (optimized for finetuning)
- Batch size: 4800 frames (2400 × 2 accumulation)
- Warmup: 2000 updates
- Target: Loss < 0.15 for high quality

### Upload to HuggingFace

```bash
python scripts/utils/upload_to_hub.py \
    --model-name btsee/oron-tts-mongolian \
    --checkpoint-dir /workspace/output/ckpts/mongolian-tts
```

## License

Apache 2.0

## Citation

```bibtex
@software{orontts2026,
  author = {OronTTS Team},
  title = {OronTTS: Mongolian Text-to-Speech with F5-TTS},
  year = {2026},
  url = {https://github.com/btseee/oron-tts}
}
```

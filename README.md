# OronTTS: Mongolian Text-to-Speech

High-quality Mongolian (Khalkha dialect) text-to-speech using **F5-TTS** with Conditional Flow Matching.

## Features

- **F5-TTS Backend**: Official F5-TTS as git submodule for voice cloning
- **Mongolian Support**: Khalkha text normalization and phonemization
- **Multi-Speaker**: Zero-shot voice cloning from reference audio
- **DeepFilterNet**: Audio denoising for recordings

## Quick Start

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/btseee/oron-tts.git
cd oron-tts

# Or if already cloned:
git submodule update --init --recursive

# Install OronTTS
pip install -e .

# Install F5-TTS from submodule
pip install -e third_party/F5-TTS
```

### Inference

```bash
# Basic synthesis
python scripts/inference/infer.py \
    --ref-audio reference.wav \
    --ref-text "Энэ бол жишээ текст" \
    --gen-text "Сайн байна уу" \
    --output output.wav

# With custom checkpoint
python scripts/inference/infer.py \
    --ckpt-file path/to/checkpoint.pt \
    --ref-audio reference.wav \
    --gen-text "Сайн байна уу" \
    --output output.wav

# Batch synthesis
python scripts/inference/infer.py \
    --ref-audio reference.wav \
    --ref-text "Энэ бол жишээ текст" \
    --input-file texts.txt \
    --output-dir outputs/audio
```

## Project Structure

```text
oron-tts/
├── configs/
│   └── config.yaml          # Configuration
├── src/
│   ├── data/                # Audio, text processing
│   │   ├── audio.py         # Audio preprocessing
│   │   └── cleaner.py       # Mongolian text normalization
│   └── utils/               # Checkpoints, logging, hub
├── scripts/
│   ├── inference/infer.py   # Inference script
│   ├── data/                # Dataset preparation
│   └── setup/               # Environment setup
└── third_party/
    └── F5-TTS/              # Official F5-TTS (submodule)
```

## Python API

```python
import sys
from pathlib import Path

# Add F5-TTS to path
sys.path.insert(0, str(Path("third_party/F5-TTS/src")))

from f5_tts.api import F5TTS
from src.data import MongolianTextCleaner

# Initialize
tts = F5TTS()
cleaner = MongolianTextCleaner()

# Clean Mongolian text
text = cleaner("2024 онд 5 км")

# Synthesize
wav, sr, spec = tts.infer(
    ref_file="reference.wav",
    ref_text="Энэ бол жишээ текст",
    gen_text=text,
    file_wave="output.wav",
)
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

For training custom models, use the official F5-TTS training pipeline:

```bash
cd third_party/F5-TTS
python src/f5_tts/train/train.py --help
```

See the [F5-TTS documentation](https://github.com/SWivid/F5-TTS) for training details.

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

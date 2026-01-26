# OronTTS: Mongolian Text-to-Speech

High-quality Mongolian (Khalkha dialect) text-to-speech using **F5-TTS** with Conditional Flow Matching.

## Features

- **F5-TTS Architecture**: Flow matching with Diffusion Transformer backbone
- **Mongolian Support**: Khalkha text normalization and phonemization
- **Multi-Speaker**: Multiple speakers with gender separation
- **DeepFilterNet**: Audio denoising for recordings
- **Flash Attention 2**: Optimized training
- **HuggingFace**: Dataset streaming and model hosting

## Quick Start

### Docker (Recommended)

```bash
# Set HuggingFace token
export HF_TOKEN=hf_xxx

# Build and run
docker-compose up -d

# Start training
docker exec -it oron-tts accelerate launch scripts/training/train.py \
  --config configs/config.yaml \
  --output-dir outputs \
  --hub-repo btsee/oron-tts \
  --hf-dataset btsee/common-voices-24-mn
```

### Runpod

```bash
# Run setup script
curl -sSL https://raw.githubusercontent.com/btseee/oron-tts/main/scripts/setup/setup_runpod.sh | bash

# Train
cd /workspace/oron-tts
accelerate launch scripts/training/train.py \
  --config configs/config.yaml \
  --output-dir outputs \
  --hub-repo btsee/oron-tts \
  --hf-dataset btsee/common-voices-24-mn
```

### Local

```bash
# Clone
git clone https://github.com/btseee/oron-tts.git
cd oron-tts

# Install
pip install -e ".[dev]"

# Build espeak-ng from source (see scripts/setup/setup_runpod.sh)

# Train
accelerate launch scripts/training/train.py \
  --config configs/config.yaml \
  --output-dir outputs \
  --hub-repo btsee/oron-tts \
  --hf-dataset btsee/common-voices-24-mn
```

## Project Structure

```
oron-tts/
├── configs/
│   └── config.yaml         # Model configuration
├── src/
│   ├── core/              # CFM, DiT, F5-TTS model
│   ├── data/              # Audio, text processing, dataset
│   ├── modules/           # Attention, embeddings
│   └── utils/             # Checkpoints, logging, hub
└── scripts/
    ├── training/train.py
    ├── data/prepare_cv_dataset.py
    └── setup/setup_runpod.sh
```

## Model Architecture

| Component | Value |
|-----------|-------|
| Parameters | ~300M |
| Layers | 22 |
| Attention Heads | 16 |
| Embedding Dim | 1024 |
| Max Sequence | 4096 tokens |

## Training Arguments

```bash
--config          # Path to config.yaml
--output-dir      # Output directory
--hub-repo        # HuggingFace repo for checkpoints
--hf-dataset      # HuggingFace dataset name
--resume          # Resume from checkpoint
```

## Inference

```python
from src.core import F5TTS

model = F5TTS.from_pretrained("btsee/oron-tts")
mel = model.synthesize(phonemes, speaker_id=0, cfg_scale=2.0)
```

## Mongolian Text Processing

- **Numeric expansion**: `123` → `нэг зуун хорин гурав`
- **Unicode normalization**: NFC form
- **Punctuation standardization**

```python
from src.data import MongolianTextCleaner

cleaner = MongolianTextCleaner()
text = cleaner("2024 онд 5 км")
```

## Dataset Preparation

```bash
python scripts/data/prepare_cv_dataset.py \
  --output-repo btsee/common-voices-24-mn \
  --min-duration 1.0 \
  --max-duration 15.0 \
  --quality-filter  # Only upvotes>0, downvotes=0
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

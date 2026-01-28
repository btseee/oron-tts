# OronTTS

Mongolian Cyrillic (Khalkha) multi-speaker Text-to-Speech system using VITS architecture.

## Features

- **VITS Architecture**: End-to-end TTS with variational inference and adversarial training
- **Multi-speaker**: Support for distinct male and female voices
- **Mongolian Text Processing**: Custom rule-based phonemizer for Cyrillic script
- **Number Normalization**: Comprehensive Mongolian number-to-text transliteration
- **Audio Denoising**: DeepFilterNet integration for preprocessing non-professional recordings
- **Hugging Face Integration**: Dataset and model hub support

## Installation

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Project Structure

```text
oron-tts/
├── src/
│   ├── data/           # Dataset wrappers, denoising, preprocessing
│   ├── models/         # VITS architecture components
│   ├── training/       # Training loop, losses, checkpointing
│   └── utils/          # Audio processing, text normalization
├── scripts/
│   ├── prepare.py      # Dataset preparation
│   ├── train.py        # Model training
│   └── infer.py        # Inference/synthesis
└── configs/            # YAML configuration files
```

## Usage

### 1. Dataset Preparation (Local)

Clean and denoise audio from Common Voice and MBSpeech datasets:

```bash
python scripts/prepare.py \
    --output-dir data/processed \
    --dataset all \
    --upload \
    --hf-repo btsee/oron-tts-dataset
```

### 2. Training (RunPod/Cloud)

Train the VITS model:

```bash
# Single GPU
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --from-hf \
    --dataset btsee/oron-tts-dataset \
    --push-to-hub \
    --hf-repo btsee/oron-tts-model

# Multi-GPU
python scripts/train.py \
    --config configs/vits_runpod.yaml \
    --from-hf \
    --dataset btsee/oron-tts-dataset \
    --num-gpus 4
```

### 3. Inference

Generate speech from text:

```bash
python scripts/infer.py \
    --checkpoint checkpoints/vits_best.pt \
    --text "Сайн байна уу" \
    --speaker 0 \
    --output output.wav
```

Speaker IDs:

- `0`: Female voice
- `1`: Male voice

## Mongolian Number Examples

| Input | Output                  |
| ----- | ----------------------- |
| 10    | арван                   |
| 25    | хорин тав               |
| 100   | зуун                    |
| 1-р   | нэгдүгээр               |
| 2024  | хоёр мянга хорин дөрөв  |

## Configuration

Key hyperparameters in `configs/vits_base.yaml`:

```yaml
sample_rate: 22050
batch_size: 16
learning_rate: 0.0002
model:
  hidden_channels: 192
  n_layers: 6
  n_heads: 2
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check src/ scripts/

# Format
ruff format src/ scripts/
isort src/ scripts/
```

## License

MIT

## Citation

If you use OronTTS in your research, please cite:

```bibtex
@software{orontts2024,
  title = {OronTTS: Mongolian Text-to-Speech},
  year = {2024},
  url = {https://github.com/btsee/oron-tts}
}
```

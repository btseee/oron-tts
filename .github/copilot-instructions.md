# OronTTS Project Instructions

## General Principles
- **Language:** Python 3.11.
- **Style:** Functional and modular. Avoid monolithic files.
- **Tone:** Blunt, technical, directive. 
- **Documentation:** Minimal, meaningful comments only. Focus on self-documenting code.

## Project Structure
- `src/data/`: Text normalization, DeepFilterNet denoising, and Hugging Face Dataset wrappers.
- `src/models/`: VITS architecture components (Encoder, Flow, Stochastic Duration Predictor, HiFi-GAN Decoder).
- `src/training/`: Training logic, loss functions (adversarial, KL, spectral), and RunPod setup.
- `src/utils/`: Audio processing, number transliteration, and checkpoint management.
- `configs/`: YAML/JSON files for hyperparameters.
- `scripts/`: Entry points for `prepare.py`, `train.py`, and `infer.py`.

## Technical Specifics
- **Number Normalization:** `src/utils/number_norm.py` must handle ordinal and cardinal Mongolian numbers.
- **Audio Cleaning:** Use `df-net` (DeepFilterNet) to preprocess non-professional recordings before training.
- **No Coqui:** All model code must be native PyTorch or based on original VITS implementations.
- **Hardware Target:** Local CPU/GPU for prep; NVIDIA A100/L40 on RunPod for training.

## Formatting and Quality
- **Linter:** Ruff.
- **Formatter:** Black.
- **Imports:** Sorted via isort.
- **Typing:** Strict Python type hints required for all functions.

## Workflow
1. Local prep: `python scripts/prepare.py` (Clean -> Denoise -> Upload to HF).
2. Cloud training: `python scripts/train.py` (Pull from HF -> Train -> Push Model).
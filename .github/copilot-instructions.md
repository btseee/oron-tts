# OronTTS Copilot Instructions

## 1. Environment & Stack
- **Language:** Python 3.11
- **Framework:** PyTorch 2.4.0 (via F5-TTS)
- **Compute:** CUDA 12.4.1
- **Architecture:** F5-TTS (official repo as git submodule)
- **Local Only:** DeepFilterNet, espeak-ng (for data cleaning)

## 2. Project Structure
- `configs/`: YAML configuration files.
- `third_party/F5-TTS/`: Official F5-TTS repository (git submodule).
- `src/data/`:
    - `cleaner.py`: Mongolian text normalization & numeric expansion.
    - `audio.py`: DeepFilterNet pipeline & resampling.
- `src/utils/`: Logging, HF API wrappers.
- `scripts/training/`: Training scripts wrapping F5-TTS.
- `scripts/inference/`: Inference using F5-TTS API with Mongolian text cleaning.
- `scripts/data/`: Dataset preparation scripts (local only).
- `scripts/setup/`: Environment setup scripts.

## 3. Coding Standards
- **Style:** Clean Architecture. No global state.
- **Typing:** Strict static typing for all function signatures.
- **Documentation:** Terse, functional comments. Avoid redundant descriptions.
- **Logic:** Favor `pathlib` over `os`. Use `match` statements for speaker/gender metadata.

## 4. Mongolian Specifics
- **Normalization:** Expand numbers to Cyrillic text following Khalkha declension rules.
- **Phonemes:** Interface with `espeak-ng` using the Mongolian (`mn`) data.

## 5. Workflow & Dependencies
- **Local (`pip install -e ".[local]"`):** Data cleaning, preprocessing, dataset upload.
- **Cloud (RunPod/Docker):** Training only. Install F5-TTS from submodule.
- **Submodule:** F5-TTS provides all training/inference dependencies.
- **HuggingFace:** Models and datasets stored on HF Hub.
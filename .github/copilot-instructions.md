# OronTTS Copilot Instructions

## 1. Environment & Stack

- **Language:** Python 3.11
- **Framework:** PyTorch 2.4.0 (via F5-TTS)
- **Compute:** CUDA 12.4.1
- **Architecture:** F5-TTS (official repo as git submodule at `third_party/F5-TTS/`)
- **Local Only:** DeepFilterNet, espeak-ng (for data cleaning)
- **Audio:** HuggingFace datasets with soundfile decoder (`HF_AUDIO_DECODER=soundfile`)

## 2. Project Structure

- `configs/`: YAML configuration files
- `third_party/F5-TTS/`: Official F5-TTS repository (git submodule)
- `src/data/`:
  - `cleaner.py`: Mongolian text normalization & numeric expansion
  - `audio.py`: DeepFilterNet pipeline & resampling
  - `dataset.py`: Dataset handling utilities
- `src/utils/`:
  - `logging.py`: Rich console logging
  - `hub.py`: HuggingFace Hub API wrappers
  - `checkpoint.py`: Model checkpoint utilities
- `scripts/training/`: Training scripts wrapping F5-TTS
  - `train.py`: Main training script with optimized hyperparameters
  - `train.sh`: Shell wrapper for RunPod
- `scripts/inference/`:
  - `infer.py`: Inference using F5-TTS API with Mongolian text cleaning
- `scripts/data/`:
  - `prepare_combined_dataset.py`: Combine mbspeech + Common Voice datasets- `scripts/setup/`:
    - `runpod_setup.sh`: RunPod environment setup script- `scripts/utils/`:
  - `upload_to_hub.py`: Upload models/datasets to HuggingFace Hub
- `.vscode/`: VS Code configuration
  - `settings.json`: Python, Ruff, MyPy configuration
  - `launch.json`: Debug configurations for all scripts
  - `tasks.json`: Build, lint, test, training tasks
  - `extensions.json`: Recommended extensions

## 3. Coding Standards

- **Style:** Clean Architecture. No global state.
- **Typing:** Strict static typing for all function signatures. Use `from __future__ import annotations`.
- **Documentation:** Terse, functional comments. Avoid redundant descriptions.
- **Logic:** Favor `pathlib` over `os`. Use `match` statements for speaker/gender metadata.
- **Formatting:** Ruff for linting and formatting (100 char line length)
- **Type Checking:** MyPy with strict mode
- **Testing:** Pytest with coverage reporting

## 4. Mongolian Specifics

- **Normalization:** Expand numbers to Cyrillic text following Khalkha declension rules
- **Phonemes:** Interface with `espeak-ng` using the Mongolian (`mn`) data
- **Text Processing:** `MongolianTextCleaner` in `src/data/cleaner.py`
- **Dataset:** Combined mbspeech (3.8k samples) + Common Voice (best male/female voices)

## 5. Workflow & Dependencies

- **Local (`pip install -e ".[local]"`):** Data cleaning, preprocessing, dataset upload
- **Cloud (RunPod/Docker):** Training only. Install F5-TTS from submodule
- **Submodule:** F5-TTS provides all training/inference dependencies
- **HuggingFace:** Models and datasets stored on HF Hub (user: `btsee`)
- **Environment Variables:**
  - `PYTHONPATH`: `${workspaceFolder}:${workspaceFolder}/third_party/F5-TTS/src`
  - `HF_AUDIO_DECODER`: `soundfile`
  - `CUDA_VISIBLE_DEVICES`: `0` (for training)

## 6. Training Strategy

- **Method:** Finetune from pretrained F5-TTS (NOT from scratch)
- **Learning Rate:** 7.5e-6 (low for stability)
- **Batch Size:** 4800 frames (2400 × 2 accumulation)
- **Warmup:** 2000 updates
- **Target:** Loss < 0.15 for high quality
- **Output:** `/workspace/output/` (checkpoints, logs, TensorBoard)

## 7. VS Code Integration

- **Launch Configs:** Training, inference, dataset prep, tests
- **Tasks:** Lint, format, type check, test, install dependencies
- **Extensions:** Python, Ruff, MyPy, Jupyter, GitHub Copilot
- **Debugging:** Use debugpy with `justMyCode=false` to debug F5-TTS internals

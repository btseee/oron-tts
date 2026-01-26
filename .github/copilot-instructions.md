# OronTTS Copilot Instructions

## 1. Environment & Stack
- **Language:** Python 3.11
- **Framework:** PyTorch 2.4.0
- **Compute:** CUDA 12.4.1
- **Architecture:** F5-TTS (DiT-based CFM)
- **Audio Prep:** DeepFilterNet
- **Phonemizer:** espeak-ng (source build)

## 2. Project Structure
- `configs/`: JSON/YAML for model hyperparameters (light vs. hq).
- `src/core/`: CFM solvers, DiT backbone, and ODE integration.
- `src/data/`:
    - `cleaner.py`: Mongolian text normalization & numeric expansion.
    - `audio.py`: DeepFilterNet pipeline & resampling.
    - `dataset.py`: HF Hub integration & multi-speaker formatting.
- `src/modules/`: Attention mechanisms (RoPE, Flash), Embeddings.
- `src/utils/`: Logging, HF API wrappers, checkpointing.
- `scripts/`: `train.py`, `infer.py`, `setup_runpod.sh`.

## 3. Coding Standards
- **Style:** Clean Architecture. No global state.
- **Typing:** Strict static typing for all function signatures.
- **Documentation:** Terse, functional comments. Avoid redundant descriptions.
- **Logic:** Favor `pathlib` over `os`. Use `match` statements for speaker/gender metadata.
- **Efficiency:** Ensure `torch.compile` compatibility in the DiT forward pass.

## 4. Mongolian Specifics
- **Normalization:** Expand numbers to Cyrillic text following Khalkha declension rules.
- **Phonemes:** Interface with `espeak-ng` using the Mongolian (`mn`) data.

## 5. Workflow Constraints
- **Local:** Preprocessing and dataset upload to HuggingFace.
- **Cloud:** Training on Runpod.io. 
- **Persistence:** All training state must sync to HuggingFace Hub.
- **Infrastructure:** `setup_runpod.sh` must handle system dependencies (`cmake`, `espeak-ng`, `libsndfile`).
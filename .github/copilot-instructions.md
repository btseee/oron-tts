# OronTTS Copilot Instructions

## Project Context
* **Name:** OronTTS
* **Goal:** Mongolian Cyrillic (Khalkha) VITS2 TTS.
* **Target Runtime:** Python 3.14.
* **Deployment:** Runpod.io (Training), Local (Prep).

## Code Style & Standards
* **Formatting:** Enforce PEP 8. Use `ruff` configuration. Line length 88.
* **Typing:** Strict static typing. Use `typing.TypeVar`, `typing.Protocol`, and Python 3.12+ generic syntax (e.g., `def fn[T](x: T) -> T`).
* **Comments:** Functional and terse. Explain *why*, not *what*.
* **Docstrings:** Google Style. Mandatory for all public modules and classes.

## Architecture Boundaries
1.  **Preprocessing (`src/preprocessing`)**:
    * Strict separation between Audio (DeepFilterNet) and Text (Normalization) logic.
    * Mongolian number transliteration must handle cases strictly.
2.  **Dataset (`src/dataset`)**:
    * Abstraction layer over `huggingface_datasets`.
    * Output format must match VITS2 dataloader requirements directly.
3.  **Model (`src/model`)**:
    * Implement VITS2 logic. Keep PyTorch modules distinct from training logic.
    * Do not hardcode hyperparameters; load strictly from `configs/*.json`.

## Specific Library Usage
* **Audio:** `torchaudio`, `librosa` (latest).
* **Cleaning:** `DeepFilterNet`.
* **Phonemization:** Wrapper around `espeak-ng` binary.
* **Training:** `pytorch-lightning` or raw `torch` training loop (optimized for VRAM).
* **Config:** `pydantic` for configuration validation.

## Workflow Rules
* **Runpod Compatibility:** All paths must be relative or environment-variable driven. Assume `/workspace` as root in cloud.
* **HuggingFace:** Use `huggingface_hub` `HfApi` for all artifact transfers.
* **Error Handling:** Fail fast. Raise custom exceptions defined in `src/exceptions.py`.

## Forbidden Patterns
* Do not use deprecated audio libraries (e.g., `pydub` unless necessary).
* Do not hardcode file paths (use `pathlib`).
* Do not include massive commented-out blocks.
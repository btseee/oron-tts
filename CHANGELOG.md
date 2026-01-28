# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Initial VITS architecture implementation for Mongolian TTS
- Multi-speaker support with distinct male and female voices
- DeepFilterNet integration for audio denoising
- Mongolian text normalization (Khalkha Cyrillic)
- Comprehensive number-to-text transliteration (cardinal and ordinal)
- Custom rule-based phonemizer for Mongolian Cyrillic
- HuggingFace dataset and model hub integration
- RunPod cloud training configuration and scripts
- Structured logging for container environments
  - Timestamped log output for RunPod/cloud platforms
  - Configurable tqdm enable/disable for local vs cloud training
  - Detailed loss component logging (mel, KL, duration, discriminator, generator)
- Multi-GPU distributed training support
- Mixed precision (FP16) training with gradient scaling
- TensorBoard integration for training monitoring
- Checkpoint management with automatic saving and HuggingFace Hub push

### Fixed

- PyTorch 2.x compatibility: Updated `torch.amp` imports to use correct submodules
  - Changed to `torch.amp.autocast_mode.autocast`
  - Changed to `torch.amp.grad_scaler.GradScaler`
- NaN loss issues during training:
  - Added log variance clamping in text encoder (±10.0 range)
  - Added log variance clamping in posterior encoder (±10.0 range)
  - Added log variance clamping in flow layers (±10.0 range)
  - Added log variance clamping in ElementwiseAffine module (±10.0 range)
  - Added numerical stability to KL divergence loss computation
  - Added clamping to VITS forward pass log computations
- Rational quadratic spline numerical stability:
  - Fixed derivative tensor shape (num_bins * 3 + 1)
  - Added proper index clamping in spline transform
  - Added discriminant clamping to prevent negative square roots
- Audio waveform handling:
  - Fixed torchcodec AudioDecoder integration
  - Added proper audio segment slicing for discriminator input
  - Fixed channel dimension handling for discriminator
- Dataset loading:
  - Auto-detection of text columns in HuggingFace datasets
  - Support for multiple text column names (text, sentence, sentence_norm, etc.)
  - Proper handling of AudioDecoder objects from torchcodec

### Changed

- Training configuration:
  - Optimized batch sizes for RTX 5070 Ti (12) and RTX 4090 (24)
  - Configured segment sizes for memory efficiency
  - Added CUDA optimizations (cudnn_benchmark, use_tf32, torch.compile)
- Logging behavior:
  - Local training uses tqdm progress bars by default
  - Cloud/RunPod training uses structured timestamp logs
  - Configurable log intervals for different environments

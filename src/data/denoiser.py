"""Audio denoiser using DeepFilterNet."""

from pathlib import Path
from typing import Final

import numpy as np
import torch
import torchaudio

DF_SAMPLE_RATE: Final[int] = 48000


class AudioDenoiser:
    def __init__(self, target_sr: int = 22050) -> None:
        self.target_sr = target_sr
        self._model = None
        self._df_state = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        try:
            from df.enhance import enhance, init_df
            self._df_state, self._model, _ = init_df()
            self._enhance_fn = enhance
            self._initialized = True
        except ImportError as e:
            raise ImportError(
                "DeepFilterNet not installed. Install with: pip install deepfilternet"
            ) from e

    def denoise(self, audio: torch.Tensor | np.ndarray, sr: int) -> torch.Tensor:
        self._lazy_init()

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if sr != DF_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DF_SAMPLE_RATE)

        enhanced = self._enhance_fn(self._df_state, self._model, audio)  # type: ignore

        if self.target_sr != DF_SAMPLE_RATE:
            enhanced = torchaudio.functional.resample(enhanced, DF_SAMPLE_RATE, self.target_sr)

        return enhanced.squeeze(0)

    def denoise_file(self, input_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        enhanced = self.denoise(audio.squeeze(0), sr)
        torchaudio.save(str(output_path), enhanced.unsqueeze(0), self.target_sr)
        return output_path

    def process_batch(
        self,
        input_paths: list[Path],
        output_dir: Path,
        progress: bool = True,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        if progress:
            from tqdm import tqdm
            iterator = tqdm(input_paths, desc="Denoising")
        else:
            iterator = input_paths

        for input_path in iterator:
            output_path = output_dir / f"{input_path.stem}_denoised.wav"
            try:
                self.denoise_file(input_path, output_path)
                output_paths.append(output_path)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

        return output_paths

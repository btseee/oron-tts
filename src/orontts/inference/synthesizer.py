"""High-level synthesizer interface for VITS2."""

from pathlib import Path
from typing import Literal, Self

import torch

from orontts.constants import SAMPLE_RATE
from orontts.dataset.hf_integration import download_checkpoint
from orontts.exceptions import InferenceError, ModelError
from orontts.inference.audio_output import AudioOutput
from orontts.model.config import VITS2Config
from orontts.model.vits2 import VITS2
from orontts.preprocessing.phonemizer import text_to_phoneme_ids
from orontts.preprocessing.text import normalize_text


class Synthesizer:
    """High-level synthesizer for VITS2 TTS.

    Provides a simple interface for text-to-speech synthesis with
    automatic text normalization and phonemization.

    Attributes:
        model: VITS2 model instance.
        config: Model configuration.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model: VITS2,
        config: VITS2Config,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize synthesizer.

        Args:
            model: Loaded VITS2 model.
            config: Model configuration.
            device: Device for inference.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # Remove weight norm for faster inference
        self.model.remove_weight_norm()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> Self:
        """Load synthesizer from checkpoint file.

        Args:
            checkpoint_path: Path to .ckpt or .pt file.
            device: Device for inference.

        Returns:
            Initialized Synthesizer.

        Raises:
            ModelError: If loading fails.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise ModelError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Extract config
            if "config" in checkpoint:
                config = VITS2Config.model_validate(checkpoint["config"])
            else:
                # Try loading config from adjacent file
                config_path = checkpoint_path.parent / "config.json"
                if config_path.exists():
                    config = VITS2Config.load(config_path)
                else:
                    raise ModelError("No config found in checkpoint or adjacent file")

            # Create model
            model = VITS2(config)

            # Load state dict
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Handle Lightning module prefix
                state_dict = {
                    k.replace("generator.", "", 1) if k.startswith("generator.") else k: v
                    for k, v in state_dict.items()
                    if not k.startswith("discriminator.")
                }
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            return cls(model, config, device)

        except Exception as e:
            raise ModelError(f"Failed to load checkpoint: {e}") from e

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str | torch.device = "cpu",
        cache_dir: Path | None = None,
    ) -> Self:
        """Load synthesizer from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "orontts/mongolian-vits2").
            device: Device for inference.
            cache_dir: Local cache directory.

        Returns:
            Initialized Synthesizer.
        """
        try:
            checkpoint_path = download_checkpoint(
                repo_id=model_id,
                filename="model.ckpt",
                local_dir=cache_dir,
            )
            return cls.from_checkpoint(checkpoint_path, device)

        except Exception as e:
            raise ModelError(f"Failed to load from Hub: {e}") from e

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        normalize: bool = True,
        phonemize: bool = True,
    ) -> AudioOutput:
        """Synthesize speech from text.

        Args:
            text: Input text in Mongolian Cyrillic.
            speaker_id: Speaker ID for multi-speaker models.
            noise_scale: Noise scale for sampling (higher = more variation).
            length_scale: Duration scale (higher = slower speech).
            noise_scale_w: Noise scale for duration prediction.
            normalize: Whether to normalize text before synthesis.
            phonemize: Whether to convert text to phonemes.

        Returns:
            AudioOutput containing generated audio.

        Raises:
            InferenceError: If synthesis fails.
        """
        try:
            # Preprocess text
            if normalize:
                text = normalize_text(text)

            # Convert to phoneme IDs
            if phonemize:
                phoneme_ids = text_to_phoneme_ids(text)
            else:
                # Assume text is already phoneme string
                from orontts.constants import PHONEME_TO_ID
                phoneme_ids = [PHONEME_TO_ID.get(c, 0) for c in text]

            # Prepare tensors
            phoneme_tensor = torch.LongTensor([phoneme_ids]).to(self.device)
            phoneme_lengths = torch.LongTensor([len(phoneme_ids)]).to(self.device)

            speaker_tensor = None
            if self.config.n_speakers > 1:
                speaker_tensor = torch.LongTensor([speaker_id]).to(self.device)

            # Generate audio
            audio = self.model.infer(
                phoneme_ids=phoneme_tensor,
                phoneme_lengths=phoneme_lengths,
                speaker_ids=speaker_tensor,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
            )

            # Convert to AudioOutput
            audio = audio.squeeze().cpu()

            return AudioOutput(
                audio=audio,
                sample_rate=self.config.audio.sample_rate,
                speaker_id=speaker_id,
                text=text,
            )

        except Exception as e:
            raise InferenceError(f"Synthesis failed: {e}") from e

    @torch.inference_mode()
    def synthesize_batch(
        self,
        texts: list[str],
        speaker_ids: list[int] | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> list[AudioOutput]:
        """Synthesize multiple texts.

        Args:
            texts: List of input texts.
            speaker_ids: List of speaker IDs (same length as texts).
            noise_scale: Noise scale for sampling.
            length_scale: Duration scale.

        Returns:
            List of AudioOutput objects.
        """
        if speaker_ids is None:
            speaker_ids = [0] * len(texts)

        outputs = []
        for text, sid in zip(texts, speaker_ids, strict=True):
            output = self.synthesize(
                text=text,
                speaker_id=sid,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )
            outputs.append(output)

        return outputs

    def list_speakers(self) -> list[int]:
        """Get list of available speaker IDs."""
        return list(range(self.config.n_speakers))

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self.config.audio.sample_rate

    def to(self, device: str | torch.device) -> Self:
        """Move model to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        return self


class LightweightSynthesizer(Synthesizer):
    """Optimized synthesizer for low-latency inference.

    Uses torch.compile and other optimizations for faster synthesis.
    """

    def __init__(
        self,
        model: VITS2,
        config: VITS2Config,
        device: str | torch.device = "cpu",
        compile_model: bool = True,
    ) -> None:
        super().__init__(model, config, device)

        # Compile model for faster inference
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                # Compilation may fail on some platforms
                pass

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        compile_model: bool = True,
    ) -> Self:
        """Load lightweight synthesizer from checkpoint."""
        # Use parent class loading logic
        synth = super().from_checkpoint(checkpoint_path, device)

        # Convert to lightweight
        return cls(
            model=synth.model,
            config=synth.config,
            device=device,
            compile_model=compile_model,
        )


class StreamingSynthesizer(Synthesizer):
    """Synthesizer for streaming audio generation.

    Generates audio in chunks for real-time applications.
    """

    def __init__(
        self,
        model: VITS2,
        config: VITS2Config,
        device: str | torch.device = "cpu",
        chunk_size: int = 8192,
    ) -> None:
        super().__init__(model, config, device)
        self.chunk_size = chunk_size

    def synthesize_stream(
        self,
        text: str,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ):
        """Generate audio as a stream of chunks.

        Args:
            text: Input text.
            speaker_id: Speaker ID.
            noise_scale: Noise scale.
            length_scale: Length scale.

        Yields:
            Audio chunks as numpy arrays.
        """
        # Generate full audio first
        output = self.synthesize(
            text=text,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )

        # Yield in chunks
        audio = output.numpy
        for i in range(0, len(audio), self.chunk_size):
            yield audio[i : i + self.chunk_size]

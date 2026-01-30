"""Inference script for OronTTS."""

import argparse
from pathlib import Path

import torch
import torchaudio

from src.models.vits import VITS
from src.utils.audio import AudioProcessor
from src.utils.phonemizer import MongolianPhonemizer
from src.utils.text_cleaner import TextCleaner


class OronTTSSynthesizer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        sample_rate: int = 22050,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.text_cleaner = TextCleaner()
        self.phonemizer = MongolianPhonemizer()
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

    def _load_model(self, checkpoint_path: str | Path) -> VITS:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        config = checkpoint.get("config", {})
        model_config = config.get("model", {})

        n_speakers = model_config.get("n_speakers", 2)

        model = VITS(
            n_vocab=self.phonemizer.vocab_size,
            spec_channels=config.get("n_mels", 80),
            segment_size=config.get("segment_size", 32),
            inter_channels=model_config.get("inter_channels", 192),
            hidden_channels=model_config.get("hidden_channels", 192),
            filter_channels=model_config.get("filter_channels", 768),
            n_heads=model_config.get("n_heads", 2),
            n_layers=model_config.get("n_layers", 6),
            kernel_size=model_config.get("kernel_size", 3),
            p_dropout=model_config.get("p_dropout", 0.1),
            resblock=model_config.get("resblock", "1"),
            resblock_kernel_sizes=model_config.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=model_config.get(
                "resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            ),
            upsample_rates=model_config.get("upsample_rates", [8, 8, 2, 2]),
            upsample_initial_channel=model_config.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=model_config.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            n_speakers=n_speakers,
            gin_channels=model_config.get("gin_channels", 256),
            use_sdp=model_config.get("use_sdp", True),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        return model

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        cleaned_text = self.text_cleaner.clean(text)
        text_ids = self.text_cleaner.text_to_sequence(cleaned_text)

        x = torch.LongTensor(text_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(text_ids)]).to(self.device)
        sid = torch.LongTensor([speaker_id]).to(self.device)

        with torch.no_grad():
            audio, attn, mask, _ = self.model.infer(
                x,
                x_lengths,
                sid=sid,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )

        audio = audio.squeeze(0).squeeze(0).cpu()
        audio = self.audio_processor.normalize_audio(audio)
        return audio

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> Path:
        audio = self.synthesize(
            text,
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), audio.unsqueeze(0), self.sample_rate)
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OronTTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--speaker", type=int, default=0, help="0=female, 1=male")
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    synthesizer = OronTTSSynthesizer(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print(f"Synthesizing: {args.text}")
    output_path = synthesizer.synthesize_to_file(
        text=args.text,
        output_path=args.output,
        speaker_id=args.speaker,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
    )
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    main()

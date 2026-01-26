#!/usr/bin/env python3
"""OronTTS Inference Script for speech synthesis.

Usage:
    python scripts/infer.py \
        --model outputs/final_model \
        --text "Сайн байна уу" \
        --output output.wav
        
    # With speaker selection
    python scripts/infer.py \
        --model outputs/final_model \
        --text "Сайн байна уу" \
        --speaker 0 \
        --output output.wav
        
    # Batch synthesis from file
    python scripts/infer.py \
        --model outputs/final_model \
        --input-file texts.txt \
        --output-dir outputs/audio
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchaudio
from rich.console import Console
from rich.progress import track

from src.core.model import F5TTS
from src.data.audio import AudioConfig, AudioProcessor
from src.data.cleaner import MongolianPhonemizer, MongolianTextCleaner

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Synthesize speech with OronTTS")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model or HuggingFace repo ID",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="File with texts to synthesize (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch synthesis",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=0,
        help="Speaker ID for multi-speaker models",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of ODE integration steps",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["euler", "midpoint", "rk4"],
        default="euler",
        help="ODE solver method",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Output sample rate",
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Path to vocoder model (optional, uses Griffin-Lim if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile for faster inference",
    )
    
    return parser.parse_args()


class Synthesizer:
    """Speech synthesizer using OronTTS."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sample_rate: int = 24000,
        vocoder_path: str | None = None,
        compile_model: bool = False,
    ) -> None:
        """Initialize synthesizer.
        
        Args:
            model_path: Path to model checkpoint or HF repo.
            device: Device to run on.
            sample_rate: Audio sample rate.
            vocoder_path: Optional vocoder model path.
            compile_model: Use torch.compile.
        """
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        
        # Load model
        console.print(f"[bold blue]Loading model from {model_path}...[/]")
        
        if Path(model_path).exists():
            self.model = F5TTS.from_pretrained(model_path, device=device)
        else:
            # Try loading from HuggingFace Hub
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path)
            self.model = F5TTS.from_pretrained(local_path, device=device)
        
        self.model.eval()
        
        # Compile if requested
        if compile_model:
            console.print("[yellow]Compiling model...[/]")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Initialize text processing
        self.cleaner = MongolianTextCleaner()
        self.phonemizer = MongolianPhonemizer()
        
        # Initialize audio processing
        audio_config = AudioConfig(sample_rate=sample_rate)
        self.audio_processor = AudioProcessor(audio_config)
        
        # Load vocoder if provided
        self.vocoder = None
        if vocoder_path is not None:
            console.print(f"[bold blue]Loading vocoder from {vocoder_path}...[/]")
            self.vocoder = self._load_vocoder(vocoder_path)
        
        console.print("[bold green]Ready for synthesis![/]")
    
    def _load_vocoder(self, path: str) -> torch.nn.Module:
        """Load neural vocoder."""
        # Placeholder for HiFi-GAN or other vocoder loading
        # In practice, load your specific vocoder here
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Return vocoder model
        raise NotImplementedError("Vocoder loading not implemented. Use Griffin-Lim for now.")
    
    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        cfg_scale: float = 2.0,
        num_steps: int = 32,
        method: str = "euler",
    ) -> torch.Tensor:
        """Synthesize speech from text.
        
        Args:
            text: Input text in Mongolian.
            speaker_id: Speaker index.
            cfg_scale: CFG scale.
            num_steps: ODE steps.
            method: ODE solver.
            
        Returns:
            Audio waveform tensor.
        """
        # Clean and phonemize text
        text_clean = self.cleaner(text)
        phonemes_str = self.phonemizer(text_clean)
        phoneme_ids = self.phonemizer.phoneme_to_ids(phonemes_str)
        
        # Prepare inputs
        phonemes = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
        speaker_ids = torch.tensor([speaker_id], dtype=torch.long, device=self.device)
        
        # Generate mel-spectrogram
        with torch.inference_mode():
            mel = self.model.synthesize(
                phonemes=phonemes,
                speaker_ids=speaker_ids,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                method=method,
            )
        
        # Convert to audio
        mel = mel.squeeze(0).cpu()
        audio = self.audio_processor.mel_to_audio(mel, self.vocoder)
        
        return audio
    
    def save_audio(self, audio: torch.Tensor, path: str) -> None:
        """Save audio tensor to file.
        
        Args:
            audio: Audio waveform.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(str(path), audio, self.sample_rate)


def main() -> None:
    """Main inference function."""
    args = parse_args()
    
    # Validate arguments
    if args.text is None and args.input_file is None:
        console.print("[red]Error: Must provide --text or --input-file[/]")
        sys.exit(1)
    
    # Initialize synthesizer
    synth = Synthesizer(
        model_path=args.model,
        device=args.device,
        sample_rate=args.sample_rate,
        vocoder_path=args.vocoder,
        compile_model=args.compile,
    )
    
    # Single text synthesis
    if args.text is not None:
        console.print(f"\n[bold]Text:[/] {args.text}")
        
        start_time = time.time()
        audio = synth.synthesize(
            text=args.text,
            speaker_id=args.speaker,
            cfg_scale=args.cfg_scale,
            num_steps=args.steps,
            method=args.method,
        )
        elapsed = time.time() - start_time
        
        synth.save_audio(audio, args.output)
        
        duration = audio.size(-1) / args.sample_rate
        rtf = elapsed / duration
        
        console.print(f"[green]Saved to {args.output}[/]")
        console.print(f"[dim]Duration: {duration:.2f}s | Generation: {elapsed:.2f}s | RTF: {rtf:.3f}[/]")
    
    # Batch synthesis
    elif args.input_file is not None:
        input_path = Path(args.input_file)
        output_dir = Path(args.output_dir or "outputs/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_path) as f:
            texts = [line.strip() for line in f if line.strip()]
        
        console.print(f"\n[bold]Synthesizing {len(texts)} texts...[/]\n")
        
        total_time = 0.0
        total_duration = 0.0
        
        for i, text in track(enumerate(texts), total=len(texts), description="Synthesizing"):
            start_time = time.time()
            audio = synth.synthesize(
                text=text,
                speaker_id=args.speaker,
                cfg_scale=args.cfg_scale,
                num_steps=args.steps,
                method=args.method,
            )
            elapsed = time.time() - start_time
            
            output_path = output_dir / f"{i:04d}.wav"
            synth.save_audio(audio, output_path)
            
            total_time += elapsed
            total_duration += audio.size(-1) / args.sample_rate
        
        rtf = total_time / total_duration
        console.print(f"\n[green]Batch synthesis complete![/]")
        console.print(f"[dim]Total duration: {total_duration:.1f}s | Total time: {total_time:.1f}s | Avg RTF: {rtf:.3f}[/]")


if __name__ == "__main__":
    main()

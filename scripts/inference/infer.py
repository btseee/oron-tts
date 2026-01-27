#!/usr/bin/env python3
"""OronTTS Inference Script for Mongolian speech synthesis using F5-TTS.

Usage:
    python scripts/inference/infer.py \
        --ref-audio reference.wav \
        --ref-text "Энэ бол жишээ текст" \
        --gen-text "Сайн байна уу" \
        --output output.wav
        
    # With custom model checkpoint
    python scripts/inference/infer.py \
        --model path/to/checkpoint.pt \
        --ref-audio reference.wav \
        --ref-text "Энэ бол жишээ текст" \
        --gen-text "Сайн байна уу" \
        --output output.wav
        
    # Batch synthesis from file
    python scripts/inference/infer.py \
        --ref-audio reference.wav \
        --ref-text "Энэ бол жишээ текст" \
        --input-file texts.txt \
        --output-dir outputs/audio
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root and F5-TTS to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "F5-TTS" / "src"))

import torch
from rich.console import Console
from rich.progress import track

from f5_tts.api import F5TTS
from src.data.cleaner import MongolianTextCleaner

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Synthesize Mongolian speech with OronTTS (F5-TTS)")
    
    parser.add_argument(
        "--model",
        type=str,
        default="F5TTS_v1_Base",
        help="Model name or path to checkpoint (default: F5TTS_v1_Base)",
    )
    parser.add_argument(
        "--ckpt-file",
        type=str,
        default="",
        help="Path to custom checkpoint file",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="",
        help="Path to custom vocabulary file",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        required=True,
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Transcript of reference audio (auto-transcribed if empty)",
    )
    parser.add_argument(
        "--gen-text",
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
        "--cfg-strength",
        type=float,
        default=2.0,
        help="Classifier-free guidance strength",
    )
    parser.add_argument(
        "--nfe-step",
        type=int,
        default=32,
        help="Number of function evaluations (ODE steps)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor for generation",
    )
    parser.add_argument(
        "--ode-method",
        type=str,
        default="euler",
        help="ODE solver method",
    )
    parser.add_argument(
        "--remove-silence",
        action="store_true",
        help="Remove silence from generated audio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (auto-detected if not specified)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable Mongolian text cleaning/normalization",
    )
    
    return parser.parse_args()


class OronTTSSynthesizer:
    """Mongolian speech synthesizer using F5-TTS."""
    
    def __init__(
        self,
        model: str = "F5TTS_v1_Base",
        ckpt_file: str = "",
        vocab_file: str = "",
        ode_method: str = "euler",
        device: str | None = None,
    ) -> None:
        """Initialize synthesizer.
        
        Args:
            model: Model name or config.
            ckpt_file: Path to checkpoint.
            vocab_file: Path to vocabulary.
            ode_method: ODE solver method.
            device: Device to run on.
        """
        console.print(f"[bold blue]Loading F5-TTS model: {model}...[/]")
        
        self.tts = F5TTS(
            model=model,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ode_method=ode_method,
            device=device,
        )
        
        # Initialize Mongolian text cleaner
        self.cleaner = MongolianTextCleaner()
        
        console.print("[bold green]Ready for synthesis![/]")
    
    def synthesize(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        cfg_strength: float = 2.0,
        nfe_step: int = 32,
        speed: float = 1.0,
        remove_silence: bool = False,
        output_path: str | None = None,
        seed: int | None = None,
        clean_text: bool = True,
    ) -> tuple:
        """Synthesize speech from text.
        
        Args:
            ref_audio: Reference audio path.
            ref_text: Reference audio transcript.
            gen_text: Text to synthesize.
            cfg_strength: CFG strength.
            nfe_step: Number of ODE steps.
            speed: Speed factor.
            remove_silence: Remove silence from output.
            output_path: Output file path.
            seed: Random seed.
            clean_text: Apply Mongolian text cleaning.
            
        Returns:
            Tuple of (wav, sample_rate, spectrogram).
        """
        # Clean text if enabled
        if clean_text:
            gen_text = self.cleaner(gen_text)
            if ref_text:
                ref_text = self.cleaner(ref_text)
        
        wav, sr, spec = self.tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            cfg_strength=cfg_strength,
            nfe_step=nfe_step,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=output_path,
            seed=seed,
        )
        
        return wav, sr, spec


def main() -> None:
    """Main inference function."""
    args = parse_args()
    
    # Validate arguments
    if args.gen_text is None and args.input_file is None:
        console.print("[red]Error: Must provide --gen-text or --input-file[/]")
        sys.exit(1)
    
    # Initialize synthesizer
    synth = OronTTSSynthesizer(
        model=args.model,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        ode_method=args.ode_method,
        device=args.device,
    )
    
    clean_text = not args.no_clean
    
    # Single text synthesis
    if args.gen_text is not None:
        console.print(f"\n[bold]Text:[/] {args.gen_text}")
        
        start_time = time.time()
        wav, sr, _ = synth.synthesize(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.gen_text,
            cfg_strength=args.cfg_strength,
            nfe_step=args.nfe_step,
            speed=args.speed,
            remove_silence=args.remove_silence,
            output_path=args.output,
            seed=args.seed,
            clean_text=clean_text,
        )
        elapsed = time.time() - start_time
        
        duration = len(wav) / sr
        rtf = elapsed / duration if duration > 0 else 0
        
        console.print(f"[green]Saved to {args.output}[/]")
        console.print(f"[dim]Duration: {duration:.2f}s | Generation: {elapsed:.2f}s | RTF: {rtf:.3f}[/]")
        console.print(f"[dim]Seed: {synth.tts.seed}[/]")
    
    # Batch synthesis
    elif args.input_file is not None:
        input_path = Path(args.input_file)
        output_dir = Path(args.output_dir or "outputs/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        console.print(f"\n[bold]Synthesizing {len(texts)} texts...[/]\n")
        
        total_time = 0.0
        total_duration = 0.0
        
        for i, text in track(enumerate(texts), total=len(texts), description="Synthesizing"):
            output_path = output_dir / f"{i:04d}.wav"
            
            start_time = time.time()
            wav, sr, _ = synth.synthesize(
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                gen_text=text,
                cfg_strength=args.cfg_strength,
                nfe_step=args.nfe_step,
                speed=args.speed,
                remove_silence=args.remove_silence,
                output_path=str(output_path),
                seed=args.seed,
                clean_text=clean_text,
            )
            elapsed = time.time() - start_time
            
            total_time += elapsed
            total_duration += len(wav) / sr
        
        rtf = total_time / total_duration if total_duration > 0 else 0
        console.print(f"\n[green]Batch synthesis complete![/]")
        console.print(f"[dim]Total duration: {total_duration:.1f}s | Total time: {total_time:.1f}s | Avg RTF: {rtf:.3f}[/]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""OronTTS Inference Script for Mongolian speech synthesis.

Usage:
    # Generate example with trained model
    python scripts/inference/infer.py

    # Custom text
    python scripts/inference/infer.py \
        --text "Сайн байна уу, монгол хэлээр ярьж байна"

    # With specific checkpoint
    python scripts/inference/infer.py \
        --checkpoint /workspace/output/ckpts/mongolian-tts/model_50000.pt \
        --text "Энэ бол жишээ текст" \
        --output custom_output.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ["HF_AUDIO_DECODER"] = "soundfile"

# Add project root and F5-TTS to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "F5-TTS" / "src"))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from datasets import load_dataset  # noqa: E402
from f5_tts.infer.utils_infer import (  # noqa: E402
    infer_process,
    load_checkpoint,
    load_vocoder,
)
from f5_tts.model import CFM, DiT  # noqa: E402
from rich.console import Console  # noqa: E402

from src.data.cleaner import MongolianTextCleaner  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate Mongolian speech with trained OronTTS")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: finds latest in /workspace/output)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="btsee/mongolian-tts-combined",
        help="Dataset used for training (for vocabulary)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Сайн байна уу! Би монгол хэлээр ярьж байна. 2025 онд Улаанбаатар хотод амьдардаг.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output/example_mongolian.wav",
        help="Output audio file",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio (optional, uses F5-TTS example if not provided)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="Some call me nature, others call me mother nature.",
        help="Reference audio text",
    )

    return parser.parse_args()


def find_latest_checkpoint(base_dir: str = "/workspace/output/ckpts") -> Path | None:
    """Find latest checkpoint."""
    ckpt_dir = Path(base_dir) / "mongolian-tts"
    if not ckpt_dir.exists():
        return None

    checkpoints = list(ckpt_dir.glob("model_*.pt"))
    if not checkpoints:
        # Try safetensors
        checkpoints = list(ckpt_dir.glob("model_*.safetensors"))

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]


def load_model(checkpoint_path: Path, dataset_name: str, device: str = "cuda") -> Any:
    """Load trained model with correct vocabulary."""
    console.print(f"[blue]📥 Loading vocabulary from dataset: {dataset_name}[/]")

    # Build vocabulary from dataset
    hf_dataset = load_dataset(dataset_name, split="train")
    all_texts = hf_dataset["text"]
    all_text = " ".join(all_texts)
    vocab_chars = sorted(set(all_text))
    vocab_char_map = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(vocab_char_map)

    console.print(f"[green]✓ Vocabulary: {vocab_size} characters[/]")

    # Mel spec config (must match training)
    mel_spec_kwargs = {
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 100,
        "target_sample_rate": 24000,
        "mel_spec_type": "vocos",
    }

    # Model config (F5TTS_v1_Base)
    model_cfg = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    }

    console.print("[blue]🏗️  Creating model architecture...[/]")
    transformer = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    console.print(f"[blue]📂 Loading checkpoint: {checkpoint_path.name}[/]")
    model = load_checkpoint(model, str(checkpoint_path), device, use_ema=True)

    return model


def main() -> int:
    args = parse_args()

    console.print("\n[bold blue]═" * 35)
    console.print("[bold blue]  OronTTS - Mongolian Speech Synthesis")
    console.print("[bold blue]═" * 35 + "\n")

    # Find checkpoint
    ckpt_path: Path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path_optional = find_latest_checkpoint()
        if not ckpt_path_optional:
            console.print("[red]❌ No checkpoint found in /workspace/output/ckpts/[/]")
            console.print("[yellow]Please specify --checkpoint or train a model first[/]")
            return 1
        ckpt_path = ckpt_path_optional

    if not ckpt_path.exists():
        console.print(f"[red]❌ Checkpoint not found: {ckpt_path}[/]")
        return 1

    console.print(f"[green]✓ Using checkpoint: {ckpt_path}[/]")

    # Clean text
    cleaner = MongolianTextCleaner()
    original_text = args.text
    cleaned_text = cleaner(original_text)

    console.print(f"\n[bold]Original:[/] {original_text}")
    console.print(f"[bold]Cleaned:[/]  {cleaned_text}\n")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Device: {device.upper()}[/]")

    # Load model
    model = load_model(ckpt_path, args.dataset, device)

    # Load vocoder
    console.print("[blue]🎵 Loading vocoder (Vocos)...[/]")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, local_path="")

    # Reference audio
    if args.ref_audio:
        ref_audio = args.ref_audio
    else:
        # Use F5-TTS example
        ref_audio = str(
            PROJECT_ROOT
            / "third_party"
            / "F5-TTS"
            / "src"
            / "f5_tts"
            / "infer"
            / "examples"
            / "basic"
            / "basic_ref_en.wav"
        )

    console.print(f"[dim]Reference: {Path(ref_audio).name}[/]\n")

    # Generate
    console.print("[bold green]🎤 Generating speech...[/]")

    wav, sr, _ = infer_process(
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        gen_text=cleaned_text,
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type="vocos",
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        speed=1.0,
        device=device,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), torch.tensor(wav).unsqueeze(0), sr)

    duration = len(wav) / sr

    console.print("\n[bold green]✅ Success![/]")
    console.print(f"[green]Saved to: {output_path}[/]")
    console.print(f"[dim]Duration: {duration:.2f}s | Sample rate: {sr} Hz[/]")
    console.print("\n[bold blue]═" * 35 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

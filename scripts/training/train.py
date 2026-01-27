#!/usr/bin/env python3
"""Optimized OronTTS Training for Mongolian Khalkha TTS.

Designed for maximum quality with combined dataset:
- Finetunes from pretrained F5-TTS (essential for small datasets)
- Optimized hyperparameters for low loss
- Multi-speaker male/female voices
- Output to /workspace/output

Usage:
    python scripts/training/train.py
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_AUDIO_DECODER"] = "soundfile"

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "F5-TTS" / "src"))

from datasets import load_dataset as hf_load_dataset  # noqa: E402
from f5_tts.model import CFM, DiT, Trainer  # noqa: E402
from f5_tts.model.dataset import HFDataset  # noqa: E402
from rich.console import Console  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OronTTS with optimal settings")

    parser.add_argument(
        "--dataset",
        type=str,
        default="/workspace/output/data/mongolian-tts-combined",
        help="Dataset path or name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/workspace/output", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Training epochs (more needed for from-scratch)"
    )
    parser.add_argument("--batch-size", type=int, default=2400, help="Batch size in frames")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7.5e-5,
        help="Learning rate (higher for from-scratch)",
    )
    parser.add_argument(
        "--save-every", type=int, default=2000, help="Save checkpoint every N updates"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print("\n[bold blue]═" * 35)
    console.print("[bold blue]  OronTTS Training - Mongolian Khalkha")
    console.print("[bold blue]═" * 35)
    console.print("  Strategy:    Train from scratch (vocab mismatch)")
    console.print(f"  Dataset:     {args.dataset}")
    console.print("  Samples:     ~4,900 combined (mbspeech + common-voice)")
    console.print("  Target:      Low loss with 150+ epochs")
    console.print(f"  Output:      {args.output_dir}")
    console.print("[bold blue]═" * 35 + "\n")

    # Setup paths
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "ckpts" / "mongolian-tts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Model configuration - F5TTS v1 Base
    model_cls = DiT
    model_cfg = {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "ff_mult": 2,
        "text_dim": 512,
        "conv_layers": 4,
    }

    # NOTE: We train from scratch due to vocabulary mismatch
    # Pretrained F5-TTS has 2546 vocab (English), Mongolian has ~74 chars
    # Loading pretrained weights for non-text layers would help, but
    # F5-TTS trainer doesn't support partial loading yet
    console.print("[yellow]⚠️  Training from scratch (vocab mismatch with pretrained)[/]")
    console.print("[dim]   Pretrained: 2546 chars | Mongolian: will be ~74 chars[/]")
    console.print("[dim]   This will take longer but is necessary for proper learning[/]\n")

    # Load dataset
    console.print(f"\n[blue]📂 Loading dataset: {args.dataset}[/]")

    # Check if local path or HuggingFace dataset
    if Path(args.dataset).exists():
        from datasets import load_from_disk

        hf_dataset = load_from_disk(args.dataset)
    else:
        hf_dataset = hf_load_dataset(args.dataset, split="train")

    console.print(f"[green]✓ Loaded {len(hf_dataset):,} samples[/]")

    # Build vocabulary
    console.print("[blue]🔤 Building vocabulary...[/]")
    all_texts = hf_dataset["text"]
    all_text = " ".join(all_texts)
    vocab_chars = sorted(set(all_text))
    vocab_char_map = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(vocab_char_map)
    console.print(f"[green]✓ Vocabulary: {vocab_size} characters[/]")
    console.print(f"[dim]   Sample chars: {''.join(list(vocab_chars)[:50])}...[/]")

    # Mel spectrogram config
    mel_spec_kwargs = {
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 100,
        "target_sample_rate": 24000,
        "mel_spec_type": "vocos",
    }

    # Wrap dataset
    train_dataset = HFDataset(
        hf_dataset,
        target_sample_rate=24000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    )

    # Create model
    console.print("[blue]🏗️  Creating model...[/]")
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Optimal training configuration (FROM SCRATCH)
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=1000,  # Warmup for from-scratch
        save_per_updates=args.save_every,
        last_per_updates=500,
        keep_last_n_checkpoints=10,
        checkpoint_path=str(ckpt_dir),
        batch_size_per_gpu=args.batch_size,
        batch_size_type="frame",
        max_samples=48,
        grad_accumulation_steps=2,  # Effective batch = 4800 frames
        max_grad_norm=1.0,  # Standard for from-scratch
        logger="tensorboard",
        wandb_project="oron-tts",
        wandb_run_name=f"mongolian-scratch-{args.learning_rate}",
        log_samples=True,
        bnb_optimizer=False,
    )

    console.print("\n[bold green]🚀 Starting training from scratch...[/]")
    console.print("[dim]Settings:[/]")
    console.print("[dim]  - Mode: From scratch (due to vocab mismatch)[/]")
    console.print(f"[dim]  - Learning Rate: {args.learning_rate}[/]")
    console.print(f"[dim]  - Batch Size: {args.batch_size} frames × 2 accum = 4800 effective[/]")
    console.print(f"[dim]  - Epochs: {args.epochs} (more needed for from-scratch)[/]")
    console.print("[dim]  - Warmup: 1000 updates[/]")
    console.print(f"[dim]  - Checkpoints: {ckpt_dir}[/]")
    console.print(f"[dim]  - TensorBoard: {output_dir}/runs[/]\n")

    # Train
    trainer.train(
        train_dataset,
        num_workers=4,
        resumable_with_seed=42,
    )

    console.print("\n[bold green]✅ Training complete![/]")
    console.print(f"[green]Model saved to: {ckpt_dir}[/]")


if __name__ == "__main__":
    main()

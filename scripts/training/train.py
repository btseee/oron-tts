#!/usr/bin/env python3
"""OronTTS Training Script - F5-TTS training for Mongolian.

Optimized for RTX 4090 (24GB VRAM) with HuggingFace dataset.

Usage:
    # Train from scratch on HuggingFace dataset (RTX 4090)
    python scripts/training/train.py \
        --dataset-name btsee/common-voices-24-mn \
        --epochs 500 \
        --batch-size 1800

    # Finetune from pretrained F5-TTS
    python scripts/training/train.py \
        --dataset-name btsee/common-voices-24-mn \
        --finetune \
        --epochs 100 \
        --batch-size 1800
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root and F5-TTS to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "F5-TTS" / "src"))

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OronTTS (F5-TTS for Mongolian)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="btsee/common-voices-24-mn",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )

    # Model
    parser.add_argument(
        "--exp-name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Base model architecture",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Finetune from pretrained checkpoint",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Path to pretrained checkpoint",
    )

    # Training - RTX 4090 (24GB) optimized defaults
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=7.5e-5, help="Learning rate (higher for from-scratch)")
    parser.add_argument("--batch-size", type=int, default=1800, help="Batch size in frames (1800 for RTX 4090 24GB)")
    parser.add_argument("--max-samples", type=int, default=32, help="Max samples per batch")
    parser.add_argument("--grad-accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup-updates", type=int, default=1000, help="Warmup updates")

    # Checkpointing
    parser.add_argument("--save-per-updates", type=int, default=5000, help="Save checkpoint every N updates")
    parser.add_argument("--last-per-updates", type=int, default=500, help="Save last checkpoint every N updates")
    parser.add_argument("--keep-checkpoints", type=int, default=5, help="Number of checkpoints to keep")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint directory")

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["pinyin", "char", "custom"],
        help="Tokenizer type (char for Mongolian Cyrillic)",
    )
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Custom tokenizer vocab path")

    # Logging
    parser.add_argument("--logger", type=str, default="tensorboard", choices=[None, "wandb", "tensorboard"])
    parser.add_argument("--log-samples", action="store_true", help="Log audio samples")

    # Optimization
    parser.add_argument("--bnb-optimizer", action="store_true", help="Use 8-bit Adam (saves memory)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    console.print("\n[bold blue]═══════════════════════════════════════════════════════════[/]")
    console.print("[bold blue]                    OronTTS Training                        [/]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/]")
    console.print(f"  Dataset:     {args.dataset_name}")
    console.print(f"  Model:       {args.exp_name}")
    console.print(f"  Mode:        {'Finetune' if args.finetune else 'From Scratch'}")
    console.print(f"  Batch Size:  {args.batch_size} frames")
    console.print(f"  LR:          {args.learning_rate}")
    console.print(f"  Epochs:      {args.epochs}")
    console.print("[bold blue]═══════════════════════════════════════════════════════════[/]\n")

    # Import F5-TTS components
    from datasets import load_dataset as hf_load_dataset
    from cached_path import cached_path

    from f5_tts.model import CFM, DiT, Trainer, UNetT
    from f5_tts.model.dataset import HFDataset
    from f5_tts.model.utils import get_tokenizer

    # Setup checkpoint path
    dataset_short_name = args.dataset_name.split("/")[-1]
    checkpoint_path = args.checkpoint_path or str(PROJECT_ROOT / "ckpts" / dataset_short_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    console.print(f"[dim]Checkpoints: {checkpoint_path}[/]")

    # Model configuration
    if args.exp_name == "F5TTS_v1_Base":
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        default_ckpt = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"

    elif args.exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        default_ckpt = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"

    elif args.exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        default_ckpt = "hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"

    # Handle finetuning
    if args.finetune:
        console.print("[yellow]Finetuning mode - loading pretrained checkpoint...[/]")
        ckpt_path = args.pretrain or str(cached_path(default_ckpt))
        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        dest_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(dest_checkpoint):
            console.print(f"[dim]Copying checkpoint to {dest_checkpoint}[/]")
            shutil.copy2(ckpt_path, dest_checkpoint)
        # Lower learning rate for finetuning
        if args.learning_rate == 7.5e-5:
            args.learning_rate = 1e-5
            console.print(f"[dim]Adjusted LR for finetuning: {args.learning_rate}[/]")

    # Load HuggingFace dataset
    console.print(f"\n[blue]Loading dataset: {args.dataset_name}[/]")
    hf_dataset = hf_load_dataset(args.dataset_name, split=args.dataset_split)
    console.print(f"[green]✓ Loaded {len(hf_dataset):,} samples[/]")

    # Mel spectrogram configuration
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )

    # Build vocabulary from dataset text (access column directly to avoid audio decoding)
    console.print("[blue]Building vocabulary from dataset...[/]")
    all_texts = hf_dataset["text"]  # Direct column access, no audio decoding
    all_text = " ".join(all_texts)
    vocab_chars = sorted(set(all_text))
    vocab_char_map = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(vocab_char_map)
    console.print(f"[green]✓ Vocabulary size: {vocab_size} characters[/]")

    # Wrap in F5-TTS HFDataset
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
    console.print("[blue]Creating model...[/]")
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Create trainer
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=args.warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project="oron-tts",
        wandb_run_name=f"{args.exp_name}_{dataset_short_name}",
        wandb_resume_id=None,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
    )

    # Start training
    console.print("\n[bold green]Starting training...[/]")
    console.print("[dim]Press Ctrl+C to stop. Training will resume from last checkpoint.[/]\n")

    trainer.train(
        train_dataset,
        num_workers=args.num_workers,
        resumable_with_seed=42,
    )


if __name__ == "__main__":
    main()

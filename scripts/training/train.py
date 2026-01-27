#!/usr/bin/env python3
"""OronTTS Training Script - Wrapper for F5-TTS training with Mongolian support.

Usage:
    # Finetune from pretrained F5-TTS
    python scripts/training/train.py \
        --dataset-name oron_mn \
        --finetune \
        --epochs 100 \
        --batch-size 3200

    # Train from scratch
    python scripts/training/train.py \
        --dataset-name oron_mn \
        --epochs 500 \
        --batch-size 1600
"""

from __future__ import annotations

import argparse
import os
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
        required=True,
        help="Name of the dataset (should match prepared data directory)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset directory (default: data/{dataset-name})",
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
        help="Path to pretrained checkpoint (uses default if not specified)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=3200, help="Batch size per GPU (in frames)")
    parser.add_argument(
        "--batch-size-type",
        type=str,
        default="frame",
        choices=["frame", "sample"],
        help="Batch size type",
    )
    parser.add_argument("--max-samples", type=int, default=64, help="Max samples per batch")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup-updates", type=int, default=2000, help="Warmup updates")

    # Checkpointing
    parser.add_argument("--save-per-updates", type=int, default=10000, help="Save checkpoint every N updates")
    parser.add_argument("--last-per-updates", type=int, default=1000, help="Save last checkpoint every N updates")
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to keep (-1 for all, 0 for none)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint save directory (default: ckpts/{dataset-name})",
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["pinyin", "char", "custom"],
        help="Tokenizer type (char recommended for Mongolian)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file",
    )

    # Logging
    parser.add_argument(
        "--logger",
        type=str,
        default=None,
        choices=[None, "wandb", "tensorboard"],
        help="Logger for training metrics",
    )
    parser.add_argument("--wandb-project", type=str, default="oron-tts", help="W&B project name")
    parser.add_argument("--log-samples", action="store_true", help="Log audio samples during training")

    # Optimization
    parser.add_argument("--bnb-optimizer", action="store_true", help="Use 8-bit Adam optimizer")

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    console.print("[bold blue]OronTTS Training[/]")
    console.print(f"Dataset: {args.dataset_name}")
    console.print(f"Model: {args.exp_name}")
    console.print(f"Finetune: {args.finetune}")

    # Import F5-TTS components
    import shutil

    from cached_path import cached_path

    from f5_tts.model import CFM, DiT, Trainer, UNetT
    from f5_tts.model.dataset import load_dataset
    from f5_tts.model.utils import get_tokenizer

    # Setup paths
    checkpoint_path = args.checkpoint_path or str(PROJECT_ROOT / "ckpts" / args.dataset_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Model configuration based on experiment name
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

    # Handle finetuning checkpoint
    if args.finetune:
        ckpt_path = args.pretrain or str(cached_path(default_ckpt))
        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            console.print(f"[yellow]Copying pretrained checkpoint to {file_checkpoint}[/]")
            shutil.copy2(ckpt_path, file_checkpoint)

    # Tokenizer setup
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected but no tokenizer_path provided")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    console.print(f"Vocabulary size: {vocab_size}")

    # Mel spectrogram configuration
    mel_spec_type = "vocos"
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type=mel_spec_type,
    )

    # Create model
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
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{args.exp_name}_{args.dataset_name}",
        wandb_resume_id=None,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
    )

    # Load dataset
    console.print(f"[blue]Loading dataset: {args.dataset_name}[/]")
    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    # Start training
    console.print("[bold green]Starting training...[/]")
    trainer.train(
        train_dataset,
        resumable_with_seed=42,
    )


if __name__ == "__main__":
    main()

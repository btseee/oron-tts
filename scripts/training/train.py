#!/usr/bin/env python3
"""OronTTS Training Script with HuggingFace Accelerate.

Usage:
    # Single GPU
    python scripts/train.py --config configs/light.yaml
    
    # Multi-GPU with accelerate
    accelerate launch scripts/train.py --config configs/hq.yaml
    
    # Resume training
    python scripts/train.py --config configs/hq.yaml --resume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.core.model import F5TTS, F5TTSConfig
from src.data.audio import AudioConfig
from src.data.dataset import OronDataCollator, OronDataset
from src.utils.checkpoint import AccelerateCheckpointManager
from src.utils.hub import HubConfig, HubManager
from src.utils.logging import TrainingLogger, get_progress_bar


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train OronTTS")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (overrides config)",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="HuggingFace Hub repo for checkpoint sync",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> F5TTS:
    """Build F5-TTS model from config."""
    model_cfg = config["model"]
    
    f5_config = F5TTSConfig(
        mel_dim=model_cfg.get("mel_dim", 100),
        phoneme_vocab_size=model_cfg.get("phoneme_vocab_size", 256),
        dim=model_cfg.get("dim", 1024),
        depth=model_cfg.get("depth", 22),
        num_heads=model_cfg.get("num_heads", 16),
        ff_mult=model_cfg.get("ff_mult", 4.0),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=model_cfg.get("max_seq_len", 4096),
        num_speakers=model_cfg.get("num_speakers", 1),
        speaker_dim=model_cfg.get("speaker_dim", 256),
        sigma_min=model_cfg.get("sigma_min", 1e-4),
        use_flash_attn=model_cfg.get("use_flash_attn", True),
    )
    
    return F5TTS(f5_config)


def build_dataloader(
    config: dict,
    split: str = "train",
    hf_dataset: str | None = None,
) -> DataLoader:
    """Build data loader from config."""
    data_cfg = config["data"]
    audio_cfg = config["audio"]
    train_cfg = config["training"]
    
    audio_config = AudioConfig(
        sample_rate=audio_cfg.get("sample_rate", 24000),
        n_fft=audio_cfg.get("n_fft", 1024),
        hop_length=audio_cfg.get("hop_length", 256),
        win_length=audio_cfg.get("win_length", 1024),
        n_mels=audio_cfg.get("n_mels", 100),
        denoise=audio_cfg.get("denoise", True),
    )
    
    dataset = OronDataset(
        manifest_path=data_cfg.get(f"{split}_manifest"),
        hf_dataset_name=hf_dataset or data_cfg.get("hf_dataset"),
        hf_split=split,
        audio_config=audio_config,
        max_duration_sec=data_cfg.get("max_duration_sec", 30.0),
        min_duration_sec=data_cfg.get("min_duration_sec", 0.5),
    )
    
    collator = OronDataCollator()
    
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=collator,
        drop_last=(split == "train"),
    )


def build_optimizer(model: F5TTS, config: dict) -> AdamW:
    """Build AdamW optimizer from config."""
    train_cfg = config["training"]
    
    return AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        eps=train_cfg.get("eps", 1e-8),
    )


def build_scheduler(optimizer: AdamW, config: dict):
    """Build learning rate scheduler from config."""
    train_cfg = config["training"]
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    max_steps = train_cfg.get("max_steps", 100000)
    
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=1e-7,
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def train_step(
    model: F5TTS,
    batch,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Perform a single training step."""
    mel = batch.mel
    phonemes = batch.phonemes
    speaker_ids = batch.speaker_ids
    mask = batch.mask
    
    # Forward pass
    losses = model.compute_loss(
        mel=mel,
        phonemes=phonemes,
        speaker_ids=speaker_ids,
        mask=mask,
    )
    
    loss = losses["loss"]
    
    # Backward pass
    accelerator.backward(loss)
    
    return {"loss": loss.item()}


def main() -> None:
    """Main training loop."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    train_cfg = config["training"]
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
    )
    
    # Setup logging
    output_dir = Path(args.output_dir)
    logger = TrainingLogger(
        log_dir=output_dir / "logs",
    )
    
    if accelerator.is_main_process:
        logger.log_hyperparameters(config)
    
    # Build components
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    
    # Initialize Hub Manager
    hub_repo_id = args.hub_repo or config.get("hub", {}).get("repo_id")
    hub_manager = None
    if hub_repo_id and accelerator.is_main_process:
        hub_manager = HubManager(HubConfig(repo_id=hub_repo_id))
    scheduler = build_scheduler(optimizer, config)
    train_loader = build_dataloader(config, "train", args.hf_dataset)
    
    if accelerator.is_main_process:
        logger.log_model_summary(model.num_parameters)
    
    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    # Compile model if requested
    if config["model"].get("compile_model", False):
        model = torch.compile(model, mode=config.get("advanced", {}).get("compile_mode", "default"))
    
    # Checkpoint management
    ckpt_manager = AccelerateCheckpointManager(
        output_dir=output_dir / "checkpoints",
        max_checkpoints=train_cfg.get("max_checkpoints", 5),
    )
    
    # Hub sync is initialized earlier
    
    # Resume if requested
    global_step = 0
    if args.resume:
        global_step = ckpt_manager.load(accelerator)
        if accelerator.is_main_process:
            logger.logger.info(f"Resumed from step {global_step}")
    
    # Training loop
    max_steps = train_cfg.get("max_steps", 100000)
    save_steps = train_cfg.get("save_steps", 5000)
    log_steps = train_cfg.get("log_steps", 100)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    
    model.train()
    
    with get_progress_bar() as progress:
        task = progress.add_task("Training", total=max_steps)
        progress.update(task, advance=global_step)
        
        while global_step < max_steps:
            for batch in train_loader:
                with accelerator.accumulate(model):
                    metrics = train_step(model, batch, accelerator)
                    
                    # Gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(task, advance=1)
                    
                    # Logging
                    if global_step % log_steps == 0:
                        metrics["lr"] = scheduler.get_last_lr()[0]
                        if accelerator.is_main_process:
                            logger.log_metrics(metrics, global_step)
                    
                    # Checkpointing
                    if global_step % save_steps == 0:
                        ckpt_manager.save(accelerator, global_step, metrics)
                        
                        # Sync to Hub
                        if hub_manager is not None and accelerator.is_main_process:
                            hub_manager.upload_training_state(
                                output_dir,
                                commit_message=f"Step {global_step}",
                            )
                    
                    if global_step >= max_steps:
                        break
    
    # Final save
    if accelerator.is_main_process:
        ckpt_manager.save(accelerator, global_step, metrics)
        
        # Save final model
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir / "final_model")
        
        if hub_manager is not None:
            hub_manager.upload_training_state(output_dir, "Final model")
            hub_manager.upload_model_card(
                model_name=config["model"].get("name", "OronTTS"),
                description="Mongolian TTS model trained with F5-TTS architecture.",
            )
        
        logger.finish()


if __name__ == "__main__":
    main()

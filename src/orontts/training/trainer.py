"""Training script and utilities."""

import os
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from orontts.dataset.loader import TTSDataModule
from orontts.model.config import VITS2Config, load_config
from orontts.training.callbacks import (
    AudioLoggingCallback,
    CheckpointCallback,
    LearningRateMonitor,
)
from orontts.training.lightning_module import VITS2LightningModule


def create_trainer(
    config: VITS2Config,
    output_dir: Path,
    max_epochs: int | None = None,
    devices: int | str = "auto",
    precision: str = "16-mixed",
    accumulate_grad_batches: int = 1,
    val_check_interval: float | int = 1.0,
    log_every_n_steps: int = 50,
    gradient_clip_val: float | None = None,
    enable_progress_bar: bool = True,
    resume_from: str | Path | None = None,
) -> L.Trainer:
    """Create a configured Lightning Trainer.

    Args:
        config: VITS2 configuration.
        output_dir: Directory for outputs (checkpoints, logs).
        max_epochs: Maximum training epochs.
        devices: GPU devices to use.
        precision: Training precision.
        accumulate_grad_batches: Gradient accumulation steps.
        val_check_interval: Validation check interval.
        log_every_n_steps: Logging frequency.
        gradient_clip_val: Gradient clipping value.
        enable_progress_bar: Whether to show progress bar.
        resume_from: Checkpoint path to resume from.

    Returns:
        Configured Lightning Trainer.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_epochs = max_epochs or config.training.epochs
    gradient_clip_val = gradient_clip_val or config.training.grad_clip

    # Callbacks
    callbacks = [
        CheckpointCallback(
            dirpath=output_dir / "checkpoints",
            monitor="val/loss_mel",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        AudioLoggingCallback(
            log_every_n_steps=1000,
            sample_rate=config.audio.sample_rate,
        ),
    ]

    if enable_progress_bar:
        callbacks.append(RichProgressBar())

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
        version=config.model_name,
    )

    # Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        default_root_dir=str(output_dir),
    )

    return trainer


def train(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    batch_size: int | None = None,
    num_workers: int = 4,
    max_epochs: int | None = None,
    devices: int | str = "auto",
    precision: str = "16-mixed",
    resume_from: str | Path | None = None,
) -> None:
    """Train VITS2 model.

    Args:
        config_path: Path to configuration JSON file.
        data_dir: Path to dataset directory.
        output_dir: Path for outputs.
        batch_size: Override batch size from config.
        num_workers: Data loading workers.
        max_epochs: Override max epochs from config.
        devices: GPU devices to use.
        precision: Training precision.
        resume_from: Checkpoint to resume from.
    """
    # Load configuration
    config = load_config(config_path)

    # Override batch size if provided
    if batch_size is not None:
        config.training.batch_size = batch_size

    # Create data module
    data_module = TTSDataModule(
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        num_workers=num_workers,
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        n_mels=config.audio.n_mels,
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        mel_fmin=config.audio.mel_fmin,
        mel_fmax=config.audio.mel_fmax,
    )

    # Create model
    model = VITS2LightningModule(config)

    # Create trainer
    trainer = create_trainer(
        config=config,
        output_dir=Path(output_dir),
        max_epochs=max_epochs,
        devices=devices,
        precision=precision,
        resume_from=resume_from,
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=str(resume_from) if resume_from else None,
    )

    # Save final model
    final_path = Path(output_dir) / "checkpoints" / "final.ckpt"
    trainer.save_checkpoint(final_path)

    # Save config
    config.save(Path(output_dir) / "config.json")


def resume_training(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path | None = None,
    max_epochs: int | None = None,
    devices: int | str = "auto",
) -> None:
    """Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        data_dir: Path to dataset directory.
        output_dir: Path for outputs. Uses checkpoint directory if None.
        max_epochs: Additional epochs to train.
        devices: GPU devices to use.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = output_dir or checkpoint_path.parent.parent

    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = VITS2Config.model_validate(checkpoint["config"])

    # Resume training
    train(
        config_path=output_dir / "config.json",
        data_dir=data_dir,
        output_dir=output_dir,
        max_epochs=max_epochs,
        devices=devices,
        resume_from=checkpoint_path,
    )


# Allow import of torch in this module
import torch

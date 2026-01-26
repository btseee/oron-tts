"""Checkpoint management for training resumption and model saving."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class TrainingState:
    """Training state for checkpointing."""

    epoch: int
    global_step: int
    best_loss: float
    config: dict[str, Any]


class CheckpointManager:
    """Manager for saving and loading training checkpoints.

    Features:
    - Automatic checkpoint rotation (keep last N)
    - Best model tracking
    - Full training state preservation
    - HuggingFace Hub integration
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory for checkpoints.
            max_checkpoints: Maximum checkpoints to keep.
            save_optimizer: Include optimizer state in checkpoints.
            save_scheduler: Include scheduler state in checkpoints.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self.checkpoints: list[Path] = []
        self._load_checkpoint_list()

    def _load_checkpoint_list(self) -> None:
        """Load existing checkpoint list from disk."""
        checkpoints_file = self.output_dir / "checkpoints.json"
        if checkpoints_file.exists():
            with open(checkpoints_file) as f:
                data = json.load(f)
                self.checkpoints = [Path(p) for p in data.get("checkpoints", [])]

    def _save_checkpoint_list(self) -> None:
        """Save checkpoint list to disk."""
        checkpoints_file = self.output_dir / "checkpoints.json"
        with open(checkpoints_file, "w") as f:
            json.dump(
                {"checkpoints": [str(p) for p in self.checkpoints]},
                f,
                indent=2,
            )

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        state: TrainingState | None = None,
        is_best: bool = False,
        prefix: str = "checkpoint",
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Learning rate scheduler to save.
            state: Training state metadata.
            is_best: Mark as best checkpoint.
            prefix: Checkpoint filename prefix.

        Returns:
            Path to saved checkpoint.
        """
        step = state.global_step if state else 0
        checkpoint_name = f"{prefix}_step{step:08d}.pt"
        checkpoint_path = self.output_dir / checkpoint_name

        # Build checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }

        if self.save_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if self.save_scheduler and scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if state is not None:
            checkpoint["training_state"] = asdict(state)

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self.checkpoints.append(checkpoint_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        # Save best model separately
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        self._save_checkpoint_list()

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: _LRScheduler | None = None,
        checkpoint_path: str | Path | None = None,
        load_best: bool = False,
        strict: bool = True,
        device: str | torch.device = "cpu",
    ) -> TrainingState | None:
        """Load a checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to restore state.
            scheduler: Scheduler to restore state.
            checkpoint_path: Specific checkpoint to load.
            load_best: Load best model instead of latest.
            strict: Strict state dict loading.
            device: Device to load checkpoint to.

        Returns:
            Training state if available.
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.output_dir / "best_model.pt"
            elif self.checkpoints:
                checkpoint_path = self.checkpoints[-1]
            else:
                return None

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return training state
        if "training_state" in checkpoint:
            return TrainingState(**checkpoint["training_state"])

        return None

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to latest checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1]
        return None

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.checkpoints) > 0 or (self.output_dir / "best_model.pt").exists()


class AccelerateCheckpointManager:
    """Checkpoint manager compatible with HuggingFace Accelerate.

    Uses Accelerate's save/load state for distributed training.
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_checkpoints: int = 5,
    ) -> None:
        """Initialize accelerate checkpoint manager.

        Args:
            output_dir: Directory for checkpoints.
            max_checkpoints: Maximum checkpoints to keep.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(
        self,
        accelerator,
        global_step: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save checkpoint using Accelerate.

        Args:
            accelerator: HuggingFace Accelerator instance.
            global_step: Current training step.
            metrics: Optional metrics to save.

        Returns:
            Path to checkpoint directory.
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{global_step}"

        # Save accelerate state
        accelerator.save_state(str(checkpoint_dir))

        # Save metrics
        if metrics is not None:
            metrics_file = checkpoint_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        # Rotate checkpoints
        self._rotate_checkpoints()

        return checkpoint_dir

    def load(self, accelerator, checkpoint_dir: str | Path | None = None) -> int:
        """Load checkpoint using Accelerate.

        Args:
            accelerator: HuggingFace Accelerator instance.
            checkpoint_dir: Specific checkpoint to load.

        Returns:
            Training step from checkpoint.
        """
        if checkpoint_dir is None:
            checkpoint_dir = self._get_latest_checkpoint()

        if checkpoint_dir is None:
            return 0

        checkpoint_dir = Path(checkpoint_dir)
        accelerator.load_state(str(checkpoint_dir))

        # Extract step from directory name
        step = int(checkpoint_dir.name.split("-")[-1])
        return step

    def _get_latest_checkpoint(self) -> Path | None:
        """Get latest checkpoint directory."""
        checkpoints = list(self.output_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
        return checkpoints[-1]

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoint directories."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )

        while len(checkpoints) > self.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            if old_checkpoint.exists():
                import shutil

                shutil.rmtree(old_checkpoint)

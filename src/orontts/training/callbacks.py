"""Training utilities and callbacks for VITS2."""

from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor as LRMonitor


class CheckpointCallback(ModelCheckpoint):
    """Enhanced checkpoint callback with model export."""

    def __init__(
        self,
        dirpath: str | Path,
        filename: str = "vits2-{epoch:04d}-{val/loss_mel:.4f}",
        monitor: str = "val/loss_mel",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dirpath=str(dirpath),
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            **kwargs,
        )

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Add extra metadata to checkpoint."""
        checkpoint["training_step"] = trainer.global_step
        checkpoint["epoch"] = trainer.current_epoch


class AudioLoggingCallback(Callback):
    """Callback for logging audio samples during training."""

    def __init__(
        self,
        log_every_n_steps: int = 1000,
        num_samples: int = 4,
        sample_rate: int = 22050,
    ) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self.sample_rate = sample_rate

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Log audio samples periodically."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        if trainer.logger is None:
            return

        # Generate samples
        pl_module.eval()
        with torch.no_grad():
            phoneme_ids = batch["phoneme_ids"][: self.num_samples]
            phoneme_lengths = batch["phoneme_lengths"][: self.num_samples]
            speaker_ids = batch.get("speaker_ids")
            if speaker_ids is not None:
                speaker_ids = speaker_ids[: self.num_samples]

            generated = pl_module(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                speaker_ids=speaker_ids,
            )

            # Log to tensorboard if available
            if hasattr(trainer.logger, "experiment"):
                for i in range(min(self.num_samples, generated.shape[0])):
                    audio = generated[i].squeeze().cpu()
                    trainer.logger.experiment.add_audio(
                        f"generated/sample_{i}",
                        audio,
                        trainer.global_step,
                        sample_rate=self.sample_rate,
                    )

        pl_module.train()


class LearningRateMonitor(LRMonitor):
    """Learning rate monitor with custom logging."""

    def __init__(self, logging_interval: str = "step") -> None:
        super().__init__(logging_interval=logging_interval)


class GradientStatsCallback(Callback):
    """Callback for monitoring gradient statistics."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient statistics."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = {}

        # Compute gradient norms for key modules
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                # Group by module
                module_name = name.split(".")[0]
                if module_name not in grad_norms:
                    grad_norms[module_name] = []
                grad_norms[module_name].append(grad_norm)

        # Log average gradient norm per module
        for module_name, norms in grad_norms.items():
            avg_norm = sum(norms) / len(norms)
            pl_module.log(
                f"gradients/{module_name}_norm",
                avg_norm,
                on_step=True,
                logger=True,
            )


class EMACallback(Callback):
    """Exponential Moving Average callback for smoother inference."""

    def __init__(self, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema_params: dict[str, torch.Tensor] = {}

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            for name, param in pl_module.generator.named_parameters():
                if name not in self.ema_params:
                    self.ema_params[name] = param.data.clone()
                else:
                    self.ema_params[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def apply_ema(self, module: torch.nn.Module) -> None:
        """Apply EMA parameters to module for inference."""
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.ema_params:
                    param.data.copy_(self.ema_params[name])

    def save_ema_checkpoint(self, path: Path, pl_module: L.LightningModule) -> None:
        """Save EMA checkpoint."""
        state_dict = {}
        for name, param in pl_module.generator.named_parameters():
            if name in self.ema_params:
                state_dict[name] = self.ema_params[name]
            else:
                state_dict[name] = param.data

        torch.save({"state_dict": state_dict}, path)

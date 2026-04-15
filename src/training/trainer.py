"""F5-TTS Trainer with EMA, gradient clipping, and Weights & Biases logging."""

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from src.models.f5tts import F5TTS
from src.utils.checkpoint import CheckpointManager


class F5Trainer:
    """Single-optimizer trainer for F5-TTS flow matching.

    No discriminator, no GAN — just CFM MSE loss + EMA.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: F5TTS,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.model = model.to(device)
        if world_size > 1:
            self.model = DDP(model, device_ids=[rank])  # type: ignore[assignment]

        self.train_loader = train_loader
        self.val_loader = val_loader

        lr = config.get("learning_rate", 1e-4)
        betas = tuple(config.get("betas", [0.9, 0.999]))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=0.01
        )

        warmup_steps = config.get("warmup_steps", 1000)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        self.ema = (
            ExponentialMovingAverage(self.model.parameters(), decay=config.get("ema_decay", 0.9999))
            if self.is_main
            else None
        )

        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_tqdm = config.get("use_tqdm", True)
        self.grad_accum = config.get("grad_accumulation_steps", 1)

        self.global_step = 0
        self.epoch = 0

        if self.is_main:
            self._wandb_run = None
            if _WANDB_AVAILABLE and config.get("wandb_project"):
                self._wandb_run = _wandb.init(  # type: ignore[union-attr]
                    project=config["wandb_project"],
                    name=config.get("wandb_run_name"),
                    config=config,
                    resume="allow",
                    dir=log_dir,
                )
            self.checkpoint_manager: CheckpointManager | None = CheckpointManager(
                checkpoint_dir, model_name="f5tts"
            )
            self.logger = self._setup_logger()
        else:
            self._wandb_run = None
            self.checkpoint_manager = None
            self.logger = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("F5Trainer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        return logger

    def train_step(self, batch: dict[str, torch.Tensor], accum_step: int) -> float | None:
        """Single forward+backward step; returns loss on final accum step, else None."""
        mel = batch["mel"].to(self.device)  # [B, n_mels, T]
        text_ids = batch["text_ids"].to(self.device)  # [B, T]
        mel_lengths = batch["mel_lengths"].to(self.device)  # [B]

        if torch.isnan(mel).any() or torch.isinf(mel).any():
            return None

        self.model.train()
        loss = self.model(mel, text_ids, mel_lengths) / self.grad_accum
        loss.backward()

        if (accum_step + 1) % self.grad_accum == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update()
            return loss.item() * self.grad_accum

        return None

    def train_epoch(self, total_epochs: int) -> float:
        self.optimizer.zero_grad()
        total_loss = 0.0
        n = 0
        accum_step = -1

        disable = not self.is_main or not self.use_tqdm
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch + 1}/{total_epochs}", disable=disable
        )

        for accum_step, batch in enumerate(pbar):
            loss_val = self.train_step(batch, accum_step)

            if loss_val is not None:
                total_loss += loss_val
                n += 1
                self.global_step += 1

                if (
                    self.is_main
                    and self._wandb_run is not None
                    and self.global_step % self.config.get("log_interval", 100) == 0
                ):
                    self._wandb_run.log(
                        {"train/loss": loss_val, "lr": self.optimizer.param_groups[0]["lr"]},
                        step=self.global_step,
                    )

                if (
                    self.is_main
                    and self.logger
                    and not self.use_tqdm
                    and self.global_step % self.config.get("log_interval", 100) == 0
                ):
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(f"Step {self.global_step} | loss={loss_val:.4f} | lr={lr:.2e}")

                if self.is_main and self.use_tqdm:
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Flush any accumulated gradients that didn't complete a full accumulation window
        if accum_step >= 0 and (accum_step + 1) % self.grad_accum != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update()

        self.epoch += 1
        return total_loss / max(n, 1)

    def train(self, num_epochs: int, save_interval: int = 5) -> None:
        for _ in range(num_epochs):
            if self.world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)  # type: ignore

            avg_loss = self.train_epoch(total_epochs=num_epochs)

            if self.is_main:
                msg = f"Epoch {self.epoch}/{num_epochs} | avg_loss={avg_loss:.4f}"
                if self.logger:
                    self.logger.info(msg)
                if self._wandb_run is not None:
                    self._wandb_run.log({"train/epoch_loss": avg_loss}, step=self.global_step)

                if self.epoch % save_interval == 0:
                    self.save_checkpoint()

        self.finish()

    def finish(self) -> None:
        """Flush and close the wandb run if active."""
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                mel = batch["mel"].to(self.device)
                text_ids = batch["text_ids"].to(self.device)
                mel_lengths = batch["mel_lengths"].to(self.device)
                loss = self.model(mel, text_ids, mel_lengths)
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    def save_checkpoint(self, is_best: bool = False) -> Path | None:
        if self.checkpoint_manager is None:
            return None
        raw_model: nn.Module = (
            self.model.module if isinstance(self.model, DDP) else self.model  # type: ignore[union-attr]
        )
        ema_state = None
        if self.ema:
            with self.ema.average_parameters():
                ema_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}

        return self.checkpoint_manager.save(
            step=self.global_step,
            model=raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_state=ema_state,
            config=self.config,
            is_best=is_best,
        )

    def load_checkpoint(self, path: str | Path | None = None, load_best: bool = False) -> None:
        if self.checkpoint_manager is None:
            raise ValueError("CheckpointManager not initialized")
        raw_model: nn.Module = self.model.module if isinstance(self.model, DDP) else self.model
        info = self.checkpoint_manager.load(
            model=raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            path=path,
            load_best=load_best,
            device=self.device,
        )
        self.global_step = info["step"]

    def push_to_hub(self, repo_id: str, token: str | None = None) -> str:
        if self.checkpoint_manager is None:
            raise ValueError("CheckpointManager not initialized")
        return self.checkpoint_manager.push_to_hub(repo_id, token=token)

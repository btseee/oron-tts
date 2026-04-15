"""F5-TTS Trainer with AMP, EMA, gradient clipping, and console logging."""

import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from src.models.f5tts import F5TTS
from src.utils.checkpoint import CheckpointManager


def _detect_amp_dtype(device: str) -> torch.dtype | None:
    """Pick the best AMP dtype for the current GPU, or None to disable."""
    if device == "cpu" or not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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

        # AMP (automatic mixed precision)
        self.amp_dtype = _detect_amp_dtype(device)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.amp_dtype == torch.float16
            else None
        )

        self.global_step = 0
        self.epoch = 0
        self._best_val: float = float("inf")

        if self.is_main:
            self.checkpoint_manager: CheckpointManager | None = CheckpointManager(
                checkpoint_dir, model_name="f5tts"
            )
            self.logger = self._setup_logger()
        else:
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
        mel = batch["mel"].to(self.device, non_blocking=True)  # [B, n_mels, T]
        text_ids = batch["text_ids"].to(self.device, non_blocking=True)  # [B, T]
        mel_lengths = batch["mel_lengths"].to(self.device, non_blocking=True)  # [B]

        if torch.isnan(mel).any() or torch.isinf(mel).any():
            return None

        self.model.train()

        with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            loss = self.model(mel, text_ids, mel_lengths) / self.grad_accum

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (accum_step + 1) % self.grad_accum == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
        epoch_start = time.monotonic()

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

                log_interval = self.config.get("log_interval", 100)

                if self.is_main and self.global_step % log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]

                    if self.logger and not self.use_tqdm:
                        vram_str = ""
                        if torch.cuda.is_available():
                            vram_gb = torch.cuda.max_memory_allocated() / 1e9
                            vram_str = f" | vram={vram_gb:.1f}GB"
                        self.logger.info(
                            f"Step {self.global_step} | loss={loss_val:.4f} | lr={lr:.2e}{vram_str}"
                        )

                if self.is_main and self.use_tqdm:
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

        # Flush any accumulated gradients that didn't complete a full accumulation window
        if accum_step >= 0 and (accum_step + 1) % self.grad_accum != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update()

        self.epoch += 1
        epoch_time = time.monotonic() - epoch_start

        if self.is_main and self.logger:
            samples = len(self.train_loader.dataset)  # type: ignore[arg-type]
            throughput = samples / epoch_time if epoch_time > 0 else 0.0
            self.logger.info(
                f"  ↳ epoch {self.epoch}: {epoch_time:.1f}s | "
                f"{throughput:.0f} samples/s | avg_loss={total_loss / max(n, 1):.4f}"
            )

        return total_loss / max(n, 1)

    def train(self, num_epochs: int, save_interval: int = 5) -> None:
        if self.is_main and self.logger:
            dtype_str = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "float32"
            self.logger.info(
                f"Training: epochs {self.epoch}\u2192{num_epochs}, AMP={dtype_str}, "
                f"grad_accum={self.grad_accum}, device={self.device}"
            )

        start_epoch = self.epoch
        train_start = time.monotonic()

        for _ in range(self.epoch, num_epochs):
            if self.world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)  # type: ignore

            avg_loss = self.train_epoch(total_epochs=num_epochs)

            if self.is_main:
                # Validate with EMA weights if available
                val_loss = self._validate_with_ema()
                if val_loss > 0:
                    is_best = val_loss < self._best_val
                    if is_best:
                        self._best_val = val_loss
                else:
                    is_best = False

                elapsed = time.monotonic() - train_start
                epochs_done = self.epoch - start_epoch
                if epochs_done > 0:
                    remaining = elapsed / epochs_done * (num_epochs - self.epoch)
                else:
                    remaining = 0.0
                eta_h, eta_m = divmod(int(remaining), 3600)
                eta_m = eta_m // 60

                val_str = f" | val_loss={val_loss:.4f}" if val_loss > 0 else ""
                if self.logger:
                    self.logger.info(
                        f"Epoch {self.epoch}/{num_epochs} | "
                        f"avg_loss={avg_loss:.4f}{val_str} | "
                        f"ETA={eta_h}h{eta_m:02d}m"
                    )

                if self.epoch % save_interval == 0:
                    self.save_checkpoint(is_best=is_best)

        self.finish()

    def finish(self) -> None:
        """Final cleanup hook (placeholder for future use)."""
        pass

    def _validate_with_ema(self) -> float:
        """Run validation using EMA weights if available, else raw weights."""
        if self.val_loader is None:
            return 0.0
        if self.ema:
            with self.ema.average_parameters():
                return self.validate()
        return self.validate()

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                mel = batch["mel"].to(self.device, non_blocking=True)
                text_ids = batch["text_ids"].to(self.device, non_blocking=True)
                mel_lengths = batch["mel_lengths"].to(self.device, non_blocking=True)
                with torch.amp.autocast(
                    self.device, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
                ):
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
            extra_state={"epoch": self.epoch, "best_val": self._best_val},
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
        self.epoch = info.get("epoch", 0)
        self._best_val = info.get("best_val", float("inf"))

    def push_to_hub(self, repo_id: str, token: str | None = None) -> str:
        if self.checkpoint_manager is None:
            raise ValueError("CheckpointManager not initialized")
        return self.checkpoint_manager.push_to_hub(repo_id, token=token)

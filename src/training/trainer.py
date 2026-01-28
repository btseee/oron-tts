"""VITS Trainer with multi-GPU support and HF Hub integration."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.discriminator import MultiPeriodDiscriminator
from src.models.vits import VITS
from src.training.losses import discriminator_loss, feature_loss, generator_loss, kl_loss, mel_loss
from src.utils.audio import AudioProcessor
from src.utils.checkpoint import CheckpointManager


class VITSTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        model: VITS,
        discriminator: MultiPeriodDiscriminator,
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
        self.discriminator = discriminator.to(device)

        if world_size > 1:
            self.model = DDP(model, device_ids=[rank])
            self.discriminator = DDP(discriminator, device_ids=[rank])

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.audio_processor = AudioProcessor(
            sample_rate=config.get("sample_rate", 22050),
            n_fft=config.get("n_fft", 1024),
            hop_length=config.get("hop_length", 256),
            win_length=config.get("win_length", 1024),
            n_mels=config.get("n_mels", 80),
        )

        self._setup_optimizers()
        self._setup_schedulers()

        self.scaler = GradScaler("cuda", enabled=config.get("fp16", True))

        if self.is_main:
            self.writer = SummaryWriter(log_dir)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.writer = None
            self.checkpoint_manager = None

        self.global_step = 0
        self.epoch = 0
        
        # Setup logging for container logs (RunPod)
        self.use_tqdm = config.get("use_tqdm", True)
        if self.is_main:
            self.logger = self._setup_logger()
        else:
            self.logger = None

    def _setup_optimizers(self) -> None:
        lr = self.config.get("learning_rate", 2e-4)
        betas = tuple(self.config.get("betas", [0.8, 0.99]))
        eps = self.config.get("eps", 1e-9)

        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
        )
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
        )

    def _setup_schedulers(self) -> None:
        gamma = self.config.get("lr_decay", 0.999875)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_g, gamma=gamma
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer_d, gamma=gamma
        )
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("OronTTS")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler for container logs
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Format with timestamp for RunPod logs
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        self.discriminator.train()

        x = batch["text_ids"].to(self.device)
        x_lengths = batch["text_lengths"].to(self.device)
        y = batch["specs"].to(self.device)
        y_lengths = batch["spec_lengths"].to(self.device)
        wav = batch["audios"].to(self.device)
        sid = batch["speaker_ids"].to(self.device)

        with autocast("cuda", enabled=self.config.get("fp16", True)):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                (z, z_p, m_p, logs_p),
                (m_q, logs_q, y_mask, z_slice),
            ) = self.model(x, x_lengths, y, y_lengths, sid)

            y_mel = self._slice_segments(
                y, ids_slice, self.config.get("segment_size", 32)
            )
            y_hat_mel = self.audio_processor.mel_spectrogram(y_hat.squeeze(1))

            # Slice real audio waveforms to match generated audio
            hop_length = self.config.get("hop_length", 256)
            segment_size = self.config.get("segment_size", 32)
            wav_segment_length = segment_size * hop_length
            wav_slice = self._slice_audio_segments(wav, ids_slice, wav_segment_length, hop_length)
            wav_slice = wav_slice.unsqueeze(1)  # Add channel dim: [B, 1, T]

            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(wav_slice, y_hat.detach())

            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        self.optimizer_d.zero_grad()
        self.scaler.scale(loss_disc).backward()
        self.scaler.unscale_(self.optimizer_d)
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer_d)

        with autocast("cuda", enabled=self.config.get("fp16", True)):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(wav_slice, y_hat)

            loss_dur = torch.sum(l_length.float())
            loss_mel = mel_loss(y_mel, y_hat_mel)
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        self.optimizer_g.zero_grad()
        self.scaler.scale(loss_gen_all).backward()
        self.scaler.unscale_(self.optimizer_g)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        return {
            "loss_disc": loss_disc.item(),
            "loss_gen": loss_gen.item(),
            "loss_fm": loss_fm.item(),
            "loss_mel": loss_mel.item(),
            "loss_dur": loss_dur.item(),
            "loss_kl": loss_kl.item(),
            "loss_total": loss_gen_all.item(),
        }

    def _slice_segments(
        self, y: torch.Tensor, ids_slice: torch.Tensor, segment_size: int
    ) -> torch.Tensor:
        b, c, t = y.shape
        ret = torch.zeros(b, c, segment_size, device=y.device, dtype=y.dtype)
        for i in range(b):
            idx_str = ids_slice[i].item()
            idx_end = min(idx_str + segment_size, t)
            actual_len = idx_end - idx_str
            ret[i, :, :actual_len] = y[i, :, idx_str:idx_end]
        return ret

    def _slice_audio_segments(
        self, wav: torch.Tensor, ids_slice: torch.Tensor, segment_length: int, hop_length: int
    ) -> torch.Tensor:
        """Slice audio waveforms to match spectrogram segments."""
        b, t = wav.shape
        ret = torch.zeros(b, segment_length, device=wav.device, dtype=wav.dtype)
        for i in range(b):
            idx_str = ids_slice[i].item() * hop_length
            idx_end = min(idx_str + segment_length, t)
            actual_len = idx_end - idx_str
            if actual_len > 0:
                ret[i, :actual_len] = wav[i, idx_str:idx_end]
        return ret

    def train_epoch(self) -> dict[str, float]:
        epoch_losses = {}
        num_batches = 0
        
        if self.is_main and self.logger:
            self.logger.info(f"Starting Epoch {self.epoch + 1}")

        # Use tqdm if enabled, otherwise plain iterator
        disable_tqdm = not self.is_main or not self.use_tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", disable=disable_tqdm)
        
        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch)

            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value
            num_batches += 1

            if self.is_main and self.global_step % self.config.get("log_interval", 100) == 0:
                self._log_training(losses)
                
                # Log to container logs (RunPod)
                if self.logger and not self.use_tqdm:
                    lr = self.optimizer_g.param_groups[0]['lr']
                    self.logger.info(
                        f"Step {self.global_step} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
                        f"Loss: {losses['loss_total']:.4f} | Mel: {losses['loss_mel']:.4f} | "
                        f"KL: {losses['loss_kl']:.4f} | Dur: {losses['loss_dur']:.4f} | LR: {lr:.6f}"
                    )

            self.global_step += 1

            if self.is_main and self.use_tqdm:
                pbar.set_postfix(
                    loss=f"{losses['loss_total']:.4f}",
                    mel=f"{losses['loss_mel']:.4f}",
                )

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        self.scheduler_g.step()
        self.scheduler_d.step()
        self.epoch += 1

        return epoch_losses

    def train(self, num_epochs: int, save_interval: int = 1) -> None:
        for _ in range(num_epochs):
            if self.world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)  # type: ignore

            epoch_losses = self.train_epoch()

            if self.is_main:
                epoch_summary = (
                    f"Epoch {self.epoch} Complete | "
                    f"Avg Loss: {epoch_losses['loss_total']:.4f} | "
                    f"Mel: {epoch_losses['loss_mel']:.4f} | "
                    f"KL: {epoch_losses['loss_kl']:.4f} | "
                    f"Dur: {epoch_losses['loss_dur']:.4f} | "
                    f"Disc: {epoch_losses['loss_disc']:.4f} | "
                    f"Gen: {epoch_losses['loss_gen']:.4f}"
                )
                if self.logger:
                    self.logger.info(epoch_summary)
                else:
                    print(epoch_summary)

                if self.epoch % save_interval == 0:
                    if self.logger:
                        self.logger.info(f"Saving checkpoint at epoch {self.epoch}")
                    self.save_checkpoint(is_best=False)

                if self.val_loader and self.writer:
                    val_loss = self.validate()
                    self.writer.add_scalar("val/loss", val_loss, self.global_step)

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["text_ids"].to(self.device)
                x_lengths = batch["text_lengths"].to(self.device)
                y = batch["specs"].to(self.device)
                y_lengths = batch["spec_lengths"].to(self.device)
                sid = batch["speaker_ids"].to(self.device)

                (
                    y_hat,
                    l_length,
                    attn,
                    ids_slice,
                    x_mask,
                    (z, z_p, m_p, logs_p),
                    (m_q, logs_q, y_mask, z_slice),
                ) = self.model(x, x_lengths, y, y_lengths, sid)

                y_mel = self._slice_segments(y, ids_slice, self.config.get("segment_size", 32))
                y_hat_mel = self.audio_processor.mel_spectrogram(y_hat.squeeze(1))
                loss = mel_loss(y_mel, y_hat_mel)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _log_training(self, losses: dict[str, float]) -> None:
        if self.writer is None:
            return
        for key, value in losses.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)

        self.writer.add_scalar(
            "lr", self.optimizer_g.param_groups[0]["lr"], self.global_step
        )

    def save_checkpoint(self, is_best: bool = False) -> Path | None:
        if self.checkpoint_manager is None:
            return None
        model: nn.Module = self.model.module if hasattr(self.model, "module") else self.model  # type: ignore
        return self.checkpoint_manager.save(
            step=self.global_step,
            model=model,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
            loss=None,
            config=self.config,
            is_best=is_best,
        )

    def load_checkpoint(self, path: str | Path | None = None, load_best: bool = False) -> None:
        if self.checkpoint_manager is None:
            raise ValueError("CheckpointManager not initialized")
        model: nn.Module = self.model.module if hasattr(self.model, "module") else self.model  # type: ignore
        info = self.checkpoint_manager.load(
            model=model,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
            path=path,
            load_best=load_best,
            device=self.device,
        )
        self.global_step = info["step"]

    def push_to_hub(self, repo_id: str, token: str | None = None) -> str:
        if self.checkpoint_manager is None:
            raise ValueError("CheckpointManager not initialized")
        return self.checkpoint_manager.push_to_hub(repo_id, token=token)

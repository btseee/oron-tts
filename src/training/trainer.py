"""VITS Trainer with multi-GPU support and HF Hub integration.

Implements numerically stable training with:
- Discriminator warmup to stabilize early training
- Aggressive gradient clipping and NaN detection
- Proper loss weighting for training from scratch
- FP16 training with careful handling of GAN components
"""

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.discriminator import MultiPeriodDiscriminator
from src.models.vits import VITS
from src.training.losses import (
    discriminator_loss,
    duration_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    mel_loss,
)
from src.utils.audio import AudioProcessor
from src.utils.checkpoint import CheckpointManager


def _check_nan_inf(tensor: torch.Tensor, name: str = "") -> bool:
    """Check if tensor contains NaN or Inf values."""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return bool(has_nan or has_inf)


def _safe_backward(
    loss: torch.Tensor,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: float = 1.0,
    clip_value: float | None = None,
) -> bool:
    """Perform backward pass with comprehensive NaN checking and gradient clipping.

    Returns True if backward was successful, False if NaN/Inf detected.
    """
    if _check_nan_inf(loss, "loss"):
        optimizer.zero_grad()
        return False

    # Scale and backward
    scaler.scale(loss).backward()

    # Unscale for gradient clipping
    scaler.unscale_(optimizer)

    # Check for NaN gradients before clipping
    for param in model.parameters():
        if param.grad is not None:
            if _check_nan_inf(param.grad, "gradient"):
                optimizer.zero_grad()
                return False
            # Optional: clip individual gradient values
            if clip_value is not None:
                param.grad.data.clamp_(-clip_value, clip_value)

    # Gradient norm clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    return True


class VITSTrainer:
    """VITS Trainer with stability improvements for training from scratch."""

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

        # Initialize models with weight initialization
        self.model = model.to(device)
        self.discriminator = discriminator.to(device)

        # Apply weight initialization for stable training from scratch
        self._init_weights(self.model)
        self._init_weights(self.discriminator)

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
        
        # Use separate scalers for generator and discriminator (more stable)
        use_amp = config.get("fp16", False)  # Default to FP32 for stability
        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.scaler_d = GradScaler("cuda", enabled=use_amp)

        from torch_ema import ExponentialMovingAverage

        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=0.9999
        ) if self.is_main else None

        if self.is_main:
            self.writer = SummaryWriter(log_dir)
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.writer = None
            self.checkpoint_manager = None

        self.global_step = 0
        self.epoch = 0

        # Training stability settings
        self.disc_warmup_steps = config.get("disc_warmup_steps", 1000)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.grad_clip_value = config.get("grad_clip_value", 5.0)

        # Loss weights for balanced training
        self.loss_weights = {
            "mel": config.get("loss_weight_mel", 1.0),
            "kl": config.get("loss_weight_kl", 1.0),
            "dur": config.get("loss_weight_dur", 1.0),
            "fm": config.get("loss_weight_fm", 1.0),
            "gen": config.get("loss_weight_gen", 1.0),
        }

        # NaN recovery tracking
        self.nan_count = 0
        self.max_nan_tolerance = 50  # Skip training if too many NaN batches

        # Setup logging for container logs (RunPod)
        self.use_tqdm = config.get("use_tqdm", True)
        if self.is_main:
            self.logger = self._setup_logger()
        else:
            self.logger = None

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for stable training from scratch.

        Uses Xavier/Glorot initialization for linear layers and
        orthogonal initialization for RNNs, which helps prevent
        gradient explosion in deep networks.
        """
        for name, param in module.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:
                    # Xavier initialization for matrices
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif len(param.shape) == 1:
                    # Small uniform for 1D weights
                    nn.init.uniform_(param, -0.1, 0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _setup_optimizers(self) -> None:
        # Lower learning rate for stable training from scratch
        lr = self.config.get("learning_rate", 1e-4)  # Reduced from 2e-4
        betas = tuple(self.config.get("betas", [0.8, 0.99]))
        eps = self.config.get("eps", 1e-8)  # Slightly larger epsilon

        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.01,  # Add weight decay for regularization
        )
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
        )

    def _setup_schedulers(self) -> None:
        gamma = self.config.get("lr_decay", 0.999)

        warmup_epochs = 2
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return gamma ** (epoch - warmup_epochs)

        self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_g, lr_lambda=lr_lambda, last_epoch=-1
        )
        self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_d, lr_lambda=lr_lambda, last_epoch=-1
        )
        
        self._scheduler_initialized = True

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
        """Single training step with comprehensive stability measures.

        Implements:
        - Discriminator warmup (skip disc training for first N steps)
        - Pre-forward NaN checks on input
        - Individual loss NaN handling
        - Gradient value clipping in addition to norm clipping
        - Proper loss weighting
        """
        self.model.train()
        self.discriminator.train()
        
        # Track whether optimizers actually stepped (for scheduler synchronization)
        optimizer_g_stepped = False
        optimizer_d_stepped = False

        # Zero result for NaN recovery
        zero_losses = {
            "loss_disc": 0.0, "loss_gen": 0.0, "loss_fm": 0.0,
            "loss_mel": 0.0, "loss_dur": 0.0, "loss_kl": 0.0, "loss_total": 0.0,
            "_optimizer_g_stepped": False,
            "_optimizer_d_stepped": False,
        }

        # Move batch to device with NaN checking
        x = batch["text_ids"].to(self.device)
        x_lengths = batch["text_lengths"].to(self.device)
        y = batch["specs"].to(self.device)
        y_lengths = batch["spec_lengths"].to(self.device)
        wav = batch["audios"].to(self.device)
        sid = batch["speaker_ids"].to(self.device)

        # Check input spectrograms for NaN/Inf
        if _check_nan_inf(y, "input_spec") or _check_nan_inf(wav, "input_wav"):
            if self.is_main and self.logger:
                self.logger.warning(f"Step {self.global_step}: NaN in input, skipping batch")
            self.nan_count += 1
            return zero_losses

        use_amp = self.config.get("fp16", False)

        # ========== FORWARD PASS ==========
        try:
            with autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
                (
                    y_hat,
                    l_length,
                    attn,
                    ids_slice,
                    x_mask,
                    (z, z_p, m_p, logs_p),
                    (m_q, logs_q, y_mask, z_slice),
                ) = self.model(x, x_lengths, y, y_lengths, sid)

            # Check model outputs for NaN
            if _check_nan_inf(y_hat, "y_hat") or _check_nan_inf(z_p, "z_p"):
                if self.is_main and self.logger:
                    self.logger.warning(f"Step {self.global_step}: NaN in model output, skipping")
                self.nan_count += 1
                return zero_losses

            # Compute mel spectrograms for loss
            y_mel = self._slice_segments(
                y, ids_slice, self.config.get("segment_size", 32)
            )
            y_hat_mel = self.audio_processor.mel_spectrogram(y_hat.squeeze(1).float())

            # Slice real audio waveforms to match generated audio
            hop_length = self.config.get("hop_length", 256)
            segment_size = self.config.get("segment_size", 32)
            wav_segment_length = segment_size * hop_length
            wav_slice = self._slice_audio_segments(wav, ids_slice, wav_segment_length, hop_length)
            wav_slice = wav_slice.unsqueeze(1)  # Add channel dim: [B, 1, T]

        except RuntimeError as e:
            if self.is_main and self.logger:
                self.logger.error(f"Step {self.global_step}: Runtime error in forward: {e}")
            self.nan_count += 1
            return zero_losses

        # ========== DISCRIMINATOR UPDATE ==========
        # Skip discriminator update during warmup phase
        in_warmup = self.global_step < self.disc_warmup_steps

        if not in_warmup:
            self.optimizer_d.zero_grad()

            with autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
                y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(wav_slice, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            if not _check_nan_inf(loss_disc, "loss_disc"):
                success = _safe_backward(
                    loss_disc,
                    self.scaler_d,
                    self.optimizer_d,
                    self.discriminator,
                    max_grad_norm=self.max_grad_norm,
                    clip_value=self.grad_clip_value,
                )
                if success:
                    # GradScaler.step() returns None, but we know it stepped if success=True
                    self.scaler_d.step(self.optimizer_d)
                    optimizer_d_stepped = True
                self.scaler_d.update()
            else:
                loss_disc = torch.zeros(1, device=self.device)
        else:
            loss_disc = torch.zeros(1, device=self.device)

        # ========== GENERATOR UPDATE ==========
        self.optimizer_g.zero_grad()

        with autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
            # Re-run discriminator for generator update (with gradients)
            if not in_warmup:
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(wav_slice, y_hat)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
            else:
                # During warmup, skip adversarial losses
                loss_fm = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
                loss_gen = torch.zeros(1, device=self.device, requires_grad=True).squeeze()

            # Compute reconstruction losses
            loss_mel = mel_loss(y_mel, y_hat_mel)
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
            loss_dur = torch.mean(l_length) if l_length.numel() > 1 else l_length.squeeze()

            # Check individual losses for NaN and replace with zeros
            if _check_nan_inf(loss_mel, "loss_mel"):
                loss_mel = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            if _check_nan_inf(loss_kl, "loss_kl"):
                loss_kl = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            if _check_nan_inf(loss_dur, "loss_dur"):
                loss_dur = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            if _check_nan_inf(loss_fm, "loss_fm"):
                loss_fm = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            if _check_nan_inf(loss_gen, "loss_gen"):
                loss_gen = torch.zeros(1, device=self.device, requires_grad=True).squeeze()

            # Weighted loss combination
            w = self.loss_weights
            loss_gen_all = (
                w["gen"] * loss_gen +
                w["fm"] * loss_fm +
                w["mel"] * loss_mel +
                w["dur"] * loss_dur +
                w["kl"] * loss_kl
            )

        # Check total loss
        if _check_nan_inf(loss_gen_all, "loss_total"):
            if self.is_main and self.logger:
                self.logger.warning(f"Step {self.global_step}: NaN in total loss, skipping")
            self.nan_count += 1
            self.optimizer_g.zero_grad()
            return zero_losses

        # Backward with safety checks
        success = _safe_backward(
            loss_gen_all,
            self.scaler,
            self.optimizer_g,
            self.model,
            max_grad_norm=self.max_grad_norm,
            clip_value=self.grad_clip_value,
        )
        
        if not in_warmup and self.global_step % 2 == 0:
            self.optimizer_g.zero_grad()
            
            with autocast("cuda", enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
                # Re-run for second gen update
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(wav_slice, y_hat)
                loss_fm_2 = feature_loss(fmap_r, fmap_g)
                loss_gen_2, _ = generator_loss(y_d_hat_g)
                
                # Only adversarial losses, skip mel/kl/dur
                w = self.loss_weights
                loss_gen_extra = w["gen"] * loss_gen_2 + w["fm"] * loss_fm_2
            
            if not _check_nan_inf(loss_gen_extra, "loss_gen_extra"):
                _safe_backward(
                    loss_gen_extra,
                    self.scaler,
                    self.optimizer_g,
                    self.model,
                    max_grad_norm=self.max_grad_norm,
                    clip_value=self.grad_clip_value,
                )
                self.scaler.step(self.optimizer_g)
            self.scaler.update()
                
        if success:
            if self.ema:
                self.ema.update()
            self.scaler.step(self.optimizer_g)            
            optimizer_g_stepped = True
        else:
            if self.is_main and self.logger:
                self.logger.warning(f"Step {self.global_step}: NaN gradient, skipping update")
            self.nan_count += 1

        self.scaler.update()

        # Reset NaN count on successful step
        if success:
            self.nan_count = max(0, self.nan_count - 1)

        return {
            "loss_disc": loss_disc.item() if torch.is_tensor(loss_disc) else loss_disc,
            "loss_gen": loss_gen.item() if torch.is_tensor(loss_gen) else loss_gen,
            "loss_fm": loss_fm.item() if torch.is_tensor(loss_fm) else loss_fm,
            "loss_mel": loss_mel.item() if torch.is_tensor(loss_mel) else loss_mel,
            "loss_dur": loss_dur.item() if torch.is_tensor(loss_dur) else loss_dur,
            "loss_kl": loss_kl.item() if torch.is_tensor(loss_kl) else loss_kl,
            "loss_total": loss_gen_all.item() if torch.is_tensor(loss_gen_all) else loss_gen_all,
            "_optimizer_g_stepped": optimizer_g_stepped,
            "_optimizer_d_stepped": optimizer_d_stepped,
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

    def train_epoch(self, total_epochs: int) -> dict[str, float]:
        epoch_losses = {}
        num_batches = 0
        
        # Track optimizer steps for proper scheduler synchronization
        epoch_optimizer_g_stepped = False
        epoch_optimizer_d_stepped = False

        if self.is_main and self.logger:
            self.logger.info(f"Starting Epoch {self.epoch + 1}/{total_epochs}")

        # Use tqdm if enabled, otherwise plain iterator
        disable_tqdm = not self.is_main or not self.use_tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{total_epochs}", disable=disable_tqdm)

        for batch_idx, batch in enumerate(pbar):
            losses = self.train_step(batch)
            
            # Track if optimizers stepped during this epoch
            if losses.get("_optimizer_g_stepped", False):
                epoch_optimizer_g_stepped = True
            if losses.get("_optimizer_d_stepped", False):
                epoch_optimizer_d_stepped = True

            for key, value in losses.items():
                if not key.startswith("_"):  # Skip internal tracking keys
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

        self.epoch += 1

        if epoch_optimizer_g_stepped:
            self.scheduler_g.step()
        if epoch_optimizer_d_stepped:
            self.scheduler_d.step()

        return epoch_losses

    def train(self, num_epochs: int, save_interval: int = 1) -> None:
        for _ in range(num_epochs):
            if self.world_size > 1 and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)  # type: ignore

            epoch_losses = self.train_epoch(total_epochs=num_epochs)

            if self.is_main:
                epoch_summary = (
                    f"Epoch {self.epoch}/{num_epochs} Complete | "
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

        if self.ema:
            with self.ema.average_parameters():
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
        else:
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

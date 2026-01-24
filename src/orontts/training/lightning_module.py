"""Lightning module for VITS2 training."""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from orontts.model.config import VITS2Config
from orontts.model.vits2 import VITS2, VITS2Discriminator
from orontts.training.losses import (
    MelSpectrogramLoss,
    discriminator_loss,
    duration_loss,
    feature_matching_loss,
    generator_loss,
    kl_divergence_loss,
)


class VITS2LightningModule(L.LightningModule):
    """PyTorch Lightning module for VITS2 training.

    Handles the complete training loop including generator and discriminator
    optimization with proper scheduling.
    """

    def __init__(
        self,
        config: VITS2Config,
        learning_rate: float | None = None,
    ) -> None:
        """Initialize the Lightning module.

        Args:
            config: VITS2 configuration.
            learning_rate: Override learning rate from config.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["config"])

        self.config = config
        self.lr = learning_rate or config.training.learning_rate

        # Models
        self.generator = VITS2(config)
        self.discriminator = VITS2Discriminator(config)

        # Losses
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mels=config.audio.n_mels,
            f_min=config.audio.mel_fmin,
            f_max=config.audio.mel_fmax,
        )

        # For manual optimization
        self.automatic_optimization = False

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass for inference."""
        return self.generator.infer(
            phoneme_ids,
            phoneme_lengths,
            speaker_ids=speaker_ids,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Training step with alternating generator/discriminator updates."""
        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        # Unpack batch
        phoneme_ids = batch["phoneme_ids"]
        phoneme_lengths = batch["phoneme_lengths"]
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        audio = batch["audio"]
        speaker_ids = batch.get("speaker_ids")

        # Generator forward pass
        outputs = self.generator(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel=mel,
            mel_lengths=mel_lengths,
            speaker_ids=speaker_ids,
        )

        # Slice real audio to match generated
        audio_slice = self._slice_audio(
            audio, outputs["ids_slice"], self.config.training.segment_size
        )

        # ===== Discriminator Update =====
        opt_d.zero_grad()

        # Detach generator output for discriminator update
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
            audio_slice.unsqueeze(1), outputs["audio"].detach()
        )

        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_rs, y_d_gs)

        self.manual_backward(loss_disc)
        self.clip_gradients(opt_d, gradient_clip_val=self.config.training.grad_clip)
        opt_d.step()

        # ===== Generator Update =====
        opt_g.zero_grad()

        # Re-run discriminator with gradients for generator
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
            audio_slice.unsqueeze(1), outputs["audio"]
        )

        # Losses
        loss_gen, losses_gen = generator_loss(y_d_gs)
        loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
        loss_mel = self.mel_loss(outputs["audio"], audio_slice.unsqueeze(1))
        loss_kl = kl_divergence_loss(
            outputs["z_p"],
            outputs["z_logs"],
            outputs["x_m"],
            outputs["x_logs"],
            outputs["z_mask"],
        )
        loss_dur = duration_loss(
            outputs["log_duration_pred"],
            outputs["duration_target"],
            outputs["x_mask"],
        )

        # Total generator loss
        loss_g = (
            loss_gen * self.config.training.adversarial_weight
            + loss_fm * self.config.training.feature_matching_weight
            + loss_mel * self.config.training.mel_weight
            + loss_kl * self.config.training.kl_weight
            + loss_dur * self.config.training.duration_weight
        )

        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=self.config.training.grad_clip)
        opt_g.step()

        # Learning rate scheduling
        if self.trainer.is_last_batch:
            sch_g.step()
            sch_d.step()

        # Logging
        self.log_dict(
            {
                "train/loss_g": loss_g,
                "train/loss_d": loss_disc,
                "train/loss_gen": loss_gen,
                "train/loss_fm": loss_fm,
                "train/loss_mel": loss_mel,
                "train/loss_kl": loss_kl,
                "train/loss_dur": loss_dur,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Validation step."""
        phoneme_ids = batch["phoneme_ids"]
        phoneme_lengths = batch["phoneme_lengths"]
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        audio = batch["audio"]
        speaker_ids = batch.get("speaker_ids")

        # Forward pass
        outputs = self.generator(
            phoneme_ids=phoneme_ids,
            phoneme_lengths=phoneme_lengths,
            mel=mel,
            mel_lengths=mel_lengths,
            speaker_ids=speaker_ids,
        )

        audio_slice = self._slice_audio(
            audio, outputs["ids_slice"], self.config.training.segment_size
        )

        # Compute losses
        loss_mel = self.mel_loss(outputs["audio"], audio_slice.unsqueeze(1))
        loss_kl = kl_divergence_loss(
            outputs["z_p"],
            outputs["z_logs"],
            outputs["x_m"],
            outputs["x_logs"],
            outputs["z_mask"],
        )

        self.log_dict(
            {
                "val/loss_mel": loss_mel,
                "val/loss_kl": loss_kl,
            },
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss_mel": loss_mel, "loss_kl": loss_kl}

    def configure_optimizers(self) -> tuple[list, list]:
        """Configure optimizers and schedulers."""
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.lr,
            betas=self.config.training.betas,
            eps=self.config.training.eps,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=self.config.training.betas,
            eps=self.config.training.eps,
        )

        sch_g = torch.optim.lr_scheduler.ExponentialLR(
            opt_g, gamma=self.config.training.lr_decay
        )
        sch_d = torch.optim.lr_scheduler.ExponentialLR(
            opt_d, gamma=self.config.training.lr_decay
        )

        return [opt_g, opt_d], [sch_g, sch_d]

    def _slice_audio(
        self,
        audio: torch.Tensor,
        ids_start: torch.Tensor,
        segment_size: int,
    ) -> torch.Tensor:
        """Slice audio segments to match generator output."""
        batch_size = audio.shape[0]
        hop_length = self.config.audio.hop_length

        # Convert mel frame indices to audio sample indices
        ids_start_audio = ids_start * hop_length

        segments = torch.zeros(batch_size, segment_size, device=audio.device, dtype=audio.dtype)
        for i in range(batch_size):
            start = ids_start_audio[i].item()
            end = min(start + segment_size, audio.shape[1])
            length = end - start
            segments[i, :length] = audio[i, start:end]

        return segments

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add config to checkpoint."""
        checkpoint["config"] = self.config.model_dump()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load config from checkpoint."""
        if "config" in checkpoint:
            self.config = VITS2Config.model_validate(checkpoint["config"])

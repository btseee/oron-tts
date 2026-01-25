"""Lightning module for VITS2 training."""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from orontts.model.config import VITS2Config
from orontts.model.vits2 import VITS2, VITS2Discriminator
from orontts.utils import slice_segments
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

        # Log audio samples (first batch only)
        if batch_idx == 0 and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_audio"):
            # Run inference on first sample
            with torch.no_grad():
                gen_audio = self.generator.infer(
                    phoneme_ids[:1],
                    phoneme_lengths[:1],
                    speaker_ids=speaker_ids[:1] if speaker_ids is not None else None,
                )
                
                self.logger.experiment.add_audio(
                    "val/audio_gen",
                    gen_audio[0],
                    self.global_step,
                    sample_rate=self.config.audio.sample_rate,
                )
                self.logger.experiment.add_audio(
                    "val/audio_real",
                    audio[:1],
                    self.global_step,
                    sample_rate=self.config.audio.sample_rate,
                )
                
                # Optional: Log Mel Spectrograms
                try:
                    import matplotlib.pyplot as plt
                    
                    # Helper to plot
                    def plot_mel(mel_data):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        im = ax.imshow(mel_data.cpu().numpy(), aspect='auto', origin='lower')
                        fig.colorbar(im)
                        fig.canvas.draw()
                        data = torch.from_numpy(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8))
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plt.close(fig)
                        return data.permute(2, 0, 1)

                    # We need numpy
                    import numpy as np
                    
                    # Generate mel from generated audio for comparison
                    gen_mel = self.mel_loss.mel_spectrogram(gen_audio)
                    
                    self.logger.experiment.add_image(
                        "val/mel_gen",
                        plot_mel(gen_mel[0]),
                        self.global_step,
                    )
                    self.logger.experiment.add_image(
                        "val/mel_real",
                        plot_mel(mel[0]),
                        self.global_step,
                    )
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Failed to log images: {e}")
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
        # Convert mel frame indices to audio sample indices
        ids_start_audio = ids_start * self.config.audio.hop_length
        
        # Audio is [B, T], need [B, 1, T] for slice_segments
        audio_unsqueezed = audio.unsqueeze(1)
        sliced = slice_segments(audio_unsqueezed, ids_start_audio, segment_size)
        
        # Return [B, segment_size]
        return sliced.squeeze(1)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add config to checkpoint."""
        checkpoint["config"] = self.config.model_dump()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load config from checkpoint."""
        if "config" in checkpoint:
            self.config = VITS2Config.model_validate(checkpoint["config"])

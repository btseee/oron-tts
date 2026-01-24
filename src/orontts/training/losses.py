"""Loss functions for VITS2 training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_matching_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps.

    Args:
        fmap_r: Real audio feature maps from discriminator.
        fmap_g: Generated audio feature maps from discriminator.

    Returns:
        Scalar loss tensor.
    """
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        for rl, gl in zip(dr, dg, strict=True):
            loss = loss + F.l1_loss(gl, rl.detach())
    return loss


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Discriminator loss for real and generated audio.

    Args:
        disc_real_outputs: Discriminator outputs for real audio.
        disc_generated_outputs: Discriminator outputs for generated audio.

    Returns:
        Tuple of (total_loss, real_losses, fake_losses).
    """
    loss = 0.0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=True):
        # Real should be classified as 1, fake as 0
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Generator adversarial loss.

    Args:
        disc_outputs: Discriminator outputs for generated audio.

    Returns:
        Tuple of (total_loss, individual_losses).
    """
    loss = 0.0
    gen_losses = []

    for dg in disc_outputs:
        # Generator wants discriminator to classify fake as 1
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses


def kl_divergence_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence loss between posterior and prior.

    Args:
        z_p: Samples from posterior (passed through flow).
        logs_q: Log variance of posterior.
        m_p: Mean of prior.
        logs_p: Log variance of prior.
        z_mask: Mask for valid positions.

    Returns:
        Scalar KL divergence loss.
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    kl = kl / torch.sum(z_mask)

    return kl


def duration_loss(
    log_duration_pred: torch.Tensor,
    duration_target: torch.Tensor,
    x_mask: torch.Tensor,
) -> torch.Tensor:
    """Duration prediction loss.

    Args:
        log_duration_pred: Predicted log duration.
        duration_target: Target duration from alignment.
        x_mask: Text mask.

    Returns:
        Scalar duration loss.
    """
    log_duration_target = torch.log(duration_target.float() + 1e-6)
    loss = F.mse_loss(log_duration_pred.squeeze(1), log_duration_target, reduction="none")
    loss = torch.sum(loss * x_mask.squeeze(1)) / torch.sum(x_mask)
    return loss


class MelSpectrogramLoss(nn.Module):
    """Mel spectrogram reconstruction loss."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.mel_spec = torch.nn.Sequential(
            # Will use torchaudio transforms
        )

        # Store config for later use
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mel spectrogram L1 loss.

        Args:
            y_hat: Generated audio [B, 1, T].
            y: Target audio [B, 1, T].

        Returns:
            Scalar mel loss.
        """
        import torchaudio

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
        ).to(y.device)

        mel_y = mel_transform(y.squeeze(1))
        mel_y_hat = mel_transform(y_hat.squeeze(1))

        mel_y = torch.log(torch.clamp(mel_y, min=1e-5))
        mel_y_hat = torch.log(torch.clamp(mel_y_hat, min=1e-5))

        return F.l1_loss(mel_y_hat, mel_y)

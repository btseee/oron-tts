"""Loss functions for VITS training.

Implements numerically stable loss functions following best practices from:
- Original VITS paper (Kim et al., 2021)
- NaturalSpeech / VITS2 improvements
- Practical stability fixes for training from scratch
"""

import torch
import torch.nn.functional as F


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps.

    Uses float32 accumulation and detaches real features to prevent
    gradients from flowing back through discriminator during generator update.
    """
    device = fmap_r[0][0].device
    loss = torch.zeros(1, device=device, dtype=torch.float32)
    count = 0

    for dr, dg in zip(fmap_r, fmap_g, strict=False):
        for rl, gl in zip(dr, dg, strict=False):
            # Skip if either contains NaN/Inf
            if not torch.isfinite(rl).all() or not torch.isfinite(gl).all():
                continue
            # Detach real features - only train generator to match
            rl_det = rl.float().detach()
            gl_f = gl.float()
            # L1 loss with clamping for stability
            diff = torch.abs(rl_det - gl_f)
            loss = loss + torch.mean(diff.clamp(max=50.0))
            count += 1

    if count == 0:
        return torch.zeros(1, device=device, requires_grad=True).squeeze()

    return (loss / count).squeeze() * 2.0


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Least-squares GAN discriminator loss with numerical stability.

    Uses label smoothing (0.9 instead of 1.0) for real samples
    to improve training stability.
    """
    device = disc_real_outputs[0].device
    loss = torch.zeros(1, device=device, dtype=torch.float32)
    r_losses: list[float] = []
    g_losses: list[float] = []

    real_label = 0.9  # Label smoothing for stability

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs, strict=False):
        dr_f = dr.float()
        dg_f = dg.float()  # Do NOT detach - discriminator needs gradients from both real and fake

        # Clamp outputs to prevent extreme values
        dr_f = torch.clamp(dr_f, min=-10.0, max=10.0)
        dg_f = torch.clamp(dg_f, min=-10.0, max=10.0)

        r_loss = torch.mean((real_label - dr_f) ** 2)
        g_loss = torch.mean(dg_f ** 2)

        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    # Normalize by number of discriminators for consistent loss scaling
    num_discriminators = len(disc_real_outputs)
    return (loss / num_discriminators).squeeze(), r_losses, g_losses


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Least-squares GAN generator loss with numerical stability."""
    device = disc_outputs[0].device
    loss = torch.zeros(1, device=device, dtype=torch.float32)
    gen_losses: list[torch.Tensor] = []

    for dg in disc_outputs:
        dg_f = dg.float()
        # Clamp to prevent extreme gradients
        dg_f = torch.clamp(dg_f, min=-10.0, max=10.0)
        gen_loss = torch.mean((1.0 - dg_f) ** 2)
        gen_losses.append(gen_loss)
        loss = loss + gen_loss

    # Normalize by number of discriminators for consistent loss scaling
    num_discriminators = len(disc_outputs)
    return (loss / num_discriminators).squeeze(), gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence loss with comprehensive numerical stability.

    Implements KL(q||p) where:
    - q is the posterior (from audio encoder)
    - p is the prior (from text encoder)

    Uses log-space computations and aggressive clamping to prevent
    overflow/underflow in exp() operations.
    """
    # Ensure float32 for numerical stability
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # Aggressive clamping of log-variances (prevents exp overflow)
    # log(var) in range [-7, 7] means var in range [~0.001, ~1100]
    logs_p_clamped = torch.clamp(logs_p, min=-7.0, max=7.0)
    logs_q_clamped = torch.clamp(logs_q, min=-7.0, max=7.0)

    # KL divergence: 0.5 * (log(var_p/var_q) + var_q/var_p + (m_p-z_p)^2/var_p - 1)
    # In log space: logs_p - logs_q - 0.5 + 0.5*(z_p-m_p)^2*exp(-2*logs_p)

    # Compute variance ratio term in log space
    log_var_ratio = logs_p_clamped - logs_q_clamped

    # Compute squared difference term with stable exp
    # exp(-2*logs_p) = 1/var_p
    neg_2_logs_p = -2.0 * logs_p_clamped
    neg_2_logs_p = torch.clamp(neg_2_logs_p, min=-14.0, max=14.0)
    inv_var_p = torch.exp(neg_2_logs_p)

    diff_sq = (z_p - m_p) ** 2
    diff_sq = torch.clamp(diff_sq, max=100.0)  # Prevent extreme squared differences

    # Combine KL terms
    kl = log_var_ratio - 0.5 + 0.5 * diff_sq * inv_var_p

    # Clamp individual KL values before summing
    kl = torch.clamp(kl, min=-50.0, max=50.0)

    # Apply mask and compute mean
    kl_masked = kl * z_mask
    mask_sum = torch.sum(z_mask).clamp(min=1.0)
    kl_mean = torch.sum(kl_masked) / mask_sum

    # Final clamp - KL should be non-negative in expectation
    return torch.clamp(kl_mean, min=0.0, max=100.0)


def mel_loss(
    y_mel: torch.Tensor,
    y_g_hat_mel: torch.Tensor,
) -> torch.Tensor:
    """Mel spectrogram reconstruction loss with NaN handling.

    Uses L1 loss with reduced weight (compared to original 45x) for
    better balance with other losses during early training.
    """
    y_mel_f = y_mel.float()
    y_g_hat_mel_f = y_g_hat_mel.float()

    # Check for NaN/Inf and skip if present
    if not torch.isfinite(y_mel_f).all() or not torch.isfinite(y_g_hat_mel_f).all():
        return torch.zeros(1, device=y_mel.device, requires_grad=True).squeeze()

    # L1 loss with reduced weight for small datasets
    # Original VITS uses 45x, reduced to 10x for better KL balance
    loss = F.l1_loss(y_mel_f, y_g_hat_mel_f)
    return loss * 10.0


def duration_loss(
    logw: torch.Tensor,
    logw_gt: torch.Tensor, 
    x_mask: torch.Tensor,
) -> torch.Tensor:
    """Duration prediction loss.
    
    Args:
        logw: Predicted log durations [B, 1, T]
        logw_gt: Ground truth log durations [B, 1, T]
        x_mask: Text mask [B, 1, T]
    """
    if not torch.isfinite(logw).all() or not torch.isfinite(logw_gt).all():
        return torch.zeros(1, device=logw.device, requires_grad=True).squeeze()
    
    loss = torch.sum((logw - logw_gt) ** 2, [1, 2]) / torch.sum(x_mask, [1, 2])
    return torch.mean(loss)

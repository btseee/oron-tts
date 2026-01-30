"""Loss functions for VITS training."""

import torch
import torch.nn.functional as F


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fmap_r[0][0].device, requires_grad=True)
    count = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if torch.isnan(rl).any() or torch.isnan(gl).any():
                continue
            loss = loss + torch.mean(torch.abs(rl - gl).clamp(max=100.0))
            count += 1
    return (loss / max(count, 1)) * 2


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # Clamp log variances for numerical stability
    logs_p = torch.clamp(logs_p, min=-10.0, max=10.0)
    logs_q = torch.clamp(logs_q, min=-10.0, max=10.0)

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.clamp(kl, min=-100.0, max=100.0)  # Prevent extreme values
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask).clamp(min=1.0)
    return torch.clamp(l, min=0.0, max=1000.0)  # Final clamp


def mel_loss(
    y_mel: torch.Tensor,
    y_g_hat_mel: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(y_mel, y_g_hat_mel) * 45

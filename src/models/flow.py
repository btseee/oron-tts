"""Optimal-Transport Conditional Flow Matching for F5-TTS."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFM(nn.Module):
    """Wraps a DiT backbone with OT-CFM training and Euler-ODE inference.

    Training path (Lipman et al. 2022, rectified flow convention):
        x_t = (1 - t) * noise + t * data,   t ~ U[0, 1]
        target velocity  v = data - noise
        loss = MSE(predicted_v, v)     (mean over non-padding positions)

    Inference:
        Start at z_0 = noise, integrate dz/dt = v_θ(z_t, t, cond) from t=0→1
        using simple Euler steps or higher-order ODE solvers.
    """

    def __init__(self, backbone: nn.Module, sigma_min: float = 1e-4) -> None:
        super().__init__()
        self.backbone = backbone
        self.sigma_min = sigma_min

    def forward(
        self,
        x1: torch.Tensor,
        text_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute CFM training loss.

        Args:
            x1: Clean mel spectrograms [B, n_mels, T] (data distribution).
            text_ids: Token IDs zero-padded to length T [B, T].
            mask: Boolean mask [B, T], True = valid mel frame.

        Returns:
            Scalar CFM MSE loss averaged over valid frames.
        """
        B, C, T = x1.shape
        x1_t = x1.transpose(1, 2)  # [B, T, C]

        noise = torch.randn_like(x1_t)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)

        # Interpolate: x_t = (1-t)*noise + t*data
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * noise + t_expand * x1_t

        # Target velocity
        v_target = x1_t - noise  # [B, T, C]

        # Predict velocity via backbone
        v_pred = self.backbone(x_t, t, text_ids, mask)  # [B, T, C]

        if mask is not None:
            m = mask[:, :, None].float()  # [B, T, 1]
            loss = F.mse_loss(v_pred * m, v_target * m, reduction="sum") / (m.sum() * C + 1e-8)
        else:
            loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.inference_mode()
    def sample(
        self,
        text_ids: torch.Tensor,
        n_mels: int,
        target_len: int,
        n_steps: int = 32,
        mask: torch.Tensor | None = None,
        ref_mel: torch.Tensor | None = None,
        ref_len: int = 0,
    ) -> torch.Tensor:
        """Generate mel spectrogram via Euler ODE integration.

        Args:
            text_ids: Token IDs [B, T_total] (ref + target, zero-padded).
            n_mels: Number of mel bins.
            target_len: Number of target frames to generate.
            n_steps: Number of Euler integration steps.
            mask: Valid frame mask [B, T_total].
            ref_mel: Reference clean mel [B, n_mels, T_ref] for voice cloning.
            ref_len: Number of reference frames (T_ref).

        Returns:
            Generated mel [B, n_mels, target_len].
        """
        B = text_ids.shape[0]
        T_total = ref_len + target_len
        device = text_ids.device

        # Initialise: reference is clean, target starts as noise
        x = torch.randn(B, T_total, n_mels, device=device)
        if ref_mel is not None and ref_len > 0:
            x[:, :ref_len, :] = ref_mel.transpose(1, 2)[:, :ref_len, :]

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            t = torch.full((B,), t_val, device=device, dtype=x.dtype)
            v = self.backbone(x, t, text_ids, mask)

            # Only update target frames (not reference region)
            if ref_len > 0:
                x[:, ref_len:, :] = x[:, ref_len:, :] + v[:, ref_len:, :] * dt
            else:
                x = x + v * dt

        # Return target portion only, transposed to [B, n_mels, T]
        return x[:, ref_len:, :].transpose(1, 2)

"""Conditional Flow Matching (CFM) for TTS synthesis.

Implements optimal transport CFM with linear interpolation paths
between noise and mel-spectrogram targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConditionalFlowMatcher(nn.Module):
    """Conditional Flow Matching with optimal transport paths.

    Learns the velocity field v(x_t, t | c) that transports noise
    to mel-spectrograms conditioned on phoneme sequences.
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        ot_ode: bool = True,
    ) -> None:
        """Initialize CFM.

        Args:
            sigma_min: Minimum noise level for numerical stability.
            ot_ode: Use optimal transport ODE (linear paths) vs VP-ODE.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.ot_ode = ot_ode

    def compute_mu_t(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute mean of conditional probability path p_t(x|x_0, x_1).

        For OT-CFM: μ_t = (1 - t) * x_0 + t * x_1

        Args:
            x_0: Source samples (noise). Shape: (B, T, D).
            x_1: Target samples (mel). Shape: (B, T, D).
            t: Time steps in [0, 1]. Shape: (B,) or (B, 1, 1).

        Returns:
            Interpolated samples at time t.
        """
        t = t.view(-1, 1, 1)  # Broadcast over (T, D)
        return (1 - t) * x_0 + t * x_1

    def compute_sigma_t(self, t: Tensor) -> Tensor:
        """Compute std of conditional path (constant for OT-CFM).

        Args:
            t: Time steps. Shape: (B,).

        Returns:
            Standard deviation at time t.
        """
        del t  # Unused for OT-CFM
        return torch.tensor(self.sigma_min)

    def sample_conditional_pt(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
        epsilon: Tensor | None = None,
    ) -> Tensor:
        """Sample from conditional path p_t(x|x_0, x_1).

        Args:
            x_0: Noise samples. Shape: (B, T, D).
            x_1: Target mel-spectrograms. Shape: (B, T, D).
            t: Time steps. Shape: (B,).
            epsilon: Optional pre-sampled noise for x_t perturbation.

        Returns:
            Samples x_t from the conditional path.
        """
        mu_t = self.compute_mu_t(x_0, x_1, t)
        sigma_t = self.compute_sigma_t(t)

        if epsilon is None:
            epsilon = torch.randn_like(mu_t)

        return mu_t + sigma_t * epsilon

    def compute_conditional_ut(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute target velocity u_t(x|x_0, x_1).

        For OT-CFM with linear paths: u_t = x_1 - x_0

        Args:
            x_0: Noise samples. Shape: (B, T, D).
            x_1: Target samples. Shape: (B, T, D).
            t: Time steps. Shape: (B,).

        Returns:
            Target velocity field.
        """
        del t  # Velocity is constant for linear paths
        return x_1 - x_0

    def forward(
        self,
        x_1: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample training data for CFM loss.

        Args:
            x_1: Target mel-spectrograms. Shape: (B, T, D).
            mask: Optional padding mask. Shape: (B, T).

        Returns:
            Tuple of (x_t, t, u_t, x_0) for loss computation.
        """
        batch_size = x_1.size(0)
        device = x_1.device

        # Sample time uniformly in [0, 1]
        t = torch.rand(batch_size, device=device)

        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # Apply mask to noise if provided
        if mask is not None:
            x_0 = x_0 * mask.unsqueeze(-1)

        # Get noisy samples x_t and target velocity u_t
        x_t = self.sample_conditional_pt(x_0, x_1, t)
        u_t = self.compute_conditional_ut(x_0, x_1, t)

        return x_t, t, u_t, x_0


class CFMLoss(nn.Module):
    """Flow matching loss for training."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        v_pred: Tensor,
        u_target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute CFM loss: E[||v_θ(x_t, t) - u_t||²].

        Args:
            v_pred: Predicted velocity from model. Shape: (B, T, D).
            u_target: Target velocity from CFM. Shape: (B, T, D).
            mask: Optional padding mask. Shape: (B, T).

        Returns:
            Scalar loss value.
        """
        loss = (v_pred - u_target).pow(2)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / mask.sum().clamp(min=1)
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()

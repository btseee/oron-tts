"""ODE solvers for sampling from trained flow matching models.

Provides Euler and adaptive RK45 integrators for velocity-based ODEs.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import Tensor
from torchdiffeq import odeint


class ODESolver:
    """ODE solver for flow matching inference.

    Integrates the ODE: dx/dt = v_θ(x_t, t) from t=0 to t=1.
    """

    def __init__(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        method: Literal["euler", "midpoint", "rk4", "dopri5"] = "euler",
        num_steps: int = 32,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
        """Initialize ODE solver.

        Args:
            velocity_fn: Model predicting velocity v(x, t).
            method: Integration method.
            num_steps: Number of steps for fixed-step methods.
            atol: Absolute tolerance for adaptive methods.
            rtol: Relative tolerance for adaptive methods.
        """
        self.velocity_fn = velocity_fn
        self.method = method
        self.num_steps = num_steps
        self.atol = atol
        self.rtol = rtol

    def solve(
        self,
        x_0: Tensor,
        t_span: tuple[float, float] = (0.0, 1.0),
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Solve ODE from noise to mel-spectrogram.

        Args:
            x_0: Initial noise. Shape: (B, T, D).
            t_span: (t_start, t_end) for integration.
            return_trajectory: Return full trajectory.

        Returns:
            Final sample x_1, or (x_1, trajectory) if return_trajectory.
        """
        device = x_0.device
        t_start, t_end = t_span

        if self.method in ("dopri5",):
            # Adaptive solver via torchdiffeq
            return self._solve_adaptive(x_0, t_span, return_trajectory)

        # Fixed-step solvers
        dt = (t_end - t_start) / self.num_steps
        t_points = torch.linspace(t_start, t_end, self.num_steps + 1, device=device)

        trajectory = [x_0] if return_trajectory else None
        x = x_0

        for i in range(self.num_steps):
            t = t_points[i]
            x = self._step(x, t, dt)
            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, torch.stack(trajectory, dim=0)
        return x

    def _step(self, x: Tensor, t: float, dt: float) -> Tensor:
        """Single integration step."""
        device = x.device
        t_tensor = torch.full((x.size(0),), t, device=device)

        match self.method:
            case "euler":
                v = self.velocity_fn(x, t_tensor)
                return x + dt * v

            case "midpoint":
                v1 = self.velocity_fn(x, t_tensor)
                x_mid = x + 0.5 * dt * v1
                t_mid = torch.full((x.size(0),), t + 0.5 * dt, device=device)
                v2 = self.velocity_fn(x_mid, t_mid)
                return x + dt * v2

            case "rk4":
                return self._rk4_step(x, t, dt)

            case _:
                raise ValueError(f"Unknown method: {self.method}")

    def _rk4_step(self, x: Tensor, t: float, dt: float) -> Tensor:
        """Fourth-order Runge-Kutta step."""
        device = x.device

        t0 = torch.full((x.size(0),), t, device=device)
        t_half = torch.full((x.size(0),), t + 0.5 * dt, device=device)
        t1 = torch.full((x.size(0),), t + dt, device=device)

        k1 = self.velocity_fn(x, t0)
        k2 = self.velocity_fn(x + 0.5 * dt * k1, t_half)
        k3 = self.velocity_fn(x + 0.5 * dt * k2, t_half)
        k4 = self.velocity_fn(x + dt * k3, t1)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _solve_adaptive(
        self,
        x_0: Tensor,
        t_span: tuple[float, float],
        return_trajectory: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Solve using adaptive torchdiffeq solver."""
        device = x_0.device
        t_eval = torch.tensor([t_span[0], t_span[1]], device=device)

        # Wrapper for torchdiffeq interface
        def odefunc(t: Tensor, x: Tensor) -> Tensor:
            t_batch = t.expand(x.size(0))
            return self.velocity_fn(x, t_batch)

        trajectory = odeint(
            odefunc,
            x_0,
            t_eval,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
        )

        x_1 = trajectory[-1]

        if return_trajectory:
            return x_1, trajectory
        return x_1


class CFGSampler:
    """Classifier-Free Guidance sampler for conditional generation."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg_scale: float = 2.0,
        method: Literal["euler", "midpoint", "rk4", "dopri5"] = "euler",
        num_steps: int = 32,
    ) -> None:
        """Initialize CFG sampler.

        Args:
            model: F5-TTS model with forward(x_t, t, phonemes, speaker_ids).
            cfg_scale: Classifier-free guidance scale.
            method: ODE integration method.
            num_steps: Number of integration steps.
        """
        self.model = model
        self.cfg_scale = cfg_scale
        self.solver = ODESolver(
            velocity_fn=self._velocity_with_cfg,
            method=method,
            num_steps=num_steps,
        )
        self._current_cond: dict | None = None

    def _velocity_with_cfg(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity with classifier-free guidance."""
        assert self._current_cond is not None

        phonemes = self._current_cond["phonemes"]
        speaker_ids = self._current_cond["speaker_ids"]
        mask = self._current_cond.get("mask")

        # Conditional prediction
        v_cond = self.model(x, t, phonemes, speaker_ids, mask)

        if self.cfg_scale == 1.0:
            return v_cond

        # Unconditional prediction (zero phonemes)
        null_phonemes = torch.zeros_like(phonemes)
        v_uncond = self.model(x, t, null_phonemes, speaker_ids, mask)

        # CFG interpolation
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    @torch.inference_mode()
    def sample(
        self,
        phonemes: Tensor,
        speaker_ids: Tensor | None = None,
        mel_length: int | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Generate mel-spectrogram from phonemes.

        Args:
            phonemes: Phoneme indices. Shape: (B, T).
            speaker_ids: Speaker indices. Shape: (B,).
            mel_length: Target mel length (if different from phoneme length).
            mask: Padding mask. Shape: (B, T).

        Returns:
            Generated mel-spectrogram. Shape: (B, T, mel_dim).
        """
        device = phonemes.device
        batch_size, seq_len = phonemes.shape
        mel_dim = self.model.proj_out.out_features

        if mel_length is None:
            mel_length = seq_len

        # Sample initial noise
        x_0 = torch.randn(batch_size, mel_length, mel_dim, device=device)

        # Store conditioning for velocity function
        self._current_cond = {
            "phonemes": phonemes,
            "speaker_ids": speaker_ids,
            "mask": mask,
        }

        # Integrate ODE
        x_1 = self.solver.solve(x_0)

        self._current_cond = None
        return x_1

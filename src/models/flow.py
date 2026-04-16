"""Optimal-Transport Conditional Flow Matching for F5-TTS.

Random span masking for infilling training, classifier-free guidance (CFG)
training dropout, CFG inference with configurable strength, sway sampling,
and Euler ODE integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def _lens_to_mask(lens: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """Convert sequence lengths to boolean mask [B, N]."""
    if length is None:
        length = int(lens.amax().item())
    seq = torch.arange(length, device=lens.device)
    return seq[None, :] < lens[:, None]


def _mask_from_frac_lengths(
    lens: torch.Tensor, frac_lengths: torch.Tensor
) -> torch.Tensor:
    """Generate random span masks from fractional lengths.

    For each sample, masks a contiguous span of frac_lengths[i] * lens[i] frames
    starting at a random position within the valid region.
    """
    lengths = (frac_lengths * lens).long()
    max_start = lens - lengths
    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    max_len = lens.max()
    seq = torch.arange(max_len, device=lens.device).long()
    return (seq[None, :] >= start[:, None]) & (seq[None, :] < end[:, None])


class CFM(nn.Module):
    """Conditional Flow Matching with infilling training and CFG."""

    def __init__(
        self,
        backbone: nn.Module,
        sigma: float = 0.0,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        n_mels: int = 100,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.sigma = sigma
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = frac_lengths_mask
        self.n_mels = n_mels

    def forward(
        self,
        inp: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training forward pass with random span masking and CFG dropout.

        Args:
            inp: Mel spectrograms [B, n_mels, T] (channels-first).
            text_ids: Token IDs [B, Nt].
            lens: Mel lengths [B] (long). If None, all frames are valid.

        Returns:
            Scalar CFM loss (MSE on masked span).
        """
        # [B, n_mels, T] → [B, T, n_mels]
        if inp.ndim == 3 and inp.shape[1] == self.n_mels:
            inp = inp.transpose(1, 2)

        batch, seq_len, dtype, device = (
            inp.shape[0],
            inp.shape[1],
            inp.dtype,
            inp.device,
        )

        if lens is None:
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        mask = _lens_to_mask(lens, length=seq_len)

        # Random span mask for infilling (mask 70-100% of frames)
        frac_lengths = torch.zeros(batch, device=device).float().uniform_(
            *self.frac_lengths_mask
        )
        rand_span_mask = _mask_from_frac_lengths(lens, frac_lengths)
        rand_span_mask = rand_span_mask & mask

        x1 = inp
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        x0 = torch.randn_like(x1)
        time = torch.rand(batch, dtype=dtype, device=device)

        # OT-CFM interpolation: x_t = (1-t)*x0 + t*x1
        t = time[:, None, None]
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # CFG dropout
        drop_audio_cond = torch.rand(()).item() < self.audio_drop_prob
        drop_text = torch.rand(()).item() < self.cond_drop_prob
        if drop_text:
            drop_audio_cond = True

        pred = self.backbone(
            x=phi,
            cond=cond,
            text=text_ids,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
        )

        # Loss on masked span only
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean()

    @torch.inference_mode()
    def sample(
        self,
        cond: torch.Tensor,
        text_ids: torch.Tensor,
        duration: torch.Tensor | int,
        *,
        lens: torch.Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 1.0,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        max_duration: int = 65536,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Generate mel spectrogram via Euler ODE with optional CFG.

        Args:
            cond: Conditioning mel [B, T_ref, n_mels] (reference audio region).
            text_ids: Token IDs [B, Nt].
            duration: Target total sequence length [B] or scalar int.
            lens: Reference audio lengths [B]; frames [0:lens[i]] are conditioning.
            steps: Number of ODE integration steps.
            cfg_strength: CFG strength (0 = no guidance).
            sway_sampling_coef: Sway sampling coefficient (None = uniform timesteps).
            seed: Random seed for reproducible noise.
            max_duration: Maximum allowed duration.

        Returns:
            (output_mel, trajectory) where output_mel is [B, T, n_mels].
        """
        self.eval()

        batch, cond_seq_len, device = cond.shape[0], cond.shape[1], cond.device

        if lens is None:
            lens = torch.full(
                (batch,), cond_seq_len, device=device, dtype=torch.long
            )

        # Conditioning mask: True where reference audio exists
        cond_mask = _lens_to_mask(lens)

        if isinstance(duration, int):
            duration = torch.full(
                (batch,), duration, device=device, dtype=torch.long
            )
        duration = duration.clamp(max=max_duration)
        max_dur = int(duration.amax().item())

        # Pad conditioning to max duration
        cond = F.pad(cond, (0, 0, 0, max_dur - cond_seq_len), value=0.0)
        cond_mask = F.pad(
            cond_mask, (0, max_dur - cond_mask.shape[-1]), value=False
        )
        cond_mask_3d = cond_mask.unsqueeze(-1)  # [B, T, 1]
        step_cond = torch.where(cond_mask_3d, cond, torch.zeros_like(cond))

        # Attention mask for padding
        attn_mask = _lens_to_mask(duration) if batch > 1 else None

        # ODE velocity function
        def fn(t_val: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            if cfg_strength < 1e-5:
                return self.backbone(
                    x=x,
                    cond=step_cond,
                    text=text_ids,
                    time=t_val,
                    mask=attn_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            # CFG: predict conditioned + unconditioned
            pred_cfg = self.backbone(
                x=x,
                cond=step_cond,
                text=text_ids,
                time=t_val,
                mask=attn_mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # Initial noise (per-sample seeding for batch consistency)
        y0_list: list[torch.Tensor] = []
        for dur in duration:
            if seed is not None:
                torch.manual_seed(seed)
            y0_list.append(
                torch.randn(
                    int(dur.item()),
                    self.n_mels,
                    device=device,
                    dtype=step_cond.dtype,
                )
            )
        y0 = pad_sequence(y0_list, padding_value=0, batch_first=True)

        # Timestep schedule
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # Euler ODE integration
        trajectory: list[torch.Tensor] = [y0]
        x = y0
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            t_batch = t[i].expand(batch)
            v = fn(t_batch, x)
            x = x + v * dt
            trajectory.append(x)

        self.backbone.clear_cache()

        # Replace conditioning region with original
        out = torch.where(cond_mask_3d, cond, x)

        return out, trajectory

"""VITS: Variational Inference with adversarial learning for end-to-end Text-to-Speech."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Generator
from src.models.duration_predictor import StochasticDurationPredictor
from src.models.encoder import TextEncoder
from src.models.flow import ResidualCouplingBlock
from src.models.posterior import PosteriorEncoder


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    ids = torch.arange(max_length, device=length.device)
    return ids.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


class VITS(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_sdp: bool = True,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            from src.models.duration_predictor import DurationPredictor
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 1 else None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            logs_p_clamped = torch.clamp(logs_p, min=-10.0, max=10.0)
            s_p_sq_r = torch.exp(-2 * logs_p_clamped)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p_clamped, dim=1, keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = self._maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = self._rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, l_length, attn, ids_slice, x_mask, (z, z_p, m_p, logs_p), (m_q, logs_q, y_mask, z_slice)

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 1 else None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).unsqueeze(1))

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        logs_p_clamped = torch.clamp(logs_p, min=-10.0, max=10.0)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p_clamped) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def _rand_slice_segments(
        self, x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, t = x.size()
        ids_str_max = x_lengths - segment_size + 1
        ids_str = (torch.rand([b], device=x.device) * ids_str_max.float()).long()
        ids_str = torch.clamp(ids_str, min=0)

        ret = torch.zeros(b, d, segment_size, device=x.device, dtype=x.dtype)
        for i in range(b):
            idx_str = ids_str[i].item()
            idx_end = idx_str + segment_size
            ret[i] = x[i, :, idx_str:idx_end]
        return ret, ids_str

    def _maximum_path(self, neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = neg_cent.device
        dtype = neg_cent.dtype
        neg_cent_np = neg_cent.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        path = self._maximum_path_numpy(neg_cent_np, mask_np)
        return torch.from_numpy(path).to(device=device, dtype=dtype)

    def _maximum_path_numpy(self, value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        b, t_t, t_s = value.shape
        direction = np.zeros(value.shape, dtype=np.int64)
        v = np.zeros((b, t_t), dtype=np.float32)
        x_range = np.arange(t_t, dtype=np.float32).reshape(1, -1)

        for j in range(t_s):
            v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=-np.inf)[:, :-1]
            v1 = v
            max_mask = v1 >= v0
            v = np.where(max_mask, v1, v0)
            direction[:, :, j] = max_mask

            index_mask = x_range <= j
            v = np.where(index_mask * mask[:, :, j], v + value[:, :, j], -np.inf)

        path = np.zeros(value.shape, dtype=np.float32)
        index = mask[:, :, -1].sum(axis=1).astype(np.int64) - 1

        for j in range(t_s - 1, -1, -1):
            for k in range(b):
                path[k, index[k], j] = 1
                if j > 0 and direction[k, index[k], j] == 0 and index[k] > 0:
                    index[k] -= 1
        return path

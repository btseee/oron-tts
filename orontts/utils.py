"""Utility functions for OronTTS."""

import torch
import torch.nn.functional as F


def sequence_mask(
    lengths: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    """Create a sequence mask from lengths.

    Args:
        lengths: Tensor of sequence lengths [B].
        max_len: Maximum length. If None, uses max of lengths.

    Returns:
        Boolean mask tensor [B, T].
    """
    max_len = max_len or int(lengths.max().item())
    ids = torch.arange(0, max_len, device=lengths.device)
    return ids < lengths.unsqueeze(1)


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    """Convert pad shape from nested list to flat list for F.pad.

    Args:
        pad_shape: Nested list [[left, right], [top, bottom], ...].

    Returns:
        Flat list [left, right, top, bottom, ...].
    """
    return [item for sublist in reversed(pad_shape) for item in sublist]


def slice_segments(
    x: torch.Tensor,
    ids_start: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Slice segments from tensor.

    Args:
        x: Input tensor [B, C, T].
        ids_start: Start indices [B].
        segment_size: Segment size.

    Returns:
        Sliced segments [B, C, segment_size].
    """
    batch_size, channels, _ = x.shape
    device = x.device

    # Create grid for gathering [B, 1, segment_size]
    # Expand to [B, 1, segment_size]
    arange = torch.arange(segment_size, device=device).reshape(1, 1, -1)
    
    # [B, 1, 1]
    starts = ids_start.reshape(batch_size, 1, 1)
    
    # Broadcast to [B, C, segment_size]
    # We want indices: [B, segment_size] broadcasted to [B, C, segment_size]
    indices = arange + starts
    indices = indices.expand(batch_size, channels, -1)
    
    # Gather
    return torch.gather(x, 2, indices)


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments from tensor.

    Args:
        x: Input tensor [B, C, T].
        x_lengths: Sequence lengths [B].
        segment_size: Segment size.

    Returns:
        Tuple of (segments [B, C, segment_size], start_indices [B]).
    """
    batch_size = x.shape[0]
    
    # Calculate max starting point for each batch item [B]
    max_starts = (x_lengths - segment_size).clamp(min=0)
    
    # Sample start indices [B]
    ids_start = (torch.rand(batch_size, device=x.device) * (max_starts + 1)).long()
    
    segments = slice_segments(x, ids_start, segment_size)
    return segments, ids_start


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: int,
) -> torch.Tensor:
    """Fused gated activation for WaveNet.

    Args:
        input_a: First input tensor.
        input_b: Second input tensor (conditioning).
        n_channels: Number of channels (half of input).

    Returns:
        Gated activation output.
    """
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def get_padding(
    kernel_size: int,
    dilation: int = 1,
) -> int:
    """Calculate padding for 'same' convolution.

    Args:
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.

    Returns:
        Padding size.
    """
    return (kernel_size * dilation - dilation) // 2


def init_weights(
    module: torch.nn.Module,
    mean: float = 0.0,
    std: float = 0.01,
) -> None:
    """Initialize module weights with normal distribution.

    Args:
        module: PyTorch module to initialize.
        mean: Mean of normal distribution.
        std: Standard deviation.
    """
    if isinstance(module, torch.nn.Conv1d | torch.nn.Linear):
        module.weight.data.normal_(mean, std)
        if module.bias is not None:
            module.bias.data.zero_()


def clip_grad_value_(
    parameters: torch.nn.utils.clip_grad.ParamsT,
    clip_value: float,
) -> None:
    """Clip gradient values.

    Args:
        parameters: Model parameters.
        clip_value: Maximum gradient value.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is not None:
            p.grad.data.clamp_(-clip_value, clip_value)

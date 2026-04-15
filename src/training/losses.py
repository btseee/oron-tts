"""Loss functions for F5-TTS training.

The CFM (Conditional Flow Matching) objective is a simple MSE on the velocity
field.  This module exists as a thin wrapper so the trainer can remain
agnostic to the exact loss formulation and future additions (e.g. perceptual
loss on vocoder outputs) can be plugged in cleanly.
"""

import torch
import torch.nn.functional as F


def cfm_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted and target velocity fields.

    Args:
        v_pred:   Predicted velocity [B, T, C].
        v_target: Ground-truth velocity [B, T, C].
        mask:     Valid-frame mask [B, T]; if None all frames are used.

    Returns:
        Scalar loss.
    """
    if mask is not None:
        m = mask[:, :, None].float()
        return F.mse_loss(v_pred * m, v_target * m, reduction="sum") / (
            m.sum() * v_pred.shape[-1] + 1e-8
        )
    return F.mse_loss(v_pred, v_target)

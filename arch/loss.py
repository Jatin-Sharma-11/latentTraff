"""CLFMv2 loss — simplified (no DAG / sparsity losses)."""

import torch
from basicts.metrics import masked_mae


def clfm_v2_loss(prediction: torch.Tensor, target: torch.Tensor,
                 null_val: float = 0.0,
                 smoothness_loss: torch.Tensor = None) -> torch.Tensor:
    """Combined loss for CLFMv2.

    Total = MAE(pred, target) + smoothness

    The smoothness loss is pre-weighted by its coefficient in the model.

    Args:
        prediction:      [B, L, N, C] model predictions.
        target:          [B, L, N, C] ground truth.
        null_val:        value to mask in MAE.
        smoothness_loss: PDE evolution smoothness (pre-weighted).

    Returns:
        total_loss: scalar.
    """
    base_loss = masked_mae(prediction, target, null_val)

    total = base_loss
    if smoothness_loss is not None:
        total = total + smoothness_loss

    return total

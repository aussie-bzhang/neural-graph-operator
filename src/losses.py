"""
Loss functions for GNO training (paper §3.3).

Implements:
  - Weighted alignment loss  (Eq. 9)
  - Function reconstruction loss  (Eq. 10)
  - Regularization  (Eq. 11)
  - Combined total loss  (Eq. 12)
"""

import torch
import torch.nn.functional as F


def weighted_alignment_loss(K: torch.Tensor, Y: torch.Tensor,
                            pos_weight: float = 5.0) -> torch.Tensor:
    """
    Weighted binary cross-entropy for sparse alignment labels (paper Eq. 9).

    Args:
        K : (n_src, n_tgt)  predicted alignment scores in [0, 1]
        Y : (n_src, n_tgt)  ground-truth alignment (0, 0.5, 0.7, 1.0)
        pos_weight : weight multiplier for positive pairs

    The paper showed this is the single most critical component:
    removing it drops F1 from 0.5433 to 0.0030 (-99.4%).
    """
    # Build per-element weights: positive entries get higher weight
    weights = torch.ones_like(Y)
    weights[Y > 0] = pos_weight

    # Treat Y as soft labels → binary_cross_entropy handles [0,1] targets
    # Clamp K to avoid log(0)
    K_clamped = K.clamp(1e-7, 1 - 1e-7)
    bce = -(Y * torch.log(K_clamped) + (1 - Y) * torch.log(1 - K_clamped))
    loss = (weights * bce).sum() / weights.sum()
    return loss


def reconstruction_loss(v_hat: torch.Tensor,
                        v_true: torch.Tensor) -> torch.Tensor:
    """Function reconstruction loss (paper Eq. 10)."""
    return F.mse_loss(v_hat, v_true)


def regularization_loss(K: torch.Tensor,
                        alpha: float = 0.001,
                        beta: float = 0.001) -> torch.Tensor:
    """
    Regularization (paper Eq. 11):
      α ||K||²_F  +  β Σ_i H(K(i, ·))
    """
    frob = alpha * K.pow(2).sum()

    # Row-wise entropy: encourages sparse, focused attention
    K_row = K / (K.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -beta * (K_row * (K_row + 1e-8).log()).sum()

    return frob + entropy


def gno_total_loss(K: torch.Tensor, Y: torch.Tensor,
                   v_hat: torch.Tensor, v_true: torch.Tensor,
                   pos_weight: float = 5.0,
                   gamma: float = 0.1,
                   alpha: float = 0.001,
                   beta: float = 0.001) -> dict:
    """
    Combined loss (paper Eq. 12):
        L_total = L_align + γ L_recon + L_reg

    Returns dict with total loss and individual components for logging.
    """
    L_align = weighted_alignment_loss(K, Y, pos_weight)
    L_recon = reconstruction_loss(v_hat, v_true)
    L_reg   = regularization_loss(K, alpha, beta)

    L_total = L_align + gamma * L_recon + L_reg

    return {
        "total": L_total,
        "align": L_align.item(),
        "recon": L_recon.item(),
        "reg":   L_reg.item(),
    }

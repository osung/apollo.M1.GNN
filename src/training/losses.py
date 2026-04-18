"""Link-ranking losses for the GNN encoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted Bayesian Personalized Ranking.

    pos_scores: (B,) inner products of positive (u, i) pairs.
    neg_scores: (B,) or (B, K) inner products of negative pairs.
    weights:    (B,) optional per-sample weights (e.g. relation weights).

    Returns the mean of  -weight * log σ(s_pos - s_neg)  over all (B, K).
    """
    if neg_scores.dim() == 2:
        diff = pos_scores.unsqueeze(1) - neg_scores
    elif neg_scores.dim() == 1:
        diff = pos_scores - neg_scores
    else:
        raise ValueError(f"neg_scores must be 1-D or 2-D, got shape {tuple(neg_scores.shape)}")

    loss = -F.logsigmoid(diff)
    if weights is not None:
        if loss.dim() == 2:
            loss = loss * weights.unsqueeze(1)
        else:
            loss = loss * weights
    return loss.mean()

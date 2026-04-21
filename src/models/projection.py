"""Cold-start projection MLP.

Maps raw `norm_embed` features to the GNN-learned latent space `z`, so
nodes not present in the training graph can be scored by the same FAISS
indices the trained nodes use.

Training protocol (see docs/model.md):
  1. Train GNN on the bipartite heterogeneous graph; save z for every
     training-graph node (project and company).
  2. For each node type, form (norm_embed, z) pairs across all
     training-graph nodes.
  3. Train a small MLP per node type: norm_embed -> z_hat, minimizing
     MSE + cosine-alignment hybrid.
  4. At inference, for a cold node, compute z_hat = MLP(norm_embed) and
     insert into the retrieval index. Warm nodes keep the GNN z.

A separate MLP is used per node type because project and company
distributions in the z space differ in scale and geometry.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionMLP(nn.Module):
    """Small feed-forward network: norm_embed -> z.

    Architecture (num_layers=2, default):
        Linear(input_dim -> hidden_dim) -> GELU -> Dropout ->
        Linear(hidden_dim -> output_dim)

    For num_layers >= 3, additional (Linear, GELU, Dropout) blocks are
    inserted before the final output Linear.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def cosine_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hybrid loss: alpha * MSE + (1 - alpha) * (1 - cosine_similarity).

    - MSE targets exact reconstruction (matters when z is not L2-normalized).
    - Cosine targets angular alignment (matters for retrieval via inner
      product, which is our main downstream task).
    """
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=eps).mean()
    return alpha * mse + (1.0 - alpha) * cos

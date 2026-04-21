"""SeHGNN encoder adapted for bipartite heterogeneous graphs.

Paper: Yang et al., "Simple and Efficient Heterogeneous Graph Neural Network"
(NeurIPS 2023).

Original SeHGNN philosophy:
  1. Pre-compute multi-hop aggregated features ONCE (non-parametric)
  2. Learnable fusion across hops via Transformer
  3. No graph propagation during training forward pass -> very fast

Why this fits our problem:
  - Our graph is static during training, so precomputation pays off
  - Full-batch forward in our trainer is expensive; SeHGNN removes it
  - Bipartite + 4 edge types is simple enough that the per-metapath
    expansion in the original paper collapses to per-hop aggregation

Our adaptations:
  - Precompute h_0 (raw norm_embed), h_1, ..., h_K where each h_k is the
    k-hop propagated feature via sym-normalized adjacency summed over all
    edge types (same shape as input).
  - Project each hop with a shared linear, add a learnable hop-position
    embedding, and fuse with a small Transformer encoder layer over the
    hop axis (per-node).
  - Mean-pool fused hop representations, then linear output projection.
  - Cache is built lazily on first forward and reused thereafter (raw
    hop features only; the projection + fusion are parametric and run
    fresh each forward).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _symmetric_norm(
    edge_index: torch.Tensor, num_src: int, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row, col, w) for D_src^-1/2 · A · D_dst^-1/2."""
    row, col = edge_index[0], edge_index[1]
    deg_src = torch.zeros(num_src, dtype=torch.float, device=edge_index.device)
    deg_src.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    deg_dst = torch.zeros(num_dst, dtype=torch.float, device=edge_index.device)
    deg_dst.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
    deg_src_inv = deg_src.pow(-0.5)
    deg_dst_inv = deg_dst.pow(-0.5)
    deg_src_inv[torch.isinf(deg_src_inv)] = 0.0
    deg_dst_inv[torch.isinf(deg_dst_inv)] = 0.0
    w = deg_src_inv[row] * deg_dst_inv[col]
    return row, col, w


class SeHGNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_hops: int,
        metadata: Metadata,
        node_types: Iterable[str] = ("project", "company"),
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads}) for Transformer fusion"
            )
        self.num_hops = num_hops
        self.normalize_output = normalize_output
        self.node_types = tuple(node_types)
        self.edge_types = tuple(metadata[1])
        self.hidden_dim = hidden_dim

        # Shared projection: raw features -> hidden_dim (applied per hop)
        self.input_proj = nn.ModuleDict(
            {nt: nn.Linear(input_dim, hidden_dim) for nt in self.node_types}
        )

        # Learnable hop position embedding: (K+1, hidden_dim)
        self.hop_embedding = nn.Parameter(
            torch.randn(num_hops + 1, hidden_dim) * 0.02
        )

        # Fusion across hops: small Transformer encoder layer per node
        self.fusion = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.output_proj = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types}
        )

        # Cache for precomputed raw k-hop features per node type.
        # Built lazily on first forward(); reused thereafter.
        self._cached_raw_hops: Optional[dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def _precompute_raw_hops(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute (K+1)-hop raw propagated features per node type.

        Returns dict[node_type] -> (N_nt, K+1, input_dim).
        Each hop uses the sum of sym-normalized messages across all
        edge types that land on that node type. Non-parametric.
        """
        current = {nt: x_dict[nt] for nt in self.node_types}
        hops = {nt: [current[nt]] for nt in self.node_types}

        for _ in range(self.num_hops):
            new_h = {nt: torch.zeros_like(current[nt]) for nt in self.node_types}
            for et in self.edge_types:
                src_type, _, dst_type = et
                if (
                    src_type not in self.node_types
                    or dst_type not in self.node_types
                ):
                    continue
                edge_index = edge_index_dict.get(et)
                if edge_index is None or edge_index.shape[1] == 0:
                    continue
                x_src = current[src_type]
                num_src = x_src.shape[0]
                num_dst = current[dst_type].shape[0]
                row, col, w = _symmetric_norm(edge_index, num_src, num_dst)
                msg = x_src[row] * w.unsqueeze(-1)
                idx = col.unsqueeze(-1).expand(-1, x_src.shape[1])
                new_h[dst_type].scatter_add_(0, idx, msg)
            current = new_h
            for nt in self.node_types:
                hops[nt].append(current[nt])

        return {nt: torch.stack(hops[nt], dim=1) for nt in self.node_types}

    def invalidate_cache(self) -> None:
        """Drop the cached raw hops (e.g., if the graph changes)."""
        self._cached_raw_hops = None

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        # Invalidate cache if the input moved to a different device than
        # where the cache was built. This happens in train_gnn.py, which
        # runs a dummy CPU forward to materialize lazy PyG layers before
        # model.to(device). The cache, being a plain attribute rather
        # than a registered buffer, does not follow .to().
        any_nt = self.node_types[0]
        input_device = x_dict[any_nt].device
        if (
            self._cached_raw_hops is not None
            and self._cached_raw_hops[any_nt].device != input_device
        ):
            self._cached_raw_hops = None

        if self._cached_raw_hops is None:
            self._cached_raw_hops = self._precompute_raw_hops(
                x_dict, edge_index_dict
            )

        result: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            # (N, K+1, input_dim) -> (N, K+1, hidden_dim) via shared linear
            raw = self._cached_raw_hops[nt]
            proj = self.input_proj[nt](raw)
            # Add hop position embedding, broadcast across nodes
            proj = proj + self.hop_embedding.unsqueeze(0)
            # Transformer attends across the K+1 hop axis per node
            fused = self.fusion(proj)  # (N, K+1, hidden_dim)
            # Mean pool across hops
            pooled = fused.mean(dim=1)
            result[nt] = self.output_proj[nt](pooled)

        if self.normalize_output:
            result = {k: F.normalize(v, p=2, dim=-1) for k, v in result.items()}
        return result

    @torch.no_grad()
    def encode_all(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(x_dict, edge_index_dict)

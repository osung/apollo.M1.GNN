"""R-GCN (Relational GCN) encoder for bipartite heterogeneous graphs.

Paper: Schlichtkrull et al., "Modeling Relational Data with Graph
Convolutional Networks" (ESWC 2018) — the canonical heterogeneous-GNN
baseline. R-GCN learns one transformation matrix W_r per edge type
("relation") plus a self-loop W_0, and aggregates per-relation messages
with destination-degree normalization:

    h_v^(l+1) = σ( W_0^(l) · h_v^(l)
                  + Σ_r Σ_{u in N_v^r} 1/|N_v^r| · W_r^(l) · h_u^(l) )

For our bipartite project↔company graph with `royalty / commercial /
performance / similarity` (and their reverses), each relation gets its
own W_r so the model can encode "this project's royalty partners
matter differently from its sim partners" — exactly the property
edge-type weights probe at message level.

Adaptations vs the paper:
  - Bipartite: input/output projections are per-node-type so project
    and company features can have different scales.
  - Optional MP edge-weight injection (multiply per-edge norm by the
    relation scalar before scatter_add) to keep the same --mp-edge-weights
    semantics as GFM/LightGCN/SeHGNN.
  - No basis decomposition: with 8 edge types (4 forward + 4 reverse)
    parameter blow-up is small (~500K at h=256), so the basis trick
    isn't worth the implementation cost.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _et_key(et: tuple) -> str:
    """Flatten ('project', 'royalty', 'company') into a ModuleDict-safe key."""
    return "__".join(et)


def _dst_degree_inv(
    edge_index: torch.Tensor, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row, col, w) where w = 1/deg_dst[col]. Used for the
    per-relation normalization 1/|N_v^r| from the R-GCN paper."""
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_dst, dtype=torch.float, device=edge_index.device)
    deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    w = deg_inv[col]
    return row, col, w


class RGCNLayer(nn.Module):
    """One R-GCN convolution layer for the bipartite heterogeneous graph."""

    def __init__(
        self,
        hidden_dim: int,
        metadata: Metadata,
        node_types: tuple[str, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = tuple(metadata[1])
        self.dropout = dropout

        # Self-loop transformation W_0 per node type
        self.w_self = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )
        # Per-relation transformation W_r — the defining R-GCN parameter
        self.w_rel = nn.ModuleDict(
            {_et_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types}
        )

    def forward(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        out = {nt: self.w_self[nt](h_dict[nt]) for nt in self.node_types}

        for et in self.edge_types:
            src_type, _, dst_type = et
            if src_type not in self.node_types or dst_type not in self.node_types:
                continue
            edge_index = edge_index_dict.get(et)
            if edge_index is None or edge_index.shape[1] == 0:
                continue

            x_src = h_dict[src_type]
            num_dst = h_dict[dst_type].shape[0]
            d = x_src.shape[1]

            row, col, w = _dst_degree_inv(edge_index, num_dst)
            w = w.to(x_src.dtype)
            # Optional relation-level scalar weight for MP weighting
            if edge_weight_dict is not None:
                ew = edge_weight_dict.get(et)
                if ew is not None:
                    w = w * ew.to(x_src.dtype)

            msg = x_src[row] * w.unsqueeze(-1)  # (E, d) — normalized neighbor features
            idx = col.unsqueeze(-1).expand(-1, d)
            agg = torch.zeros(num_dst, d, device=x_src.device, dtype=x_src.dtype)
            agg.scatter_add_(0, idx, msg)

            # Apply per-relation transformation W_r and sum into destination
            out[dst_type] = out[dst_type] + self.w_rel[_et_key(et)](agg)

        out = {nt: F.relu(v) for nt, v in out.items()}
        if self.training and self.dropout > 0:
            out = {nt: F.dropout(v, p=self.dropout) for nt, v in out.items()}
        return out


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        metadata: Metadata,
        node_types: Iterable[str] = ("project", "company"),
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.node_types = tuple(node_types)
        self.num_layers = num_layers
        self.normalize_output = normalize_output

        self.input_proj = nn.ModuleDict(
            {nt: nn.Linear(input_dim, hidden_dim) for nt in self.node_types}
        )
        self.layers = nn.ModuleList(
            [
                RGCNLayer(hidden_dim, metadata, self.node_types, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, output_dim) for nt in self.node_types}
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        h = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}
        for layer in self.layers:
            h = layer(h, edge_index_dict, edge_weight_dict)
        z = {nt: self.output_proj[nt](h[nt]) for nt in self.node_types}
        if self.normalize_output:
            z = {k: F.normalize(v, p=2, dim=-1) for k, v in z.items()}
        return z

    @torch.no_grad()
    def encode_all(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(x_dict, edge_index_dict, edge_weight_dict)

"""GFM (Graph Factorization Machine) encoder for bipartite heterogeneous graphs.

Paper: Su et al., "Graph Factorization Machines for Cross-Domain Recommendation"
(arXiv 2007.05911). We follow the formulation where each GFM layer aggregates
neighborhood information via FM-style second-order interactions on top of a
standard first-order sum.

Core layer update for destination node v:
    first_order(v)  = Σ_{u in N(v)} w_{uv} · h_u
    second_order(v) = 0.5 · ((Σ m)^2 - Σ m^2)       with m = w_{uv} · h_u
    out(v) = W_self · h_v + W_lin · first_order(v) + W_fm · second_order(v)

The identity Σ_{i<j} m_i ⊙ m_j = 0.5 · ((Σ m)^2 − Σ m^2) (Rendle 2010)
lets us compute pairwise interactions among all neighbors in O(|N(v)|·d)
instead of O(|N(v)|^2·d). Element-wise, so it returns a d-dim vector.

Adaptations for this project:
  - Bipartite heterogeneous: separate W_lin and W_fm per edge type, results
    summed across edge types at each destination node.
  - Symmetric normalization (D_src^-1/2 A D_dst^-1/2) on edge messages so
    high-degree nodes don't dominate — same regularization as LightGCN.
  - Per-node-type LayerNorm after each layer so the FM quadratic term's
    magnitude doesn't blow up across stacked layers.
  - input_proj / output_proj mirror GNNEncoder so the cold-start projection
    MLP is a drop-in replacement (same output shape).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _symmetric_norm_weight(
    edge_index: torch.Tensor, num_src: int, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row, col, w) for D_src^-1/2 · A · D_dst^-1/2 on one relation."""
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


def _et_key(et: tuple) -> str:
    # Flatten ('project', 'royalty', 'company') into a ModuleDict-safe key
    return "__".join(et)


class GFMLayer(nn.Module):
    """One heterogeneous GFM convolution layer."""

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

        self.w_self = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )
        self.w_lin = nn.ModuleDict(
            {_et_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types}
        )
        self.w_fm = nn.ModuleDict(
            {_et_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types}
        )
        # LayerNorm absorbs the scale of the FM quadratic term when layers
        # are stacked; empirically required for num_layers >= 2 stability.
        self.out_norm = nn.ModuleDict(
            {nt: nn.LayerNorm(hidden_dim) for nt in node_types}
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
            num_src = x_src.shape[0]
            num_dst = h_dict[dst_type].shape[0]
            d = x_src.shape[1]

            row, col, w = _symmetric_norm_weight(edge_index, num_src, num_dst)
            # Cast sym-norm weight to x_src.dtype so that under bf16 autocast
            # the message m_e and the accumulator sum_msg agree on dtype.
            # Without this, w stays fp32 → m_e gets promoted to fp32 → scatter_add
            # into bf16 sum_msg fails with "self.dtype != src.dtype".
            w = w.to(x_src.dtype)
            # Multiply by relation-specific edge weight (royalty=1.0, ..., sim=0.25
            # by default) so high-priority edges propagate more signal per hop.
            # When edge_weight_dict is None (e.g., loading legacy graph without
            # edge_weight attr), falls back to uniform edge importance.
            if edge_weight_dict is not None:
                ew = edge_weight_dict.get(et)
                if ew is not None:
                    w = w * ew.to(x_src.dtype)
            m_e = x_src[row] * w.unsqueeze(-1)  # (E, d): weighted messages

            idx = col.unsqueeze(-1).expand(-1, d)
            sum_msg = torch.zeros(num_dst, d, device=x_src.device, dtype=x_src.dtype)
            sum_msg.scatter_add_(0, idx, m_e)

            sqsum_msg = torch.zeros_like(sum_msg)
            sqsum_msg.scatter_add_(0, idx, m_e * m_e)

            # FM identity: Σ_{i<j} m_i ⊙ m_j = 0.5 · ((Σ m)^2 − Σ m^2)
            fm_msg = 0.5 * (sum_msg * sum_msg - sqsum_msg)

            k = _et_key(et)
            out[dst_type] = (
                out[dst_type]
                + self.w_lin[k](sum_msg)
                + self.w_fm[k](fm_msg)
            )

        out = {nt: F.relu(self.out_norm[nt](v)) for nt, v in out.items()}
        if self.training and self.dropout > 0:
            out = {nt: F.dropout(v, p=self.dropout) for nt, v in out.items()}
        return out


class GFMEncoder(nn.Module):
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
                GFMLayer(hidden_dim, metadata, self.node_types, dropout)
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
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict, edge_weight_dict)
        z_dict = {nt: self.output_proj[nt](h_dict[nt]) for nt in self.node_types}
        if self.normalize_output:
            z_dict = {k: F.normalize(z, p=2, dim=-1) for k, z in z_dict.items()}
        return z_dict

    @torch.no_grad()
    def encode_all(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(x_dict, edge_index_dict, edge_weight_dict)

"""LightGCN encoder adapted for bipartite heterogeneous graphs with features.

Paper: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation" (SIGIR 2020).

Key simplifications vs GCN/SAGE:
  - No feature transformation (no W matrix in layers)
  - No nonlinearity (no ReLU between layers)
  - Layer combination: final = mean(h_0, h_1, ..., h_L)

Our adaptations for this project:
  - Start with input_proj(norm_embed) to use content features as h_0, unlike
    the paper's learnable ID embeddings. This preserves cold-start
    capability and lets us compare fairly against SAGE/GCN/HGT backbones.
  - Heterogeneous propagation: per-relation symmetric normalization
    (D_src^-1/2 · A · D_dst^-1/2) implemented via scatter_add. Messages
    from all edge types are summed per layer (same as HeteroConv aggr='sum').
  - FAISS-friendly output: optional L2 normalization (matches other encoders).

PyG's built-in LGConv doesn't accept the (x_src, x_dst) tuple that
HeteroConv passes for bipartite edges, so we write the propagation directly.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _symmetric_norm(
    edge_index: torch.Tensor, num_src: int, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row, col, w) for D_src^-1/2 · A · D_dst^-1/2 normalization.

    For a bipartite edge (src, dst), the weight is deg(src)^-1/2 * deg(dst)^-1/2
    where deg(src) is the src-side out-degree and deg(dst) the dst-side
    in-degree (both of this particular relation).
    """
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


class LightGCNEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        metadata: Metadata,
        node_types: Iterable[str] = ("project", "company"),
        normalize_output: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.normalize_output = normalize_output
        self.node_types = tuple(node_types)
        self.hidden_dim = hidden_dim

        # norm_embed -> hidden_dim (acts as h_0). Paper uses learnable ID
        # embeddings here; we use a linear projection of content features so
        # that cold-start projection can reuse the same space.
        self.input_proj = nn.ModuleDict(
            {nt: nn.Linear(input_dim, hidden_dim) for nt in self.node_types}
        )

        # Edge types kept as ordered tuples so forward iteration is stable
        # and compatible with PyG HeteroData edge_index_dict keys.
        self.edge_types = tuple(metadata[1])

    def _propagate_one_layer(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        new_h: dict[str, torch.Tensor] = {
            nt: torch.zeros_like(h_dict[nt]) for nt in self.node_types
        }
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
            row, col, w = _symmetric_norm(edge_index, num_src, num_dst)

            # message = w * x_src[row]  shape (E, D)
            msg = x_src[row] * w.unsqueeze(-1)
            # scatter_add into dst positions
            idx = col.unsqueeze(-1).expand(-1, x_src.shape[1])
            new_h[dst_type].scatter_add_(0, idx, msg)
        return new_h

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        # h_0: project features into embedding space
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}

        # Collect h_0 ... h_L for layer combination
        all_h = [{nt: h_dict[nt] for nt in self.node_types}]

        for _ in range(self.num_layers):
            h_dict = self._propagate_one_layer(h_dict, edge_index_dict)
            all_h.append({nt: h_dict[nt] for nt in self.node_types})

        # Layer combination: mean across L+1 layers (paper default)
        combined: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            stacked = torch.stack([h[nt] for h in all_h], dim=0)
            combined[nt] = stacked.mean(dim=0)

        if self.normalize_output:
            combined = {
                k: F.normalize(v, p=2, dim=-1) for k, v in combined.items()
            }
        return combined

    @torch.no_grad()
    def encode_all(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(x_dict, edge_index_dict)

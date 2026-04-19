"""Heterogeneous bipartite GNN encoder with pluggable layer backend.

Layer types (select via `layer_type`):
- `sage`  : HeteroConv + SAGEConv per relation
- `gcn`   : HeteroConv + GraphConv per relation (edge-weight aware)
- `gat`   : HeteroConv + GATConv per relation
- `hgt`   : HGTConv (Heterogeneous Graph Transformer, native per-type attention)

All variants share the same input/output shape contract:
    forward(x_dict, edge_index_dict) -> z_dict
where `z` is L2-normalized so that inner product = cosine similarity,
matching the FAISS serving path.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GraphConv, HeteroConv, HGTConv, SAGEConv
from torch_geometric.typing import Metadata


LAYER_TYPES = ("sage", "gcn", "gat", "hgt")


def _build_layer(
    layer_type: str,
    hidden_dim: int,
    metadata: Metadata,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    edge_types = metadata[1]
    if layer_type == "sage":
        return HeteroConv(
            {et: SAGEConv((-1, -1), hidden_dim) for et in edge_types},
            aggr="sum",
        )
    if layer_type == "gcn":
        return HeteroConv(
            {et: GraphConv((-1, -1), hidden_dim) for et in edge_types},
            aggr="sum",
        )
    if layer_type == "gat":
        return HeteroConv(
            {
                et: GATConv(
                    (-1, -1),
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    add_self_loops=False,
                    dropout=dropout,
                )
                for et in edge_types
            },
            aggr="sum",
        )
    if layer_type == "hgt":
        return HGTConv(hidden_dim, hidden_dim, metadata, num_heads)
    raise ValueError(f"unknown layer_type: {layer_type!r} (expected one of {LAYER_TYPES})")


class GNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        metadata: Metadata,
        layer_type: str = "sage",
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize_output: bool = True,
        node_types: Iterable[str] = ("project", "company"),
    ):
        super().__init__()
        self.layer_type = layer_type
        self.dropout = dropout
        self.normalize_output = normalize_output
        self.node_types = tuple(node_types)

        self.input_proj = nn.ModuleDict(
            {nt: nn.Linear(input_dim, hidden_dim) for nt in self.node_types}
        )
        self.layers = nn.ModuleList(
            [
                _build_layer(layer_type, hidden_dim, metadata, num_heads, dropout)
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
    ) -> dict[str, torch.Tensor]:
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            h_dict = {k: F.relu(h) for k, h in h_dict.items()}
            if self.training and self.dropout > 0:
                h_dict = {k: F.dropout(h, p=self.dropout) for k, h in h_dict.items()}
        z_dict = {nt: self.output_proj[nt](h_dict[nt]) for nt in self.node_types}
        if self.normalize_output:
            z_dict = {k: F.normalize(z, p=2, dim=-1) for k, z in z_dict.items()}
        return z_dict

    @torch.no_grad()
    def encode_all(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        """Evaluation-time forward pass that disables dropout."""
        self.eval()
        return self.forward(x_dict, edge_index_dict)

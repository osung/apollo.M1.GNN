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


class FastHGTConv(HGTConv):
    """HGTConv with the per-relation key/value transform done as one batched matmul per
    edge type.

    PyG applies the transform with `HeteroLinear`, which (without pyg-lib) writes each
    (edge type, head) slice into a preallocated tensor. The backward pass of every such
    slice assignment copies the whole tensor, so its cost grows with
    (number of edge types)^2 x nodes and dominates training once the graph has many
    relations. Parameters and results are identical to HGTConv.
    """

    def _construct_src_node_feat(self, k_dict, v_dict, edge_index_dict):
        heads, dim = self.heads, self.out_channels // self.heads
        n_types = len(self.edge_types)
        # HeteroLinear type index of (head h, edge type e) is h * n_types + e
        w_k = self.k_rel.weight.view(heads, n_types, dim, dim)
        w_v = self.v_rel.weight.view(heads, n_types, dim, dim)
        ks, vs, offset, cumsum = [], [], {}, 0
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            e = self.edge_types_map[edge_type]
            offset[edge_type] = cumsum
            cumsum += k_dict[src].size(0)
            # (N, H, D) -> (H, N, D) @ (H, D, D) -> (N, H, D)
            ks.append(torch.bmm(k_dict[src].transpose(0, 1), w_k[:, e]).transpose(0, 1))
            vs.append(torch.bmm(v_dict[src].transpose(0, 1), w_v[:, e]).transpose(0, 1))
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0), offset


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
        # GraphConv defaults to aggr='add', which blows up on our
        # high-degree heterogeneous graph (hundreds of neighbors per node,
        # ~12M similarity edges). Without output L2-normalization the
        # embeddings explode and BPR loss diverges. Use 'mean' to match
        # SAGEConv's aggregation style.
        return HeteroConv(
            {et: GraphConv((-1, -1), hidden_dim, aggr="mean") for et in edge_types},
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
        return FastHGTConv(hidden_dim, hidden_dim, metadata, num_heads)
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
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        # GCN's GraphConv accepts an edge_weight kwarg, so we forward the dict
        # via HeteroConv's per-relation kwarg routing. SAGEConv/GATConv/HGTConv
        # don't accept edge_weight, so we drop it for those layer types to
        # avoid TypeError ("got an unexpected keyword argument 'edge_weight'").
        # edge_index_dict may also be a list with one dict per layer (mini-batch training:
        # src/training/neighbor_sampler.py). Edge weights are not used in that mode.
        per_layer = isinstance(edge_index_dict, (list, tuple))
        if per_layer and len(edge_index_dict) != len(self.layers):
            raise ValueError(f"got {len(edge_index_dict)} edge dicts for "
                             f"{len(self.layers)} layers")
        pass_edge_weights = (
            self.layer_type == "gcn" and edge_weight_dict is not None and not per_layer
        )
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}
        for i, layer in enumerate(self.layers):
            layer_edges = edge_index_dict[i] if per_layer else edge_index_dict
            if pass_edge_weights:
                h_dict = layer(
                    h_dict, layer_edges, edge_weight_dict=edge_weight_dict
                )
            else:
                h_dict = layer(h_dict, layer_edges)
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
        edge_weight_dict: dict | None = None,
        chunk_nodes: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluation-time forward pass that disables dropout.

        chunk_nodes: compute each layer for that many target nodes (per node type) at a
        time, using all of their in-edges. The result equals the full-graph pass, but only
        one chunk's edges are in memory at once (edge weights are not supported here).
        """
        self.eval()
        if not chunk_nodes:
            return self.forward(x_dict, edge_index_dict, edge_weight_dict)
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}
        n_max = max(h.shape[0] for h in h_dict.values())
        for layer in self.layers:
            out = {nt: torch.empty_like(h) for nt, h in h_dict.items()}
            for lo in range(0, n_max, chunk_nodes):
                hi = lo + chunk_nodes
                part = {
                    et: ei[:, (ei[1] >= lo) & (ei[1] < hi)]
                    for et, ei in edge_index_dict.items()
                }
                res = layer(h_dict, part)
                for nt in out:
                    out[nt][lo:hi] = F.relu(res[nt][lo:hi]).to(out[nt].dtype)
            h_dict = out
        z_dict = {nt: self.output_proj[nt](h_dict[nt]) for nt in self.node_types}
        if self.normalize_output:
            z_dict = {k: F.normalize(z, p=2, dim=-1) for k, z in z_dict.items()}
        return z_dict

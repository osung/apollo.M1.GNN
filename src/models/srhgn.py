"""SR-HGN (Semantic- and Relation-Aware HGN) encoder for bipartite heterogeneous graphs.

Paper: Wang et al., "SR-HGN: Semantic- and Relation-Aware Heterogeneous
Graph Neural Network" (Expert Systems with Applications, Vol 224, 2023).
The same-journal predecessor with the most directly comparable scope —
heterogeneous GNN for representation learning that doesn't rely on
hand-crafted meta-paths.

SR-HGN's two-level attention:

  1. Node-level (per-relation) attention: for each edge type r, run a
     GAT-style attention over a node v's neighbors connected via r.
     This produces a per-relation representation h_v^r.

  2. Type-level (semantic) attention: combine the {h_v^r}_r vectors
     into a single h_v using a learnable per-node-type query that
     scores each relation's contribution.

The original paper uses meta-path-free per-relation attention plus a
semantic+relation attention mix. This implementation keeps the
two-level structure with a single attention head and a softmax-based
type-level mixer for memory efficiency on our 12M+ similarity-edge
bipartite graph.

Adaptations:
  - Bipartite + per-node-type input/output projection (matches our
    GFM/RGCN/LightGCN contract).
  - Optional --mp-edge-weights (added to attention logits before
    softmax via log(ew), so the per-relation prior multiplies inside
    the softmax denominator).
  - Single-head attention only — multi-head would multiply edge-side
    activations by num_heads and OOM at sim=10. We keep the door
    open for a future num_heads parameter without changing the API.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _et_key(et: tuple) -> str:
    return "__".join(et)


def _scatter_softmax(
    score: torch.Tensor,    # (E,) edge-level logits
    index: torch.Tensor,    # (E,) destination node index for each edge
    num_nodes: int,
) -> torch.Tensor:
    """Numerically stable softmax over edges sharing the same destination
    node. Returns (E,) attention weights summing to 1 per dst node."""
    # max-shift for stability
    max_per_node = torch.full(
        (num_nodes,), float("-inf"), device=score.device, dtype=score.dtype
    )
    max_per_node.scatter_reduce_(0, index, score, reduce="amax", include_self=False)
    max_per_node = torch.where(
        torch.isinf(max_per_node), torch.zeros_like(max_per_node), max_per_node
    )
    score = score - max_per_node[index]
    score_exp = torch.exp(score)

    sum_per_node = torch.zeros(num_nodes, device=score.device, dtype=score.dtype)
    sum_per_node.scatter_add_(0, index, score_exp)
    # Avoid div-by-0 for nodes with no in-edges (shouldn't happen if we
    # only score edges that exist, but defensive).
    sum_per_node = torch.where(
        sum_per_node == 0, torch.ones_like(sum_per_node), sum_per_node
    )
    return score_exp / sum_per_node[index]


class SRHGNLayer(nn.Module):
    """One SR-HGN convolution layer for bipartite heterogeneous graphs."""

    def __init__(
        self,
        hidden_dim: int,
        metadata: Metadata,
        node_types: tuple[str, ...],
        dropout: float = 0.1,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = tuple(metadata[1])
        self.dropout = dropout
        self.leaky_slope = leaky_slope

        # Per-relation linear transform W_r used for both message and attention.
        self.w_rel = nn.ModuleDict(
            {_et_key(et): nn.Linear(hidden_dim, hidden_dim, bias=False)
             for et in self.edge_types}
        )
        # Per-relation attention vectors (split into src/dst halves so
        # the score factorizes as a_src·W·h_u + a_dst·W·h_v, matching
        # GAT's bilinear attention).
        self.a_src = nn.ParameterDict(
            {_et_key(et): nn.Parameter(torch.empty(hidden_dim))
             for et in self.edge_types}
        )
        self.a_dst = nn.ParameterDict(
            {_et_key(et): nn.Parameter(torch.empty(hidden_dim))
             for et in self.edge_types}
        )

        # Self-loop W_0 per node type
        self.w_self = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )

        # Type-level (semantic) attention query — one per node type. The
        # attention score for relation r at node v is
        #   q_nt · tanh(h_v^r)
        # then softmax over r.
        self.q_type = nn.ParameterDict(
            {nt: nn.Parameter(torch.empty(hidden_dim)) for nt in node_types}
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.a_src.values():
            nn.init.normal_(p, mean=0.0, std=0.02)
        for p in self.a_dst.values():
            nn.init.normal_(p, mean=0.0, std=0.02)
        for p in self.q_type.values():
            nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        edge_weight_dict: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        # === Step 1: per-relation node-level attention ===
        # Track (relation_aggregate, relation_scalar_weight) per dst node type.
        # The scalar weight is constant within a relation in our setup
        # (royalty=1.0, sim=0.25, ...) so taking ew[0] is well-defined; we
        # use it at the type-level softmax below — within-relation softmax
        # would silently absorb a uniform-per-edge constant.
        rel_vecs: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            nt: [] for nt in self.node_types
        }

        for et in self.edge_types:
            src_type, _, dst_type = et
            if src_type not in self.node_types or dst_type not in self.node_types:
                continue
            edge_index = edge_index_dict.get(et)
            if edge_index is None or edge_index.shape[1] == 0:
                continue

            x_src = h_dict[src_type]
            x_dst = h_dict[dst_type]
            num_dst = x_dst.shape[0]
            d = x_src.shape[1]
            row, col = edge_index[0], edge_index[1]
            key = _et_key(et)

            # W_r · h applied once per node, then sliced for edges.
            wx_src = self.w_rel[key](x_src)        # (N_src, d)
            wx_dst = self.w_rel[key](x_dst)        # (N_dst, d)

            # GAT-style bilinear attention factored into two halves:
            #   score = a_src · (W·h_u) + a_dst · (W·h_v)
            score_src = (wx_src * self.a_src[key]).sum(dim=-1)   # (N_src,)
            score_dst = (wx_dst * self.a_dst[key]).sum(dim=-1)   # (N_dst,)
            score = score_src[row] + score_dst[col]              # (E,)
            score = F.leaky_relu(score, self.leaky_slope)

            attn = _scatter_softmax(score, col, num_dst)         # (E,)

            # Aggregate W·h_u with attention weights into (N_dst, d).
            msg = wx_src[row] * attn.unsqueeze(-1)
            idx = col.unsqueeze(-1).expand(-1, d)
            agg = torch.zeros(num_dst, d, device=x_src.device, dtype=x_src.dtype)
            agg.scatter_add_(0, idx, msg)

            # Per-relation scalar weight for type-level injection.
            relation_w = torch.tensor(1.0, device=agg.device, dtype=agg.dtype)
            if edge_weight_dict is not None:
                ew = edge_weight_dict.get(et)
                if ew is not None and ew.numel() > 0:
                    relation_w = ew[0].to(agg.dtype)
            rel_vecs[dst_type].append((agg, relation_w))

        # === Step 2: type-level (semantic) attention over relations ===
        out = {nt: self.w_self[nt](h_dict[nt]) for nt in self.node_types}

        for nt in self.node_types:
            if not rel_vecs[nt]:
                continue
            # (N, num_rels, d)
            stacked = torch.stack([v for v, _ in rel_vecs[nt]], dim=1)
            # Semantic attention score per (node, relation):
            #   q_nt · tanh(h_v^r)
            scores = torch.tanh(stacked) @ self.q_type[nt]       # (N, num_rels)
            # Inject relation-level scalar weight as log-bias on the
            # relation softmax. royalty (w=1.0) gets +0, sim (w=0.25)
            # gets ~-1.39 → softmax pulls mass away from low-priority
            # relations, mirroring the GFM/RGCN MPW semantics.
            rel_weights = torch.stack([w for _, w in rel_vecs[nt]])  # (num_rels,)
            scores = scores + torch.log(rel_weights.clamp(min=1e-8)).unsqueeze(0)

            type_attn = F.softmax(scores, dim=-1)                # (N, num_rels)
            weighted = (stacked * type_attn.unsqueeze(-1)).sum(dim=1)  # (N, d)
            out[nt] = out[nt] + weighted

        # ELU activation (matches GAT/SR-HGN convention) + dropout
        out = {nt: F.elu(v) for nt, v in out.items()}
        if self.training and self.dropout > 0:
            out = {nt: F.dropout(v, p=self.dropout) for nt, v in out.items()}
        return out


class SRHGNEncoder(nn.Module):
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
                SRHGNLayer(hidden_dim, metadata, self.node_types, dropout)
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

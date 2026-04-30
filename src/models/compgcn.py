"""CompGCN encoder for bipartite heterogeneous graphs.

Paper: Vashishth et al., "Composition-Based Multi-Relational Graph
Convolutional Networks" (ICLR 2020).

Key idea vs R-GCN:
  R-GCN learns one weight matrix W_r per relation, which blows up
  parameters when there are many relations. CompGCN keeps W as just
  three direction-specific matrices (forward / inverse / self-loop)
  and instead represents each relation as a *vector* h_r that is
  composed with the node feature via a composition function φ.

  h_v^(l+1) = σ( W_S · h_v^(l)
                + W_O · agg_{u→v via forward r} 1/|N| · φ(h_u, h_r)
                + W_I · agg_{u→v via reverse r} 1/|N| · φ(h_u, h_r) )

  Composition operators in the paper: subtraction (Trans-E),
  element-wise multiplication (DistMult), circular correlation
  (HolE). We expose all three; subtraction is the default since it's
  the simplest and most often the strongest in the paper's experiments.

Adaptations for our bipartite project↔company graph:
  - "Forward" = src is project, dst is company; "inverse" = src is
    company, dst is project. Self-loop is per-node-type (separate
    W_S^{project}, W_S^{company}) so the two sides can scale features
    independently.
  - Each edge type (royalty, commercial, performance, similarity, plus
    reverses) gets its own learned relation vector h_r — even though
    the W matrices are shared by direction, the per-relation vector
    still lets the model differentiate edge types.
  - Optional --mp-edge-weights multiplied into the dst-degree-inverse
    normalization weight, matching GFM/RGCN/LightGCN/SeHGNN.
  - Output is L2-normalized for FAISS retrieval, matching the rest of
    the model lineup.

Notes:
  - The relation embeddings h_r are learned alongside everything else;
    no pretraining required.
  - For circular correlation we fall back to fp32 inside the FFT call
    even under bf16 autocast — torch's complex FFTs aren't bf16-safe.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.typing import Metadata


def _et_key(et: tuple) -> str:
    return "__".join(et)


def _dst_degree_inv(
    edge_index: torch.Tensor, num_dst: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (row, col, w) where w = 1/deg_dst[col]. R-GCN-style
    per-relation destination-degree normalization."""
    row, col = edge_index[0], edge_index[1]
    deg = torch.zeros(num_dst, dtype=torch.float, device=edge_index.device)
    deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float))
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    return row, col, deg_inv[col]


def _ccorr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular correlation φ(a, b) = IFFT(conj(FFT(a)) ⊙ FFT(b)).

    Operates on the last dim. Falls back to fp32 internally because
    torch's complex FFTs aren't bf16-safe; result is cast back to
    a's dtype so downstream stays in autocast.
    """
    in_dtype = a.dtype
    a_f = a.float()
    b_f = b.float()
    A = torch.fft.fft(a_f, dim=-1)
    B = torch.fft.fft(b_f, dim=-1)
    out = torch.fft.ifft(torch.conj(A) * B, dim=-1).real
    return out.to(in_dtype)


def _compose(h_u: torch.Tensor, h_r: torch.Tensor, op: str) -> torch.Tensor:
    """Composition function φ(h_u, h_r). h_u is (E, d), h_r is (d,)
    or (E, d). Returns (E, d)."""
    if op == "sub":
        return h_u - h_r
    if op == "mult":
        return h_u * h_r
    if op == "ccorr":
        return _ccorr(h_u, h_r)
    raise ValueError(f"unknown composition op: {op!r}")


class CompGCNLayer(nn.Module):
    """One CompGCN convolution layer for the bipartite heterogeneous graph."""

    def __init__(
        self,
        hidden_dim: int,
        metadata: Metadata,
        node_types: tuple[str, ...],
        dropout: float = 0.1,
        comp_op: str = "sub",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = tuple(metadata[1])
        self.dropout = dropout
        self.comp_op = comp_op

        # Three direction-specific weight matrices (the defining
        # CompGCN parameter sharing across relations).
        self.w_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Per-node-type self-loop weight (= W_S^{nt}). The paper uses a
        # single W_S; we split per node type so project/company can
        # scale their own features independently — matches our other
        # encoders' contract.
        self.w_self = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )
        # Per-relation embedding h_r — learned vector per edge type.
        self.rel_emb = nn.ParameterDict(
            {_et_key(et): nn.Parameter(torch.empty(hidden_dim))
             for et in self.edge_types}
        )
        # Relation embeddings transformed each layer (the paper updates
        # h_r per layer via a learnable W_rel).
        self.w_rel = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.rel_emb.values():
            nn.init.xavier_uniform_(p.unsqueeze(0))

    def _is_forward(self, src_type: str, dst_type: str) -> bool:
        """Treat (project → company) as the forward direction.
        Inverse is (company → project)."""
        return src_type == "project" and dst_type == "company"

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
            if edge_weight_dict is not None:
                ew = edge_weight_dict.get(et)
                if ew is not None:
                    w = w * ew.to(x_src.dtype)

            # Composition: φ(h_u, h_r_eff). The paper transforms h_r
            # by a learnable W_rel each layer (h_r^(l+1) = W_rel · h_r^(l));
            # we apply that transform here as a stateless local op so
            # the per-layer W_rel still gets gradient signal while
            # rel_emb itself remains a clean learnable parameter
            # (no mid-forward state mutation).
            h_r = self.rel_emb[_et_key(et)].to(x_src.dtype)
            h_r_eff = self.w_rel(h_r)
            comp = _compose(x_src[row], h_r_eff, self.comp_op)    # (E, d)

            # Direction-specific transform W_O or W_I, then weighted scatter
            comp = self.w_out(comp) if self._is_forward(src_type, dst_type) \
                   else self.w_in(comp)
            msg = comp * w.unsqueeze(-1)
            idx = col.unsqueeze(-1).expand(-1, d)
            agg = torch.zeros(num_dst, d, device=x_src.device, dtype=x_src.dtype)
            agg.scatter_add_(0, idx, msg)

            out[dst_type] = out[dst_type] + agg

        out = {nt: F.elu(v) for nt, v in out.items()}
        if self.training and self.dropout > 0:
            out = {nt: F.dropout(v, p=self.dropout) for nt, v in out.items()}
        return out


class CompGCNEncoder(nn.Module):
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
        comp_op: str = "sub",
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
                CompGCNLayer(hidden_dim, metadata, self.node_types, dropout, comp_op)
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

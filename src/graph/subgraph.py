"""Reduced experiment graph: keep only the nodes that take part in the given relations.

The full graph has ~430k projects and ~390k companies, most of which have no observed
relation at all. `induced_subgraph` keeps the nodes that appear in at least one edge of
`node_relations` (training edges and held-out edges together), re-indexes them and keeps
every edge whose two endpoints survive. Node types and the bipartite invariant are
unchanged: only project-company edges exist before and after.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT


@dataclass(frozen=True)
class SubgraphResult:
    graph: HeteroData
    held_out: HeteroData
    project_index: np.ndarray  # new project index -> index in the full graph
    company_index: np.ndarray  # new company index -> index in the full graph


def _endpoints(data: HeteroData, relations: tuple[str, ...]) -> tuple[list, list]:
    projects, companies = [], []
    for rel in relations:
        key = (NODE_TYPE_PROJECT, rel, NODE_TYPE_COMPANY)
        if key in data.edge_types:
            ei = data[key].edge_index.numpy()
            projects.append(ei[0])
            companies.append(ei[1])
    return projects, companies


def _restrict(data: HeteroData, new_index: dict[str, torch.Tensor]) -> HeteroData:
    """Copy `data` onto the kept nodes (new_index: old index -> new index, -1 = dropped)."""
    out = HeteroData()
    for node_type in data.node_types:
        keep = new_index[node_type] >= 0
        n_old = int(new_index[node_type].numel())
        for attr, value in data[node_type].items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == n_old:
                out[node_type][attr] = value[keep]
        out[node_type].num_nodes = int(keep.sum())
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        ei = data[edge_type].edge_index
        new_src, new_dst = new_index[src][ei[0]], new_index[dst][ei[1]]
        keep = (new_src >= 0) & (new_dst >= 0)
        n_edges = int(ei.shape[1])
        out[edge_type].edge_index = torch.stack([new_src[keep], new_dst[keep]])
        for attr, value in data[edge_type].items():
            if attr != "edge_index" and torch.is_tensor(value) and value.shape[0] == n_edges:
                out[edge_type][attr] = value[keep]
    return out


def induced_subgraph(
    graph: HeteroData, held_out: HeteroData, node_relations: tuple[str, ...]
) -> SubgraphResult:
    """Keep the nodes with an edge in `node_relations` (training or held-out)."""
    p_tr, c_tr = _endpoints(graph, node_relations)
    p_ho, c_ho = _endpoints(held_out, node_relations)
    if not p_tr + p_ho:
        raise ValueError(f"no edges found for relations {node_relations}")
    kept = {
        NODE_TYPE_PROJECT: np.unique(np.concatenate(p_tr + p_ho)),
        NODE_TYPE_COMPANY: np.unique(np.concatenate(c_tr + c_ho)),
    }
    new_index = {}
    for node_type, idx in kept.items():
        mapping = torch.full((graph[node_type].num_nodes,), -1, dtype=torch.long)
        mapping[torch.from_numpy(idx)] = torch.arange(len(idx))
        new_index[node_type] = mapping
    return SubgraphResult(
        graph=_restrict(graph, new_index),
        held_out=_restrict(held_out, new_index),
        project_index=kept[NODE_TYPE_PROJECT],
        company_index=kept[NODE_TYPE_COMPANY],
    )

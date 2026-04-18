from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.loader import load_edges, load_nodes, stack_embeddings
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
    NodeMap,
)
from src.graph.split import split_held_out


REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


@dataclass
class BuildResult:
    graph: HeteroData
    held_out: HeteroData
    project_map: NodeMap
    company_map: NodeMap
    drop_counts: dict[str, int]


def _map_edges(
    df: pd.DataFrame, project_map: NodeMap, company_map: NodeMap
) -> tuple[np.ndarray, int]:
    """Map string ids to indices. Drop rows whose ids are not in either map.

    Returns (edge_index of shape (2, E), n_dropped). E is after dedup.
    """
    total = len(df)
    src = df["project_id"].map(project_map.id_to_idx)
    dst = df["company_id"].map(company_map.id_to_idx)
    mask = src.notna() & dst.notna()
    src = src[mask].to_numpy(dtype=np.int64)
    dst = dst[mask].to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.unique(edge_index, axis=1)
    return edge_index, total - int(mask.sum())


def _to_tensor(edge_index: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(edge_index, dtype=np.int64))


def _add_relation(
    data: HeteroData,
    edge_type: str,
    edge_index: np.ndarray,
    weight: float,
    add_reverse: bool,
    weights_as_attr: bool,
) -> None:
    """Add a (project, <edge_type>, company) relation, optionally with reverse."""
    ei = _to_tensor(edge_index)
    rel = (NODE_TYPE_PROJECT, edge_type, NODE_TYPE_COMPANY)
    data[rel].edge_index = ei
    if weights_as_attr:
        data[rel].edge_weight = torch.full(
            (ei.shape[1],), fill_value=float(weight), dtype=torch.float32
        )

    if add_reverse:
        rev_ei = ei.flip(0).contiguous()
        rev = (NODE_TYPE_COMPANY, f"rev_{edge_type}", NODE_TYPE_PROJECT)
        data[rev].edge_index = rev_ei
        if weights_as_attr:
            data[rev].edge_weight = torch.full(
                (rev_ei.shape[1],), fill_value=float(weight), dtype=torch.float32
            )


def build_graph(paths: dict, graph_cfg: dict) -> BuildResult:
    """Build the base bipartite heterogeneous graph (3 real edge types + held-out split).

    Returns a BuildResult with:
      - graph: PyG HeteroData (training edges only)
      - held_out: PyG HeteroData (held-out edges, same node set)
      - project_map, company_map: id <-> index mapping
    """
    id_cols = paths["id_columns"]
    edge_cols = paths["edge_columns"]

    project_df = load_nodes(paths["raw"]["projects"], id_cols["project"])
    company_df = load_nodes(paths["raw"]["companies"], id_cols["company"])

    project_map = NodeMap.from_ids(project_df[id_cols["project"]].tolist())
    company_map = NodeMap.from_ids(company_df[id_cols["company"]].tolist())

    project_x = torch.from_numpy(stack_embeddings(project_df))
    company_x = torch.from_numpy(stack_embeddings(company_df))

    edge_files = {
        EDGE_ROYALTY: paths["raw"]["edges_royalty"],
        EDGE_COMMERCIAL: paths["raw"]["edges_commercial"],
        EDGE_PERFORMANCE: paths["raw"]["edges_performance"],
    }

    all_edges: dict[str, np.ndarray] = {}
    drop_counts: dict[str, int] = {}
    for et in REAL_EDGE_TYPES:
        df = load_edges(
            edge_files[et],
            project_col=edge_cols[et]["project"],
            company_col=edge_cols[et]["company"],
        )
        edge_index, n_dropped = _map_edges(df, project_map, company_map)
        all_edges[et] = edge_index
        drop_counts[et] = n_dropped

    held_cfg = graph_cfg.get("held_out", {})
    apply_to = set(held_cfg.get("apply_to", []))
    ratio = float(held_cfg.get("ratio", 0.1))
    seed = int(held_cfg.get("seed", 42))
    do_split = bool(held_cfg.get("enabled", True))

    train_edges: dict[str, np.ndarray] = {}
    held_edges: dict[str, np.ndarray] = {}
    for et, ei in all_edges.items():
        if do_split and et in apply_to:
            tr, ho = split_held_out(ei, ratio=ratio, seed=seed)
        else:
            tr = ei
            ho = np.zeros((2, 0), dtype=np.int64)
        train_edges[et] = tr
        held_edges[et] = ho

    add_reverse = bool(graph_cfg.get("reverse_edges", {}).get("enabled", True))
    weights_as_attr = bool(graph_cfg.get("edge_weights_as_attr", True))

    graph = HeteroData()
    graph[NODE_TYPE_PROJECT].x = project_x
    graph[NODE_TYPE_COMPANY].x = company_x
    graph[NODE_TYPE_PROJECT].num_nodes = project_x.shape[0]
    graph[NODE_TYPE_COMPANY].num_nodes = company_x.shape[0]

    for et in REAL_EDGE_TYPES:
        weight = float(graph_cfg["edge_types"][et]["weight"])
        _add_relation(
            graph,
            edge_type=et,
            edge_index=train_edges[et],
            weight=weight,
            add_reverse=add_reverse,
            weights_as_attr=weights_as_attr,
        )

    held_out = HeteroData()
    held_out[NODE_TYPE_PROJECT].num_nodes = project_x.shape[0]
    held_out[NODE_TYPE_COMPANY].num_nodes = company_x.shape[0]
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        held_out[rel].edge_index = _to_tensor(held_edges[et])

    return BuildResult(
        graph=graph,
        held_out=held_out,
        project_map=project_map,
        company_map=company_map,
        drop_counts=drop_counts,
    )


def save_graph(
    result: BuildResult,
    graph_path: str | Path,
    held_out_path: str | Path,
    node_maps_path: str | Path | None = None,
) -> None:
    import pickle

    graph_path = Path(graph_path)
    held_out_path = Path(held_out_path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    held_out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(result.graph, graph_path)
    torch.save(result.held_out, held_out_path)

    if node_maps_path is None:
        node_maps_path = graph_path.with_name("node_maps.pkl")
    with open(node_maps_path, "wb") as f:
        pickle.dump(
            {
                "project_ids": result.project_map.idx_to_id,
                "company_ids": result.company_map.idx_to_id,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def summarize(result: BuildResult) -> str:
    g = result.graph
    ho = result.held_out
    lines = [
        f"nodes: project={g[NODE_TYPE_PROJECT].num_nodes:,}  company={g[NODE_TYPE_COMPANY].num_nodes:,}",
        f"project.x: {tuple(g[NODE_TYPE_PROJECT].x.shape)} {g[NODE_TYPE_PROJECT].x.dtype}",
        f"company.x: {tuple(g[NODE_TYPE_COMPANY].x.shape)} {g[NODE_TYPE_COMPANY].x.dtype}",
        "edges (train):",
    ]
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = g[rel].edge_index
        lines.append(f"  {et:12s}  E={ei.shape[1]:>10,}")
    lines.append("edges (held_out):")
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = ho[rel].edge_index
        lines.append(f"  {et:12s}  E={ei.shape[1]:>10,}")
    lines.append(f"pre-split drops (id not in node map): {result.drop_counts}")
    lines.append(f"edge types in graph: {list(g.edge_types)}")
    return "\n".join(lines)

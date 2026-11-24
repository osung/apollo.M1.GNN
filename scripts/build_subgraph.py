"""Entrypoint: build the reduced experiment graph (config/graph.yaml: `subgraph`).

Writes graph.pt, held_out.pt, node_maps.pkl and the full-graph index of every kept node to
`subgraph.output_dir`, together with paths.yaml / graph.yaml copies whose processed paths
point into that directory. Every other script then works on the reduced graph through
`--paths <dir>/paths.yaml --graph-cfg <dir>/graph.yaml` (scripts/run_gnn_recipe.sh: SUBGRAPH).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT
from src.graph.subgraph import induced_subgraph
from src.utils import load_yaml


def _relocate(value: str, out_dir: Path) -> str:
    return str(out_dir / Path(value).name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-cfg", default="config/graph.yaml")
    args = parser.parse_args()
    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph_cfg)
    cfg = graph_cfg["subgraph"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = Path(paths["processed"]["graph"])
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(graph_path.with_name("held_out.pt"), weights_only=False)
    with open(graph_path.with_name("node_maps.pkl"), "rb") as f:
        maps = pickle.load(f)

    result = induced_subgraph(graph, held_out, tuple(cfg["node_relations"]))
    torch.save(result.graph, out_dir / "graph.pt")
    torch.save(result.held_out, out_dir / "held_out.pt")
    with open(out_dir / "node_maps.pkl", "wb") as f:
        pickle.dump(
            {
                "project_ids": [maps["project_ids"][i] for i in result.project_index],
                "company_ids": [maps["company_ids"][i] for i in result.company_index],
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    np.savez(out_dir / "full_graph_index.npz",
             project=result.project_index, company=result.company_index)

    paths["processed"] = {k: _relocate(v, out_dir) for k, v in paths["processed"].items()}
    sim_cfg = graph_cfg.get("similarity_edges") or {}
    if sim_cfg.get("cache_path"):
        sim_cfg["cache_path"] = _relocate(sim_cfg["cache_path"], out_dir)
    for aux in (graph_cfg.get("auxiliary_edges") or {}).values():
        if isinstance(aux, dict) and aux.get("cache_path"):
            aux["cache_path"] = _relocate(aux["cache_path"], out_dir)
    for name, content in (("paths.yaml", paths), ("graph.yaml", graph_cfg)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False)

    print(f"[subgraph] node relations: {list(cfg['node_relations'])} -> {out_dir}")
    for node_type in (NODE_TYPE_PROJECT, NODE_TYPE_COMPANY):
        print(f"[subgraph] {node_type}: {graph[node_type].num_nodes:,} -> "
              f"{result.graph[node_type].num_nodes:,}")
    pairs = (("train", graph, result.graph), ("held-out", held_out, result.held_out))
    for name, full, sub in pairs:
        for edge_type in sub.edge_types:
            if edge_type[0] == NODE_TYPE_PROJECT:
                print(f"[subgraph] {name} {edge_type[1]}: {full[edge_type].edge_index.shape[1]:,}"
                      f" -> {sub[edge_type].edge_index.shape[1]:,}")


if __name__ == "__main__":
    main()

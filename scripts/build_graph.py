"""Entrypoint: load raw pickles and build data/processed/graph.pt + held_out.pt.

Wired by `make data`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.builder import build_graph, save_graph, summarize
from src.utils import load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph", default="config/graph.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph)

    if args.seed is not None:
        set_seed(args.seed)

    print("[build_graph] loading data and building base graph...")
    result = build_graph(paths, graph_cfg)

    graph_path = paths["processed"]["graph"]
    held_out_path = str(Path(graph_path).with_name("held_out.pt"))
    node_maps_path = str(Path(graph_path).with_name("node_maps.pkl"))
    save_graph(result, graph_path, held_out_path, node_maps_path)

    print(summarize(result))
    print(f"[build_graph] saved graph     -> {graph_path}")
    print(f"[build_graph] saved held_out  -> {held_out_path}")


if __name__ == "__main__":
    main()

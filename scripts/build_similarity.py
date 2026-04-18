"""Compute top-k similarity edges from norm_embed and cache to .npz.

Outputs (config/paths.yaml: processed.graph sibling, default sim_edges.npz):
    edge_index: (2, E) int64, oriented (project_idx, company_idx)
    topk, direction, drop_overlap, n_project, n_company  (metadata scalars)

Scale notes for this dataset (~800K per side, dim=768):
- IndexFlatIP, topk=100, single-threaded FAISS: ~1-3 hours on CPU.
- --approximate (IVF) cuts this to ~5-15 min with ~95%+ recall.
- --threads N raises FAISS thread count (default 1 for safety when torch
  is also in-process; multi-thread is usually safe in this script
  because message-passing torch ops are never called after graph load).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch

faiss.omp_set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)
from src.graph.similarity import build_similarity_edges
from src.utils import load_yaml

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


def _load_from_graph(graph_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"[sim] loading graph {graph_path}")
    g = torch.load(graph_path, weights_only=False)
    px = g[NODE_TYPE_PROJECT].x
    cx = g[NODE_TYPE_COMPANY].x
    if px.dtype != torch.float32:
        print(f"[sim] casting project.x {px.dtype} -> float32")
        px = px.float()
    if cx.dtype != torch.float32:
        print(f"[sim] casting company.x {cx.dtype} -> float32")
        cx = cx.float()
    project_x = np.ascontiguousarray(px.numpy())
    company_x = np.ascontiguousarray(cx.numpy())

    known_list = []
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        known_list.append(g[rel].edge_index.numpy())
    known = np.concatenate(known_list, axis=1) if known_list else np.zeros((2, 0), dtype=np.int64)
    del g
    return project_x, company_x, known


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-cfg", default="config/graph.yaml")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output", default=None,
                        help="override output npz path (default: sim_edges.npz next to graph)")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--direction", default=None, choices=["p2c", "c2p", "both"])
    parser.add_argument("--drop-overlap", dest="drop_overlap",
                        default=None, action="store_true")
    parser.add_argument("--no-drop-overlap", dest="drop_overlap",
                        action="store_false")
    parser.add_argument("--approximate", action="store_true",
                        help="use IVF index (fast, ~95%% recall)")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=None,
                        help="FAISS threads (default 1; raise for speed if safe)")
    args = parser.parse_args()

    if args.threads is not None:
        faiss.omp_set_num_threads(int(args.threads))

    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph_cfg)
    sim_cfg = graph_cfg.get("similarity_edges", {}) or {}

    graph_path = args.graph_path or paths["processed"]["graph"]
    topk = args.topk or int(sim_cfg.get("topk", 100))
    direction = args.direction or sim_cfg.get("direction", "both")
    drop_overlap = (
        args.drop_overlap if args.drop_overlap is not None
        else bool(sim_cfg.get("drop_overlap_with_known_edges", True))
    )
    output = Path(
        args.output
        or sim_cfg.get("cache_path")
        or str(Path(paths["processed"]["graph"]).with_name("sim_edges.npz"))
    )

    project_x, company_x, known_edges = _load_from_graph(graph_path)
    print(
        f"[sim] projects={project_x.shape[0]:,}  "
        f"companies={company_x.shape[0]:,}  dim={project_x.shape[1]}"
    )
    print(
        f"[sim] config: topk={topk}  direction={direction}  "
        f"drop_overlap={drop_overlap}  approximate={args.approximate}"
    )

    t0 = time.perf_counter()
    result = build_similarity_edges(
        project_x=project_x,
        company_x=company_x,
        topk=topk,
        direction=direction,
        known_edges=known_edges,
        drop_overlap=drop_overlap,
        approximate=args.approximate,
        batch=args.batch,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        edge_index=result.edge_index,
        topk=np.int64(topk),
        direction=np.array(direction, dtype="<U10"),
        drop_overlap=np.bool_(drop_overlap),
        n_project=np.int64(project_x.shape[0]),
        n_company=np.int64(company_x.shape[0]),
    )

    total = time.perf_counter() - t0
    size_mb = output.stat().st_size / 1e6
    print(
        f"[sim] raw={result.n_raw:,}  dedup={result.n_dedup:,}  "
        f"dropped_overlap={result.n_dropped_overlap:,}  final={result.n_final:,}"
    )
    print(f"[sim] total elapsed: {total:.1f}s")
    print(f"[sim] saved -> {output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

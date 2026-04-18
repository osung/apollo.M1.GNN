"""Evaluate the memory-based neighborhood CF baseline on held-out edges.

Usage: python scripts/baseline_cf.py
       (reads config/paths.yaml, config/graph.yaml, config/train.yaml)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

faiss.omp_set_num_threads(1)

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.neighborhood_cf import CFConfig, NeighborhoodCF
from src.eval.ranking import evaluate, group_ground_truth
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)
from src.utils import load_yaml

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


def _collect_edges(graph, edge_types) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for et in edge_types:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        out[et] = graph[rel].edge_index.numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-cfg", default="config/graph.yaml")
    parser.add_argument("--train-cfg", default="config/train.yaml")
    parser.add_argument("--k-neighbors", type=int, default=50)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="optional cap on number of eval queries for quick runs")
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph_cfg)
    train_cfg = load_yaml(args.train_cfg)

    graph_path = paths["processed"]["graph"]
    held_out_path = str(Path(graph_path).with_name("held_out.pt"))

    print(f"[cf] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    print(f"[cf] loading held_out {held_out_path}")
    held_out = torch.load(held_out_path, weights_only=False)

    project_x = graph[NODE_TYPE_PROJECT].x.numpy()
    company_x = graph[NODE_TYPE_COMPANY].x.numpy()
    train_edges = _collect_edges(graph, REAL_EDGE_TYPES)
    relation_weights = {
        et: float(graph_cfg["edge_types"][et]["weight"]) for et in REAL_EDGE_TYPES
    }

    cf = NeighborhoodCF(
        project_x=project_x,
        company_x=company_x,
        edges_per_relation=train_edges,
        relation_weights=relation_weights,
        config=CFConfig(k_neighbors=args.k_neighbors, topk=args.topk),
    )
    print(f"[cf] built CF: projects={project_x.shape[0]:,}, companies={company_x.shape[0]:,}")

    held_edges = _collect_edges(held_out, REAL_EDGE_TYPES)

    per_relation_metrics: dict[str, dict[str, float]] = {}
    for et in REAL_EDGE_TYPES:
        ei = held_edges[et]
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction="project_to_company")
        query_ids = list(gt.keys())
        if args.limit_queries:
            query_ids = query_ids[: args.limit_queries]
        print(f"[cf] evaluating {et}: {len(query_ids)} query projects")
        preds, _ = cf.recommend_companies(np.asarray(query_ids, dtype=np.int64))
        per_relation_metrics[et] = evaluate(preds, query_ids, gt, ks=(10, args.topk))

    print("\n=== Memory-based Neighborhood CF (project -> company) ===")
    for et, m in per_relation_metrics.items():
        print(f"  {et:12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()

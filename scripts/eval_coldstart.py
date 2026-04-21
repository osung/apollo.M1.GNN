"""Evaluate cold-start retrieval quality using projection MLPs.

Two regimes supported via --mode:

  1. `edge_holdout` (default): re-use the standard held-out edges from
     `held_out.pt`. Nodes remain warm. This is the canonical "warm"
     evaluation and matches `scripts/train_gnn.py` final eval.

  2. `node_holdout`: split a random subset of query-side nodes as cold.
     Their GNN z is replaced with projection_mlp(norm_embed). Retrieval
     then runs exactly like warm mode. This measures how well the MLP
     preserves retrieval quality for truly unseen nodes.

The point of having both modes in one script is to let us report the
quality gap (cold vs warm) for a single trained GNN + MLP pair.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ranking import evaluate, group_ground_truth
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)
from src.models.projection import ProjectionMLP
from src.utils import load_yaml

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


def _batched_topk(
    query_ids: np.ndarray,
    z_src: np.ndarray,
    z_dst: np.ndarray,
    topk: int,
    batch: int = 256,
) -> np.ndarray:
    Q = query_ids.shape[0]
    out = np.empty((Q, topk), dtype=np.int64)
    for start in range(0, Q, batch):
        end = min(start + batch, Q)
        ids = query_ids[start:end]
        scores = z_src[ids] @ z_dst.T
        part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def _load_mlp(ckpt_path: Path, device: str) -> ProjectionMLP:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ProjectionMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _project_cold(
    model: ProjectionMLP,
    norm_embed: np.ndarray,
    indices: np.ndarray,
    device: str,
    batch: int = 4096,
) -> np.ndarray:
    """Run projection MLP on the selected cold-node features, returning
    predicted z aligned to `indices`."""
    out = np.empty((indices.shape[0], model.output_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, indices.shape[0], batch):
            e = min(s + batch, indices.shape[0])
            x = torch.from_numpy(norm_embed[indices[s:e]]).to(device)
            pred = model(x).detach().cpu().numpy()
            out[s:e] = pred
    return out


def _evaluate_edge_holdout(
    z_p: np.ndarray,
    z_c: np.ndarray,
    held_out,
    direction: str,
    topk: int,
) -> dict[str, dict[str, float]]:
    """Standard warm evaluation — identical to train_gnn.py final eval."""
    z_query, z_cand = (z_p, z_c) if direction == "p2c" else (z_c, z_p)
    gt_dir = "project_to_company" if direction == "p2c" else "company_to_project"
    results: dict[str, dict[str, float]] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = held_out[rel].edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction=gt_dir)
        qids = np.asarray(list(gt.keys()), dtype=np.int64)
        preds = _batched_topk(qids, z_query, z_cand, topk=topk)
        results[et] = evaluate(preds, qids.tolist(), gt, ks=(10, topk))
    return results


def _evaluate_node_holdout(
    z_p_warm: np.ndarray,
    z_c_warm: np.ndarray,
    norm_p: np.ndarray,
    norm_c: np.ndarray,
    held_out,
    cold_proj: np.ndarray,
    cold_comp: np.ndarray,
    mlp_proj: ProjectionMLP | None,
    mlp_comp: ProjectionMLP | None,
    direction: str,
    topk: int,
    device: str,
) -> dict[str, dict[str, float]]:
    """Cold evaluation: replace z for held-out nodes with MLP projections.

    Query-side is the "cold" side — those queries' z is from MLP; all
    candidates keep their warm GNN z.
    """
    z_p = z_p_warm.copy()
    z_c = z_c_warm.copy()
    if mlp_proj is not None and cold_proj.size > 0:
        z_p[cold_proj] = _project_cold(mlp_proj, norm_p, cold_proj, device)
    if mlp_comp is not None and cold_comp.size > 0:
        z_c[cold_comp] = _project_cold(mlp_comp, norm_c, cold_comp, device)

    z_query, z_cand = (z_p, z_c) if direction == "p2c" else (z_c, z_p)
    gt_dir = "project_to_company" if direction == "p2c" else "company_to_project"
    cold_set = set(cold_proj.tolist()) if direction == "p2c" else set(cold_comp.tolist())

    results: dict[str, dict[str, float]] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = held_out[rel].edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction=gt_dir)
        # Restrict to queries that are in the cold set so we measure
        # cold-only retrieval. (Edges where both endpoints are warm
        # would just replicate the warm evaluation.)
        qids = np.asarray(
            [q for q in gt.keys() if int(q) in cold_set], dtype=np.int64
        )
        if qids.size == 0:
            continue
        preds = _batched_topk(qids, z_query, z_cand, topk=topk)
        results[et] = evaluate(preds, qids.tolist(), gt, ks=(10, topk))
    return results


def _print_metrics(label: str, m: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {label} ===")
    if not m:
        print("  (no queries fell into this segment)")
        return
    for et, mm in m.items():
        print(f"  {et:12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in mm.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument(
        "--graph-path", default=None,
        help="override paths.yaml processed.graph",
    )
    parser.add_argument("--project-z", default=None)
    parser.add_argument("--company-z", default=None)
    parser.add_argument(
        "--mlp-project", default=None,
        help="projection_mlp_project.pt path",
    )
    parser.add_argument(
        "--mlp-company", default=None,
        help="projection_mlp_company.pt path",
    )
    parser.add_argument(
        "--mode", default="edge_holdout",
        choices=["edge_holdout", "node_holdout"],
    )
    parser.add_argument(
        "--cold-fraction", type=float, default=0.1,
        help="node_holdout mode: fraction of nodes to treat as cold",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--direction", default="both", choices=["p2c", "c2p", "both"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    graph_path = args.graph_path or paths["processed"]["graph"]
    held_out_path = str(Path(graph_path).with_name("held_out.pt"))
    print(f"[coldstart-eval] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(held_out_path, weights_only=False)

    norm_p = graph[NODE_TYPE_PROJECT].x.float().numpy()
    norm_c = graph[NODE_TYPE_COMPANY].x.float().numpy()

    pz = args.project_z or paths["processed"]["project_emb"]
    cz = args.company_z or paths["processed"]["company_emb"]
    print(f"[coldstart-eval] loading z: {pz}")
    z_p = np.load(pz).astype(np.float32)
    print(f"[coldstart-eval] loading z: {cz}")
    z_c = np.load(cz).astype(np.float32)

    directions = ["p2c", "c2p"] if args.direction == "both" else [args.direction]

    if args.mode == "edge_holdout":
        for d in directions:
            m = _evaluate_edge_holdout(z_p, z_c, held_out, d, args.topk)
            _print_metrics(f"edge_holdout [{d}] K={args.topk}", m)
        return

    # ---- node_holdout ----
    mlp_proj = None
    mlp_comp = None
    if args.mlp_project:
        print(f"[coldstart-eval] loading project MLP {args.mlp_project}")
        mlp_proj = _load_mlp(Path(args.mlp_project), args.device)
    if args.mlp_company:
        print(f"[coldstart-eval] loading company MLP {args.mlp_company}")
        mlp_comp = _load_mlp(Path(args.mlp_company), args.device)

    if mlp_proj is None and mlp_comp is None:
        raise SystemExit(
            "node_holdout mode needs at least one of --mlp-project / --mlp-company"
        )

    rng = np.random.default_rng(args.seed)
    n_p = z_p.shape[0]
    n_c = z_c.shape[0]
    cold_proj = (
        rng.choice(n_p, size=int(args.cold_fraction * n_p), replace=False)
        if mlp_proj is not None else np.zeros(0, dtype=np.int64)
    )
    cold_comp = (
        rng.choice(n_c, size=int(args.cold_fraction * n_c), replace=False)
        if mlp_comp is not None else np.zeros(0, dtype=np.int64)
    )
    print(
        f"[coldstart-eval] cold sets: projects={cold_proj.size:,} / {n_p:,}  "
        f"companies={cold_comp.size:,} / {n_c:,}"
    )

    # Cold-only evaluation
    for d in directions:
        m_cold = _evaluate_node_holdout(
            z_p, z_c, norm_p, norm_c, held_out,
            cold_proj, cold_comp,
            mlp_proj, mlp_comp,
            d, args.topk, args.device,
        )
        _print_metrics(f"node_holdout cold-only [{d}] K={args.topk}", m_cold)

    # Reference warm-all for the same held-out edges
    for d in directions:
        m_warm = _evaluate_edge_holdout(z_p, z_c, held_out, d, args.topk)
        _print_metrics(f"reference warm-all [{d}] K={args.topk}", m_warm)


if __name__ == "__main__":
    main()

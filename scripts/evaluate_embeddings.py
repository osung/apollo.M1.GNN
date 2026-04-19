"""Evaluate saved z embeddings (.npy) against held_out.pt.

Use this to check convergence of a training run after downloading the
per-epoch z snapshots without re-running the model. Works for any pair
of (project, company) embedding arrays — GNN checkpoints, baseline
outputs, or hand-built ones — as long as they're L2-normalized dense
float arrays.

Examples:
    # Single epoch
    python scripts/evaluate_embeddings.py \\
        --project-z data/processed/checkpoints/project_z_sage_h64_l2_epoch005.npy \\
        --company-z data/processed/checkpoints/company_z_sage_h64_l2_epoch005.npy

    # Sweep all epochs in a checkpoint dir
    python scripts/evaluate_embeddings.py --sweep-dir data/processed/checkpoints
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ranking import evaluate, group_ground_truth
from src.graph.schema import (
    EDGE_COMMERCIAL, EDGE_PERFORMANCE, EDGE_ROYALTY,
    NODE_TYPE_COMPANY, NODE_TYPE_PROJECT,
)
from src.utils import load_yaml

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


def _batched_topk(query_ids, z_src, z_dst, topk, bs=256):
    Q = query_ids.shape[0]
    out = np.empty((Q, topk), dtype=np.int64)
    for start in range(0, Q, bs):
        end = min(start + bs, Q)
        ids = query_ids[start:end]
        scores = z_src[ids] @ z_dst.T
        part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def _evaluate(z_p, z_c, held_out, topk, direction):
    if direction == "p2c":
        z_q, z_cand = z_p, z_c
        gt_direction = "project_to_company"
    else:
        z_q, z_cand = z_c, z_p
        gt_direction = "company_to_project"

    out = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = held_out[rel].edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction=gt_direction)
        query_ids = list(gt.keys())
        preds = _batched_topk(
            np.asarray(query_ids, dtype=np.int64), z_q, z_cand, topk=topk,
        )
        out[et] = evaluate(preds, query_ids, gt, ks=(10, topk))
    return out


def _print_row(tag: str, metrics: dict, topk: int):
    for et in REAL_EDGE_TYPES:
        m = metrics.get(et)
        if not m:
            continue
        print(f"  {tag}  {et:11s}  "
              f"R@10={m['recall@10']:.4f}  N@10={m['ndcg@10']:.4f}  "
              f"R@{topk}={m[f'recall@{topk}']:.4f}  N@{topk}={m[f'ndcg@{topk}']:.4f}")


def _load_z(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _load_held_out(paths_yaml: str):
    paths = load_yaml(paths_yaml)
    held_out_path = str(Path(paths["processed"]["graph"]).with_name("held_out.pt"))
    return torch.load(held_out_path, weights_only=False)


def _sweep_pairs(ckpt_dir: Path) -> list[tuple[int, Path, Path]]:
    """Find matching (project_z, company_z) .npy pairs per epoch in a directory."""
    pattern = re.compile(r"(project|company)_z_(.+)_epoch(\d+)\.npy$")
    buckets: dict[tuple[str, int], dict[str, Path]] = {}
    for p in ckpt_dir.glob("*.npy"):
        m = pattern.match(p.name)
        if not m:
            continue
        side, tag, ep = m.group(1), m.group(2), int(m.group(3))
        buckets.setdefault((tag, ep), {})[side] = p
    pairs = []
    for (tag, ep), d in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        if "project" in d and "company" in d:
            pairs.append((ep, d["project"], d["company"]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--project-z", default=None,
                        help="path to project_z_*.npy (single-file mode)")
    parser.add_argument("--company-z", default=None,
                        help="path to company_z_*.npy (single-file mode)")
    parser.add_argument("--sweep-dir", default=None,
                        help="directory of *_epoch###.npy pairs to sweep")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--direction", default="both", choices=["p2c", "c2p", "both"])
    parser.add_argument("--output", default=None, help="optional JSON summary path")
    args = parser.parse_args()

    held_out = _load_held_out(args.paths)

    work = []
    if args.sweep_dir:
        for ep, pz, cz in _sweep_pairs(Path(args.sweep_dir)):
            work.append((f"epoch{ep:03d}", pz, cz))
    else:
        if not (args.project_z and args.company_z):
            parser.error("supply --project-z and --company-z, or --sweep-dir")
        work.append(("single", Path(args.project_z), Path(args.company_z)))

    summary: dict[str, dict] = {}
    directions = ["p2c", "c2p"] if args.direction == "both" else [args.direction]

    for tag, pz_path, cz_path in work:
        print(f"\n[eval] {tag}: {pz_path.name}  +  {cz_path.name}")
        z_p = _load_z(pz_path)
        z_c = _load_z(cz_path)
        per_tag = {}
        for d in directions:
            metrics = _evaluate(z_p, z_c, held_out, topk=args.topk, direction=d)
            print(f"  [{d}]")
            _print_row(d, metrics, args.topk)
            per_tag[d] = metrics
        summary[tag] = per_tag

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[eval] wrote summary to {args.output}")


if __name__ == "__main__":
    main()

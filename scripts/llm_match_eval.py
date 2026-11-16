"""Step 3 of the embedding + LLM matching baseline: metrics on a common query set.

Compares, on exactly the same held-out queries (those fully scored by the LLM):
  - pro-sroberta        : top-N by norm_embed similarity
  - pro-sroberta+LLM    : the same N candidates re-ranked by LLM score
  - any saved GNN z     : --z-tag lightgcn_h128_l2 --z-tag hgt_h128_l2_nh4 ...

    python scripts/llm_match_eval.py --direction p2c --z-tag lightgcn_h128_l2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.llm_match import GT_DIRECTION, load_scores, rerank, topn_candidates
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--config", default="config/llm_match.yaml")
    parser.add_argument("--direction", default="p2c", choices=["p2c", "c2p"])
    parser.add_argument("--z-tag", action="append", default=[],
                        help="experiment tag of saved project_z_<tag>.npy / company_z_<tag>.npy")
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.config)
    ks = tuple(int(k) for k in cfg["eval"]["ks"])
    out_dir = Path(cfg["output_dir"])
    d = args.direction

    cand = np.load(out_dir / f"candidates_{d}.npz")
    query_idx, cand_idx, base = cand["query_idx"], cand["cand_idx"], cand["base_score"]
    n = cand_idx.shape[1]
    scores = load_scores(sorted(out_dir.glob(f"scores_{d}_shard*.jsonl"))).get(d, {})

    # keep only queries whose N candidates were all sent to the LLM
    rows = [i for i, q in enumerate(query_idx.tolist()) if len(scores.get(q, {})) >= n]
    if not rows:
        raise SystemExit(f"no fully scored queries for direction {d} in {out_dir}")
    print(f"[llm-eval] {d}: {len(rows):,} of {len(query_idx):,} prepared queries fully scored "
          f"(top-{n} candidates each)")
    q_sel = query_idx[rows]
    preds: dict[str, np.ndarray] = {"pro-sroberta": cand_idx[rows]}
    reranked = np.empty((len(rows), n), dtype=np.int64)
    for r, i in enumerate(rows):
        s = scores[int(query_idx[i])]
        llm = np.asarray([s[int(c)] for c in cand_idx[i]], dtype=np.float32)
        reranked[r] = rerank(cand_idx[i], base[i], llm)
    preds["pro-sroberta+LLM"] = reranked

    z_p_path, z_c_path = Path(paths["processed"]["project_emb"]), Path(paths["processed"]["company_emb"])
    for tag in args.z_tag:
        z_p = np.load(z_p_path.with_name(f"{z_p_path.stem}_{tag}.npy"))
        z_c = np.load(z_c_path.with_name(f"{z_c_path.stem}_{tag}.npy"))
        z_q, z_cand = (z_p, z_c) if d == "p2c" else (z_c, z_p)
        preds[tag], _ = topn_candidates(z_q[q_sel], z_cand, max(ks))

    held_out = torch.load(Path(paths["processed"]["graph"]).with_name("held_out.pt"),
                          weights_only=False)
    q_pos = {int(q): r for r, q in enumerate(q_sel.tolist())}
    results: dict[str, dict[str, dict[str, float]]] = {}
    n_queries: dict[str, int] = {}
    for et in REAL_EDGE_TYPES:
        ei = held_out[(NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)].edge_index.numpy()
        gt = group_ground_truth(ei, direction=GT_DIRECTION[d])
        qs = [q for q in gt if q in q_pos]
        n_queries[et] = len(qs)
        if not qs:
            continue
        sel = [q_pos[q] for q in qs]
        results[et] = {name: evaluate(p[sel], qs, gt, ks=ks) for name, p in preds.items()}

    for et, per_method in results.items():
        print(f"\n=== {et} [{d}]  queries={n_queries[et]:,} ===")
        for name, m in per_method.items():
            print(f"  {name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    miss = sum(1 for q in q_sel.tolist() for v in scores[int(q)].values() if v == 0)
    out_path = out_dir / f"metrics_{d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"direction": d, "topn": int(n), "n_queries": n_queries,
                   "llm_model": cfg["llm"]["model"],
                   "promise_weight": cfg["candidates"]["promise_weight"],
                   "zero_score_pairs": miss, "metrics": results}, f, ensure_ascii=False, indent=1)
    print(f"\n[llm-eval] saved {out_path}")


if __name__ == "__main__":
    main()

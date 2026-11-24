"""Step 3 of the retrieval + LLM matching comparison: metrics on a common query set.

For every retriever with a candidates_<direction>_<name>.npz file (bge-m3,
LightGCN, HGT, LightFM, ...) reports, on exactly the same held-out queries
(those whose candidates were all scored by the LLM):
  - <name>       : the retriever's own top-N order
  - <name>+LLM   : the same N candidates re-ranked by LLM score

    python scripts/llm_match_eval.py --direction p2c
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

from src.baselines.llm_match import (
    GT_DIRECTION,
    NAME_MASK,
    load_candidates,
    load_scores,
    read_jsonl,
    rerank,
)
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
    parser.add_argument("--topn", type=int, default=None,
                        help="LLM re-ranking depth: re-rank each retriever's top-N "
                             "(default candidates.topn; deeper ones are in candidates.llm_depths)")
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.config)
    ks = tuple(int(k) for k in cfg["eval"]["ks"])
    out_dir = Path(cfg["output_dir"])
    d = args.direction
    base_topn = int(cfg["candidates"]["topn"])
    depth = args.topn or base_topn

    cands = load_candidates(out_dir, d)
    if not cands:
        raise SystemExit(f"no candidates_{d}_*.npz in {out_dir}; run llm_match_prepare.py first")
    query_idx = next(iter(cands.values()))["query_idx"]
    cands = {k: c for k, c in cands.items() if (c["query_idx"] == query_idx).all()}
    scores = load_scores(sorted(out_dir.glob(f"scores_{d}_shard*.jsonl"))).get(d, {})

    rerank_names = [n for n in (cfg.get("llm_rerank") or list(cands)) if n in cands]
    shallow = [n for n in rerank_names if cands[n]["cand_idx"].shape[1] < depth]
    if shallow:
        raise SystemExit(f"--topn {depth} exceeds the saved candidates of {shallow}")

    # keep only queries for which every LLM-re-ranked retriever's candidates were all scored
    def scored(i: int) -> bool:
        s = scores.get(int(query_idx[i]), {})
        return all(int(x) in s for n in rerank_names
                   for x in cands[n]["cand_idx"][i, :depth])

    rows = [i for i in range(len(query_idx)) if scored(i)]
    if not rows:
        raise SystemExit(f"no fully scored queries for direction {d} in {out_dir}")
    print(f"[llm-eval] {d}: {len(rows):,} of {len(query_idx):,} prepared queries fully scored "
          f"to depth {depth}; retrievers: {', '.join(cands)}; LLM re-ranks: {', '.join(rerank_names)}")
    q_sel = query_idx[rows]
    preds: dict[str, np.ndarray] = {}
    for name, c in cands.items():
        cand_idx, base = c["cand_idx"], c["base_score"]
        preds[name] = cand_idx[rows]
        if name not in rerank_names:
            continue
        reranked = np.empty_like(preds[name][:, :depth])
        for r, i in enumerate(rows):
            s = scores[int(query_idx[i])]
            llm = np.asarray([s[int(x)] for x in cand_idx[i, :depth]], dtype=np.float32)
            reranked[r] = rerank(cand_idx[i, :depth], base[i, :depth], llm)
        preds[f"{name}+LLM"] = reranked

    def ks_for(name: str) -> tuple[int, ...]:
        # the re-ranked retrievers also report @depth: the share of queries whose held-out
        # partner is among the candidates the LLM saw (the ceiling of re-ranking)
        base_name = name.removesuffix("+LLM")
        return tuple(sorted({*ks, depth})) if base_name in rerank_names else ks

    held_out = torch.load(Path(paths["processed"]["graph"]).with_name("held_out.pt"),
                          weights_only=False)
    q_pos = {int(q): r for r, q in enumerate(q_sel.tolist())}
    results: dict[str, dict[str, dict[str, float]]] = {}
    n_queries: dict[str, int] = {}
    gt_all: dict[int, set[int]] = {}
    for et in REAL_EDGE_TYPES:
        ei = held_out[(NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)].edge_index.numpy()
        gt = group_ground_truth(ei, direction=GT_DIRECTION[d])
        qs = [q for q in gt if q in q_pos]
        n_queries[et] = len(qs)
        if not qs:
            continue
        sel = [q_pos[q] for q in qs]
        results[et] = {name: evaluate(p[sel], qs, gt, ks=ks_for(name))
                       for name, p in preds.items()}
        for q, items in gt.items():
            gt_all.setdefault(q, set()).update(items)
    # all relations together: a candidate is relevant if linked by any held-out relation
    n_queries["all"] = len(q_pos)
    qs_all = [q for q in q_sel.tolist() if q in gt_all]
    results["all"] = {name: evaluate(p[[q_pos[q] for q in qs_all]], qs_all, gt_all,
                                     ks=ks_for(name))
                      for name, p in preds.items()}

    # Split by whether the query project's document named a linked company (masked as
    # NAME_MASK in the prompts). The node-feature embedding was built from the unmasked
    # description, so name matching inflates every retriever on the first group.
    jobs_path = out_dir / f"jobs_{d}.jsonl"
    if d == "p2c" and jobs_path.exists():
        named = {int(j["query"]) for j in read_jsonl(jobs_path) if NAME_MASK in j["src_text"]}
        for label, keep in (("name-in-desc", True), ("no-name", False)):
            qs = [q for q in qs_all if (q in named) == keep]
            n_queries[label] = len(qs)
            if qs:
                results[label] = {
                    name: evaluate(p[[q_pos[q] for q in qs]], qs, gt_all, ks=ks_for(name))
                    for name, p in preds.items()
                }

    for et, per_method in results.items():
        print(f"\n=== {et} [{d}]  queries={n_queries[et]:,} ===")
        for name, m in per_method.items():
            print(f"  {name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    miss = sum(1 for q in q_sel.tolist() for v in scores[int(q)].values() if v == 0)
    out_path = out_dir / (f"metrics_{d}.json" if depth == base_topn
                          else f"metrics_{d}_top{depth}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"direction": d, "topn": depth, "n_queries": n_queries,
                   "llm_rerank": rerank_names,
                   "llm_model": cfg["llm"]["model"],
                   "promise_weight": cfg["candidates"]["promise_weight"],
                   "zero_score_pairs": miss, "metrics": results}, f, ensure_ascii=False, indent=1)
    print(f"\n[llm-eval] saved {out_path}")


if __name__ == "__main__":
    main()

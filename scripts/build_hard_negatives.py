"""Hard-negative mining from sim_edges_top-100.

Concept
-------
Among content-similar (norm_embed) candidates, pick the ones that the
other modalities (keyword list, industry code, name) agree are actually
unrelated. These are safer hard negatives than "content-similar and
simply unobserved", because we have positive evidence (not just absence
of observation) that they don't belong together.

Pipeline per query project `p`:
    1. Collect candidate companies from sim_edges_top-100 that pair with `p`.
    2. Drop known positives (including held-out ones).
    3. Score each remaining candidate by:
           unrelatedness(p, c) = 1
             - w_keyword  * Jaccard(kw_p, kw_c)
             - w_industry * 1{kw_p ∩ tokens(industry_c) != ∅}
             - w_name     * Jaccard(char_bigrams(name_p), char_bigrams(name_c))
    4. Take top-`topk` highest-unrelatedness pairs.

Output
------
npz with fields:
    edge_index: (2, E) int32, orientation (project_idx, company_idx)
    topk, w_kw, w_ind, w_name: scalar metadata
    n_projects, n_companies: graph dimensions
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT
from src.utils import load_yaml

REAL_RELATIONS = ("royalty", "commercial", "performance")


def _to_set(x) -> frozenset:
    if x is None:
        return frozenset()
    if isinstance(x, float) and pd.isna(x):
        return frozenset()
    try:
        return frozenset(str(t).strip() for t in x if t is not None and str(t).strip())
    except TypeError:
        s = str(x).strip()
        return frozenset({s}) if s else frozenset()


def _char_ngrams(s, n: int = 2) -> frozenset:
    if not isinstance(s, str):
        return frozenset()
    cleaned = re.sub(r"[^가-힣a-zA-Z0-9]", "", s)
    if len(cleaned) < n:
        return frozenset()
    return frozenset(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def _industry_tokens(s) -> frozenset:
    if not isinstance(s, str):
        return frozenset()
    return frozenset(
        t.strip() for t in re.split(r"[\s,/()·\-]+", s)
        if t and len(t.strip()) >= 2
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def _build_feature_lookup(df, idx_col: str, node_ids: list, columns: list[str]) -> dict:
    """Return dict[col_name -> list indexed by graph-index]."""
    tables = {}
    df_indexed = df if df.index.name == idx_col else df.set_index(idx_col)
    dicts = {col: df_indexed[col].to_dict() for col in columns}
    for col in columns:
        d = dicts[col]
        tables[col] = [d.get(nid) for nid in node_ids]
    return tables


def mine_hard_negatives(
    sim_ei: np.ndarray,
    n_projects: int,
    n_companies: int,
    project_kws: list,
    company_kws: list,
    project_ngrams: list,
    company_ngrams: list,
    company_industry: list,
    positive_map: dict[int, set[int]],
    topk: int,
    w_kw: float,
    w_ind: float,
    w_name: float,
) -> tuple[np.ndarray, dict]:
    # Group candidates by source project
    print(f"[hardneg] grouping {sim_ei.shape[1]:,} sim edges by project ...")
    per_project: dict[int, list[int]] = defaultdict(list)
    src = sim_ei[0]
    dst = sim_ei[1]
    for p, c in zip(src.tolist(), dst.tolist()):
        per_project[int(p)].append(int(c))

    candidate_counts = [len(v) for v in per_project.values()]
    print(
        f"[hardneg] {len(per_project):,} projects with candidates  "
        f"(avg {np.mean(candidate_counts):.1f}, median {int(np.median(candidate_counts))}, "
        f"max {max(candidate_counts)})"
    )

    hard_src = []
    hard_dst = []
    t0 = time.perf_counter()
    n_processed = 0
    n_skipped_nocand = 0
    n_skipped_allpos = 0

    for p_idx in range(n_projects):
        cand = per_project.get(p_idx)
        if not cand:
            n_skipped_nocand += 1
            continue
        pos = positive_map.get(p_idx, set())
        cand = [c for c in set(cand) if c not in pos]
        if not cand:
            n_skipped_allpos += 1
            continue

        kw_p = project_kws[p_idx]
        ng_p = project_ngrams[p_idx]

        scores = np.empty(len(cand), dtype=np.float32)
        for i, c in enumerate(cand):
            j_kw = _jaccard(kw_p, company_kws[c])
            ind_c = company_industry[c]
            ind_hit = 1.0 if (kw_p and ind_c and (kw_p & ind_c)) else 0.0
            j_name = _jaccard(ng_p, company_ngrams[c])
            scores[i] = 1.0 - w_kw * j_kw - w_ind * ind_hit - w_name * j_name

        k = min(topk, len(cand))
        top_idx = np.argpartition(-scores, kth=k - 1)[:k] if k < len(cand) else np.arange(len(cand))
        # Sort top-k desc by score
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        for idx in top_idx:
            hard_src.append(p_idx)
            hard_dst.append(cand[idx])

        n_processed += 1
        if n_processed % 20_000 == 0:
            elapsed = time.perf_counter() - t0
            rate = n_processed / elapsed
            eta = (n_projects - n_processed) / max(rate, 1e-9)
            print(
                f"[hardneg] {n_processed:,}/{n_projects:,} "
                f"({n_processed/n_projects*100:.1f}%)  "
                f"{rate:.0f}/s  ETA {eta:.0f}s"
            )

    edge_index = np.stack(
        [
            np.array(hard_src, dtype=np.int32),
            np.array(hard_dst, dtype=np.int32),
        ],
        axis=0,
    )
    stats = {
        "n_pairs": int(edge_index.shape[1]),
        "n_projects_processed": n_processed,
        "n_projects_no_candidates": n_skipped_nocand,
        "n_projects_all_positive": n_skipped_allpos,
        "elapsed_s": time.perf_counter() - t0,
    }
    return edge_index, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--sim-path", default="data/processed/sim_edges_top100.npz")
    parser.add_argument("--output", default="data/processed/hard_negatives_p2c.npz")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--w-keyword", type=float, default=0.5)
    parser.add_argument("--w-industry", type=float, default=0.3)
    parser.add_argument("--w-name", type=float, default=0.2)
    parser.add_argument("--graph-path", default=None)
    args = parser.parse_args()

    paths = load_yaml(args.paths)

    # 1. sim edges
    print(f"[hardneg] loading sim edges: {args.sim_path}")
    z = np.load(args.sim_path)
    sim_ei = z["edge_index"].astype(np.int64)
    n_p_sim = int(z["n_project"])
    n_c_sim = int(z["n_company"])
    print(f"  sim pairs: {sim_ei.shape[1]:,}")

    # 2. node maps
    graph_path = args.graph_path or paths["processed"]["graph"]
    nm_path = Path(graph_path).with_name("node_maps.pkl")
    print(f"[hardneg] loading node maps: {nm_path}")
    with open(nm_path, "rb") as f:
        nm = pickle.load(f)
    project_ids = nm["project_ids"]
    company_ids = nm["company_ids"]
    N_p = len(project_ids)
    N_c = len(company_ids)
    if (N_p, N_c) != (n_p_sim, n_c_sim):
        raise RuntimeError(
            f"graph/sim dim mismatch: graph=({N_p},{N_c}) vs sim=({n_p_sim},{n_c_sim})"
        )

    # 3. raw project/company metadata
    proj_id_col = paths["id_columns"]["project"]
    comp_id_col = paths["id_columns"]["company"]
    print(f"[hardneg] loading projects pkl  -> {paths['raw']['projects']}")
    proj_df = pd.read_pickle(paths["raw"]["projects"])
    print(f"[hardneg] loading companies pkl -> {paths['raw']['companies']}")
    comp_df = pd.read_pickle(paths["raw"]["companies"])

    print(f"[hardneg] building per-node feature lookups ...")
    t_feat = time.perf_counter()
    p_kw_dict = dict(zip(proj_df[proj_id_col], proj_df["키워드_리스트"]))
    p_name_dict = dict(zip(proj_df[proj_id_col], proj_df["과제명"]))
    c_kw_dict = dict(zip(comp_df[comp_id_col], comp_df["키워드_리스트"]))
    c_name_dict = dict(zip(comp_df[comp_id_col], comp_df["한글업체명"]))
    c_ind_dict = dict(zip(comp_df[comp_id_col], comp_df["10차산업코드명"]))

    project_kws = [_to_set(p_kw_dict.get(pid)) for pid in project_ids]
    project_ngrams = [_char_ngrams(p_name_dict.get(pid)) for pid in project_ids]
    company_kws = [_to_set(c_kw_dict.get(cid)) for cid in company_ids]
    company_ngrams = [_char_ngrams(c_name_dict.get(cid)) for cid in company_ids]
    company_industry = [_industry_tokens(c_ind_dict.get(cid)) for cid in company_ids]
    print(f"[hardneg]   feature tables built in {time.perf_counter() - t_feat:.1f}s")

    # 4. positive filter (train + held-out)
    print(f"[hardneg] loading graph for positive filter: {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    positive_map: dict[int, set[int]] = defaultdict(set)
    for et in REAL_RELATIONS:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = graph[rel].edge_index.numpy()
        for p, c in zip(ei[0].tolist(), ei[1].tolist()):
            positive_map[int(p)].add(int(c))
    held_out_path = Path(graph_path).with_name("held_out.pt")
    if held_out_path.exists():
        ho = torch.load(held_out_path, weights_only=False)
        for et in REAL_RELATIONS:
            rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
            ei = ho[rel].edge_index.numpy()
            for p, c in zip(ei[0].tolist(), ei[1].tolist()):
                positive_map[int(p)].add(int(c))
    print(f"[hardneg]   {sum(len(v) for v in positive_map.values()):,} positives across "
          f"{len(positive_map):,} projects")

    # 5. mine
    edge_index, stats = mine_hard_negatives(
        sim_ei=sim_ei,
        n_projects=N_p,
        n_companies=N_c,
        project_kws=project_kws,
        company_kws=company_kws,
        project_ngrams=project_ngrams,
        company_ngrams=company_ngrams,
        company_industry=company_industry,
        positive_map=positive_map,
        topk=args.topk,
        w_kw=args.w_keyword,
        w_ind=args.w_industry,
        w_name=args.w_name,
    )

    # 6. save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        edge_index=edge_index,
        topk=np.int64(args.topk),
        w_kw=np.float32(args.w_keyword),
        w_ind=np.float32(args.w_industry),
        w_name=np.float32(args.w_name),
        n_projects=np.int64(N_p),
        n_companies=np.int64(N_c),
    )
    size_mb = out.stat().st_size / 1e6
    print()
    print(f"[hardneg] saved -> {out}  ({size_mb:.1f} MB)")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

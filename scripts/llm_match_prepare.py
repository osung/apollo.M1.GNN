"""Step 1 of the embedding + LLM matching baseline: candidates and scoring jobs.

For the sampled held-out queries, retrieve top-N opposite-type nodes by
node-feature embedding (`embedding` in paths.yaml, e.g. bge-m3) similarity and write the LLM scoring jobs.

Candidate sources ("retrievers"), each giving its own top-N per query:
  <embedding.name>      node-feature similarity (always), e.g. bge-m3
  config `retrievers`   saved GNN embeddings, e.g. lightgcn: lightgcn_h128_l2
  external files        candidates_<direction>_<name>.npz already in output_dir
                        (e.g. LightFM: scripts/baseline_lightfm.py --save-candidates)

Every (query, candidate) pair in the union is scored by the LLM once.

Outputs (config/llm_match.yaml: output_dir):
  candidates_<direction>_<retriever>.npz   query_idx (Q,), cand_idx (Q,N), base_score (Q,N)
  jobs_<direction>.jsonl                   one line per prompt (<= llm.batch candidates)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.llm_match import (
    build_text_lookup,
    held_out_queries,
    iter_jobs,
    load_candidates,
    save_candidates,
    topn_candidates,
    union_candidates,
)
from src.data.loader import embedding_config
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)
from src.utils import load_yaml

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)
PROJECT_COLS = ("과제명", "제출년도", "키워드_리스트", "과제설명문")
COMPANY_COLS = ("한글업체명", "10차산업코드명", "키워드_리스트", "기업설명문")


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--config", default="config/llm_match.yaml")
    parser.add_argument("--max-per-relation", type=int, default=None,
                        help="override queries.max_per_relation (0 = all held-out queries)")
    parser.add_argument("--topn", type=int, default=None, help="override candidates.topn")
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.config)
    qcfg, ccfg = cfg["queries"], cfg["candidates"]
    max_per_rel = qcfg["max_per_relation"]
    if args.max_per_relation is not None:
        max_per_rel = args.max_per_relation or None
    topn = args.topn or int(ccfg["topn"])
    if topn < max(cfg["eval"]["ks"]):
        raise SystemExit(f"candidates.topn={topn} must be >= max(eval.ks)={max(cfg['eval']['ks'])}")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = Path(paths["processed"]["graph"])
    log(f"loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(graph_path.with_name("held_out.pt"), weights_only=False)
    with open(graph_path.with_name("node_maps.pkl"), "rb") as f:
        node_maps = pickle.load(f)
    project_ids = [str(x) for x in node_maps["project_ids"]]
    company_ids = [str(x) for x in node_maps["company_ids"]]
    project_x = graph[NODE_TYPE_PROJECT].x.numpy()
    company_x = graph[NODE_TYPE_COMPANY].x.numpy()
    # every known project-company pair (train + held-out), used only for name masking
    linked: dict[int, set[int]] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        for g in (graph, held_out):
            ei = g[rel].edge_index.numpy()
            for p_i, c_i in zip(ei[0].tolist(), ei[1].tolist(), strict=True):
                linked.setdefault(p_i, set()).add(c_i)
    del graph

    embed_name = embedding_config(paths)[2]
    promise_w = float(ccfg["promise_weight"])
    mask_names = bool(cfg["texts"].get("mask_linked_company_names", True))
    need_p: set[int] = set()
    need_c: set[int] = set()
    selected: dict[str, tuple[np.ndarray, list[list[int]]]] = {}
    z_p_path = Path(paths["processed"]["project_emb"])
    z_c_path = Path(paths["processed"]["company_emb"])
    for direction in qcfg["directions"]:
        query_idx = held_out_queries(held_out, direction, REAL_EDGE_TYPES, max_per_rel,
                                     int(qcfg["seed"]))
        q_x, c_x = (project_x, company_x) if direction == "p2c" else (company_x, project_x)
        promise = None
        if promise_w > 0.0:
            raw = paths["raw"]["companies" if direction == "p2c" else "projects"]
            id_col = paths["id_columns"]["company" if direction == "p2c" else "project"]
            ids = company_ids if direction == "p2c" else project_ids
            df = pd.read_pickle(raw)[[id_col, "유망성점수"]]
            promise = (
                df.set_index(df[id_col].astype(str))["유망성점수"].reindex(ids).to_numpy()
            )
        log(f"[{direction}] {len(query_idx):,} queries -> top-{topn} candidates per retriever")
        cand_idx, base = topn_candidates(
            q_x[query_idx], c_x, topn, cand_promise=promise, promise_weight=promise_w,
            device=ccfg["device"], batch_size=int(ccfg["batch_size"]),
        )
        save_candidates(out_dir, direction, embed_name, query_idx, cand_idx, base)
        for name, tag in (cfg.get("retrievers") or {}).items():
            zp_file = z_p_path.with_name(f"{z_p_path.stem}_{tag}.npy")
            zc_file = z_c_path.with_name(f"{z_c_path.stem}_{tag}.npy")
            if not (zp_file.exists() and zc_file.exists()):
                log(f"[{direction}] ! no saved embeddings for '{name}' ({zp_file.name}); "
                    f"skipped - re-run after training to add it")
                continue
            z_p = np.load(zp_file)
            z_c = np.load(zc_file)
            z_q, z_cand = (z_p, z_c) if direction == "p2c" else (z_c, z_p)
            cand_idx, base = topn_candidates(
                z_q[query_idx], z_cand, topn,
                device=ccfg["device"], batch_size=int(ccfg["batch_size"]),
            )
            save_candidates(out_dir, direction, name, query_idx, cand_idx, base)
            del z_p, z_c

        cands = load_candidates(out_dir, direction)
        for name, c in list(cands.items()):
            same = c["query_idx"].shape == query_idx.shape and (c["query_idx"] == query_idx).all()
            if not same or c["cand_idx"].shape[1] != topn:
                log(f"[{direction}] ! skipping stale candidates of '{name}' "
                    f"(different query sample or topn; regenerate it)")
                del cands[name]
        rerank_names = cfg.get("llm_rerank") or list(cands)
        absent = [n for n in rerank_names if n not in cands]
        if absent:
            raise SystemExit(f"llm_rerank retrievers without candidates: {absent}")
        union = union_candidates([cands[n]["cand_idx"] for n in rerank_names],
                                 int(qcfg["seed"]))
        n_pairs = sum(len(u) for u in union)
        log(f"[{direction}] retrievers: {', '.join(cands)}; LLM re-ranks "
            f"{', '.join(rerank_names)} -> {n_pairs:,} unique pairs "
            f"({n_pairs / len(union):.0f} per query)")
        selected[direction] = (query_idx, union)
        q_set = set(query_idx.tolist())
        c_set = {c for u in union for c in u}
        (need_p if direction == "p2c" else need_c).update(q_set)
        (need_c if direction == "p2c" else need_p).update(c_set)
        if mask_names and direction == "p2c":
            for q in q_set:
                need_c.update(linked.get(q, ()))
    del project_x, company_x

    # Document texts: raw tables give name/keywords for every node, the
    # *_with_desc tables add descriptions where available.
    def lookup(raw_path: str, desc_path: str, id_col: str, need: set[str], cols) -> dict:
        tables = []
        for p in (raw_path, desc_path):
            log(f"loading texts {p}")
            df = pd.read_pickle(p)
            tables.append((df.drop(columns=[c for c in df.columns if "embed" in c]), id_col))
            del df
        return build_text_lookup(tables, need, cols)

    p_need = {project_ids[i] for i in need_p}
    c_need = {company_ids[i] for i in need_c}
    project_text = lookup(paths["raw"]["projects"], cfg["texts"]["project_desc"],
                          paths["id_columns"]["project"], p_need, PROJECT_COLS)
    company_text = lookup(paths["raw"]["companies"], cfg["texts"]["company_desc"],
                          paths["id_columns"]["company"], c_need, COMPANY_COLS)
    p_desc = sum(1 for v in project_text.values() if v.get("과제설명문"))
    c_desc = sum(1 for v in company_text.values() if v.get("기업설명문"))
    log(f"texts: project {len(p_need):,} nodes ({p_desc:,} with description), "
        f"company {len(c_need):,} nodes ({c_desc:,} with description)")

    def company_name(c_i: int) -> str:
        return str(company_text.get(company_ids[c_i], {}).get("한글업체명", ""))

    for direction, (query_idx, cand_idx) in selected.items():
        path = out_dir / f"jobs_{direction}.jsonl"
        mask: dict[int, list[str]] = {}
        if mask_names:
            for q in query_idx.tolist():
                if direction == "p2c":   # hide the linked companies' names in the query project
                    mask[q] = [company_name(c) for c in linked.get(q, ())]
                else:                    # hide the query company's name in candidate projects
                    mask[q] = [company_name(q)]
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for job in iter_jobs(direction, query_idx, cand_idx, project_ids, company_ids,
                                 project_text, company_text, int(cfg["llm"]["batch"]),
                                 mask=mask):
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
                n += 1
        log(f"[{direction}] wrote {n:,} jobs -> {path}")


if __name__ == "__main__":
    main()

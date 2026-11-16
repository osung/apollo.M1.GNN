"""Build derived project-company edge caches for message passing.

    python scripts/build_aux_edges.py --kind keyword
    python scripts/build_aux_edges.py --kind inherited --device cuda

keyword   : keyword-set Jaccard, top-k per project and per company
inherited : partners of the most similar same-type nodes (node-feature inner product)

Pairs that are an observed training pair or a held-out pair are never emitted, so
neither kind can hand a held-out pair to the model. Parameters come from
config/graph.yaml: auxiliary_edges.
"""
from __future__ import annotations

import argparse
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

from src.graph.aux_edges import inherited_edges, keyword_edges, normalize_keywords
from src.graph.schema import (
    EVAL_EDGE_TYPES,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
    OBSERVED_EDGE_TYPES,
    pair_keys,
)
from src.utils import load_yaml

KEYWORD_COL = "키워드_리스트"


def _pairs(data, edge_types) -> np.ndarray:
    parts = [
        data[(NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)].edge_index.numpy()
        for et in edge_types
        if (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY) in data.edge_types
    ]
    return np.unique(np.concatenate(parts, axis=1), axis=1)


def _keyword_sets(path: str, id_col: str, ids: list[str]) -> list[set[str]]:
    df = pd.read_pickle(path)[[id_col, KEYWORD_COL]]
    lookup = dict(zip(df[id_col].astype(str), df[KEYWORD_COL], strict=True))
    return [normalize_keywords(lookup.get(i)) for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-cfg", default="config/graph.yaml")
    parser.add_argument("--kind", required=True, choices=["keyword", "inherited"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    paths = load_yaml(args.paths)
    cfg = load_yaml(args.graph_cfg)["auxiliary_edges"][args.kind]

    graph_path = Path(paths["processed"]["graph"])
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(graph_path.with_name("held_out.pt"), weights_only=False)
    n_p = graph[NODE_TYPE_PROJECT].num_nodes
    n_c = graph[NODE_TYPE_COMPANY].num_nodes
    observed = _pairs(graph, OBSERVED_EDGE_TYPES)
    held = _pairs(held_out, EVAL_EDGE_TYPES)
    held_keys = pair_keys(held, n_c)
    if np.isin(pair_keys(observed, n_c), held_keys).any():
        raise SystemExit("training graph contains held-out pairs; rebuild the graph first")
    exclude = np.unique(np.concatenate([pair_keys(observed, n_c), held_keys]))
    print(f"[aux:{args.kind}] observed training pairs {observed.shape[1]:,}, "
          f"held-out pairs {held.shape[1]:,} (both excluded from the output)")

    t0 = time.time()
    if args.kind == "keyword":
        with open(graph_path.with_name("node_maps.pkl"), "rb") as f:
            maps = pickle.load(f)
        ids = paths["id_columns"]
        p_kw = _keyword_sets(paths["raw"]["projects"], ids["project"],
                             [str(x) for x in maps["project_ids"]])
        c_kw = _keyword_sets(paths["raw"]["companies"], ids["company"],
                             [str(x) for x in maps["company_ids"]])
        edge_index, score = keyword_edges(
            p_kw, c_kw, topk=int(cfg["topk"]), min_shared=int(cfg["min_shared"]),
            min_jaccard=float(cfg["min_jaccard"]),
            max_df=int(cfg["max_keyword_df"]) if cfg.get("max_keyword_df") else None,
            exclude_keys=exclude,
        )
    else:
        edge_index, score = inherited_edges(
            graph[NODE_TYPE_PROJECT].x.numpy(), graph[NODE_TYPE_COMPANY].x.numpy(), observed,
            k_neighbors=int(cfg["k_neighbors"]), top_m=int(cfg["top_m"]),
            min_similarity=float(cfg["min_similarity"]), exclude_keys=exclude,
            device=args.device,
        )

    assert not np.isin(pair_keys(edge_index, n_c), held_keys).any()
    out = Path(cfg["cache_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edge_index=edge_index.astype(np.int64), score=score,
             n_project=np.int64(n_p), n_company=np.int64(n_c), kind=np.array(args.kind))
    deg_p = np.bincount(edge_index[0], minlength=n_p)
    deg_c = np.bincount(edge_index[1], minlength=n_c)
    print(f"[aux:{args.kind}] {edge_index.shape[1]:,} edges in {time.time() - t0:.0f}s -> {out}")
    print(f"[aux:{args.kind}] projects with an edge {np.mean(deg_p > 0):.1%} "
          f"(mean {deg_p.mean():.1f}), companies {np.mean(deg_c > 0):.1%} "
          f"(mean {deg_c.mean():.1f}); score p10/p50/p90 = "
          + "/".join(f"{np.quantile(score, q):.3f}" for q in (0.1, 0.5, 0.9)))


if __name__ == "__main__":
    main()

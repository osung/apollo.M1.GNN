"""Generic top-K retrieval and ranking metrics.

Designed to be reused across all models (baselines + GNN):
- Any model produces (query_idx -> top-K candidate indices).
- This module scores those predictions against a held-out ground-truth set.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def group_ground_truth(
    edge_index: np.ndarray, direction: str = "project_to_company"
) -> dict[int, set[int]]:
    """Turn held-out (src, dst) pairs into {query_idx -> {relevant_idx, ...}}.

    direction='project_to_company': rows of edge_index[0] are queries.
    direction='company_to_project': rows of edge_index[1] are queries.
    """
    out: dict[int, set[int]] = defaultdict(set)
    if edge_index.shape[1] == 0:
        return out
    if direction == "project_to_company":
        q, c = edge_index[0], edge_index[1]
    elif direction == "company_to_project":
        q, c = edge_index[1], edge_index[0]
    else:
        raise ValueError(direction)
    for qi, ci in zip(q.tolist(), c.tolist()):
        out[int(qi)].add(int(ci))
    return dict(out)


def recall_at_k(topk_preds: np.ndarray, gt: dict[int, set[int]], query_ids: Iterable[int]) -> float:
    """Mean Recall@K over the given queries.

    topk_preds: (Q, K) array of candidate indices, one row per query_id.
    gt: {query_idx -> {relevant_idx, ...}}.
    query_ids: same order as topk_preds rows.
    """
    hits, total = 0.0, 0
    for row, qi in zip(topk_preds, query_ids):
        rel = gt.get(int(qi))
        if not rel:
            continue
        pred_set = set(int(x) for x in row if x >= 0)
        hits += len(pred_set & rel) / len(rel)
        total += 1
    return hits / total if total else float("nan")


def ndcg_at_k(topk_preds: np.ndarray, gt: dict[int, set[int]], query_ids: Iterable[int]) -> float:
    """Binary-relevance NDCG@K averaged over queries with at least one ground-truth."""
    scores, total = 0.0, 0
    for row, qi in zip(topk_preds, query_ids):
        rel = gt.get(int(qi))
        if not rel:
            continue
        k = len(row)
        gains = np.array(
            [1.0 if int(c) in rel else 0.0 for c in row.tolist()], dtype=np.float64
        )
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float((gains * discounts).sum())
        ideal_hits = min(len(rel), k)
        idcg = float(discounts[:ideal_hits].sum())
        if idcg > 0:
            scores += dcg / idcg
            total += 1
    return scores / total if total else float("nan")


def evaluate(
    topk_preds: np.ndarray, query_ids: Iterable[int], gt: dict[int, set[int]], ks=(10, 100)
) -> dict[str, float]:
    """Convenience: compute Recall@k and NDCG@k for several cutoffs."""
    query_ids = list(query_ids)
    max_k = max(ks)
    if topk_preds.shape[1] < max_k:
        raise ValueError(f"preds have only {topk_preds.shape[1]} columns, need {max_k}")
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(topk_preds[:, :k], gt, query_ids)
        out[f"ndcg@{k}"] = ndcg_at_k(topk_preds[:, :k], gt, query_ids)
    return out

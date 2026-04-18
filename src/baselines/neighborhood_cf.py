"""Memory-based neighborhood CF over a bipartite graph.

Given norm_embed for both node types, we compute top-k **same-type neighbors**
(project↔project or company↔company) and aggregate their interactions
(royalty/commercial/performance) — weighted by both relation weight and
neighbor similarity — to score opposite-type candidates.

Score (project->company):

    score(p, c) = Σ_{p' in Nk(p)}  sim(p, p') · Σ_r w_r · I_r(p', c)

where Nk(p) is the top-k most similar projects to p by norm_embed cosine,
and I_r is the binary indicator of relation r between p' and c.

Cold-start: if the query project has no connected neighbors at all, the score
row can still be non-zero via the similar-projects' interactions; if none of
the neighbors have interactions either, the row is zero and we fall back to
the zero-training two-tower score (norm_embed_p · norm_embed_c).
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from scipy import sparse


@dataclass
class CFConfig:
    k_neighbors: int = 50
    topk: int = 100
    include_self: bool = False
    fallback_two_tower: bool = True
    query_batch_size: int = 256


class NeighborhoodCF:
    def __init__(
        self,
        project_x: np.ndarray,
        company_x: np.ndarray,
        edges_per_relation: dict[str, np.ndarray],
        relation_weights: dict[str, float],
        config: CFConfig | None = None,
    ):
        if project_x.dtype != np.float32:
            project_x = project_x.astype(np.float32)
        if company_x.dtype != np.float32:
            company_x = company_x.astype(np.float32)

        self.project_x = np.ascontiguousarray(project_x)
        self.company_x = np.ascontiguousarray(company_x)
        self.cfg = config or CFConfig()

        n_p, d = self.project_x.shape
        n_c, d_c = self.company_x.shape
        if d != d_c:
            raise ValueError(f"embed dim mismatch: project {d}, company {d_c}")
        self.n_p, self.n_c, self.d = n_p, n_c, d

        self._project_index = faiss.IndexFlatIP(d)
        self._project_index.add(self.project_x)
        self._company_index = faiss.IndexFlatIP(d)
        self._company_index.add(self.company_x)

        R = sparse.csr_matrix((n_p, n_c), dtype=np.float32)
        for et, edges in edges_per_relation.items():
            if edges.shape[1] == 0:
                continue
            w = float(relation_weights.get(et, 1.0))
            data = np.full(edges.shape[1], w, dtype=np.float32)
            R_et = sparse.coo_matrix(
                (data, (edges[0], edges[1])), shape=(n_p, n_c)
            ).tocsr()
            R = R + R_et
        self.R = R.tocsr()
        self.R_T = self.R.T.tocsr()

    def _lookup_neighbors(
        self, index: faiss.Index, query_vecs: np.ndarray, k: int, self_indices: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (sim, idx), both (Q, k). Optionally exclude self by id."""
        search_k = k + (1 if self_indices is not None and not self.cfg.include_self else 0)
        sim, idx = index.search(query_vecs, search_k)
        if self_indices is not None and not self.cfg.include_self:
            sim, idx = self._drop_self(sim, idx, self_indices, k)
        return sim, idx

    @staticmethod
    def _drop_self(
        sim: np.ndarray, idx: np.ndarray, self_indices: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        Q = idx.shape[0]
        out_sim = np.empty((Q, k), dtype=sim.dtype)
        out_idx = np.empty((Q, k), dtype=idx.dtype)
        for q in range(Q):
            mask = idx[q] != self_indices[q]
            kept_idx = idx[q][mask][:k]
            kept_sim = sim[q][mask][:k]
            if kept_idx.shape[0] < k:
                pad = k - kept_idx.shape[0]
                kept_idx = np.concatenate([kept_idx, np.full(pad, -1, dtype=idx.dtype)])
                kept_sim = np.concatenate([kept_sim, np.zeros(pad, dtype=sim.dtype)])
            out_sim[q] = kept_sim
            out_idx[q] = kept_idx
        return out_sim, out_idx

    def recommend_companies(self, project_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each project in `project_indices`, return (top_company_idx, top_score).

        Shape: (Q, topk) each. -1 padding if fewer than topk available.
        Processed in batches of `cfg.query_batch_size` to bound memory.
        """
        return self._recommend(
            queries=project_indices,
            query_x=self.project_x,
            index=self._project_index,
            R_query_to_target=self.R,
            target_x=self.company_x,
            n_targets=self.n_c,
        )

    def recommend_projects(self, company_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reverse direction: for each company, recommend top-k projects."""
        return self._recommend(
            queries=company_indices,
            query_x=self.company_x,
            index=self._company_index,
            R_query_to_target=self.R_T,
            target_x=self.project_x,
            n_targets=self.n_p,
        )

    def _recommend(
        self,
        queries: np.ndarray,
        query_x: np.ndarray,
        index: faiss.Index,
        R_query_to_target: sparse.csr_matrix,
        target_x: np.ndarray,
        n_targets: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.int64)
        Q = queries.shape[0]
        topk = self.cfg.topk
        bs = max(1, self.cfg.query_batch_size)

        out_idx = np.empty((Q, topk), dtype=np.int64)
        out_scores = np.empty((Q, topk), dtype=np.float32)

        for start in range(0, Q, bs):
            end = min(start + bs, Q)
            batch_ids = queries[start:end]
            batch_vec = np.ascontiguousarray(query_x[batch_ids])
            sim, idx = self._lookup_neighbors(
                index, batch_vec, self.cfg.k_neighbors, batch_ids
            )
            B = end - start
            scores = np.zeros((B, n_targets), dtype=np.float32)
            for q in range(B):
                valid = idx[q] >= 0
                if not valid.any():
                    continue
                neighbor_ids = idx[q][valid]
                neighbor_sim = sim[q][valid].astype(np.float32)
                slice_R = R_query_to_target[neighbor_ids]
                scores[q] = neighbor_sim @ slice_R
            if self.cfg.fallback_two_tower:
                empty = scores.sum(axis=1) == 0
                if empty.any():
                    scores[empty] = batch_vec[empty] @ target_x.T
            idx_b, sc_b = _topk(scores, topk)
            out_idx[start:end] = idx_b
            out_scores[start:end] = sc_b

        return out_idx, out_scores


def _topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Argpartition + sort for top-k per row. Returns (indices, scores) both (Q, k)."""
    Q, N = scores.shape
    k = min(k, N)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)
    return top_idx.astype(np.int64), top_scores

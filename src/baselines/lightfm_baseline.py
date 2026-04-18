"""LightFM hybrid factorization baseline.

Three identity strategies via `identity_mode`:

- `none`  — content-only. node embedding = norm_embed @ learned projection.
            Small model (~50K params/side), cold-start natural, no per-id
            memory.
- `warm`  — identity features only for warm nodes (those with ≥1 training
            edge). Inference extends to all nodes via the content
            projection; warm nodes additionally get their learned id
            embedding. Recommended on this dataset (~97% cold companies).
- `all`   — identity for every node. Maximal capacity, highest memory
            (~10 GB sparse on this data); use only if RAM allows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from lightfm import LightFM
from scipy import sparse


IdentityMode = Literal["none", "warm", "all"]


@dataclass
class LightFMConfig:
    no_components: int = 128
    loss: str = "warp"
    learning_rate: float = 0.05
    item_alpha: float = 1e-6
    user_alpha: float = 1e-6
    epochs: int = 20
    num_threads: int = 1
    identity_mode: IdentityMode = "none"
    query_batch_size: int = 256


def _interaction_matrix(
    edges_per_relation: dict[str, np.ndarray],
    relation_weights: dict[str, float],
    n_users: int,
    n_items: int,
) -> tuple[sparse.coo_matrix, sparse.coo_matrix]:
    """Combine relations into a single sparse (users x items) interaction matrix.

    Returns (interactions, weights). Interactions is binary (1 on any edge);
    weights carries the relation weight for BPR/WARP sample weighting.
    Duplicate (u, i) from different relations keep the **max** weight.
    """
    rows, cols, ws = [], [], []
    for et, edges in edges_per_relation.items():
        if edges.shape[1] == 0:
            continue
        w = float(relation_weights.get(et, 1.0))
        rows.append(edges[0])
        cols.append(edges[1])
        ws.append(np.full(edges.shape[1], w, dtype=np.float32))
    if not rows:
        z = sparse.coo_matrix((n_users, n_items), dtype=np.float32)
        return z, z
    rows_cat = np.concatenate(rows)
    cols_cat = np.concatenate(cols)
    ws_cat = np.concatenate(ws)

    weight_lookup: dict[tuple[int, int], float] = {}
    for r, c, w in zip(rows_cat.tolist(), cols_cat.tolist(), ws_cat.tolist()):
        key = (int(r), int(c))
        prev = weight_lookup.get(key, 0.0)
        if w > prev:
            weight_lookup[key] = w

    keys = np.array(list(weight_lookup.keys()), dtype=np.int64)
    vals = np.array(list(weight_lookup.values()), dtype=np.float32)
    uniq_rows, uniq_cols = keys[:, 0], keys[:, 1]

    interactions = sparse.coo_matrix(
        (np.ones_like(vals), (uniq_rows, uniq_cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    weights = sparse.coo_matrix(
        (vals, (uniq_rows, uniq_cols)), shape=(n_users, n_items), dtype=np.float32
    )
    return interactions, weights


def _node_features(content: np.ndarray, use_identity: bool) -> sparse.csr_matrix:
    """Build the LightFM feature matrix for one node side.

    - use_identity=True: identity (N x N) ⊕ content (N x D).
    - use_identity=False: content only.
    """
    n, _ = content.shape
    content_csr = sparse.csr_matrix(content.astype(np.float32))
    if not use_identity:
        return content_csr
    identity = sparse.identity(n, format="csr", dtype=np.float32)
    return sparse.hstack([identity, content_csr], format="csr")


def _warm_indices(
    edges_per_relation: dict[str, np.ndarray], side: int
) -> np.ndarray:
    """Collect unique warm node indices on one side of the bipartite graph.

    side=0 takes edges[0] (projects), side=1 takes edges[1] (companies).
    """
    parts = []
    for _, edges in edges_per_relation.items():
        if edges.shape[1] == 0:
            continue
        parts.append(edges[side])
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64)


class LightFMBaseline:
    def __init__(self, config: LightFMConfig | None = None):
        self.cfg = config or LightFMConfig()
        self.model: LightFM | None = None
        self.n_users: int = 0
        self.n_items: int = 0

        self._user_features: sparse.csr_matrix | None = None
        self._item_features: sparse.csr_matrix | None = None

        # warm-mode caches: dense embeddings/bias for every full-size node
        self._user_emb_full: np.ndarray | None = None
        self._user_bias_full: np.ndarray | None = None
        self._item_emb_full: np.ndarray | None = None
        self._item_bias_full: np.ndarray | None = None

    def fit(
        self,
        project_x: np.ndarray,
        company_x: np.ndarray,
        edges_per_relation: dict[str, np.ndarray],
        relation_weights: dict[str, float],
    ) -> "LightFMBaseline":
        n_p, n_c = project_x.shape[0], company_x.shape[0]
        self.n_users, self.n_items = n_p, n_c

        if self.cfg.identity_mode == "warm":
            return self._fit_warm(project_x, company_x, edges_per_relation, relation_weights)
        return self._fit_full(project_x, company_x, edges_per_relation, relation_weights)

    def _fit_full(
        self, project_x, company_x, edges_per_relation, relation_weights
    ) -> "LightFMBaseline":
        use_identity = self.cfg.identity_mode == "all"
        self._user_features = _node_features(project_x, use_identity)
        self._item_features = _node_features(company_x, use_identity)

        interactions, weights = _interaction_matrix(
            edges_per_relation, relation_weights, self.n_users, self.n_items
        )
        self.model = self._build_model()
        self.model.fit(
            interactions.tocoo(),
            sample_weight=weights.tocoo(),
            user_features=self._user_features,
            item_features=self._item_features,
            epochs=self.cfg.epochs,
            num_threads=self.cfg.num_threads,
            verbose=True,
        )
        return self

    def _fit_warm(
        self, project_x, company_x, edges_per_relation, relation_weights
    ) -> "LightFMBaseline":
        warm_u = _warm_indices(edges_per_relation, side=0)
        warm_i = _warm_indices(edges_per_relation, side=1)
        n_warm_u = warm_u.shape[0]
        n_warm_i = warm_i.shape[0]
        print(f"[lightfm:warm] warm users={n_warm_u:,} items={n_warm_i:,}")

        u_full_to_warm = -np.ones(self.n_users, dtype=np.int64)
        u_full_to_warm[warm_u] = np.arange(n_warm_u)
        i_full_to_warm = -np.ones(self.n_items, dtype=np.int64)
        i_full_to_warm[warm_i] = np.arange(n_warm_i)

        warm_edges: dict[str, np.ndarray] = {}
        for et, edges in edges_per_relation.items():
            if edges.shape[1] == 0:
                warm_edges[et] = edges
                continue
            warm_edges[et] = np.stack(
                [u_full_to_warm[edges[0]], i_full_to_warm[edges[1]]]
            )

        user_features_warm = _node_features(project_x[warm_u], use_identity=True)
        item_features_warm = _node_features(company_x[warm_i], use_identity=True)

        interactions, weights = _interaction_matrix(
            warm_edges, relation_weights, n_warm_u, n_warm_i
        )

        self.model = self._build_model()
        self.model.fit(
            interactions.tocoo(),
            sample_weight=weights.tocoo(),
            user_features=user_features_warm,
            item_features=item_features_warm,
            epochs=self.cfg.epochs,
            num_threads=self.cfg.num_threads,
            verbose=True,
        )

        self._precompute_warm_full_embeddings(project_x, company_x, warm_u, warm_i)
        return self

    def _precompute_warm_full_embeddings(
        self,
        project_x: np.ndarray,
        company_x: np.ndarray,
        warm_u: np.ndarray,
        warm_i: np.ndarray,
    ) -> None:
        """Reconstruct dense (n_p, k) and (n_c, k) embeddings for every node.

        For a warm node u with warm index wu:
            emb[u]  = model.user_embeddings[wu]          (identity part)
                    + norm_embed[u] @ model.user_embeddings[n_warm_u:]  (content part)
            bias[u] = model.user_biases[wu]
                    + norm_embed[u] @ model.user_biases[n_warm_u:]

        For cold nodes, only the content contribution applies.
        """
        assert self.model is not None
        n_warm_u = warm_u.shape[0]
        n_warm_i = warm_i.shape[0]

        ue = self.model.user_embeddings.astype(np.float32)
        ub = self.model.user_biases.astype(np.float32)
        ie = self.model.item_embeddings.astype(np.float32)
        ib = self.model.item_biases.astype(np.float32)

        user_content_w = ue[n_warm_u:]
        user_content_b = ub[n_warm_u:]
        item_content_w = ie[n_warm_i:]
        item_content_b = ib[n_warm_i:]

        px = project_x.astype(np.float32)
        cx = company_x.astype(np.float32)

        user_emb_full = px @ user_content_w
        user_bias_full = px @ user_content_b
        item_emb_full = cx @ item_content_w
        item_bias_full = cx @ item_content_b

        user_emb_full[warm_u] += ue[:n_warm_u]
        user_bias_full[warm_u] += ub[:n_warm_u]
        item_emb_full[warm_i] += ie[:n_warm_i]
        item_bias_full[warm_i] += ib[:n_warm_i]

        self._user_emb_full = user_emb_full
        self._user_bias_full = user_bias_full
        self._item_emb_full = item_emb_full
        self._item_bias_full = item_bias_full

    def _build_model(self) -> LightFM:
        return LightFM(
            no_components=self.cfg.no_components,
            loss=self.cfg.loss,
            learning_rate=self.cfg.learning_rate,
            item_alpha=self.cfg.item_alpha,
            user_alpha=self.cfg.user_alpha,
        )

    def _materialize_representations(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._user_emb_full is not None:
            return (
                self._user_bias_full,  # type: ignore[return-value]
                self._user_emb_full,
                self._item_bias_full,  # type: ignore[return-value]
                self._item_emb_full,
            )
        assert self.model is not None
        u_bias, u_emb = self.model.get_user_representations(self._user_features)
        i_bias, i_emb = self.model.get_item_representations(self._item_features)
        return (
            np.asarray(u_bias, dtype=np.float32),
            np.asarray(u_emb, dtype=np.float32),
            np.asarray(i_bias, dtype=np.float32),
            np.asarray(i_emb, dtype=np.float32),
        )

    def recommend_companies(
        self, project_indices: np.ndarray, topk: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each project query, return top-k company candidates."""
        if self.model is None:
            raise RuntimeError("call .fit() first")
        u_bias, u_emb, i_bias, i_emb = self._materialize_representations()
        return _batched_topk(
            query_ids=np.asarray(project_indices, dtype=np.int64),
            query_emb=u_emb,
            query_bias=u_bias,
            candidate_emb=i_emb,
            candidate_bias=i_bias,
            topk=topk,
            batch_size=self.cfg.query_batch_size,
        )

    def recommend_projects(
        self, company_indices: np.ndarray, topk: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reverse direction: for each company query, return top-k projects.

        LightFM's WARP loss trains user_embedding as *query* and item_embedding
        as *candidate*. user_bias gets no pairwise gradient (cancels out) and
        is therefore noise when used as candidate bias in this direction.
        We drop the candidate-side bias here; query-side bias is constant
        across candidates and doesn't affect ranking anyway, so the scoring
        reduces to the dot product only.
        """
        if self.model is None:
            raise RuntimeError("call .fit() first")
        _, u_emb, _, i_emb = self._materialize_representations()
        zero = np.zeros(u_emb.shape[0], dtype=np.float32)
        return _batched_topk(
            query_ids=np.asarray(company_indices, dtype=np.int64),
            query_emb=i_emb,
            query_bias=np.zeros(i_emb.shape[0], dtype=np.float32),
            candidate_emb=u_emb,
            candidate_bias=zero,
            topk=topk,
            batch_size=self.cfg.query_batch_size,
        )


def _batched_topk(
    query_ids: np.ndarray,
    query_emb: np.ndarray,
    query_bias: np.ndarray,
    candidate_emb: np.ndarray,
    candidate_bias: np.ndarray,
    topk: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    Q = query_ids.shape[0]
    bs = max(1, batch_size)
    top_idx = np.empty((Q, topk), dtype=np.int64)
    top_scores = np.empty((Q, topk), dtype=np.float32)
    for start in range(0, Q, bs):
        end = min(start + bs, Q)
        q = query_ids[start:end]
        batch_emb = query_emb[q]
        scores = batch_emb @ candidate_emb.T
        scores += query_bias[q, None]
        scores += candidate_bias[None, :]
        idx_b, sc_b = _topk_rows(scores, topk)
        top_idx[start:end] = idx_b
        top_scores[start:end] = sc_b
    return top_idx, top_scores


def _topk_rows(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    Q, N = scores.shape
    k = min(k, N)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)
    return top_idx.astype(np.int64), top_scores

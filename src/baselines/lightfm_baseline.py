"""LightFM hybrid factorization baseline.

norm_embed이 공통 latent space 벡터이므로, 각 노드를 하나의 "identity feature"로만
쓰는 순수 CF 대신, **norm_embed을 사용자/아이템 feature**로 주입하는 하이브리드 구성을
사용한다. Cold node는 identity 임베딩이 없어도 feature 임베딩만으로 점수 계산 가능.

특징:
- `no_components` 차원의 latent embedding을 사용자·아이템에 각각 학습.
- interaction weight = 간선 타입별 relation weight.
- Loss: WARP (pairwise ranking with margin).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lightfm import LightFM
from scipy import sparse


@dataclass
class LightFMConfig:
    no_components: int = 128
    loss: str = "warp"
    learning_rate: float = 0.05
    item_alpha: float = 1e-6
    user_alpha: float = 1e-6
    epochs: int = 20
    num_threads: int = 1


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


def _identity_plus_features(content: np.ndarray) -> sparse.csr_matrix:
    """Stack identity (N x N) on the left + content (N x D) as features on the right.

    LightFM treats features as additive; including identity lets warm nodes
    learn per-id embeddings on top of the shared content projection.
    """
    n, d = content.shape
    identity = sparse.identity(n, format="csr", dtype=np.float32)
    content_csr = sparse.csr_matrix(content.astype(np.float32))
    return sparse.hstack([identity, content_csr], format="csr")


class LightFMBaseline:
    def __init__(self, config: LightFMConfig | None = None):
        self.cfg = config or LightFMConfig()
        self.model: LightFM | None = None
        self.user_features: sparse.csr_matrix | None = None
        self.item_features: sparse.csr_matrix | None = None
        self.n_users: int = 0
        self.n_items: int = 0

    def fit(
        self,
        project_x: np.ndarray,
        company_x: np.ndarray,
        edges_per_relation: dict[str, np.ndarray],
        relation_weights: dict[str, float],
    ) -> "LightFMBaseline":
        n_p, n_c = project_x.shape[0], company_x.shape[0]
        self.n_users, self.n_items = n_p, n_c

        self.user_features = _identity_plus_features(project_x)
        self.item_features = _identity_plus_features(company_x)

        interactions, weights = _interaction_matrix(
            edges_per_relation, relation_weights, n_p, n_c
        )

        self.model = LightFM(
            no_components=self.cfg.no_components,
            loss=self.cfg.loss,
            learning_rate=self.cfg.learning_rate,
            item_alpha=self.cfg.item_alpha,
            user_alpha=self.cfg.user_alpha,
        )
        self.model.fit(
            interactions.tocoo(),
            sample_weight=weights.tocoo(),
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=self.cfg.epochs,
            num_threads=self.cfg.num_threads,
        )
        return self

    def recommend_companies(
        self, project_indices: np.ndarray, topk: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("call .fit() first")
        project_indices = np.asarray(project_indices, dtype=np.int64)
        Q = project_indices.shape[0]
        all_items = np.arange(self.n_items, dtype=np.int64)

        top_idx = np.empty((Q, topk), dtype=np.int64)
        top_scores = np.empty((Q, topk), dtype=np.float32)
        for i, u in enumerate(project_indices):
            scores = self.model.predict(
                int(u),
                all_items,
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=self.cfg.num_threads,
            )
            k = min(topk, scores.shape[0])
            part = np.argpartition(-scores, kth=k - 1)[:k]
            order = np.argsort(-scores[part])
            top_idx[i, :k] = part[order]
            top_scores[i, :k] = scores[part[order]]
            if k < topk:
                top_idx[i, k:] = -1
                top_scores[i, k:] = -np.inf
        return top_idx, top_scores

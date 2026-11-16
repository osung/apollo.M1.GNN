"""Training-edge sampler with per-relation weights and hybrid negative sampling.

Per batch we always emit company-side negatives (`neg_dst`) for the
project→company BPR term. When `c2p_enabled` is true we additionally
emit project-side negatives (`neg_src`) so the trainer can add a
company→project BPR term — useful for symmetric training where we
care about both retrieval directions.

Hard negatives are optional on each side. If a side has no hard-neg
map, its negatives are drawn uniformly at random.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class EdgeBatch:
    pos_src: torch.Tensor           # (B,) project indices
    pos_dst: torch.Tensor           # (B,) company indices
    neg_dst: torch.Tensor           # (B, K) company negatives (p2c direction)
    weights: torch.Tensor           # (B,)
    relation: torch.Tensor          # (B,) int64
    neg_src: Optional[torch.Tensor] = None  # (B, K) project negatives (c2p direction)


def _split(num_neg: int, ratio: float, has_hard: bool) -> tuple[int, int]:
    n_hard = int(round(num_neg * ratio)) if has_hard else 0
    return n_hard, num_neg - n_hard


class EdgeSampler:
    def __init__(
        self,
        edges_per_relation: dict[str, np.ndarray],
        relation_weights: dict[str, float],
        n_dst: int,
        num_neg: int = 5,
        batch_size: int = 4096,
        seed: int = 42,
        hard_neg_map: dict[int, np.ndarray] | None = None,
        hard_ratio: float = 0.0,
        n_src: int | None = None,
        hard_neg_map_c2p: dict[int, np.ndarray] | None = None,
        c2p_enabled: bool = False,
        warm_neg_ratio: float = 0.0,
    ):
        if not 0.0 <= warm_neg_ratio <= 1.0:
            raise ValueError(f"warm_neg_ratio must be in [0, 1], got {warm_neg_ratio}")
        # Share of the *random* negatives drawn from warm nodes (nodes with at least
        # one training edge) instead of from all nodes. Positives are always warm
        # while ~95% of companies are cold, so uniform negatives let a model reach a
        # low loss by separating warm from cold nodes without ranking within them.
        self.warm_neg_ratio = float(warm_neg_ratio)
        self.n_dst = n_dst
        self.n_src = int(n_src) if n_src else 0
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.hard_neg_map = hard_neg_map or {}
        self.hard_neg_map_c2p = hard_neg_map_c2p or {}
        self.hard_ratio = float(hard_ratio)
        self.n_hard, self.n_random = _split(
            num_neg, self.hard_ratio, bool(self.hard_neg_map)
        )
        self.n_hard_c2p, self.n_random_c2p = _split(
            num_neg, self.hard_ratio, bool(self.hard_neg_map_c2p)
        )

        self.c2p_enabled = bool(c2p_enabled or self.hard_neg_map_c2p)
        if self.c2p_enabled and self.n_src <= 0:
            raise ValueError("c2p_enabled=True requires n_src to be set (>0).")

        self.relation_names: list[str] = sorted(
            [r for r, e in edges_per_relation.items() if e.shape[1] > 0]
        )
        self.rel_idx: dict[str, int] = {r: i for i, r in enumerate(self.relation_names)}

        src_parts, dst_parts, wt_parts, rel_parts = [], [], [], []
        for r in self.relation_names:
            edges = edges_per_relation[r]
            src_parts.append(edges[0])
            dst_parts.append(edges[1])
            w = float(relation_weights.get(r, 1.0))
            wt_parts.append(np.full(edges.shape[1], w, dtype=np.float32))
            rel_parts.append(np.full(edges.shape[1], self.rel_idx[r], dtype=np.int64))

        self.src = np.concatenate(src_parts).astype(np.int64)
        self.dst = np.concatenate(dst_parts).astype(np.int64)
        self.wt = np.concatenate(wt_parts)
        self.rel = np.concatenate(rel_parts)
        self.n_edges = self.src.shape[0]
        self.warm_src = np.unique(self.src)
        self.warm_dst = np.unique(self.dst)

    def __len__(self) -> int:
        return (self.n_edges + self.batch_size - 1) // self.batch_size

    def _random_negs(self, n_rows: int, k: int, n_candidates: int,
                     warm_pool: np.ndarray) -> np.ndarray:
        """(n_rows, k) uniform negatives; the first round(k * warm_neg_ratio) columns
        are re-drawn uniformly from `warm_pool`."""
        neg = self.rng.integers(0, n_candidates, size=(n_rows, k)).astype(np.int64)
        n_warm = int(round(k * self.warm_neg_ratio))
        if n_warm > 0 and warm_pool.size > 0:
            pick = self.rng.integers(0, warm_pool.size, size=(n_rows, n_warm))
            neg[:, :n_warm] = warm_pool[pick]
        return neg

    def _sample_negs_one_side(
        self,
        queries: np.ndarray,
        n_candidates: int,
        hard_map: dict[int, np.ndarray],
        n_hard: int,
        n_random: int,
        warm_pool: np.ndarray,
    ) -> np.ndarray:
        B = queries.shape[0]
        if n_hard == 0:
            return self._random_negs(B, self.num_neg, n_candidates, warm_pool)

        neg = np.empty((B, self.num_neg), dtype=np.int64)
        if n_random > 0:
            neg[:, n_hard:] = self._random_negs(B, n_random, n_candidates, warm_pool)
        for i, q in enumerate(queries.tolist()):
            pool = hard_map.get(int(q))
            if pool is None or pool.size == 0:
                neg[i, :n_hard] = self.rng.integers(0, n_candidates, size=n_hard)
            else:
                pick = self.rng.integers(0, pool.size, size=n_hard)
                neg[i, :n_hard] = pool[pick]
        return neg

    def iter_epoch(self):
        perm = self.rng.permutation(self.n_edges)
        for start in range(0, self.n_edges, self.batch_size):
            idx = perm[start : start + self.batch_size]
            srcs = self.src[idx]
            dsts = self.dst[idx]

            neg_dst = self._sample_negs_one_side(
                srcs, self.n_dst, self.hard_neg_map, self.n_hard, self.n_random,
                self.warm_dst,
            )
            neg_src = None
            if self.c2p_enabled:
                neg_src = self._sample_negs_one_side(
                    dsts, self.n_src, self.hard_neg_map_c2p,
                    self.n_hard_c2p, self.n_random_c2p, self.warm_src,
                )

            yield EdgeBatch(
                pos_src=torch.from_numpy(srcs),
                pos_dst=torch.from_numpy(dsts),
                neg_dst=torch.from_numpy(neg_dst),
                weights=torch.from_numpy(self.wt[idx]),
                relation=torch.from_numpy(self.rel[idx]),
                neg_src=torch.from_numpy(neg_src) if neg_src is not None else None,
            )

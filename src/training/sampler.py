"""Training-edge sampler with per-relation weights and hybrid negative sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EdgeBatch:
    pos_src: torch.Tensor       # (B,)
    pos_dst: torch.Tensor       # (B,)
    neg_dst: torch.Tensor       # (B, K)
    weights: torch.Tensor       # (B,)
    relation: torch.Tensor      # (B,) int64, index into relation_names


class EdgeSampler:
    """Flatten all (relation, u, v) training edges and iterate shuffled batches.

    Negative sampling supports two modes:
    - **Random (default)**: `num_neg` destinations drawn uniformly from
      [0, n_dst). Cheap but almost always too easy for top-k ranking.
    - **Hybrid (hard + random)**: if `hard_neg_map` is provided and
      `hard_ratio > 0`, split `num_neg` between:
        - `n_hard = round(num_neg * hard_ratio)` drawn from the
          per-source hard-negative pool (with replacement to tolerate
          pools smaller than `n_hard`);
        - `n_random = num_neg - n_hard` drawn uniformly.
      Sources missing from `hard_neg_map` fall back to all-random for
      that row so the batch shape stays regular.

    The batch carries per-sample `weights` equal to the relation weight.
    """

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
    ):
        self.n_dst = n_dst
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.hard_neg_map = hard_neg_map or {}
        self.hard_ratio = float(hard_ratio)
        self.n_hard = int(round(num_neg * self.hard_ratio)) if self.hard_neg_map else 0
        self.n_random = num_neg - self.n_hard

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

    def __len__(self) -> int:
        return (self.n_edges + self.batch_size - 1) // self.batch_size

    def _sample_negatives(self, srcs: np.ndarray) -> np.ndarray:
        B = srcs.shape[0]
        if self.n_hard == 0:
            return self.rng.integers(0, self.n_dst, size=(B, self.num_neg)).astype(np.int64)

        neg = np.empty((B, self.num_neg), dtype=np.int64)
        # Random portion: draw once for the whole batch
        if self.n_random > 0:
            neg[:, self.n_hard :] = self.rng.integers(
                0, self.n_dst, size=(B, self.n_random)
            )
        for i, s in enumerate(srcs.tolist()):
            pool = self.hard_neg_map.get(int(s))
            if pool is None or pool.size == 0:
                neg[i, : self.n_hard] = self.rng.integers(
                    0, self.n_dst, size=self.n_hard
                )
            else:
                pick = self.rng.integers(0, pool.size, size=self.n_hard)
                neg[i, : self.n_hard] = pool[pick]
        return neg

    def iter_epoch(self):
        perm = self.rng.permutation(self.n_edges)
        for start in range(0, self.n_edges, self.batch_size):
            idx = perm[start : start + self.batch_size]
            srcs = self.src[idx]
            neg = self._sample_negatives(srcs)
            yield EdgeBatch(
                pos_src=torch.from_numpy(srcs),
                pos_dst=torch.from_numpy(self.dst[idx]),
                neg_dst=torch.from_numpy(neg),
                weights=torch.from_numpy(self.wt[idx]),
                relation=torch.from_numpy(self.rel[idx]),
            )

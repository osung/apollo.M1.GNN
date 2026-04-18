"""Training-edge sampler with per-relation weights and random negatives."""
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

    For each positive (u, v), sample `num_neg` random destinations uniformly
    from [0, n_dst). This is "sampled softmax"-style negative sampling; we
    do not filter existing positives (cheap to implement and bias is small
    at density ~1e-7).

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
    ):
        self.n_dst = n_dst
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

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

    def iter_epoch(self):
        perm = self.rng.permutation(self.n_edges)
        for start in range(0, self.n_edges, self.batch_size):
            idx = perm[start : start + self.batch_size]
            B = idx.shape[0]
            neg = self.rng.integers(0, self.n_dst, size=(B, self.num_neg)).astype(np.int64)
            yield EdgeBatch(
                pos_src=torch.from_numpy(self.src[idx]),
                pos_dst=torch.from_numpy(self.dst[idx]),
                neg_dst=torch.from_numpy(neg),
                weights=torch.from_numpy(self.wt[idx]),
                relation=torch.from_numpy(self.rel[idx]),
            )

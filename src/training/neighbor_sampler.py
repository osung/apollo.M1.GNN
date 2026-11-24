"""Mini-batch neighbor sampling for the bipartite heterogeneous graph.

Full-batch training keeps the activations of every edge of every layer in memory.
Here each step encodes only the batch's seed nodes: layer L needs the seeds' sampled
in-neighbors, layer L-1 the in-neighbors of those, and so on. Per target node at most
`fanout` in-edges are kept per relation, so memory is bounded by batch size x fanouts
instead of by the graph.

Two ways of choosing the kept in-edges:
  - random (default): a random subset, redrawn every epoch (`reshuffle`)
  - ranked (`edge_score_dict` given): always the `fanout` highest-scored in-edges. The
    choice is deterministic, so embeddings computed with the same sampler at inference see
    exactly the neighborhoods the model was trained on (`encode_nodes`).

Everything runs on the training device (no CPU workers, no extra libraries). Only
project-company relations exist, so sampling keeps the bipartite invariant.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

EdgeType = tuple[str, str, str]


@dataclass
class SubGraph:
    """Sampled computation graph in local node numbering.

    nodes[nt]   global ids of the local nodes (the seeds first, in the given order)
    layers[i]   edge_index_dict for encoder layer i (local ids)
    local[nt]   global id -> local id lookup (-1 for nodes outside the subgraph)
    """

    nodes: dict[str, torch.Tensor]
    layers: list[dict[EdgeType, torch.Tensor]]
    local: dict[str, torch.Tensor]


class NeighborSampler:
    def __init__(
        self,
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        num_nodes: Mapping[str, int],
        fanouts: Sequence[int],
        seed: int = 0,
        edge_score_dict: Mapping[EdgeType, torch.Tensor] | None = None,
    ) -> None:
        """fanouts[i] = in-edges kept per target node and relation at hop i+1 from the
        seeds (so fanouts[0] feeds the last encoder layer); one entry per layer.
        edge_score_dict: importance of every edge (aligned with edge_index_dict); when
        given, the highest-scored in-edges are kept instead of a random subset."""
        self.ranked = edge_score_dict is not None
        if not fanouts or any(f <= 0 for f in fanouts):
            raise ValueError(f"fanouts must be positive, got {list(fanouts)}")
        self.fanouts = [int(f) for f in fanouts]
        self.num_nodes = dict(num_nodes)
        self.edge_types = list(edge_index_dict)
        device = next(iter(edge_index_dict.values())).device
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        # in-edges grouped by target: src[ptr[t]:ptr[t + 1]] are the sources of target t
        self.src: dict[EdgeType, torch.Tensor] = {}
        self.dst: dict[EdgeType, torch.Tensor] = {}
        self.ptr: dict[EdgeType, torch.Tensor] = {}
        for et, ei in edge_index_dict.items():
            n_dst = self.num_nodes[et[2]]
            if edge_score_dict is not None:
                score = edge_score_dict[et]
                if score.shape[0] != ei.shape[1]:
                    raise ValueError(f"{et}: {score.shape[0]} scores for {ei.shape[1]} edges")
                order = torch.argsort(score, descending=True, stable=True)
                order = order[torch.argsort(ei[1][order], stable=True)]   # best first per target
            else:
                order = torch.argsort(ei[1], stable=True)
            self.src[et] = ei[0][order]
            self.dst[et] = ei[1][order]
            ptr = torch.zeros(n_dst + 1, dtype=torch.long, device=device)
            ptr[1:] = torch.cumsum(torch.bincount(ei[1], minlength=n_dst), 0)
            self.ptr[et] = ptr
        self._local = {
            nt: torch.full((n,), -1, dtype=torch.long, device=device)
            for nt, n in self.num_nodes.items()
        }

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, device=self.device, generator=self.gen)

    def reshuffle(self) -> None:
        """Randomly reorder every target's in-edges. `sample` takes a window of
        consecutive in-edges, so call this (once per epoch) to vary the subsets.
        No-op for a ranked sampler, whose in-edges stay ordered by score."""
        if self.ranked:
            return
        for et in self.edge_types:
            n = self.src[et].shape[0]
            perm = torch.argsort(self._rand(n))
            perm = perm[torch.argsort(self.dst[et][perm], stable=True)]
            self.src[et] = self.src[et][perm]   # dst stays sorted, so ptr is unchanged

    def _sample_in_edges(
        self, et: EdgeType, targets: torch.Tensor, fanout: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Up to `fanout` distinct in-edges of each target -> (src, dst) global ids."""
        ptr = self.ptr[et]
        start = ptr[targets]
        deg = ptr[targets + 1] - start
        take = deg.clamp(max=fanout)
        total = int(take.sum())
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        group = torch.repeat_interleave(torch.arange(targets.shape[0], device=self.device), take)
        within = torch.arange(total, device=self.device) - (torch.cumsum(take, 0) - take)[group]
        # a window of `take` consecutive in-edges starting at a random offset (wraps around)
        if self.ranked:                     # the list is sorted by score: take its head
            pos = start[group] + within
        else:
            offset = (self._rand(targets.shape[0]) * deg).long() \
                .clamp(max=(deg - 1).clamp(min=0))
            pos = start[group] + (offset[group] + within) % deg[group]
        return self.src[et][pos], targets[group]

    def sample(
        self,
        seeds: Mapping[str, torch.Tensor],
        drop_pairs: Mapping[EdgeType, torch.Tensor] | None = None,
        pair_base: int = 0,
    ) -> SubGraph:
        """Sample the computation graph of `seeds` ({node type: unique global ids}).

        drop_pairs: per relation, pair ids (project * pair_base + company) that must not
        appear as message-passing edges (the batch's positive pairs).
        """
        nodes = {nt: s.clone() for nt, s in seeds.items()}
        for nt in self.num_nodes:
            nodes.setdefault(nt, torch.empty(0, dtype=torch.long, device=self.device))
        hops: list[dict[EdgeType, tuple[torch.Tensor, torch.Tensor]]] = []
        for fanout in self.fanouts:
            targets = {nt: v for nt, v in nodes.items()}
            hop: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]] = {}
            found: dict[str, list[torch.Tensor]] = {nt: [] for nt in nodes}
            for et in self.edge_types:
                s_type, _, d_type = et
                src, dst = self._sample_in_edges(et, targets[d_type], fanout)
                if drop_pairs is not None and et in drop_pairs and src.numel():
                    # project is the source of forward relations, the target of rev_ ones
                    pid = (dst * pair_base + src) if et[1].startswith("rev_") \
                        else (src * pair_base + dst)
                    keep = ~torch.isin(pid, drop_pairs[et])
                    src, dst = src[keep], dst[keep]
                hop[et] = (src, dst)
                found[s_type].append(src)
            hops.append(hop)
            for nt, parts in found.items():
                if not parts:
                    continue
                cand = torch.unique(torch.cat(parts))
                known = torch.zeros(self.num_nodes[nt], dtype=torch.bool, device=self.device)
                known[nodes[nt]] = True
                nodes[nt] = torch.cat([nodes[nt], cand[~known[cand]]])

        local = self._local
        for nt, ids in nodes.items():
            local[nt][ids] = torch.arange(ids.shape[0], device=self.device)
        layers = [
            {et: torch.stack([local[et[0]][s], local[et[2]][d]]) for et, (s, d) in hop.items()}
            for hop in reversed(hops)       # the outermost hop feeds the first layer
        ]
        return SubGraph(nodes=nodes, layers=layers, local=local)

    def release(self, sub: SubGraph) -> None:
        """Reset the global -> local lookup after a step (it is shared between steps)."""
        for nt, ids in sub.nodes.items():
            self._local[nt][ids] = -1


def cosine_edge_scores(
    x_src: torch.Tensor, x_dst: torch.Tensor, edge_index: torch.Tensor, batch: int = 500_000
) -> torch.Tensor:
    """Cosine similarity of the two end nodes' features, per edge (edge importance for
    relations that carry no score of their own)."""
    parts = []
    for lo in range(0, edge_index.shape[1], batch):
        s, d = edge_index[0, lo:lo + batch], edge_index[1, lo:lo + batch]
        parts.append(torch.nn.functional.cosine_similarity(x_src[s], x_dst[d], dim=1))
    if not parts:
        return torch.empty(0, device=edge_index.device)
    return torch.cat(parts).float()


@torch.no_grad()
def encode_nodes(
    model: torch.nn.Module,
    x_dict: Mapping[str, torch.Tensor],
    sampler: NeighborSampler,
    seed_nodes: int,
) -> dict[str, torch.Tensor]:
    """Embeddings of every node, each computed on its sampled neighborhood (the same
    neighborhoods as in training when the sampler is ranked). `seed_nodes` nodes per type
    are encoded at a time."""
    model.eval()
    out: dict[str, torch.Tensor] = {}
    n_max = max(sampler.num_nodes.values())
    for lo in range(0, n_max, seed_nodes):
        seeds = {
            nt: torch.arange(lo, min(lo + seed_nodes, n), device=sampler.device)
            for nt, n in sampler.num_nodes.items() if lo < n
        }
        sub = sampler.sample(seeds)
        z = model({nt: x_dict[nt][ids] for nt, ids in sub.nodes.items()}, sub.layers)
        for nt, ids in seeds.items():
            part = z[nt][: ids.shape[0]].float()
            if nt not in out:
                out[nt] = torch.empty(sampler.num_nodes[nt], part.shape[1], device=part.device)
            out[nt][ids] = part
        sampler.release(sub)
    return out

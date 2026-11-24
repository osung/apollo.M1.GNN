"""Training loop for the heterogeneous GNN encoder (full-batch, or mini-batch with
neighbor sampling when `fanouts` is given).

Per epoch:
    1. One forward pass on the full graph -> z_dict for every node
    2. Iterate minibatches of (pos_edge, sampled_neg_edge, weight)
    3. Per minibatch: BPR loss and gradient step

Full-batch forward is feasible on this graph size (~1.6M nodes, 128-d hidden)
and avoids the overhead of subgraph sampling for ~100K positive edges.
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from src.training.losses import bpr_loss
from src.training.neighbor_sampler import NeighborSampler, cosine_edge_scores, encode_nodes
from src.training.sampler import EdgeSampler

CheckpointFn = Callable[[int, dict, nn.Module, list], None]


_AMP_DTYPES = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _amp_context(device: str, amp_dtype: str):
    """Return an autocast context, or nullcontext if amp_dtype=='none'.

    bf16 on A100/H100 doesn't need GradScaler (fp32 exponent range).
    fp16 would need one; we don't support fp16 here — bf16 is strictly
    safer for attention softmax and BPR log-sigmoid on this workload.
    """
    dt = _AMP_DTYPES.get(amp_dtype)
    if dt is None or not str(device).startswith("cuda"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dt)


@dataclass
class TrainResult:
    model: nn.Module
    z_dict: dict[str, torch.Tensor]
    history: list[dict]


def supervised_pair_ids(
    edge_index_dict: dict,
    relation_names: Iterable[str],
    src_node_type: str,
    dst_node_type: str,
    n_dst: int,
) -> dict:
    """Per supervised relation (and its `rev_` twin): id = src_idx * n_dst + dst_idx
    of every message-passing edge, so a batch's positive pairs can be located."""
    names = set(relation_names)
    out = {}
    for et, ei in edge_index_dict.items():
        s, rel, d = et
        if rel.removeprefix("rev_") not in names:
            continue
        if s == src_node_type and d == dst_node_type:
            out[et] = ei[0] * n_dst + ei[1]
        elif s == dst_node_type and d == src_node_type:
            out[et] = ei[1] * n_dst + ei[0]
    return out


def drop_batch_edges(
    edge_index_dict: dict,
    pair_ids: dict,
    batch_ids: torch.Tensor,
    edge_weight_dict: dict | None = None,
) -> tuple[dict, dict | None]:
    """Remove the batch's positive (src, dst) pairs from message passing, in every
    supervised relation and direction.

    Without this, a pair being scored is also a direct neighbor during encoding, so
    an expressive encoder can score training pairs from that edge itself — a signal
    that is absent for held-out pairs at inference time.
    """
    ei_out = dict(edge_index_dict)
    ew_out = dict(edge_weight_dict) if edge_weight_dict is not None else None
    for et, pid in pair_ids.items():
        keep = ~torch.isin(pid, batch_ids)
        ei_out[et] = edge_index_dict[et][:, keep]
        if ew_out is not None and et in ew_out:
            ew_out[et] = edge_weight_dict[et][keep]
    return ei_out, ew_out


def train_encoder(
    *,
    model: nn.Module,
    graph,  # PyG HeteroData
    sampler: EdgeSampler,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cpu",
    src_node_type: str = "project",
    dst_node_type: str = "company",
    log_every: int = 1,
    checkpoint_every: int = 0,
    on_checkpoint: Optional[CheckpointFn] = None,
    start_epoch: int = 1,
    history: Optional[list[dict]] = None,
    p2c_weight: float = 1.0,
    c2p_weight: float = 0.0,
    amp_dtype: str = "none",
    use_mp_edge_weights: bool = False,
    mask_batch_edges: bool = False,
    fanouts: Optional[list[int]] = None,
    eval_chunk_nodes: Optional[int] = None,
    seed: int = 0,
    neighbor_selection: str = "random",
    eval_seed_nodes: int = 20_000,
) -> TrainResult:
    """fanouts: per-layer neighbor fanouts; when given, each step encodes only the batch's
    nodes on a sampled subgraph instead of the whole graph (see NeighborSampler).
    eval_chunk_nodes: target nodes per chunk for the full-graph embedding pass.
    neighbor_selection: "random" (a random subset of in-edges per step; embeddings are then
    computed on the full graph) or "top_score" (always the highest-scored in-edges, for
    training and for the embeddings alike: graph[et].edge_score if present, else the cosine
    similarity of the end nodes' features).
    eval_seed_nodes: nodes per type encoded at a time for the "top_score" embeddings."""
    if neighbor_selection not in ("random", "top_score"):
        raise ValueError(f"unknown neighbor_selection {neighbor_selection!r}")
    if neighbor_selection == "top_score" and not fanouts:
        raise ValueError("neighbor_selection='top_score' needs fanouts (mini-batch training)")
    if amp_dtype not in _AMP_DTYPES:
        raise ValueError(
            f"amp_dtype must be one of {list(_AMP_DTYPES)}, got {amp_dtype!r}"
        )
    if amp_dtype == "fp16":
        raise NotImplementedError(
            "fp16 requires GradScaler; use bf16 on A100/H100 (no scaler needed, "
            "strictly safer for attention softmax and BPR log-sigmoid)."
        )
    model.to(device)

    x_dict = {nt: graph[nt].x.to(device) for nt in (src_node_type, dst_node_type)}
    edge_index_dict = {
        et: graph[et].edge_index.to(device) for et in graph.edge_types
    }
    # Per-relation edge weights from graph[et].edge_weight (royalty=1.0,
    # commercial=0.75, performance=0.5, similarity=0.25 by default).
    # Custom encoders (GFM/LightGCN/SeHGNN) multiply these into their
    # sym-norm message weights so high-priority relations propagate more
    # signal. Models that don't support this (SAGE/GCN/GAT/HGT via
    # HeteroConv) silently ignore the dict.
    # When use_mp_edge_weights=False (CLI --no-mp-edge-weights), pass None
    # so all relations contribute uniform message strength (legacy behavior).
    edge_weight_dict: dict | None = None
    if use_mp_edge_weights:
        edge_weight_dict = {}
        for et in graph.edge_types:
            ew = getattr(graph[et], "edge_weight", None)
            if ew is not None:
                edge_weight_dict[et] = ew.to(device)
        if not edge_weight_dict:
            edge_weight_dict = None

    n_dst = x_dict[dst_node_type].shape[0]
    pair_ids = (
        supervised_pair_ids(edge_index_dict, sampler.relation_names,
                            src_node_type, dst_node_type, n_dst)
        if mask_batch_edges else {}
    )

    neighbor_sampler = None
    if fanouts:
        num_nodes = {nt: x.shape[0] for nt, x in x_dict.items()}
        edge_scores = None
        if neighbor_selection == "top_score":
            edge_scores = {}
            for et, ei in edge_index_dict.items():
                stored = getattr(graph[et], "edge_score", None)
                edge_scores[et] = (
                    stored.to(device).float() if stored is not None
                    else cosine_edge_scores(x_dict[et[0]], x_dict[et[2]], ei)
                )
        neighbor_sampler = NeighborSampler(edge_index_dict, num_nodes, fanouts, seed=seed,
                                           edge_score_dict=edge_scores)
        del edge_scores
    ranked = neighbor_sampler is not None and neighbor_sampler.ranked

    def encode_all() -> dict:
        with _amp_context(device, amp_dtype):
            if ranked:      # the same top-scored neighborhoods as in training
                z = encode_nodes(model, x_dict, neighbor_sampler, eval_seed_nodes)
            elif eval_chunk_nodes:
                z = model.encode_all(x_dict, edge_index_dict, chunk_nodes=eval_chunk_nodes)
            else:
                z = model.encode_all(x_dict, edge_index_dict, edge_weight_dict)
        return {k: v.detach().float().cpu() for k, v in z.items()}

    history = list(history) if history else []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        if neighbor_sampler is not None:
            neighbor_sampler.reshuffle()

        total_loss = 0.0
        n_samples = 0
        for step, batch in enumerate(sampler.iter_epoch()):
            optimizer.zero_grad(set_to_none=True)

            # Encoder forward in autocast (bf16 on CUDA if --amp-dtype bf16).
            # HGT attention softmax and message-passing scatter ops run in
            # bf16 for speed, but we exit autocast before loss so BPR
            # log-sigmoid and score accumulation stay in fp32. Autocast
            # still unwinds backward correctly — gradients land on fp32
            # parameters regardless.
            pos_src = batch.pos_src.to(device)
            pos_dst = batch.pos_dst.to(device)
            neg_dst = batch.neg_dst.to(device)
            weights = batch.weights.to(device)

            neg_src = None
            if batch.neg_src is not None and c2p_weight > 0.0:
                neg_src = batch.neg_src.to(device)

            if neighbor_sampler is not None:
                # encode only the batch's nodes, on their sampled neighborhood
                src_parts = [pos_src] if neg_src is None else [pos_src, neg_src.reshape(-1)]
                seeds = {
                    src_node_type: torch.unique(torch.cat(src_parts)),
                    dst_node_type: torch.unique(torch.cat([pos_dst, neg_dst.reshape(-1)])),
                }
                drop = None
                if pair_ids:
                    batch_ids = pos_src * n_dst + pos_dst
                    drop = {et: batch_ids for et in pair_ids}
                sub = neighbor_sampler.sample(seeds, drop, pair_base=n_dst)
                with _amp_context(device, amp_dtype):
                    z_dict = model({nt: x_dict[nt][ids] for nt, ids in sub.nodes.items()},
                                   sub.layers)
                if step in (0, 9) and epoch == start_epoch:
                    n_nodes = sum(int(v.shape[0]) for v in sub.nodes.values())
                    n_edges = [sum(int(e.shape[1]) for e in layer.values())
                               for layer in sub.layers]
                    print(f"[train] step {step + 1}: seeds="
                          f"{sum(int(v.shape[0]) for v in seeds.values()):,} "
                          f"subgraph nodes={n_nodes:,} edges per layer={n_edges} "
                          f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)
                pos_src = sub.local[src_node_type][pos_src]
                pos_dst = sub.local[dst_node_type][pos_dst]
                neg_dst = sub.local[dst_node_type][neg_dst]
                if neg_src is not None:
                    neg_src = sub.local[src_node_type][neg_src]
                neighbor_sampler.release(sub)
            else:
                step_ei, step_ew = edge_index_dict, edge_weight_dict
                if pair_ids:
                    step_ei, step_ew = drop_batch_edges(
                        edge_index_dict, pair_ids, pos_src * n_dst + pos_dst, edge_weight_dict
                    )
                with _amp_context(device, amp_dtype):
                    z_dict = model(x_dict, step_ei, step_ew)

            if amp_dtype != "none":
                z_dict = {k: v.float() for k, v in z_dict.items()}

            z_src = z_dict[src_node_type][pos_src]             # (B, D)
            z_pos = z_dict[dst_node_type][pos_dst]             # (B, D)
            z_neg = z_dict[dst_node_type][neg_dst]             # (B, K, D)

            pos_scores = (z_src * z_pos).sum(dim=-1)           # (B,)
            neg_scores = torch.einsum("bd,bkd->bk", z_src, z_neg)  # (B, K)
            loss_p2c = bpr_loss(pos_scores, neg_scores, weights=weights)
            loss = p2c_weight * loss_p2c

            if neg_src is not None:
                z_neg_src = z_dict[src_node_type][neg_src]     # (B, K, D)
                neg_scores_c2p = torch.einsum("bd,bkd->bk", z_pos, z_neg_src)
                loss_c2p = bpr_loss(pos_scores, neg_scores_c2p, weights=weights)
                loss = loss + c2p_weight * loss_c2p

            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * pos_src.shape[0]
            n_samples += pos_src.shape[0]

        avg_loss = total_loss / max(n_samples, 1)
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "loss": avg_loss, "elapsed_s": dt})
        if epoch % log_every == 0 or epoch == epochs:
            print(
                f"[train] epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  time={dt:.1f}s"
            )

        should_checkpoint = (
            checkpoint_every
            and on_checkpoint is not None
            and epoch % checkpoint_every == 0
        )
        if should_checkpoint:
            on_checkpoint(epoch, encode_all(), model, history)
            model.train()

    return TrainResult(model=model, z_dict=encode_all(), history=history)

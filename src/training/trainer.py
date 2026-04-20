"""Full-batch training loop for the heterogeneous GNN encoder.

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
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from src.training.losses import bpr_loss
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
) -> TrainResult:
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

    history = list(history) if history else []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.perf_counter()

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
            with _amp_context(device, amp_dtype):
                z_dict = model(x_dict, edge_index_dict)

            if amp_dtype != "none":
                z_dict = {k: v.float() for k, v in z_dict.items()}

            pos_src = batch.pos_src.to(device)
            pos_dst = batch.pos_dst.to(device)
            neg_dst = batch.neg_dst.to(device)
            weights = batch.weights.to(device)

            z_src = z_dict[src_node_type][pos_src]             # (B, D)
            z_pos = z_dict[dst_node_type][pos_dst]             # (B, D)
            z_neg = z_dict[dst_node_type][neg_dst]             # (B, K, D)

            pos_scores = (z_src * z_pos).sum(dim=-1)           # (B,)
            neg_scores = torch.einsum("bd,bkd->bk", z_src, z_neg)  # (B, K)
            loss_p2c = bpr_loss(pos_scores, neg_scores, weights=weights)
            loss = p2c_weight * loss_p2c

            if batch.neg_src is not None and c2p_weight > 0.0:
                neg_src = batch.neg_src.to(device)
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
            and epoch != epochs  # final is saved by the caller
        )
        if should_checkpoint:
            ckpt_z = model.encode_all(x_dict, edge_index_dict)
            ckpt_z_cpu = {k: v.detach().cpu() for k, v in ckpt_z.items()}
            on_checkpoint(epoch, ckpt_z_cpu, model, history)
            model.train()

    z_dict = model.encode_all(x_dict, edge_index_dict)
    return TrainResult(
        model=model,
        z_dict={k: v.detach().cpu() for k, v in z_dict.items()},
        history=history,
    )

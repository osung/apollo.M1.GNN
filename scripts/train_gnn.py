"""Entrypoint: train the GNN encoder on data/processed/graph.pt.

Selects architecture via --layer-type {sage,gcn,gat,hgt} (defaults to sage).
After training, saves z embeddings and runs held-out evaluation against
the same top-100 metrics used by the baselines.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

faiss.omp_set_num_threads(1)

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ranking import evaluate, group_ground_truth
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)
from src.models.encoder import GNNEncoder
from src.training.sampler import EdgeSampler
from src.training.trainer import train_encoder
from src.utils import load_yaml, set_seed

REAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)


def _training_edges(graph) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        edges[et] = graph[rel].edge_index.numpy()
    return edges


def _evaluate_z(
    z_dict: dict[str, torch.Tensor],
    held_out,
    topk: int,
    batch_size: int = 256,
) -> dict[str, dict[str, float]]:
    z_p = z_dict[NODE_TYPE_PROJECT].numpy()
    z_c = z_dict[NODE_TYPE_COMPANY].numpy()

    results: dict[str, dict[str, float]] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = held_out[rel].edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction="project_to_company")
        query_ids = list(gt.keys())
        preds = _batched_topk(
            np.asarray(query_ids, dtype=np.int64), z_p, z_c, topk=topk, bs=batch_size
        )
        results[et] = evaluate(preds, query_ids, gt, ks=(10, topk))
    return results


def _batched_topk(
    query_ids: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray, topk: int, bs: int
) -> np.ndarray:
    Q = query_ids.shape[0]
    out = np.empty((Q, topk), dtype=np.int64)
    for start in range(0, Q, bs):
        end = min(start + bs, Q)
        ids = query_ids[start:end]
        scores = z_src[ids] @ z_dst.T
        part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-cfg", default="config/graph.yaml")
    parser.add_argument("--model-cfg", default="config/model.yaml")
    parser.add_argument("--train-cfg", default="config/train.yaml")
    parser.add_argument("--layer-type", default=None,
                        help="override model.yaml gnn.type: sage|gcn|gat|hgt")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="save intermediate z/model every N epochs (0 = disabled)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="override directory for checkpoints (default: data/processed/checkpoints)",
    )
    parser.add_argument(
        "--save-model-ckpt", action="store_true",
        help="also save model/optimizer state at each checkpoint (needed for --resume)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="path to a model_*.pt checkpoint; continues optimizer + epoch counter",
    )
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph_cfg)
    model_cfg = load_yaml(args.model_cfg)
    train_cfg = load_yaml(args.train_cfg)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    graph_path = paths["processed"]["graph"]
    held_out_path = str(Path(graph_path).with_name("held_out.pt"))
    print(f"[train_gnn] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(held_out_path, weights_only=False)

    layer_type = (args.layer_type or model_cfg["gnn"].get("type", "sage")).lower()
    gnn_cfg = model_cfg["gnn"]
    input_dim = int(gnn_cfg.get("input_dim") or graph[NODE_TYPE_PROJECT].x.shape[1])

    model = GNNEncoder(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim or gnn_cfg["hidden_dim"]),
        output_dim=int(args.output_dim or gnn_cfg["output_dim"]),
        num_layers=int(args.num_layers or gnn_cfg["num_layers"]),
        metadata=graph.metadata(),
        layer_type=layer_type,
        num_heads=int(gnn_cfg.get("num_heads", 4)),
        dropout=float(gnn_cfg.get("dropout", 0.1)),
    )
    # PyG lazy layers (SAGEConv/GATConv with in_channels=-1) need one forward
    # pass to materialize parameter shapes before anything reads .parameters().
    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict)
    n_params = sum(p.numel() for p in model.parameters() if not p.__class__.__name__.startswith("Uninit"))
    print(f"[train_gnn] layer_type={layer_type}  params={n_params:,}")

    tr_cfg = train_cfg["gnn"]
    epochs = args.epochs or int(tr_cfg["epochs"])
    lr = float(tr_cfg["lr"])
    wd = float(tr_cfg.get("weight_decay", 0.0))
    bs = int(tr_cfg["batch_size"])
    num_neg = int(tr_cfg.get("num_neg_samples", 5))

    relation_weights = {
        et: float(graph_cfg["edge_types"][et]["weight"]) for et in REAL_EDGE_TYPES
    }
    training_edges = _training_edges(graph)
    sampler = EdgeSampler(
        edges_per_relation=training_edges,
        relation_weights=relation_weights,
        n_dst=graph[NODE_TYPE_COMPANY].num_nodes,
        num_neg=num_neg,
        batch_size=bs,
        seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print(
        f"[train_gnn] epochs={epochs} batch_size={bs} num_neg={num_neg} "
        f"lr={lr} wd={wd} device={args.device}"
    )

    start_epoch = 1
    prev_history: list[dict] = []
    if args.resume:
        resume_path = Path(args.resume)
        print(f"[train_gnn] resuming from {resume_path}")
        ckpt = torch.load(resume_path, weights_only=False, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        prev_history = list(ckpt.get("history", []))
        print(
            f"[train_gnn] resumed at epoch {ckpt['epoch']}, continuing to {epochs}"
        )

    checkpoint_fn = None
    if args.checkpoint_every and args.checkpoint_every > 0:
        ckpt_dir = Path(args.checkpoint_dir or "data/processed/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{layer_type}_h{int(args.hidden_dim or gnn_cfg['hidden_dim'])}_l{int(args.num_layers or gnn_cfg['num_layers'])}"

        def checkpoint_fn(epoch_idx, z_dict_cpu, mdl, hist):
            suffix = f"{tag}_epoch{epoch_idx:03d}"
            p_path = ckpt_dir / f"project_z_{suffix}.npy"
            c_path = ckpt_dir / f"company_z_{suffix}.npy"
            np.save(p_path, z_dict_cpu[NODE_TYPE_PROJECT].numpy())
            np.save(c_path, z_dict_cpu[NODE_TYPE_COMPANY].numpy())
            print(f"[ckpt] epoch {epoch_idx}: saved {p_path.name}, {c_path.name}")
            if args.save_model_ckpt:
                m_path = ckpt_dir / f"model_{suffix}.pt"
                torch.save(
                    {
                        "epoch": epoch_idx,
                        "model": mdl.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "history": hist,
                        "config": {
                            "layer_type": layer_type,
                            "hidden_dim": int(args.hidden_dim or gnn_cfg["hidden_dim"]),
                            "output_dim": int(args.output_dim or gnn_cfg["output_dim"]),
                            "num_layers": int(args.num_layers or gnn_cfg["num_layers"]),
                            "num_heads": int(gnn_cfg.get("num_heads", 4)),
                            "input_dim": input_dim,
                        },
                    },
                    m_path,
                )
                print(f"[ckpt] epoch {epoch_idx}: saved {m_path.name}")
            h_path = ckpt_dir / f"history_{tag}.json"
            with open(h_path, "w") as f:
                import json
                json.dump(hist, f, indent=2)

        print(
            f"[train_gnn] checkpointing every {args.checkpoint_every} epochs -> {ckpt_dir}"
        )

    result = train_encoder(
        model=model,
        graph=graph,
        sampler=sampler,
        optimizer=optimizer,
        epochs=epochs,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=checkpoint_fn,
        start_epoch=start_epoch,
        history=prev_history,
    )

    out_p = Path(paths["processed"]["project_emb"])
    out_c = Path(paths["processed"]["company_emb"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_p, result.z_dict[NODE_TYPE_PROJECT].numpy())
    np.save(out_c, result.z_dict[NODE_TYPE_COMPANY].numpy())
    print(f"[train_gnn] saved project z -> {out_p}")
    print(f"[train_gnn] saved company z -> {out_c}")

    if not args.no_eval:
        metrics = _evaluate_z(result.z_dict, held_out, topk=args.topk)
        print(f"\n=== GNN [{layer_type}] (project -> company, K={args.topk}) ===")
        for et, m in metrics.items():
            print(f"  {et:12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()

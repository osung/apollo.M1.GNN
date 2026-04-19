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
SIM_EDGE_TYPE = "similarity"


def _load_hard_negatives(path: str) -> dict[int, np.ndarray]:
    """Return {project_idx -> np.ndarray of company indices}."""
    z = np.load(path)
    ei = z["edge_index"]
    src = ei[0].astype(np.int64)
    dst = ei[1].astype(np.int64)
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    dst_s = dst[order]
    change_points = np.concatenate([[True], src_s[1:] != src_s[:-1]])
    group_starts = np.where(change_points)[0]
    group_ends = np.concatenate([group_starts[1:], [src_s.size]])
    hard_map: dict[int, np.ndarray] = {}
    for s, e in zip(group_starts, group_ends):
        hard_map[int(src_s[s])] = dst_s[s:e]
    return hard_map


def _attach_similarity_edges(graph, graph_cfg: dict, override_path: str | None) -> None:
    """Load sim_edges.npz and add (project, similarity, company) + reverse.

    Similarity edges are added for **message passing only**. They are not
    used as training positives — the BPR sampler still draws from the
    three real relations (royalty/commercial/performance).
    """
    sim_cfg = graph_cfg.get("similarity_edges", {}) or {}
    default_path = sim_cfg.get("cache_path") or "data/processed/sim_edges.npz"
    sim_path = Path(override_path or default_path)
    if not sim_path.exists():
        raise FileNotFoundError(
            f"similarity cache not found: {sim_path}. "
            f"Run scripts/build_similarity.py first."
        )

    print(f"[train_gnn] loading similarity edges {sim_path}")
    z = np.load(sim_path)
    ei = np.ascontiguousarray(z["edge_index"]).astype(np.int64)

    n_p_file = int(z["n_project"])
    n_c_file = int(z["n_company"])
    n_p_graph = graph[NODE_TYPE_PROJECT].num_nodes
    n_c_graph = graph[NODE_TYPE_COMPANY].num_nodes
    if (n_p_file, n_c_file) != (n_p_graph, n_c_graph):
        raise ValueError(
            f"sim_edges dimensions ({n_p_file}, {n_c_file}) do not match "
            f"graph ({n_p_graph}, {n_c_graph}); rebuild similarity for this graph."
        )

    edge_index = torch.from_numpy(ei)
    rel = (NODE_TYPE_PROJECT, SIM_EDGE_TYPE, NODE_TYPE_COMPANY)
    graph[rel].edge_index = edge_index

    weights_as_attr = bool(graph_cfg.get("edge_weights_as_attr", True))
    sim_weight = float(
        graph_cfg.get("edge_types", {}).get(SIM_EDGE_TYPE, {}).get("weight", 0.25)
    )
    if weights_as_attr:
        graph[rel].edge_weight = torch.full(
            (edge_index.shape[1],), fill_value=sim_weight, dtype=torch.float32
        )

    add_reverse = bool(graph_cfg.get("reverse_edges", {}).get("enabled", True))
    if add_reverse:
        rev_rel = (NODE_TYPE_COMPANY, f"rev_{SIM_EDGE_TYPE}", NODE_TYPE_PROJECT)
        rev_ei = edge_index.flip(0).contiguous()
        graph[rev_rel].edge_index = rev_ei
        if weights_as_attr:
            graph[rev_rel].edge_weight = torch.full(
                (rev_ei.shape[1],), fill_value=sim_weight, dtype=torch.float32
            )

    print(
        f"[train_gnn] added similarity edges: |E|={edge_index.shape[1]:,}  "
        f"weight={sim_weight}  reverse={add_reverse}"
    )


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
    direction: str = "p2c",
    batch_size: int = 256,
) -> dict[str, dict[str, float]]:
    """Evaluate embeddings against held-out edges.

    direction = "p2c": project queries, company candidates
    direction = "c2p": company queries, project candidates
    """
    z_p = z_dict[NODE_TYPE_PROJECT].numpy()
    z_c = z_dict[NODE_TYPE_COMPANY].numpy()
    if direction == "p2c":
        z_query, z_cand = z_p, z_c
        gt_direction = "project_to_company"
    elif direction == "c2p":
        z_query, z_cand = z_c, z_p
        gt_direction = "company_to_project"
    else:
        raise ValueError(f"direction must be p2c or c2p, got {direction}")

    results: dict[str, dict[str, float]] = {}
    for et in REAL_EDGE_TYPES:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = held_out[rel].edge_index.numpy()
        if ei.shape[1] == 0:
            continue
        gt = group_ground_truth(ei, direction=gt_direction)
        query_ids = list(gt.keys())
        preds = _batched_topk(
            np.asarray(query_ids, dtype=np.int64), z_query, z_cand, topk=topk, bs=batch_size
        )
        results[et] = evaluate(preds, query_ids, gt, ks=(10, topk))
    return results


def _print_metrics(label: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {label} ===")
    for et, m in metrics.items():
        print(f"  {et:12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


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
    parser.add_argument(
        "--graph-path", default=None,
        help="override paths.yaml processed.graph (e.g. pass graph_fp16.pt directly)",
    )
    parser.add_argument("--layer-type", default=None,
                        help="override model.yaml gnn.type: sage|gcn|gat|hgt")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None,
                        help="attention heads for gat/hgt (overrides model.yaml)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--held-out-path", default=None,
        help="override held_out.pt location (default: sibling of graph path)",
    )
    parser.add_argument(
        "--direction", default="p2c", choices=["p2c", "c2p", "both"],
        help="eval direction (both runs project->company and company->project)",
    )
    parser.add_argument(
        "--save-metrics", default=None,
        help="path to write JSON with eval metrics (final run, and per-checkpoint if --eval-at-checkpoint)",
    )
    parser.add_argument(
        "--eval-at-checkpoint", action="store_true",
        help="run held-out evaluation at each checkpoint epoch (mid-training monitoring)",
    )
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
    parser.add_argument(
        "--with-similarity", action="store_true",
        help="inject (project, similarity, company) edges from sim_edges.npz into the graph "
             "for message passing. Training loss still targets royalty/commercial/performance only.",
    )
    parser.add_argument(
        "--sim-path", default=None,
        help="override sim_edges.npz location (default: graph_cfg.similarity_edges.cache_path "
             "or data/processed/sim_edges.npz)",
    )
    parser.add_argument(
        "--hard-neg-path", default=None,
        help="p2c hard negatives npz (row 0=project idx, row 1=company idx)",
    )
    parser.add_argument(
        "--hard-neg-path-c2p", default=None,
        help="c2p hard negatives npz (row 0=company idx, row 1=project idx). "
             "Auto-enables symmetric training; still pair with --c2p-weight.",
    )
    parser.add_argument(
        "--hard-ratio", type=float, default=0.5,
        help="fraction of num_neg drawn from each side's hard pool (rest random)",
    )
    parser.add_argument(
        "--num-neg", type=int, default=None,
        help="override train.yaml gnn.num_neg_samples",
    )
    parser.add_argument(
        "--c2p-weight", type=float, default=0.0,
        help="weight of the company→project BPR term added to the loss; "
             "0 disables c2p training (default). Typical: 1.0 for symmetric.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="disable L2 normalization of encoder output. Frees the BPR "
             "score from the [-1, 1] bound so hard negatives can keep "
             "providing gradient; FAISS inner-product retrieval still works.",
    )
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    graph_cfg = load_yaml(args.graph_cfg)
    model_cfg = load_yaml(args.model_cfg)
    train_cfg = load_yaml(args.train_cfg)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    graph_path = args.graph_path or paths["processed"]["graph"]
    default_held_out = str(Path(paths["processed"]["graph"]).with_name("held_out.pt"))
    held_out_path = args.held_out_path or default_held_out
    print(f"[train_gnn] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    held_out = torch.load(held_out_path, weights_only=False)

    for nt in graph.node_types:
        x = getattr(graph[nt], "x", None)
        if x is not None and x.dtype != torch.float32:
            print(f"[train_gnn] casting {nt}.x {x.dtype} -> torch.float32")
            graph[nt].x = x.float()

    if args.with_similarity:
        _attach_similarity_edges(graph, graph_cfg, args.sim_path)

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
        num_heads=int(args.num_heads or gnn_cfg.get("num_heads", 4)),
        dropout=float(gnn_cfg.get("dropout", 0.1)),
        normalize_output=not args.no_normalize,
    )
    # PyG lazy layers (SAGEConv/GATConv with in_channels=-1) need one forward
    # pass to materialize parameter shapes before anything reads .parameters().
    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict)
    n_params = sum(p.numel() for p in model.parameters() if not p.__class__.__name__.startswith("Uninit"))
    print(
        f"[train_gnn] layer_type={layer_type}  params={n_params:,}  "
        f"normalize_output={model.normalize_output}"
    )

    tr_cfg = train_cfg["gnn"]
    epochs = args.epochs or int(tr_cfg["epochs"])
    lr = float(tr_cfg["lr"])
    wd = float(tr_cfg.get("weight_decay", 0.0))
    bs = int(tr_cfg["batch_size"])
    num_neg = int(args.num_neg or tr_cfg.get("num_neg_samples", 5))

    hard_neg_map: dict[int, np.ndarray] | None = None
    if args.hard_neg_path:
        print(f"[train_gnn] loading p2c hard negatives {args.hard_neg_path}")
        hard_neg_map = _load_hard_negatives(args.hard_neg_path)
        pool_sizes = np.array([len(v) for v in hard_neg_map.values()])
        print(
            f"[train_gnn]   p2c: {len(hard_neg_map):,} projects  "
            f"(min={pool_sizes.min()}, median={int(np.median(pool_sizes))}, "
            f"max={pool_sizes.max()})"
        )

    hard_neg_map_c2p: dict[int, np.ndarray] | None = None
    if args.hard_neg_path_c2p:
        print(f"[train_gnn] loading c2p hard negatives {args.hard_neg_path_c2p}")
        hard_neg_map_c2p = _load_hard_negatives(args.hard_neg_path_c2p)
        pool_sizes = np.array([len(v) for v in hard_neg_map_c2p.values()])
        print(
            f"[train_gnn]   c2p: {len(hard_neg_map_c2p):,} companies  "
            f"(min={pool_sizes.min()}, median={int(np.median(pool_sizes))}, "
            f"max={pool_sizes.max()})"
        )

    c2p_weight = float(args.c2p_weight)
    c2p_enabled = c2p_weight > 0.0 or hard_neg_map_c2p is not None
    if c2p_enabled and c2p_weight == 0.0:
        c2p_weight = 1.0  # auto-enable with default weight when a c2p pool is provided
        print(f"[train_gnn]   c2p training auto-enabled (c2p_weight=1.0)")

    relation_weights = {
        et: float(graph_cfg["edge_types"][et]["weight"]) for et in REAL_EDGE_TYPES
    }
    training_edges = _training_edges(graph)
    sampler = EdgeSampler(
        edges_per_relation=training_edges,
        relation_weights=relation_weights,
        n_dst=graph[NODE_TYPE_COMPANY].num_nodes,
        n_src=graph[NODE_TYPE_PROJECT].num_nodes,
        num_neg=num_neg,
        batch_size=bs,
        seed=seed,
        hard_neg_map=hard_neg_map,
        hard_neg_map_c2p=hard_neg_map_c2p,
        hard_ratio=args.hard_ratio,
        c2p_enabled=c2p_enabled,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print(
        f"[train_gnn] epochs={epochs} batch_size={bs} num_neg={num_neg} "
        f"(p2c n_hard={sampler.n_hard}, n_random={sampler.n_random}; "
        f"c2p enabled={sampler.c2p_enabled}, weight={c2p_weight}, "
        f"n_hard={sampler.n_hard_c2p}, n_random={sampler.n_random_c2p}) "
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

            if args.eval_at_checkpoint:
                directions = ["p2c", "c2p"] if args.direction == "both" else [args.direction]
                ckpt_metrics: dict[str, dict] = {}
                for d in directions:
                    m = _evaluate_z(z_dict_cpu, held_out, topk=args.topk, direction=d)
                    _print_metrics(
                        f"GNN [{layer_type}] epoch {epoch_idx} [{d}] K={args.topk}", m,
                    )
                    ckpt_metrics[d] = m
                mm_path = ckpt_dir / f"metrics_{suffix}.json"
                with open(mm_path, "w") as f:
                    import json
                    json.dump(ckpt_metrics, f, indent=2)
                print(f"[ckpt] epoch {epoch_idx}: saved {mm_path.name}")

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
        c2p_weight=c2p_weight,
    )

    out_p = Path(paths["processed"]["project_emb"])
    out_c = Path(paths["processed"]["company_emb"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_p, result.z_dict[NODE_TYPE_PROJECT].numpy())
    np.save(out_c, result.z_dict[NODE_TYPE_COMPANY].numpy())
    print(f"[train_gnn] saved project z -> {out_p}")
    print(f"[train_gnn] saved company z -> {out_c}")

    if not args.no_eval:
        directions = ["p2c", "c2p"] if args.direction == "both" else [args.direction]
        final_metrics: dict[str, dict] = {}
        for d in directions:
            metrics = _evaluate_z(result.z_dict, held_out, topk=args.topk, direction=d)
            _print_metrics(f"GNN [{layer_type}] final [{d}] K={args.topk}", metrics)
            final_metrics[d] = metrics
        if args.save_metrics:
            out = Path(args.save_metrics)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                import json
                json.dump(final_metrics, f, indent=2)
            print(f"[train_gnn] saved metrics -> {out}")


if __name__ == "__main__":
    main()

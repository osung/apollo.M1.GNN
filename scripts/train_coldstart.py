"""Train cold-start projection MLPs.

Inputs:
  - `project_z.npy`, `company_z.npy`: GNN-produced embeddings for every
    training-graph node (from a completed `train_gnn.py` run).
  - `graph.pt`: source of `norm_embed` per node.

Outputs (to --output-dir, default next to the z files):
  - `projection_mlp_project.pt`
  - `projection_mlp_company.pt`

Each checkpoint holds state_dict + config so it can be loaded without
knowing the hyperparameters at call time.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT
from src.models.projection import ProjectionMLP, cosine_mse_loss
from src.utils import load_yaml, set_seed


def _train_one_side(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    val_ratio: float,
    device: str,
    log_every: int,
) -> tuple[ProjectionMLP, dict]:
    """Fit a projection MLP for one node type and return (model, history)."""
    n, in_dim = features.shape
    if targets.shape[0] != n:
        raise ValueError(
            f"{name}: feature/target row mismatch ({n} vs {targets.shape[0]})"
        )

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    x_train = torch.from_numpy(features[train_idx]).to(device)
    y_train = torch.from_numpy(targets[train_idx]).to(device)
    x_val = torch.from_numpy(features[val_idx]).to(device)
    y_val = torch.from_numpy(targets[val_idx]).to(device)

    model = ProjectionMLP(
        input_dim=in_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = x_train.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)

    history: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = 0

    print(
        f"[coldstart:{name}] n={n:,}  train={n_train:,}  val={n_val:,}  "
        f"in={in_dim}  out={output_dim}  hidden={hidden_dim}  layers={num_layers}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)

        running = 0.0
        seen = 0
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, n_train)
            sel = perm[start:end]
            xb = x_train[sel]
            yb = y_train[sel]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = cosine_mse_loss(pred, yb, alpha=alpha)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val)
            val_loss = float(cosine_mse_loss(pred_val, y_val, alpha=alpha))
            val_cos = float(
                torch.nn.functional.cosine_similarity(pred_val, y_val, dim=-1).mean()
            )
        dt = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_cos": val_cos,
                "elapsed_s": dt,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if epoch % log_every == 0 or epoch == epochs:
            print(
                f"[coldstart:{name}] epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"val_cos={val_cos:.4f}  t={dt:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(
        f"[coldstart:{name}] best val={best_val:.5f} at epoch {best_epoch}"
    )
    return model, {"history": history, "best_val": best_val, "best_epoch": best_epoch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--train-cfg", default="config/train.yaml")
    parser.add_argument(
        "--graph-path", default=None,
        help="override paths.yaml processed.graph",
    )
    parser.add_argument(
        "--project-z", default=None,
        help="project embedding .npy (default: paths.yaml processed.project_emb)",
    )
    parser.add_argument(
        "--company-z", default=None,
        help="company embedding .npy (default: paths.yaml processed.company_emb)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="where to save the two MLP .pt files (default: sibling of z files)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="optional tag appended to output filenames so multiple runs "
             "(different GNN backbones) don't clobber each other",
    )

    # MLP hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="hidden layer width (default: max(256, 2*output_dim))")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="override train.yaml projection_mlp.epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="MSE weight in MSE+cosine loss (0..1)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--side", default="both", choices=["project", "company", "both"])
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    train_cfg = load_yaml(args.train_cfg).get("projection_mlp", {})

    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8192))
    lr = args.lr or float(train_cfg.get("lr", 1e-3))
    weight_decay = args.weight_decay or float(train_cfg.get("weight_decay", 1e-6))

    graph_path = args.graph_path or paths["processed"]["graph"]
    project_z_path = args.project_z or paths["processed"]["project_emb"]
    company_z_path = args.company_z or paths["processed"]["company_emb"]

    print(f"[coldstart] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)

    project_norm = graph[NODE_TYPE_PROJECT].x.float().numpy()
    company_norm = graph[NODE_TYPE_COMPANY].x.float().numpy()

    if args.side in ("project", "both"):
        print(f"[coldstart] loading project z: {project_z_path}")
        project_z = np.load(project_z_path).astype(np.float32)
    else:
        project_z = None
    if args.side in ("company", "both"):
        print(f"[coldstart] loading company z: {company_z_path}")
        company_z = np.load(company_z_path).astype(np.float32)
    else:
        company_z = None

    out_dir = Path(args.output_dir) if args.output_dir else Path(project_z_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    set_seed(42)

    def _run(name: str, features: np.ndarray, targets: np.ndarray) -> None:
        output_dim = int(targets.shape[1])
        hidden_dim = args.hidden_dim or max(256, output_dim * 2)
        model, stats = _train_one_side(
            name,
            features=features,
            targets=targets,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            alpha=args.alpha,
            val_ratio=args.val_ratio,
            device=args.device,
            log_every=args.log_every,
        )
        ckpt_path = out_dir / f"projection_mlp_{name}{tag}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "input_dim": int(features.shape[1]),
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "alpha": args.alpha,
                    "val_ratio": args.val_ratio,
                    "best_val": stats["best_val"],
                    "best_epoch": stats["best_epoch"],
                },
                "history": stats["history"],
            },
            ckpt_path,
        )
        print(f"[coldstart:{name}] saved -> {ckpt_path}")

    if project_z is not None:
        _run("project", project_norm, project_z)
    if company_z is not None:
        _run("company", company_norm, company_z)


if __name__ == "__main__":
    main()

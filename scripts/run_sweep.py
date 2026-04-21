"""Parameter sweep driver for GNN experiments.

Fixes one architecture (--model) and sweeps over:
  - hard_count in {0, 5, 10, 20}     (number of hard negatives per positive)
  - sim_count  in {0, 5, 10, 20}     (top-k similarity edges used for MP)
  - direction  in {p2c_only, c2p_only, both}

Total: 4 * 4 * 3 = 48 runs per invocation.

Each run is launched as a subprocess so OOM / crash in one run does not
terminate the sweep. Results (final metrics + elapsed time + status) are
appended to an Excel file incrementally after every run, so partial
progress is never lost if the Colab session disconnects.

Usage (Colab):
    !python scripts/run_sweep.py sage --hidden-dim 128 --epochs 50

Specifics:
  - hard_count=0 -> no hard-neg file loaded, hard_ratio=0 (pure random).
  - hard_count=k (k>0) -> num_neg=20, hard_ratio=k/20 (so k hard + (20-k)
    random). c2p hard-neg file is loaded only when c2p direction is trained.
  - sim_count=0 -> --with-similarity is not passed (graph has only the
    three real relations + reverses).
  - direction='p2c_only' -> c2p_weight=0 (and c2p hard negs not loaded).
  - direction='c2p_only' -> p2c_weight=0.
  - direction='both' -> 1.0/1.0 symmetric.

OOM handling: subprocess stdout+stderr scanned for 'CUDA out of memory' /
'torch.OutOfMemoryError'. Those cells are filled with literal string 'OOM'
in the metric columns (instead of a number) so they stand out in Excel.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


RELATIONS = ("royalty", "commercial", "performance")
HARD_COUNTS = (0, 5, 10, 20)
SIM_COUNTS = (0, 5, 10, 20)
DIRECTIONS = (
    ("p2c_only", 1.0, 0.0),
    ("c2p_only", 0.0, 1.0),
    ("both",     1.0, 1.0),
)
METRIC_FIELDS = ("recall@10", "ndcg@10", "recall@100", "ndcg@100")


def build_command(
    *,
    model: str,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    epochs: int,
    hard_count: int,
    sim_count: int,
    p2c_w: float,
    c2p_w: float,
    graph_path: str,
    held_out_path: str,
    hard_neg_p2c_path: str,
    hard_neg_c2p_path: str,
    sim_path_template: str,
    metrics_path: Path,
) -> list[str]:
    cmd = [
        sys.executable, "scripts/train_gnn.py",
        "--graph-path", graph_path,
        "--held-out-path", held_out_path,
        "--layer-type", model,
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--epochs", str(epochs),
        "--device", "cuda",
        "--num-neg", "20",
        "--no-normalize",
        "--direction", "both",            # always evaluate both directions
        "--p2c-weight", f"{p2c_w}",
        "--c2p-weight", f"{c2p_w}",
        "--save-metrics", str(metrics_path),
        "--checkpoint-every", "0",        # no mid-training checkpoints in sweep
    ]

    # Hard negatives
    if hard_count == 0:
        cmd.extend(["--hard-ratio", "0.0"])
    else:
        hard_ratio = hard_count / 20.0
        cmd.extend([
            "--hard-ratio", f"{hard_ratio}",
            "--hard-neg-path", hard_neg_p2c_path,
        ])
        if c2p_w > 0.0:
            cmd.extend(["--hard-neg-path-c2p", hard_neg_c2p_path])

    # Similarity edges (message passing injection)
    if sim_count > 0:
        cmd.extend([
            "--with-similarity",
            "--sim-path", sim_path_template.format(k=sim_count),
        ])
    return cmd


def parse_metrics_json(path: Path) -> dict:
    """Flatten {direction: {relation: {metric: value}}} into column-per-cell."""
    with open(path) as f:
        raw = json.load(f)
    flat: dict[str, float | str] = {}
    for d in ("p2c", "c2p"):
        rel_block = raw.get(d, {})
        for r in RELATIONS:
            m = rel_block.get(r, {})
            for f_ in METRIC_FIELDS:
                key = f"{d}_{r}_{f_.replace('@', '')}"
                flat[key] = m.get(f_)
    return flat


def run_once(cmd: list[str], metrics_path: Path) -> tuple[str, float, str]:
    """Run training subprocess.

    Returns (status, elapsed_seconds, log_tail).
    status in {'OK', 'OOM', 'TIMEOUT', 'NO_METRICS', 'ERROR'}.
    """
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=6 * 3600,
        )
    except subprocess.TimeoutExpired as e:
        return "TIMEOUT", time.time() - t0, str(e)

    elapsed = time.time() - t0
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    tail = combined[-800:]

    if ("CUDA out of memory" in combined
            or "torch.OutOfMemoryError" in combined
            or "OutOfMemoryError" in combined):
        return "OOM", elapsed, tail

    if result.returncode != 0:
        return "ERROR", elapsed, tail

    if not metrics_path.exists():
        return "NO_METRICS", elapsed, tail

    return "OK", elapsed, ""


def save_incremental(rows: list[dict], excel_path: Path) -> None:
    df = pd.DataFrame(rows)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_path, index=False, engine="openpyxl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["sage", "gcn", "gat", "hgt", "lightgcn", "sehgnn"],
        help="GNN architecture to sweep",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--num-heads", type=int, default=4,
        help="only used by gat / hgt / sehgnn",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--graph-path",
        default="/content/drive/MyDrive/apollo.M1.GNN/graph.pt",
        help="override graph.pt path (Colab default)",
    )
    parser.add_argument(
        "--held-out-path", default="data/processed/held_out.pt",
    )
    parser.add_argument(
        "--hard-neg-p2c-path",
        default="data/processed/hard_negatives_p2c_clean.npz",
    )
    parser.add_argument(
        "--hard-neg-c2p-path",
        default="data/processed/hard_negatives_c2p_clean.npz",
    )
    parser.add_argument(
        "--sim-path-template",
        default="data/processed/sim_edges_top{k}.npz",
        help="template with {k} placeholder",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/apollo.M1.GNN/sweep",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip combos whose metrics json already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the commands and exit without running anything",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = (
        output_dir
        / f"sweep_{args.model}_h{args.hidden_dim}_ep{args.epochs}.xlsx"
    )

    print(f"[sweep] model={args.model}  hd={args.hidden_dim}  "
          f"ep={args.epochs}  out={excel_path}")
    total = len(HARD_COUNTS) * len(SIM_COUNTS) * len(DIRECTIONS)
    print(f"[sweep] {total} runs queued "
          f"({len(HARD_COUNTS)} hard x {len(SIM_COUNTS)} sim x "
          f"{len(DIRECTIONS)} direction)")

    rows: list[dict] = []
    idx = 0
    t_sweep = time.time()

    for hard_count in HARD_COUNTS:
        for sim_count in SIM_COUNTS:
            for dir_name, p2c_w, c2p_w in DIRECTIONS:
                idx += 1
                tag = (
                    f"{args.model}_h{args.hidden_dim}"
                    f"_hn{hard_count}_sim{sim_count}_{dir_name}"
                    f"_ep{args.epochs}"
                )
                metrics_path = output_dir / f"metrics_{tag}.json"

                print(f"\n=== [{idx}/{total}] {tag} ===")
                cmd = build_command(
                    model=args.model,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    epochs=args.epochs,
                    hard_count=hard_count,
                    sim_count=sim_count,
                    p2c_w=p2c_w,
                    c2p_w=c2p_w,
                    graph_path=args.graph_path,
                    held_out_path=args.held_out_path,
                    hard_neg_p2c_path=args.hard_neg_p2c_path,
                    hard_neg_c2p_path=args.hard_neg_c2p_path,
                    sim_path_template=args.sim_path_template,
                    metrics_path=metrics_path,
                )

                if args.dry_run:
                    print("  CMD:", " ".join(cmd))
                    continue

                if args.resume and metrics_path.exists():
                    print(f"  skip (resume): {metrics_path.name}")
                    status, elapsed, tail = "OK", 0.0, ""
                else:
                    status, elapsed, tail = run_once(cmd, metrics_path)
                    print(f"  status={status}  elapsed={elapsed:.1f}s")
                    if status != "OK":
                        print(f"  log tail: ...{tail[-300:]}")

                row: dict = {
                    "model": args.model,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "epochs": args.epochs,
                    "hard_count": hard_count,
                    "sim_count": sim_count,
                    "direction": dir_name,
                    "p2c_weight": p2c_w,
                    "c2p_weight": c2p_w,
                    "status": status,
                    "elapsed_s": round(elapsed, 1) if elapsed else 0.0,
                }

                if status == "OK" and metrics_path.exists():
                    try:
                        row.update(parse_metrics_json(metrics_path))
                    except Exception as e:  # noqa: BLE001
                        row["status"] = f"PARSE_ERROR: {e}"
                        for d in ("p2c", "c2p"):
                            for r in RELATIONS:
                                for f_ in METRIC_FIELDS:
                                    key = f"{d}_{r}_{f_.replace('@', '')}"
                                    row[key] = "PARSE_ERROR"
                else:
                    # Fill metric cells with the status marker so the Excel
                    # sheet clearly shows OOM / TIMEOUT / ERROR runs.
                    for d in ("p2c", "c2p"):
                        for r in RELATIONS:
                            for f_ in METRIC_FIELDS:
                                key = f"{d}_{r}_{f_.replace('@', '')}"
                                row[key] = status

                rows.append(row)
                save_incremental(rows, excel_path)
                print(f"  wrote {idx}/{total} rows -> {excel_path}")

    total_elapsed = time.time() - t_sweep
    print(
        f"\n[sweep] DONE  total={total}  elapsed={total_elapsed/60:.1f} min  "
        f"-> {excel_path}"
    )


if __name__ == "__main__":
    main()

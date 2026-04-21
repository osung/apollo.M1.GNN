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

Checkpointing and resume:
  - Each inner run saves a checkpoint every --checkpoint-every epochs
    (default 20) plus a final-epoch checkpoint. Files land in
    --checkpoint-dir (defaults to the sweep output dir).
  - --resume: skip any sweep cell whose metrics.json already exists
    (completed runs are reused as-is).
  - --resume-checkpoint PATH: parse the filename (e.g.
    model_sage_h128_l2_hr50_c2p100_sim10_nonorm_epoch040.pt) to figure
    out which sweep cell was mid-training. The driver skips all cells
    before it, re-enters train_gnn.py with --resume PATH for that cell
    to continue from the saved epoch, then runs the remaining cells.
    Implies --resume semantics for cells already on disk.

OOM handling: subprocess stdout+stderr scanned for 'CUDA out of memory' /
'torch.OutOfMemoryError'. Those cells are filled with literal string 'OOM'
in the metric columns (instead of a number) so they stand out in Excel.
"""
from __future__ import annotations

import argparse
import json
import re
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
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume_from: str | None,
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
        "--checkpoint-every", str(checkpoint_every),
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    if checkpoint_every > 0:
        cmd.append("--save-model-ckpt")

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

    if resume_from:
        cmd.extend(["--resume", resume_from])
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


def parse_checkpoint_name(path: Path) -> dict:
    """Reverse-engineer a sweep cell from a saved model_*.pt filename.

    Format (from train_gnn.py _build_experiment_tag):
        model_{layer}_h{hd}_l{L}[_nh{N}][_hr{R}][_p2c{P}][_c2p{C}][_sim{K}][_nonorm][_bf16]_epoch{E}.pt

    Returns dict with keys: model, hidden_dim, num_layers, num_heads,
    hard_count, sim_count, direction, epoch.
    """
    name = path.name
    m = re.match(
        r"model_([a-z]+)_h(\d+)_l(\d+)(_nh\d+)?(.*?)_epoch(\d+)\.pt$",
        name,
    )
    if not m:
        raise ValueError(f"cannot parse checkpoint filename: {name!r}")
    model = m.group(1)
    hidden_dim = int(m.group(2))
    num_layers = int(m.group(3))
    num_heads_grp = m.group(4)
    num_heads = int(num_heads_grp[3:]) if num_heads_grp else 4
    suffix = m.group(5) or ""
    epoch = int(m.group(6))

    hard_ratio = 0.0
    p2c_w = 1.0  # tag default: p2c suffix is omitted when p2c==1.0
    c2p_w = 0.0  # tag default: c2p suffix is omitted when c2p==0
    sim_count = 0

    if mm := re.search(r"_hr(\d+)", suffix):
        hard_ratio = int(mm.group(1)) / 100.0
    if mm := re.search(r"_p2c(\d+)", suffix):
        p2c_w = int(mm.group(1)) / 100.0
    if mm := re.search(r"_c2p(\d+)", suffix):
        c2p_w = int(mm.group(1)) / 100.0
    if mm := re.search(r"_sim(\d+)", suffix):
        sim_count = int(mm.group(1))

    hard_count = int(round(hard_ratio * 20))

    if p2c_w > 0 and c2p_w == 0:
        direction = "p2c_only"
    elif p2c_w == 0 and c2p_w > 0:
        direction = "c2p_only"
    elif p2c_w > 0 and c2p_w > 0:
        direction = "both"
    else:
        raise ValueError(f"Unknown direction from p2c={p2c_w}, c2p={c2p_w}")

    return {
        "model": model,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hard_count": hard_count,
        "sim_count": sim_count,
        "direction": direction,
        "epoch": epoch,
    }


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


def _fill_metric_columns(row: dict, value) -> None:
    """Populate all 24 metric cells with a scalar / status string."""
    for d in ("p2c", "c2p"):
        for r in RELATIONS:
            for f_ in METRIC_FIELDS:
                key = f"{d}_{r}_{f_.replace('@', '')}"
                row[key] = value


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
        "--checkpoint-every", type=int, default=20,
        help="save a model checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--graph-path",
        default="/content/drive/MyDrive/apollo.M1.GNN/graph.pt",
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
    )
    parser.add_argument(
        "--output-dir",
        default="/content/drive/MyDrive/apollo.M1.GNN/sweep",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip cells whose metrics json already exists",
    )
    parser.add_argument(
        "--resume-checkpoint", default=None,
        help=("path to a model_*.pt file. Driver parses the filename to "
              "locate the matching sweep cell, skips cells before it, "
              "passes --resume PATH to train_gnn.py for that cell to "
              "continue training, then runs the remaining cells"),
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

    # ---- Parse --resume-checkpoint, if given ----
    resume_cell: tuple[int, int, str] | None = None
    resume_path: str | None = None
    resume_from_epoch: int | None = None
    if args.resume_checkpoint:
        parsed = parse_checkpoint_name(Path(args.resume_checkpoint))
        if parsed["model"] != args.model:
            raise SystemExit(
                f"checkpoint model {parsed['model']!r} != --model "
                f"{args.model!r}"
            )
        if parsed["hidden_dim"] != args.hidden_dim:
            print(
                f"WARNING: checkpoint hd={parsed['hidden_dim']} "
                f"!= --hidden-dim {args.hidden_dim}"
            )
        if parsed["hard_count"] not in HARD_COUNTS:
            raise SystemExit(
                f"checkpoint hard_count={parsed['hard_count']} not in sweep "
                f"grid {HARD_COUNTS}"
            )
        if parsed["sim_count"] not in SIM_COUNTS:
            raise SystemExit(
                f"checkpoint sim_count={parsed['sim_count']} not in sweep "
                f"grid {SIM_COUNTS}"
            )
        resume_cell = (
            parsed["hard_count"],
            parsed["sim_count"],
            parsed["direction"],
        )
        resume_path = args.resume_checkpoint
        resume_from_epoch = parsed["epoch"]
        print(
            f"[sweep] resume-checkpoint parsed: {resume_cell} "
            f"at ep{resume_from_epoch}  (file={Path(resume_path).name})"
        )

    # --resume-checkpoint implies skipping completed cells via --resume
    use_skip_completed = args.resume or bool(args.resume_checkpoint)

    print(
        f"[sweep] model={args.model}  hd={args.hidden_dim}  "
        f"ep={args.epochs}  ckpt_every={args.checkpoint_every}  "
        f"out={excel_path}"
    )
    total = len(HARD_COUNTS) * len(SIM_COUNTS) * len(DIRECTIONS)
    print(
        f"[sweep] {total} cells queued  "
        f"({len(HARD_COUNTS)} hard x {len(SIM_COUNTS)} sim x "
        f"{len(DIRECTIONS)} direction)"
    )

    rows: list[dict] = []
    idx = 0
    reached_resume_cell = resume_cell is None
    t_sweep = time.time()

    for hard_count in HARD_COUNTS:
        for sim_count in SIM_COUNTS:
            for dir_name, p2c_w, c2p_w in DIRECTIONS:
                idx += 1
                current = (hard_count, sim_count, dir_name)
                tag = (
                    f"{args.model}_h{args.hidden_dim}"
                    f"_hn{hard_count}_sim{sim_count}_{dir_name}"
                    f"_ep{args.epochs}"
                )
                metrics_path = output_dir / f"metrics_{tag}.json"

                # Determine whether to pass --resume PATH to this cell
                this_resume: str | None = None
                if resume_cell is not None and current == resume_cell:
                    this_resume = resume_path
                    reached_resume_cell = True

                base_row = {
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
                }

                print(f"\n=== [{idx}/{total}] {tag} ===")

                # Case A: metrics exist and resume mode -> reuse
                if use_skip_completed and metrics_path.exists():
                    try:
                        row = {**base_row,
                               "status": "REUSED",
                               "elapsed_s": 0.0,
                               **parse_metrics_json(metrics_path)}
                        rows.append(row)
                        save_incremental(rows, excel_path)
                        print(f"  reuse (metrics exist): {metrics_path.name}")
                        continue
                    except Exception as e:  # noqa: BLE001
                        print(f"  metrics re-parse failed ({e}); re-running")

                # Case B: before the resume-checkpoint cell, no metrics
                # -> skip silently but log
                if resume_cell is not None and not reached_resume_cell:
                    row = {**base_row,
                           "status": "SKIPPED_PRE_RESUME",
                           "elapsed_s": 0.0}
                    _fill_metric_columns(row, "SKIPPED")
                    rows.append(row)
                    save_incremental(rows, excel_path)
                    print(
                        "  skip (pre resume-checkpoint cell; "
                        "no metrics on disk either)"
                    )
                    continue

                # Case C: run this cell (fresh or resumed)
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
                    checkpoint_dir=output_dir,
                    checkpoint_every=args.checkpoint_every,
                    resume_from=this_resume,
                )

                if args.dry_run:
                    print("  CMD:", " ".join(cmd))
                    if this_resume:
                        print(f"  RESUME FROM: {this_resume}")
                    continue

                if this_resume:
                    print(f"  resuming from: {this_resume}")

                status, elapsed, tail = run_once(cmd, metrics_path)
                print(f"  status={status}  elapsed={elapsed:.1f}s")
                if status != "OK":
                    print(f"  log tail: ...{tail[-300:]}")

                row = {**base_row,
                       "status": status,
                       "elapsed_s": round(elapsed, 1)}

                if status == "OK" and metrics_path.exists():
                    try:
                        row.update(parse_metrics_json(metrics_path))
                    except Exception as e:  # noqa: BLE001
                        row["status"] = f"PARSE_ERROR: {e}"
                        _fill_metric_columns(row, "PARSE_ERROR")
                else:
                    _fill_metric_columns(row, status)

                rows.append(row)
                save_incremental(rows, excel_path)
                print(f"  wrote {idx}/{total} rows -> {excel_path}")

    if resume_cell is not None and not reached_resume_cell:
        print(
            f"\nWARNING: --resume-checkpoint cell {resume_cell} never "
            f"matched any entry in the sweep grid; nothing was resumed."
        )

    total_elapsed = time.time() - t_sweep
    print(
        f"\n[sweep] DONE  total={total}  elapsed={total_elapsed/60:.1f} min  "
        f"-> {excel_path}"
    )


if __name__ == "__main__":
    main()

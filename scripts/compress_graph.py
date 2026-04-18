"""Convert node features in graph.pt from float32 to float16 (and back).

Motivation: the base graph artifact is ~4.6 GB, and ~99% of that is the
L2-normalized `norm_embed` feature matrices. These are high-entropy
floats that compress poorly with zstd/gzip (~10% reduction). Halving
the precision cuts the file roughly in half with negligible impact on
cosine similarity ranking, which is what downstream models use.

Usage:
    python scripts/compress_graph.py                       # fp32 -> fp16
    python scripts/compress_graph.py --mode decompress     # fp16 -> fp32
    python scripts/compress_graph.py --input ... --output ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def _cast_node_features(graph, dtype: torch.dtype) -> dict[str, tuple]:
    """Cast .x for every node type; return (old_dtype, new_dtype) per type."""
    changes: dict[str, tuple] = {}
    for nt in (NODE_TYPE_PROJECT, NODE_TYPE_COMPANY):
        if nt not in graph.node_types:
            continue
        if not hasattr(graph[nt], "x") or graph[nt].x is None:
            continue
        old = graph[nt].x.dtype
        graph[nt].x = graph[nt].x.to(dtype)
        changes[nt] = (old, dtype)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["compress", "decompress"], default="compress",
        help="compress: float32 -> float16; decompress: float16 -> float32",
    )
    parser.add_argument(
        "--input", default="data/processed/graph.pt",
        help="input graph file",
    )
    parser.add_argument(
        "--output", default=None,
        help="output path (default: <input>_fp16.pt or <input>_fp32.pt)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    in_size = input_path.stat().st_size
    print(f"[compress] input:  {input_path}  ({_fmt_size(in_size)})")

    print("[compress] loading...")
    graph = torch.load(input_path, weights_only=False)

    if args.mode == "compress":
        target_dtype = torch.float16
        default_suffix = "_fp16.pt"
    else:
        target_dtype = torch.float32
        default_suffix = "_fp32.pt"

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + default_suffix
    )

    changes = _cast_node_features(graph, target_dtype)
    for nt, (old, new) in changes.items():
        print(f"[compress] {nt}.x: {old} -> {new}")

    print(f"[compress] saving -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_path)

    out_size = output_path.stat().st_size
    ratio = out_size / in_size if in_size else 0.0
    print(
        f"[compress] output: {output_path}  ({_fmt_size(out_size)})  "
        f"ratio={ratio:.2%}"
    )


if __name__ == "__main__":
    main()

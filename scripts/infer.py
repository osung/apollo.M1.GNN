"""Entrypoint: return top-100 recommendations for a given node id.

Usage: `make infer QUERY=<node_id>`
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="project or company node id")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    raise NotImplementedError(f"See src/serving/retrieve.py (query={args.query})")


if __name__ == "__main__":
    main()

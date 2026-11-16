"""Build the bge-m3 embedding input with organisation names removed.

Reads the original embedding input ([kind, id, text, src]) of the bge-m3
pipeline, strips the performing organisation / linked company names from
project descriptions and each company's own name from its description
(src.data.names), and writes embed_input.pkl + shards to the work dir. The
bge-m3 pipeline's `embed` and `finalize` phases then run unchanged there.

    python scripts/build_noname_embed_input.py --paths config/paths_raw.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_edges
from src.data.names import core_name, strip_company_name, strip_project_names
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths_raw.yaml")
    args = parser.parse_args()
    paths = load_yaml(args.paths)
    cfg = paths["noname_embedding"]
    out_dir = Path(cfg["work_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.read_pickle(cfg["source_input"])
    print(f"[noname] input {len(t):,} rows")
    cid_col = paths["id_columns"]["company"]
    names = pd.read_pickle(cfg["company_names"])[[cid_col, "한글업체명"]]
    name_of = dict(zip(names[cid_col].astype(str).str.strip(), names["한글업체명"].astype(str),
                       strict=True))

    linked: dict[str, set[str]] = {}
    for rel in ("royalty", "commercial", "performance"):
        cols = paths["edge_columns"][rel]
        edges = load_edges(paths["raw"][f"edges_{rel}"], cols["project"], cols["company"])
        for pid, cid in zip(edges["project_id"].astype(str), edges["company_id"].astype(str),
                            strict=True):
            name = name_of.get(cid.strip())
            if name:
                linked.setdefault(pid.strip(), set()).add(name)

    is_p = (t["kind"] == "project").to_numpy()
    ids = t["id"].astype(str).tolist()
    old = t["text"].astype(str).tolist()
    new = [
        strip_project_names(x, linked.get(i, ())) if p else strip_company_name(x, name_of.get(i, ""))
        for x, i, p in zip(old, ids, is_p, strict=True)
    ]
    t["text"] = new
    changed = sum(a != b for a, b in zip(old, new, strict=True))
    print(f"[noname] texts changed: {changed:,} of {sum(bool(x.strip()) for x in old):,} non-empty")

    # residual check: linked company names still present in project texts
    left = sum(
        1 for x, i, p in zip(new, ids, is_p, strict=True)
        if p and any(len(core_name(n)) >= 2 and core_name(n) in x.replace(" ", "")
                     for n in linked.get(i, ()))
    )
    print(f"[noname] project texts still naming a linked company: {left:,}")

    t.to_pickle(out_dir / "embed_input.pkl")
    n = int(cfg.get("num_shards", 1))
    for s in range(n):
        t.iloc[s::n].reset_index(drop=True).to_pickle(out_dir / f"embed_input_shard{s}.pkl")
    print(f"[noname] wrote {out_dir}/embed_input.pkl (+{n} shard)")


if __name__ == "__main__":
    main()

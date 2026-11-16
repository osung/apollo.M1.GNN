"""Build the patent relation: (project, patent-applicant company) pairs.

Project ids in the patent table are raw ids; they are mapped to the canonical ids
of the multi-year merge (same rule as scripts/merge_multiyear_projects.py) so the
pairs line up with the merged node table.

    python scripts/build_patent_edges.py            # config/paths_raw.yaml: patent_edges
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_multiyear_projects import build_canonical_map
from src.data.patents import patent_edges
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths_raw.yaml")
    parser.add_argument("--min-name-len", type=int, default=16,
                        help="must match the merge run (merge_multiyear_projects.py)")
    args = parser.parse_args()
    paths = load_yaml(args.paths)
    cfg = paths["patent_edges"]
    pid_col = paths["id_columns"]["project"]
    cid_col = paths["id_columns"]["company"]

    names = pd.read_pickle(cfg["project_names"])[[pid_col, "과제명"]]
    bizno = pd.read_pickle(paths["raw"]["edges_performance"])
    canonical, stats = build_canonical_map(names, bizno, args.min_name_len)
    canonical = {str(k): str(v) for k, v in canonical.items()}
    print(f"[patent] canonical map: {stats['n_original']:,} -> {stats['n_canonical']:,} projects")

    pat = pd.read_pickle(cfg["source"])[[cfg["project_column"], cfg["applicant_column"]]]
    edges = patent_edges(pat[cfg["project_column"]], pat[cfg["applicant_column"]], canonical)
    edges = edges.rename(columns={"project_id": pid_col, "company_id": cid_col})
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_pickle(out)
    print(f"[patent] {len(pat):,} patents -> {len(edges):,} (project, company) pairs "
          f"({edges[pid_col].nunique():,} projects, {edges[cid_col].nunique():,} companies) "
          f"-> {out}")


if __name__ == "__main__":
    main()

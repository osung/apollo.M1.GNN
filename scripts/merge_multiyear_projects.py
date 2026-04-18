"""Deduplicate multi-year projects and rewrite the edge pickles.

Two merge rules, applied per `과제명`:

1. **Safe** — bizno has 수행기관 info for every candidate and the
   (name, frozenset(사업자번호)) tuple is identical. These are
   almost certainly annual phases of the same project.

2. **Specific no-org** — bizno has no 수행기관 info, but the
   name is specific enough:
     - length ≥ min_name_len (default 16)
     - doesn't match a curated GENERIC regex (administrative /
       program-label keywords like 수시연구사업, SW중심대학, 기반구축 …)
     - doesn't appear with 2+ distinct org-sets in bizno
       (cross-reference: when bizno has two different sets of
       performing organizations for the same name, the name is
       almost certainly a program label, not a research title).

Aggregation inside each merge group:
- `과제고유번호` ← lexicographically smallest id in the group (canonical)
- `norm_embed`  ← mean of member embeddings, then L2-renormalized
                   (alternative: weighted_mean by score, or max_score pick)
- `유망성점수` ← max (alternative: mean)
- `키워드_리스트` ← ordered union, duplicates removed
- other columns ← taken from the canonical row

Edges (royalty / commercial / performance / bizno) are remapped
through canonical ids; rows whose project id isn't in the map are
dropped (same behavior as the graph builder). Full-row duplicates are
removed after remap. The company table is copied unchanged.

Writes the new pickles to --output-dir (default: data/raw_merged/),
mirroring the filenames from paths.yaml.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml


GENERIC_TERMS = [
    r"수시연구", r"자유공모", r"창의기초연구", r"핵심연구지원",
    r"인력양성", r"연구인력", r"우수인력", r"학생연수", r"우수학생", r"국외연수", r"파견",
    r"중소기업.*(지원|고용|육성)", r"일반사업", r"기반구축",
    r"컨퍼런스", r"심포지엄", r"포럼", r"워크샵", r"세미나", r"학술대회",
    r"센터조성지원", r"센터운영", r"운영사업", r"운영과제",
    r"포상", r"모집", r"공모", r"초청",
    r"SW중심대학", r"BK21", r"공동연구활성화", r"정책연구",
    r"^연구기획", r"^기획연구", r"기획과제",
    r"^사업", r"사업비$",
]
GENERIC_PATTERN = re.compile("|".join(GENERIC_TERMS))


def is_specific_name(name: str, min_len: int, cross_ref_generic: set) -> bool:
    if not isinstance(name, str):
        return False
    if len(name) < min_len:
        return False
    if GENERIC_PATTERN.search(name):
        return False
    if name in cross_ref_generic:
        return False
    return True


def build_canonical_map(
    proj_df: pd.DataFrame, bizno_df: pd.DataFrame, min_name_len: int
):
    """Return (canonical_map, stats) where canonical_map[pid] = canonical_pid."""
    pid_orgs = bizno_df.groupby("과제고유번호")["사업자번호"].apply(
        lambda s: frozenset(s.tolist())
    )

    # Cross-reference: names appearing with 2+ distinct org-sets are generic
    bz_uniq = bizno_df.drop_duplicates(["과제고유번호"])[["과제고유번호", "과제명"]].copy()
    bz_uniq["_org_set"] = bz_uniq["과제고유번호"].map(pid_orgs)
    distinct_orgsets_per_name = bz_uniq.groupby("과제명")["_org_set"].nunique()
    cross_ref_generic = set(
        distinct_orgsets_per_name[distinct_orgsets_per_name >= 2].index
    )

    has_orgs_mask = proj_df["과제고유번호"].isin(pid_orgs.index)
    with_orgs = proj_df[has_orgs_mask][["과제고유번호", "과제명"]].copy()
    no_orgs = proj_df[~has_orgs_mask][["과제고유번호", "과제명"]].copy()

    canonical_map: dict = {}
    n_safe_merged = 0
    n_specific_merged = 0

    # 1. SAFE merges
    with_orgs["_orgs"] = with_orgs["과제고유번호"].map(pid_orgs)
    for (_, _), pids in with_orgs.groupby(["과제명", "_orgs"])["과제고유번호"]:
        pids = sorted(pids.tolist())
        canon = pids[0]
        if len(pids) > 1:
            n_safe_merged += len(pids) - 1
        for pid in pids:
            canonical_map[pid] = canon

    # 2. SPECIFIC no-org merges
    for name, pids in no_orgs.groupby("과제명")["과제고유번호"]:
        pids = sorted(pids.tolist())
        if len(pids) == 1 or not is_specific_name(name, min_name_len, cross_ref_generic):
            for pid in pids:
                canonical_map[pid] = pid
        else:
            canon = pids[0]
            for pid in pids:
                canonical_map[pid] = canon
            n_specific_merged += len(pids) - 1

    stats = {
        "n_original": len(proj_df),
        "n_canonical": len(set(canonical_map.values())),
        "n_safe_merged": n_safe_merged,
        "n_specific_merged": n_specific_merged,
        "n_cross_ref_generic_names": len(cross_ref_generic),
    }
    return canonical_map, stats


def _aggregate_embed(embeds: np.ndarray, scores: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":
        agg = embeds.mean(axis=0)
    elif strategy == "weighted_mean":
        w = np.clip(scores, 1e-6, None)
        w = w / w.sum()
        agg = (embeds * w[:, None]).sum(axis=0)
    elif strategy == "max_score":
        agg = embeds[int(np.argmax(scores))]
    else:
        raise ValueError(f"unknown embed strategy: {strategy!r}")
    n = float(np.linalg.norm(agg))
    if n > 0:
        agg = agg / n
    return agg.astype(np.float32)


def _aggregate_keywords(series: pd.Series) -> list:
    out: list = []
    seen: set = set()
    for kl in series:
        if kl is None or (isinstance(kl, float) and pd.isna(kl)):
            continue
        try:
            iterator = iter(kl)
        except TypeError:
            continue
        for k in iterator:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def merge_projects(
    proj_df: pd.DataFrame,
    canonical_map: dict,
    embed_strategy: str,
    score_agg: str,
) -> pd.DataFrame:
    proj_df = proj_df.copy()
    proj_df["_canon"] = proj_df["과제고유번호"].map(canonical_map)
    proj_df = proj_df[proj_df["_canon"].notna()]

    merged_rows = []
    for canon_id, sub in proj_df.groupby("_canon", sort=False):
        if len(sub) == 1:
            row = sub.iloc[0].to_dict()
            row.pop("_canon", None)
            merged_rows.append(row)
            continue

        canon_rows = sub[sub["과제고유번호"] == canon_id]
        base_row = canon_rows.iloc[0] if len(canon_rows) else sub.iloc[0]
        row = base_row.to_dict()
        row.pop("_canon", None)

        embeds = np.stack(
            [np.asarray(e, dtype=np.float32) for e in sub["norm_embed"].tolist()]
        )
        scores = sub["유망성점수"].to_numpy(dtype=np.float32)
        row["norm_embed"] = _aggregate_embed(embeds, scores, embed_strategy)

        if score_agg == "max":
            row["유망성점수"] = float(scores.max())
        elif score_agg == "mean":
            row["유망성점수"] = float(scores.mean())
        else:
            raise ValueError(f"unknown score_agg: {score_agg!r}")

        if "키워드_리스트" in sub.columns:
            row["키워드_리스트"] = _aggregate_keywords(sub["키워드_리스트"])

        row["과제고유번호"] = canon_id
        merged_rows.append(row)

    return pd.DataFrame(merged_rows)


def remap_edges(
    edge_df: pd.DataFrame, canonical_map: dict, project_col: str = "과제고유번호"
) -> pd.DataFrame:
    edge_df = edge_df.copy()
    edge_df[project_col] = edge_df[project_col].map(canonical_map)
    edge_df = edge_df.dropna(subset=[project_col])
    edge_df = edge_df.drop_duplicates()
    return edge_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--output-dir", default="data/raw_merged")
    parser.add_argument("--embed-strategy", default="mean",
                        choices=["mean", "weighted_mean", "max_score"])
    parser.add_argument("--score-agg", default="max", choices=["max", "mean"])
    parser.add_argument("--min-name-len", type=int, default=16)
    args = parser.parse_args()

    paths = load_yaml(args.paths)
    raw = paths["raw"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading projects:   {raw['projects']}")
    proj_df = pd.read_pickle(raw["projects"])
    print(f"[merge] loading bizno:      {raw['edges_performance']}")
    bizno_df = pd.read_pickle(raw["edges_performance"])

    print(f"[merge] building canonical map (min_name_len={args.min_name_len})")
    canonical_map, stats = build_canonical_map(proj_df, bizno_df, args.min_name_len)

    n_orig = stats["n_original"]
    n_can = stats["n_canonical"]
    print(f"  original:          {n_orig:,}")
    print(f"  canonical groups:  {n_can:,}")
    print(f"  reduction:         {n_orig - n_can:,} ({(n_orig - n_can)/n_orig*100:.1f}%)")
    print(f"    safe merges saved:     {stats['n_safe_merged']:,}")
    print(f"    specific merges saved: {stats['n_specific_merged']:,}")
    print(f"    cross-ref generic names flagged: {stats['n_cross_ref_generic_names']:,}")

    print(f"[merge] aggregating projects "
          f"(embed_strategy={args.embed_strategy}, score_agg={args.score_agg})")
    merged_proj = merge_projects(
        proj_df, canonical_map, args.embed_strategy, args.score_agg
    )
    assert len(merged_proj) == n_can, f"expected {n_can}, got {len(merged_proj)}"

    proj_out = out_dir / Path(raw["projects"]).name
    merged_proj.to_pickle(proj_out)
    print(f"[merge] saved merged projects -> {proj_out} ({len(merged_proj):,} rows)")

    comp_out = out_dir / Path(raw["companies"]).name
    shutil.copy(raw["companies"], comp_out)
    print(f"[merge] copied companies -> {comp_out}")

    for label, path in [
        ("royalty", raw["edges_royalty"]),
        ("commercial", raw["edges_commercial"]),
        ("performance", raw["edges_performance"]),
    ]:
        df = pd.read_pickle(path)
        n_before = len(df)
        df = remap_edges(df, canonical_map)
        out_path = out_dir / Path(path).name
        df.to_pickle(out_path)
        print(f"[merge] {label:11s}: {n_before:,} → {len(df):,} rows saved to {out_path}")

    print()
    print(f"[merge] done. Merged data in {out_dir}/")
    print("[merge] next steps:")
    print("  1. point config/paths.yaml raw.* at the merged files (or duplicate yaml)")
    print("  2. python scripts/build_graph.py")
    print("  3. python scripts/build_similarity.py (graph changed -> sim cache stale)")


if __name__ == "__main__":
    main()

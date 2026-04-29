"""Top-10 matched-pair builder for top-promise projects ↔ companies.

For each high-promise project (top X% by 유망성점수), find the top-10
candidate companies via cosine similarity in the GNN latent space.
Symmetrically for high-promise companies, find top-10 candidate projects.

Each row in the output DataFrames carries:
  - the query node's ID, name, promise score
  - rank 1..10 of recommended counterparts (ID, name, similarity score)
  - whether each rank-K pair appears in the actual royalty / commercial
    interaction tables (boolean columns)

Outputs (one set per direction):
  - results/match_top10_p2c.pkl  (project → companies)
  - results/match_top10_p2c.xlsx
  - results/match_top10_c2p.pkl  (company → projects)
  - results/match_top10_c2p.xlsx

Run from project root:
    python scripts/match_top10.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml


# Default checkpoint = the balanced champion (GFM h128 3L sim=5 ep285)
DEFAULT_PROJECT_Z = (
    "outputs/project_z_gfm_h128_l3_hr00_c2p100_sim5_nonorm_epoch285.npy"
)
DEFAULT_COMPANY_Z = (
    "outputs/company_z_gfm_h128_l3_hr00_c2p100_sim5_nonorm_epoch285.npy"
)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _batched_topk(
    z_query: np.ndarray,         # (Q, D)  L2-normalized
    z_cand: np.ndarray,           # (N, D)  L2-normalized
    topk: int,
    batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_idx, top_scores) of shape (Q, topk) each.

    Plain numpy implementation — no FAISS dependency. ~30-60 sec for
    43K queries × 800K candidates × 128 dim on a modern CPU.
    """
    Q = z_query.shape[0]
    out_idx = np.empty((Q, topk), dtype=np.int64)
    out_sc = np.empty((Q, topk), dtype=np.float32)
    for s in range(0, Q, batch):
        e = min(s + batch, Q)
        scores = z_query[s:e] @ z_cand.T            # (b, N)
        # argpartition top-k (unsorted), then sort the k partial
        part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        part_sc = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_sc, axis=1)
        out_idx[s:e] = np.take_along_axis(part, order, axis=1)
        out_sc[s:e] = np.take_along_axis(part_sc, order, axis=1)
    return out_idx, out_sc


def _load_node_maps(path: Path) -> tuple[list[str], list[str]]:
    with open(path, "rb") as f:
        nm = pickle.load(f)
    return list(nm["project_ids"]), list(nm["company_ids"])


def _extract_model_tag(z_path: str) -> str:
    """Extract the experiment tag from a 'project_z_<TAG>.npy' filename.

    Example:
        project_z_gfm_h128_l3_hr00_c2p100_sim5_nonorm_epoch285.npy
        -> 'gfm_h128_l3_hr00_c2p100_sim5_nonorm_epoch285'
    """
    name = Path(z_path).stem  # strip .npy
    for prefix in ("project_z_", "company_z_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name  # fallback: use whole stem if prefix unfamiliar


def _load_edge_pairs(
    paths_yaml: dict, side: str = "both"
) -> tuple[set, set]:
    """Return (royalty_pairs, commercial_pairs) as sets of (proj_id, comp_id)."""
    royalty_df = pd.read_pickle(paths_yaml["raw"]["edges_royalty"])
    commercial_df = pd.read_pickle(paths_yaml["raw"]["edges_commercial"])

    # Column names per docs/data_schema.md
    R_PROJ, R_COMP = "과제고유번호", "기술실시대상기관_사업자번호"
    C_PROJ, C_COMP = "과제고유번호", "사업화주체_사업자등록번호"

    royalty_pairs = set(
        zip(royalty_df[R_PROJ].astype(str), royalty_df[R_COMP].astype(str))
    )
    commercial_pairs = set(
        zip(commercial_df[C_PROJ].astype(str), commercial_df[C_COMP].astype(str))
    )
    print(
        f"[match] edges loaded: royalty={len(royalty_pairs):,}  "
        f"commercial={len(commercial_pairs):,}"
    )
    return royalty_pairs, commercial_pairs


def _build_match_dataframe(
    *,
    direction: str,                         # 'p2c' or 'c2p'
    query_idx: np.ndarray,                  # (Q,) graph indices of queries
    query_id_lookup: list[str],             # graph_idx -> raw ID (query side)
    cand_id_lookup: list[str],              # graph_idx -> raw ID (candidate side)
    query_name_map: dict[str, str],         # raw ID -> display name (query)
    cand_name_map: dict[str, str],          # raw ID -> display name (candidate)
    query_score_map: dict[str, float],      # raw ID -> 유망성점수
    top_idx: np.ndarray,                    # (Q, K) candidate graph indices
    top_score: np.ndarray,                  # (Q, K) cosine scores
    royalty_pairs: set,
    commercial_pairs: set,
    topk: int,
) -> pd.DataFrame:
    Q = query_idx.shape[0]
    rows: list[dict] = []

    if direction == "p2c":
        q_label_col, q_name_col = "과제고유번호", "과제명"
        c_id_col, c_name_col = "사업자번호", "한글업체명"
    else:
        q_label_col, q_name_col = "사업자번호", "한글업체명"
        c_id_col, c_name_col = "과제고유번호", "과제명"

    for q_pos in range(Q):
        q_gidx = int(query_idx[q_pos])
        q_id = query_id_lookup[q_gidx]
        row: dict = {
            q_label_col: q_id,
            q_name_col: query_name_map.get(q_id, ""),
            "유망성점수": query_score_map.get(q_id, np.nan),
        }
        for k in range(topk):
            c_gidx = int(top_idx[q_pos, k])
            c_id = cand_id_lookup[c_gidx]
            c_name = cand_name_map.get(c_id, "")
            score = float(top_score[q_pos, k])

            # The (project, company) pair we look up in edge sets always
            # has project first regardless of direction.
            if direction == "p2c":
                pair = (q_id, c_id)
            else:
                pair = (c_id, q_id)

            row[f"{c_id_col}_top{k+1}"] = c_id
            row[f"{c_name_col}_top{k+1}"] = c_name
            row[f"score_top{k+1}"] = round(score, 4)
            row[f"in_royalty_top{k+1}"] = pair in royalty_pairs
            row[f"in_commercial_top{k+1}"] = pair in commercial_pairs
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--project-z", default=DEFAULT_PROJECT_Z)
    parser.add_argument("--company-z", default=DEFAULT_COMPANY_Z)
    parser.add_argument(
        "--node-maps", default="data/processed/node_maps.pkl",
    )
    parser.add_argument(
        "--top-percent", type=float, default=0.10,
        help="evaluate top X fraction by 유망성점수 (1.0 = all nodes; default 0.1 = top 10%%)",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--output-dir", default="results",
    )
    parser.add_argument(
        "--out-prefix", default="match_top10",
        help="leading filename component (default: match_top10). The model "
             "tag (extracted from --project-z filename) and top-percent are "
             "appended automatically, so a typical output looks like "
             "'match_top10_p2c__gfm_h128_l3_..._epoch285__top10pct.pkl'.",
    )
    parser.add_argument(
        "--batch", type=int, default=512,
        help="query batch size for top-k search; reduce on low memory",
    )
    parser.add_argument(
        "--direction", default="both", choices=["p2c", "c2p", "both"],
    )
    args = parser.parse_args()

    paths_yaml = load_yaml(args.paths)

    # ---- Load embeddings ----
    print(f"[match] loading project z: {args.project_z}")
    z_p = np.load(args.project_z).astype(np.float32)
    print(f"[match] loading company z: {args.company_z}")
    z_c = np.load(args.company_z).astype(np.float32)
    print(f"[match]   project_z {z_p.shape}, company_z {z_c.shape}")
    z_p = _l2_normalize(z_p)
    z_c = _l2_normalize(z_c)

    # ---- Load node maps ----
    print(f"[match] loading node maps: {args.node_maps}")
    project_ids, company_ids = _load_node_maps(Path(args.node_maps))
    if len(project_ids) != z_p.shape[0]:
        raise ValueError(
            f"project_ids size {len(project_ids)} != z_p rows {z_p.shape[0]}"
        )
    if len(company_ids) != z_c.shape[0]:
        raise ValueError(
            f"company_ids size {len(company_ids)} != z_c rows {z_c.shape[0]}"
        )

    # ---- Load raw projects + companies (only the columns we need) ----
    needed_proj = ["과제고유번호", "과제명", "유망성점수"]
    needed_comp = ["사업자번호", "한글업체명", "유망성점수"]
    print(f"[match] loading projects table")
    proj_df = pd.read_pickle(paths_yaml["raw"]["projects"])
    proj_df = proj_df[needed_proj].copy()
    proj_df["과제고유번호"] = proj_df["과제고유번호"].astype(str)
    proj_name_map = dict(zip(proj_df["과제고유번호"], proj_df["과제명"]))
    proj_score_map = dict(zip(proj_df["과제고유번호"], proj_df["유망성점수"]))

    print(f"[match] loading companies table")
    comp_df = pd.read_pickle(paths_yaml["raw"]["companies"])
    comp_df = comp_df[needed_comp].copy()
    comp_df["사업자번호"] = comp_df["사업자번호"].astype(str)
    comp_name_map = dict(zip(comp_df["사업자번호"], comp_df["한글업체명"]))
    comp_score_map = dict(zip(comp_df["사업자번호"], comp_df["유망성점수"]))

    # ---- Load edge pairs for in_royalty / in_commercial ----
    royalty_pairs, commercial_pairs = _load_edge_pairs(paths_yaml)

    # ---- Identify top-promise nodes ----
    project_promise = np.array(
        [proj_score_map.get(pid, -np.inf) for pid in project_ids],
        dtype=np.float64,
    )
    company_promise = np.array(
        [comp_score_map.get(cid, -np.inf) for cid in company_ids],
        dtype=np.float64,
    )

    n_top_p = max(1, int(args.top_percent * z_p.shape[0]))
    n_top_c = max(1, int(args.top_percent * z_c.shape[0]))
    top_proj_idx = np.argpartition(-project_promise, kth=n_top_p - 1)[:n_top_p]
    top_comp_idx = np.argpartition(-company_promise, kth=n_top_c - 1)[:n_top_c]
    # Sort within the top group by promise score, descending
    top_proj_idx = top_proj_idx[
        np.argsort(-project_promise[top_proj_idx])
    ]
    top_comp_idx = top_comp_idx[
        np.argsort(-company_promise[top_comp_idx])
    ]
    print(
        f"[match] top-{args.top_percent*100:.0f}% selection: "
        f"projects={n_top_p:,}/{z_p.shape[0]:,}  "
        f"companies={n_top_c:,}/{z_c.shape[0]:,}"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a filename-safe suffix that captures model + percent so multiple
    # runs with different checkpoints / fractions don't clobber each other.
    model_tag = _extract_model_tag(args.project_z)
    pct_tag = f"top{int(round(args.top_percent * 100))}pct"
    name_suffix = f"{model_tag}__{pct_tag}"

    # ---- p2c (project queries → company candidates) ----
    if args.direction in ("p2c", "both"):
        print(f"\n[match] === p2c top-{args.topk} retrieval ===")
        t0 = time.time()
        z_q = z_p[top_proj_idx]
        idx, sc = _batched_topk(z_q, z_c, topk=args.topk, batch=args.batch)
        print(f"  retrieval: {time.time()-t0:.1f}s")

        df_p2c = _build_match_dataframe(
            direction="p2c",
            query_idx=top_proj_idx,
            query_id_lookup=project_ids,
            cand_id_lookup=company_ids,
            query_name_map=proj_name_map,
            cand_name_map=comp_name_map,
            query_score_map=proj_score_map,
            top_idx=idx,
            top_score=sc,
            royalty_pairs=royalty_pairs,
            commercial_pairs=commercial_pairs,
            topk=args.topk,
        )

        n_total_pairs = n_top_p * args.topk
        n_roy_hits = int(
            sum(df_p2c[f"in_royalty_top{k+1}"].sum() for k in range(args.topk))
        )
        n_com_hits = int(
            sum(df_p2c[f"in_commercial_top{k+1}"].sum() for k in range(args.topk))
        )
        print(
            f"  p2c: total query×top-{args.topk} pairs = {n_total_pairs:,};  "
            f"in_royalty hits = {n_roy_hits:,}  "
            f"in_commercial hits = {n_com_hits:,}"
        )

        pkl_path = out_dir / f"{args.out_prefix}_p2c__{name_suffix}.pkl"
        xlsx_path = out_dir / f"{args.out_prefix}_p2c__{name_suffix}.xlsx"
        df_p2c.to_pickle(pkl_path)
        df_p2c.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"  saved {pkl_path} ({len(df_p2c):,} rows)")
        print(f"  saved {xlsx_path}")

    # ---- c2p (company queries → project candidates) ----
    if args.direction in ("c2p", "both"):
        print(f"\n[match] === c2p top-{args.topk} retrieval ===")
        t0 = time.time()
        z_q = z_c[top_comp_idx]
        idx, sc = _batched_topk(z_q, z_p, topk=args.topk, batch=args.batch)
        print(f"  retrieval: {time.time()-t0:.1f}s")

        df_c2p = _build_match_dataframe(
            direction="c2p",
            query_idx=top_comp_idx,
            query_id_lookup=company_ids,
            cand_id_lookup=project_ids,
            query_name_map=comp_name_map,
            cand_name_map=proj_name_map,
            query_score_map=comp_score_map,
            top_idx=idx,
            top_score=sc,
            royalty_pairs=royalty_pairs,
            commercial_pairs=commercial_pairs,
            topk=args.topk,
        )

        n_total_pairs = n_top_c * args.topk
        n_roy_hits = int(
            sum(df_c2p[f"in_royalty_top{k+1}"].sum() for k in range(args.topk))
        )
        n_com_hits = int(
            sum(df_c2p[f"in_commercial_top{k+1}"].sum() for k in range(args.topk))
        )
        print(
            f"  c2p: total query×top-{args.topk} pairs = {n_total_pairs:,};  "
            f"in_royalty hits = {n_roy_hits:,}  "
            f"in_commercial hits = {n_com_hits:,}"
        )

        pkl_path = out_dir / f"{args.out_prefix}_c2p__{name_suffix}.pkl"
        xlsx_path = out_dir / f"{args.out_prefix}_c2p__{name_suffix}.xlsx"
        df_c2p.to_pickle(pkl_path)
        df_c2p.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"  saved {pkl_path} ({len(df_c2p):,} rows)")
        print(f"  saved {xlsx_path}")

    print("\n[match] DONE")


if __name__ == "__main__":
    main()

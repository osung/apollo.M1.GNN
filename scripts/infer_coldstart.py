"""Cold-start qualitative inference: new norm_embed -> top-K matches.

Use this when a new project or company (not in the training graph) has
a `norm_embed` vector available, and you want to see which existing
opposite-side nodes the trained model would recommend.

Pipeline:
    new norm_embed
        -> ProjectionMLP (per-side, trained on warm GNN z's)
        -> z_hat (predicted latent embedding)
        -> FAISS inner-product search against warm opposite-side z
        -> top-K results enriched with name/keywords/industry

Since this is a qualitative tool (no ground-truth edges for the new
node), the output is a human-readable table rather than a metric.

Two usage modes:

1. CLI on a pickle/CSV file of new nodes:
    python scripts/infer_coldstart.py \\
        --side project \\
        --input data/new_projects.pkl \\
        --mlp-ckpt data/processed/checkpoints/projection_mlp_project_<tag>.pt \\
        --warm-z   data/processed/checkpoints/company_z_<tag>.npy \\
        --top-k 20 \\
        --output results/coldstart_new_projects.xlsx

2. Python/notebook (single vector):
    from scripts.infer_coldstart import ColdStartInferencer
    inf = ColdStartInferencer(
        side="project",
        mlp_ckpt="...projection_mlp_project_*.pt",
        warm_z_opposite="...company_z_*.npy",
    )
    print(inf.recommend(new_norm_embed_vec, top_k=20))
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.projection import ProjectionMLP
from src.utils import load_yaml


# Columns we display from the raw side dataframes
_PROJECT_DISPLAY_COLS = ["과제고유번호", "과제명", "키워드_리스트", "유망성점수"]
_COMPANY_DISPLAY_COLS = [
    "사업자번호", "한글업체명", "10차산업코드명",
    "키워드_리스트", "유망성점수",
]


def _load_mlp(ckpt_path: str | Path, device: str) -> ProjectionMLP:
    """Load a saved ProjectionMLP checkpoint (state_dict + config dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ProjectionMLP(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["output_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def _truncate(s, n: int = 60) -> str:
    """Truncate long strings/lists for terminal display."""
    if isinstance(s, list):
        s = ", ".join(str(x) for x in s[:5])
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


class ColdStartInferencer:
    """End-to-end cold-start retrieval helper.

    `side` is the side of the QUERY (the new node). The retriever looks
    up the OPPOSITE side, so:
      - side="project" -> recommend companies for the new project
      - side="company" -> recommend projects for the new company
    """

    def __init__(
        self,
        *,
        side: str,
        mlp_ckpt: str | Path,
        warm_z_opposite: str | Path,
        node_maps_path: str | Path = "data/processed/node_maps.pkl",
        raw_projects_path: str | None = None,
        raw_companies_path: str | None = None,
        paths_yaml: str | Path = "config/paths.yaml",
        device: str | None = None,
    ):
        if side not in ("project", "company"):
            raise ValueError(f"side must be 'project' or 'company', got {side!r}")
        self.side = side
        self.opposite = "company" if side == "project" else "project"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load MLP for query side
        print(f"[coldstart-infer] loading MLP {mlp_ckpt}")
        self.mlp = _load_mlp(mlp_ckpt, self.device)
        print(
            f"[coldstart-infer]   in_dim={self.mlp.input_dim} "
            f"out_dim={self.mlp.output_dim} "
            f"hidden={self.mlp.hidden_dim} layers={self.mlp.num_layers}"
        )

        # Load warm z of OPPOSITE side (the candidate pool)
        print(f"[coldstart-infer] loading warm opposite z: {warm_z_opposite}")
        self.warm_z = np.load(warm_z_opposite).astype(np.float32)
        # Normalize for inner-product = cosine equivalence (training had
        # no_normalize=True, so we normalize at inference time)
        self.warm_z = _l2_normalize(self.warm_z)
        print(
            f"[coldstart-infer]   warm pool: {self.warm_z.shape[0]:,} "
            f"{self.opposite}s, dim={self.warm_z.shape[1]}"
        )

        # Load NodeMap to translate matched indices -> raw IDs
        with open(node_maps_path, "rb") as f:
            nm = pickle.load(f)
        self.id_list_opposite = list(nm[f"{self.opposite}_ids"])
        if len(self.id_list_opposite) != self.warm_z.shape[0]:
            raise ValueError(
                f"NodeMap has {len(self.id_list_opposite)} {self.opposite} ids "
                f"but warm z has {self.warm_z.shape[0]} rows"
            )

        # Optional: enrich top-K with display columns from raw dataframes
        paths = load_yaml(paths_yaml)
        proj_path = raw_projects_path or paths["raw"]["projects"]
        comp_path = raw_companies_path or paths["raw"]["companies"]
        self._df_opposite: pd.DataFrame | None = None
        try:
            opp_path = comp_path if self.opposite == "company" else proj_path
            print(f"[coldstart-infer] loading raw {self.opposite} table: {opp_path}")
            self._df_opposite = pd.read_pickle(opp_path)
            id_col = "사업자번호" if self.opposite == "company" else "과제고유번호"
            self._df_opposite = self._df_opposite.set_index(id_col, drop=False)
        except Exception as e:  # noqa: BLE001
            print(
                f"[coldstart-infer]   (raw table unavailable: {e}; "
                "id-only output)"
            )

    @torch.no_grad()
    def project(self, norm_embed: np.ndarray) -> np.ndarray:
        """Project (N, in_dim) norm_embed via the MLP, returning (N, out_dim) z_hat."""
        if norm_embed.ndim == 1:
            norm_embed = norm_embed[None, :]
        if norm_embed.shape[-1] != self.mlp.input_dim:
            raise ValueError(
                f"norm_embed dim {norm_embed.shape[-1]} != "
                f"MLP input_dim {self.mlp.input_dim}"
            )
        x = torch.from_numpy(norm_embed.astype(np.float32)).to(self.device)
        z_hat = self.mlp(x).cpu().numpy()
        return _l2_normalize(z_hat)

    def recommend(
        self,
        norm_embed: np.ndarray,
        top_k: int = 20,
    ) -> list[list[dict]]:
        """Project a single or batch of norm_embed vectors and return per-query
        top-K matches. Each match is a dict with id + score + display columns.

        Returns list-of-lists; outer list has one entry per query node.
        """
        z_hat = self.project(norm_embed)            # (Q, D), L2-normed
        scores = z_hat @ self.warm_z.T              # (Q, N), cosine since both normed
        if top_k > scores.shape[1]:
            top_k = scores.shape[1]
        # argpartition + sort for top-K (no need to sort the rest)
        part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)            # (Q, top_k)
        top_score = np.take_along_axis(part_scores, order, axis=1)   # (Q, top_k)

        all_results: list[list[dict]] = []
        for q in range(z_hat.shape[0]):
            q_results = []
            for rank in range(top_k):
                node_idx = int(top_idx[q, rank])
                node_id = self.id_list_opposite[node_idx]
                row: dict = {
                    "rank": rank + 1,
                    "score": float(top_score[q, rank]),
                    "id": node_id,
                    "graph_idx": node_idx,
                }
                if self._df_opposite is not None and node_id in self._df_opposite.index:
                    rec = self._df_opposite.loc[node_id]
                    cols = (
                        _COMPANY_DISPLAY_COLS if self.opposite == "company"
                        else _PROJECT_DISPLAY_COLS
                    )
                    for c in cols:
                        if c in rec.index and c not in row:
                            row[c] = rec[c]
                q_results.append(row)
            all_results.append(q_results)
        return all_results


def _print_results(
    query_label: str,
    results: list[dict],
    opposite: str,
    truncate: int = 60,
) -> None:
    """Pretty-print one query's top-K to stdout."""
    print(f"\n{'='*80}")
    print(f"Query [{opposite} candidates for]: {query_label}")
    print(f"{'='*80}")
    print(f"{'rank':<5}{'score':<8}{'id':<22}{'name':<40}{'keywords':<40}")
    print("-" * 115)
    for r in results:
        if opposite == "company":
            name = r.get("한글업체명", "")
            kw = r.get("키워드_리스트", "")
            ind = r.get("10차산업코드명", "")
            extra = f"{_truncate(name, 30):<32}{_truncate(kw, 30):<32}({_truncate(ind, 25)})"
        else:
            name = r.get("과제명", "")
            kw = r.get("키워드_리스트", "")
            extra = f"{_truncate(name, 40):<42}{_truncate(kw, 40):<42}"
        print(
            f"{r['rank']:<5}{r['score']:<8.4f}{_truncate(r['id'], 20):<22}{extra}"
        )


def _load_input(path: Path) -> pd.DataFrame:
    """Load a file containing new nodes. Must have a `norm_embed` column."""
    p = str(path).lower()
    if p.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif p.endswith(".csv"):
        df = pd.read_csv(path)
    elif p.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported input format: {path}")
    if "norm_embed" not in df.columns:
        raise ValueError(f"input must have a 'norm_embed' column; got {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--side", required=True, choices=["project", "company"],
        help="which side the QUERY new node is (will recommend opposite side)",
    )
    parser.add_argument("--input", required=True, help="pkl/csv/parquet of new nodes with norm_embed column")
    parser.add_argument("--mlp-ckpt", required=True, help="projection_mlp_<side>_<tag>.pt path")
    parser.add_argument(
        "--warm-z", required=True,
        help="warm z of OPPOSITE side .npy (e.g. company_z_*.npy if side=project)",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--id-col", default=None,
        help="optional column in input to display as the query label (e.g. 과제명)",
    )
    parser.add_argument(
        "--output", default=None,
        help="optional .xlsx path to save full results (one row per (query, rank))",
    )
    parser.add_argument(
        "--print-top", type=int, default=5,
        help="how many input rows to print to stdout (rest go to xlsx if given)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--paths", default="config/paths.yaml",
        help="paths.yaml for raw project/company tables (used to enrich display)",
    )
    args = parser.parse_args()

    df_in = _load_input(Path(args.input))
    print(f"[coldstart-infer] loaded {len(df_in):,} new {args.side} rows")

    inf = ColdStartInferencer(
        side=args.side,
        mlp_ckpt=args.mlp_ckpt,
        warm_z_opposite=args.warm_z,
        paths_yaml=args.paths,
        device=args.device,
    )

    norm_embed = np.stack(
        df_in["norm_embed"].apply(lambda v: np.asarray(v, dtype=np.float32)).tolist()
    )
    print(f"[coldstart-infer] projecting {norm_embed.shape}…")
    all_results = inf.recommend(norm_embed, top_k=args.top_k)

    # Choose a label column for printing (id_col -> first match in expected list)
    label_col = args.id_col
    if label_col is None:
        for c in ("과제명", "한글업체명", "과제고유번호", "사업자번호"):
            if c in df_in.columns:
                label_col = c
                break

    print_top = min(args.print_top, len(df_in))
    for i in range(print_top):
        label = str(df_in.iloc[i][label_col]) if label_col else f"row_{i}"
        _print_results(label, all_results[i], inf.opposite)

    if args.output:
        out_rows = []
        for i, results in enumerate(all_results):
            label = str(df_in.iloc[i][label_col]) if label_col else f"row_{i}"
            for r in results:
                out_rows.append({"query_idx": i, "query_label": label, **r})
        out_df = pd.DataFrame(out_rows)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_excel(out_path, index=False, engine="openpyxl")
        print(f"\n[coldstart-infer] saved full results -> {out_path}  ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()

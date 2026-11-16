from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EMBED_COL = "norm_embed"  # internal name of the node feature column, whatever its source


def _has_vector(v: object) -> bool:
    if v is None or (isinstance(v, float) and v != v):
        return False
    a = np.asarray(v)
    return a.ndim == 1 and a.size > 0 and bool(np.any(a))


def load_nodes(
    path: str | Path,
    id_col: str,
    embed_col: str = EMBED_COL,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Load a node table; the feature vector is always returned as `norm_embed`.

    embed_col: source column holding the vector (e.g. `bge_m3_embed`). When it
        is not `norm_embed`, any other `norm_embed` column in the file is replaced.
    drop_missing: drop nodes whose vector is missing (None / NaN / all-zero)
        instead of failing. Such nodes then exist nowhere in the graph, so they
        are neither trained on nor evaluated.
    """
    df = pd.read_pickle(path)
    if id_col not in df.columns:
        raise KeyError(f"id column '{id_col}' not found in {path}")
    if embed_col not in df.columns:
        raise KeyError(f"'{embed_col}' column not found in {path}")
    if embed_col != EMBED_COL:
        df = df.drop(columns=[EMBED_COL], errors="ignore").rename(columns={embed_col: EMBED_COL})
    df = df.drop_duplicates(subset=[id_col])
    has = df[EMBED_COL].map(_has_vector)
    if not has.all():
        if not drop_missing:
            raise ValueError(
                f"{int((~has).sum()):,} rows of {path} have no '{embed_col}' vector "
                "(set embedding.drop_missing: true to exclude them)"
            )
        df = df[has]
    df = df.reset_index(drop=True)
    df[EMBED_COL] = df[EMBED_COL].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


def embedding_config(paths: dict) -> tuple[str, bool, str]:
    """(source column, drop_missing, display name) from the `embedding` block of paths.yaml."""
    cfg = paths.get("embedding") or {}
    column = str(cfg.get("column", EMBED_COL))
    return column, bool(cfg.get("drop_missing", False)), str(cfg.get("name", column))


def load_edges(path: str | Path, project_col: str, company_col: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    for col in (project_col, company_col):
        if col not in df.columns:
            raise KeyError(f"edge column '{col}' not found in {path}")
    df = df[[project_col, company_col]].dropna()
    df = df.rename(columns={project_col: "project_id", company_col: "company_id"})
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def stack_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df[EMBED_COL].to_numpy()).astype(np.float32)

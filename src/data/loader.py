from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


NODE_EMBED_COLS = ["norm_embed"]


def load_nodes(path: str | Path, id_col: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if id_col not in df.columns:
        raise KeyError(f"id column '{id_col}' not found in {path}")
    if "norm_embed" not in df.columns:
        raise KeyError(f"'norm_embed' column not found in {path}")
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    df["norm_embed"] = df["norm_embed"].apply(
        lambda v: np.asarray(v, dtype=np.float32)
    )
    return df


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
    return np.stack(df["norm_embed"].to_numpy()).astype(np.float32)

"""Embedding retrieval + LLM re-ranking baseline (node-feature embedding + LLM matching).

Pipeline, mirroring the APOLLO matching workflow on held-out edges:

1. take held-out query nodes (optionally a per-relation sample),
2. retrieve the top-N opposite-type nodes by `norm_embed` cosine,
3. have an LLM score every (query, candidate) pair in batches of 20,
4. re-rank the N candidates by LLM score (ties broken by the embedding score).

Candidates already connected to the query are *not* filtered (rule #4), and
only project<->company pairs are ever formed (bipartite invariant).
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.llm_prompt import DIRECTION_SPEC, company_doc_text, project_doc_text

GT_DIRECTION = {"p2c": "project_to_company", "c2p": "company_to_project"}

NAME_MASK = "○○○"
_LEGAL_FORM = re.compile(
    r"\(주\)|\(유\)|\(재\)|\(사\)|\(합\)|㈜|주식회사|유한회사|유한책임회사|합자회사|재단법인|사단법인"
)


def core_name(name: Any) -> str:
    """Company name without its legal-form marker and whitespace: '(주)가나 다' -> '가나다'."""
    return re.sub(r"\s+", "", _LEGAL_FORM.sub("", str(name or "")))


def mask_names(text: str, names: Iterable[str]) -> str:
    """Replace company names in `text` so the LLM cannot match on the name itself.

    Project descriptions literally name the performing organisation ("(주)A에서
    수행하는 이 과제는 ..."), which would leak performance/commercial edges.
    Whitespace inside a name is ignored when matching; names shorter than two
    characters are skipped.
    """
    for core in sorted({core_name(n) for n in names}, key=len, reverse=True):
        if len(core) < 2:
            continue
        text = re.sub(r"\s*".join(re.escape(ch) for ch in core), NAME_MASK, text)
    return text


def select_queries(
    queries_per_relation: Mapping[str, Sequence[int]],
    max_per_relation: int | None,
    seed: int,
) -> np.ndarray:
    """Union of a seeded per-relation sample of query indices, in seeded random order.

    The order is random on purpose: LLM scoring proceeds in this order, so the
    queries finished at any point of a long run are a random subsample.
    """
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for rel in sorted(queries_per_relation):
        q = np.unique(np.asarray(queries_per_relation[rel], dtype=np.int64))
        if max_per_relation is not None and len(q) > max_per_relation:
            q = rng.choice(q, size=max_per_relation, replace=False)
        chosen.update(int(x) for x in q)
    return rng.permutation(np.asarray(sorted(chosen), dtype=np.int64))


def held_out_queries(held_out: Any, direction: str, edge_types: Sequence[str],
                     max_per_relation: int | None, seed: int) -> np.ndarray:
    """The query sample shared by every retriever (same call => same queries)."""
    row = 0 if direction == "p2c" else 1
    per_rel = {
        et: np.unique(held_out[("project", et, "company")].edge_index[row].numpy()).tolist()
        for et in edge_types
    }
    return select_queries(per_rel, max_per_relation, seed)


def candidate_path(out_dir: str | Path, direction: str, retriever: str) -> Path:
    return Path(out_dir) / f"candidates_{direction}_{retriever}.npz"


def save_candidates(out_dir: str | Path, direction: str, retriever: str,
                    query_idx: np.ndarray, cand_idx: np.ndarray, base_score: np.ndarray) -> Path:
    path = candidate_path(out_dir, direction, retriever)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, query_idx=query_idx, cand_idx=cand_idx, base_score=base_score)
    return path


def load_candidates(out_dir: str | Path, direction: str) -> dict[str, dict[str, np.ndarray]]:
    """All retrievers' candidate files for a direction -> {retriever: arrays}."""
    out = {}
    prefix = f"candidates_{direction}_"
    for path in sorted(Path(out_dir).glob(f"{prefix}*.npz")):
        with np.load(path) as z:
            out[path.stem[len(prefix):]] = {k: z[k] for k in z.files}
    return out


def union_candidates(cand_lists: Sequence[np.ndarray], seed: int) -> list[list[int]]:
    """Per query, the de-duplicated union of every retriever's candidates, shuffled.

    Each (query, candidate) pair is scored by the LLM once and the score is
    shared by all retrievers that proposed it. Shuffling keeps one prompt from
    holding only one retriever's (or only top-ranked) candidates.
    """
    rng = np.random.default_rng(seed)
    out = []
    for rows in zip(*cand_lists, strict=True):
        u = np.unique(np.concatenate(rows))
        out.append(rng.permutation(u).tolist())
    return out


def topn_candidates(
    query_x: np.ndarray,
    cand_x: np.ndarray,
    topn: int,
    cand_promise: np.ndarray | None = None,
    promise_weight: float = 0.0,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-N candidates per query by (1-w)*cosine + w*promise. Returns (idx, score), both (Q, N)."""
    import torch

    if topn > cand_x.shape[0]:
        raise ValueError(f"topn={topn} exceeds candidate count {cand_x.shape[0]}")
    c = torch.as_tensor(cand_x, dtype=torch.float32, device=device)
    bias = None
    if promise_weight > 0.0:
        if cand_promise is None:
            raise ValueError("promise_weight > 0 requires cand_promise")
        p = np.clip(np.nan_to_num(cand_promise.astype(np.float32), nan=50.0) / 100.0, 0.0, 1.0)
        bias = torch.as_tensor(p, device=device) * promise_weight
    idx_out, score_out = [], []
    for s in range(0, query_x.shape[0], batch_size):
        q = torch.as_tensor(query_x[s : s + batch_size], dtype=torch.float32, device=device)
        score = q @ c.T
        if bias is not None:
            score = score.clamp(0, 1) * (1.0 - promise_weight) + bias
        top = torch.topk(score, topn, dim=1)
        idx_out.append(top.indices.cpu().numpy())
        score_out.append(top.values.cpu().numpy())
    return np.concatenate(idx_out).astype(np.int64), np.concatenate(score_out).astype(np.float32)


def rerank(cand_idx: np.ndarray, base_score: np.ndarray, llm_score: np.ndarray) -> np.ndarray:
    """Order one query's candidates by LLM score desc, then embedding score desc."""
    order = np.lexsort((-base_score, -llm_score))
    return cand_idx[order]


def build_text_lookup(
    tables: Iterable[tuple[Any, str]],
    ids: set[str],
    columns: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """id -> {column: value} for the given ids.

    tables: [(DataFrame, id_column), ...] in increasing priority; later tables
    override earlier ones field by field (only with non-missing values).
    """
    out: dict[str, dict[str, Any]] = {}
    for df, id_col in tables:
        cols = [c for c in columns if c in df.columns]
        sub = df.loc[df[id_col].astype(str).isin(ids), [id_col, *cols]]
        for rec in sub.to_dict("records"):
            row = out.setdefault(str(rec[id_col]), {})
            for c in cols:
                v = rec[c]
                if v is None or (isinstance(v, float) and v != v):
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                row[c] = list(v) if isinstance(v, np.ndarray) else v
    return out


def iter_jobs(
    direction: str,
    query_idx: np.ndarray,
    cand_idx: Sequence[Sequence[int]],
    project_ids: Sequence[str],
    company_ids: Sequence[str],
    project_text: Mapping[str, Mapping[str, Any]],
    company_text: Mapping[str, Mapping[str, Any]],
    batch: int,
    mask: Mapping[int, Sequence[str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one scoring job (<= `batch` candidates) at a time, grouped by query.

    mask: query index -> company names to hide from the *project* documents of
    that query's prompts (the query project for p2c, the candidate projects
    for c2p). Company documents keep their own name.
    """
    src_label, cand_label, tech_side = DIRECTION_SPEC[direction]

    def doc(kind: str, i: int) -> str:
        if kind == "project":
            pid = str(project_ids[i])
            return project_doc_text(project_text.get(pid, {}))
        cid = str(company_ids[i])
        return company_doc_text(company_text.get(cid, {}))

    src_kind, cand_kind = ("project", "company") if direction == "p2c" else ("company", "project")
    for rank, (q, row) in enumerate(zip(query_idx.tolist(), cand_idx, strict=True)):
        row = [int(c) for c in row]
        src_text = doc(src_kind, q)
        names = (mask or {}).get(q, ())
        if names and direction == "p2c":
            src_text = mask_names(src_text, names)
        for b in range(0, len(row), batch):
            chunk = row[b : b + batch]
            cand_texts = [doc(cand_kind, c) for c in chunk]
            if names and direction == "c2p":
                cand_texts = [mask_names(t, names) for t in cand_texts]
            yield {
                "job_id": f"{direction}:{q}:{b // batch}",
                "direction": direction,
                "query": q,
                "query_rank": rank,
                "cands": chunk,
                "src_label": src_label,
                "cand_label": cand_label,
                "tech_side": tech_side,
                "src_text": src_text,
                "cand_texts": cand_texts,
            }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # a partially written last line after an interrupted run
                continue


def load_scores(paths: Iterable[str | Path]) -> dict[str, dict[int, dict[int, int]]]:
    """Merge score shards -> {direction: {query: {candidate: llm_score}}}."""
    out: dict[str, dict[int, dict[int, int]]] = {}
    for p in paths:
        for rec in read_jsonl(p):
            per_q = out.setdefault(rec["direction"], {}).setdefault(int(rec["query"]), {})
            for cand, (score, _reason) in rec["scores"].items():
                per_q[int(cand)] = int(score)
    return out

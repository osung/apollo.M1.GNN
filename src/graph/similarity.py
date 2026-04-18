"""Similarity-edge utilities: top-k nearest opposite-type nodes by norm_embed.

Logic lives here (and is tested) so the CLI script stays thin.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class SimilarityResult:
    edge_index: np.ndarray              # (2, E) int64, oriented (project_idx, company_idx)
    n_raw: int                          # edges produced before dedup
    n_dedup: int                        # after dedup (pre-overlap filter)
    n_final: int                        # after optional overlap removal
    n_dropped_overlap: int
    elapsed_s: float


def build_index(x: np.ndarray, approximate: bool = False) -> faiss.Index:
    """Build an inner-product index over `x`.

    - approximate=False: IndexFlatIP (exact, simple).
    - approximate=True:  IndexIVFFlat (nlist=1024, nprobe=16), 20-50x faster
      at ~95%+ top-k recall.
    """
    d = x.shape[1]
    x = np.ascontiguousarray(x)
    if not approximate:
        index = faiss.IndexFlatIP(d)
        index.add(x)
        return index
    nlist = min(1024, max(32, int(np.sqrt(x.shape[0]))))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(x)
    index.add(x)
    index.nprobe = min(32, max(4, nlist // 64))
    return index


def batched_search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    batch: int = 1024,
    progress_interval_s: float = 5.0,
    tag: str = "",
) -> np.ndarray:
    """Search `index` for top-k of each row in `queries`, batching to bound RAM."""
    queries = np.ascontiguousarray(queries)
    N = queries.shape[0]
    out = np.empty((N, k), dtype=np.int64)
    t0 = time.perf_counter()
    last = t0
    for s in range(0, N, batch):
        e = min(s + batch, N)
        _, I = index.search(queries[s:e], k)
        out[s:e] = I
        now = time.perf_counter()
        if now - last > progress_interval_s and e < N:
            frac = e / N
            eta = (now - t0) / frac * (1 - frac) if frac > 0 else 0
            print(f"[sim{tag}] {e:,}/{N:,} ({frac:.1%}) ETA {eta:.0f}s")
            last = now
    return out


def edges_from_topk(I: np.ndarray, direction: str) -> np.ndarray:
    """Turn a (N_query, k) result into (2, N_query*k) edges oriented (project, company).

    direction='p2c' : query=project, result=company indices.
    direction='c2p' : query=company, result=project indices. Swapped to project-first.
    """
    N, k = I.shape
    rows = np.repeat(np.arange(N, dtype=np.int64), k)
    cols = I.ravel().astype(np.int64)
    if direction == "p2c":
        return np.stack([rows, cols], axis=0)
    if direction == "c2p":
        return np.stack([cols, rows], axis=0)
    raise ValueError(f"direction must be p2c or c2p, got {direction!r}")


def _pack(a: np.ndarray, b: np.ndarray, base: int) -> np.ndarray:
    return a.astype(np.int64) * int(base) + b.astype(np.int64)


def dedup_edges(edges: np.ndarray, n_project: int, n_company: int) -> np.ndarray:
    base = max(n_project, n_company) + 1
    keys = _pack(edges[0], edges[1], base)
    uniq = np.unique(keys)
    rows = uniq // base
    cols = uniq % base
    return np.stack([rows, cols], axis=0).astype(np.int64)


def drop_edges_in(
    edges: np.ndarray, to_remove: np.ndarray, n_project: int, n_company: int
) -> tuple[np.ndarray, int]:
    if to_remove.shape[1] == 0:
        return edges, 0
    base = max(n_project, n_company) + 1
    blocked = np.unique(_pack(to_remove[0], to_remove[1], base))
    keys = _pack(edges[0], edges[1], base)
    keep = ~np.isin(keys, blocked)
    removed = int((~keep).sum())
    return edges[:, keep], removed


def build_similarity_edges(
    project_x: np.ndarray,
    company_x: np.ndarray,
    topk: int,
    direction: str = "both",
    known_edges: np.ndarray | None = None,
    drop_overlap: bool = True,
    approximate: bool = False,
    batch: int = 1024,
) -> SimilarityResult:
    """End-to-end pipeline. Returns a SimilarityResult."""
    if direction not in ("p2c", "c2p", "both"):
        raise ValueError(direction)
    n_p = project_x.shape[0]
    n_c = company_x.shape[0]

    t0 = time.perf_counter()
    pieces = []

    if direction in ("p2c", "both"):
        idx_c = build_index(company_x, approximate)
        I = batched_search(idx_c, project_x, topk, batch, tag=":p2c")
        pieces.append(edges_from_topk(I, "p2c"))
        del idx_c, I

    if direction in ("c2p", "both"):
        idx_p = build_index(project_x, approximate)
        I = batched_search(idx_p, company_x, topk, batch, tag=":c2p")
        pieces.append(edges_from_topk(I, "c2p"))
        del idx_p, I

    raw = np.concatenate(pieces, axis=1)
    n_raw = raw.shape[1]
    deduped = dedup_edges(raw, n_p, n_c)
    n_dedup = deduped.shape[1]

    n_removed = 0
    final = deduped
    if drop_overlap and known_edges is not None and known_edges.shape[1] > 0:
        final, n_removed = drop_edges_in(deduped, known_edges, n_p, n_c)

    return SimilarityResult(
        edge_index=final,
        n_raw=n_raw,
        n_dedup=n_dedup,
        n_final=final.shape[1],
        n_dropped_overlap=n_removed,
        elapsed_s=time.perf_counter() - t0,
    )

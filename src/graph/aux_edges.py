"""Derived project-company edges for message passing: keyword Jaccard and inheritance.

Both builders return project-company pairs only (bipartite invariant) and take
`exclude_keys` — the integer keys (schema.pair_keys) of pairs that must never be
produced: every observed training pair and every held-out pair. A held-out pair is
therefore never a direct neighbor through a derived edge, and the pair being
evaluated is treated exactly like a training pair.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse


def normalize_keywords(values: object) -> set[str]:
    """Keyword list -> set of case/space-insensitive tokens."""
    if values is None or isinstance(values, (float, str)):
        return set()
    try:
        out = {str(v).strip().lower().replace(" ", "") for v in values if v is not None}
    except TypeError:
        return set()
    out.discard("")
    return out


def _topk_per_row(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                  k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the k largest values of every row (ties broken by column index)."""
    if len(rows) == 0:
        return rows, cols, vals
    order = np.lexsort((cols, -vals, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    start = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
    rank = np.arange(len(rows)) - np.repeat(start, np.diff(np.r_[start, len(rows)]))
    keep = rank < k
    return rows[keep], cols[keep], vals[keep]


def _keyword_matrix(sets: Sequence[set[str]], vocab: dict[str, int]) -> sparse.csr_matrix:
    indptr, indices = [0], []
    for s in sets:
        indices.extend(sorted(vocab[t] for t in s if t in vocab))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(sets), len(vocab)))


def keyword_edges(
    project_kw: Sequence[set[str]],
    company_kw: Sequence[set[str]],
    topk: int,
    min_shared: int = 2,
    min_jaccard: float = 0.0,
    max_df: int | None = None,
    exclude_keys: np.ndarray | None = None,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k keyword-Jaccard partners for every project and for every company.

    Keywords that occur in more than `max_df` projects or companies are ignored
    (generic terms such as '소프트웨어' would otherwise connect everything); set
    sizes for the Jaccard denominator are taken after that filter. A pair needs at
    least `min_shared` common keywords. Returns (edge_index (2, E) int64, jaccard (E,)),
    the union of the project-side and company-side top-k lists.
    """
    df_p = Counter(t for s in project_kw for t in s)
    df_c = Counter(t for s in company_kw for t in s)
    shared = [t for t in df_p if t in df_c
              and (max_df is None or (df_p[t] <= max_df and df_c[t] <= max_df))]
    vocab = {t: i for i, t in enumerate(sorted(shared))}
    n_p, n_c = len(project_kw), len(company_kw)
    if not vocab:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    def usable(sets: Sequence[set[str]], df: Counter) -> np.ndarray:
        if max_df is None:
            return np.asarray([len(s) for s in sets], dtype=np.float32)
        return np.asarray([sum(1 for t in s if df[t] <= max_df) for s in sets], dtype=np.float32)

    size_p, size_c = usable(project_kw, df_p), usable(company_kw, df_c)
    mat_p, mat_c = _keyword_matrix(project_kw, vocab), _keyword_matrix(company_kw, vocab)

    def one_side(a: sparse.csr_matrix, b: sparse.csr_matrix, size_a: np.ndarray,
                 size_b: np.ndarray, flip: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bt = b.T.tocsr()
        out_r, out_c, out_v = [], [], []
        for s in range(0, a.shape[0], chunk):
            inter = (a[s:s + chunk] @ bt).tocoo()
            # int64: scipy returns int32 indices and row * n_company overflows int32
            r, c, v = inter.row.astype(np.int64) + s, inter.col.astype(np.int64), inter.data
            keep = v >= min_shared
            r, c, v = r[keep], c[keep], v[keep]
            jac = v / (size_a[r] + size_b[c] - v)
            keep = jac >= min_jaccard
            r, c, jac = r[keep], c[keep], jac[keep]
            if exclude_keys is not None and len(r):
                keys = (c * n_c + r) if flip else (r * n_c + c)
                keep = ~np.isin(keys, exclude_keys)
                r, c, jac = r[keep], c[keep], jac[keep]
            r, c, jac = _topk_per_row(r, c, jac, topk)
            out_r.append(r); out_c.append(c); out_v.append(jac)
        return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_v)

    p1, c1, v1 = one_side(mat_p, mat_c, size_p, size_c, flip=False)   # per project
    c2, p2, v2 = one_side(mat_c, mat_p, size_c, size_p, flip=True)    # per company
    return _dedup(np.concatenate([p1, p2]), np.concatenate([c1, c2]),
                  np.concatenate([v1, v2]), n_c)


def _dedup(p: np.ndarray, c: np.ndarray, v: np.ndarray,
           n_company: int) -> tuple[np.ndarray, np.ndarray]:
    keys = p.astype(np.int64) * n_company + c.astype(np.int64)
    if len(keys) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    order = np.lexsort((-v, keys))
    keys, v = keys[order], v[order]
    first = np.r_[True, keys[1:] != keys[:-1]]
    keys, v = keys[first], v[first]
    edge_index = np.stack([keys // n_company, keys % n_company]).astype(np.int64)
    return edge_index, v.astype(np.float32)


def _inherit_one_side(
    x: np.ndarray,
    partners: sparse.csr_matrix,
    k_neighbors: int,
    top_m: int,
    min_similarity: float,
    device: str,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every node of one type: the top_m opposite-type partners of its k most
    similar *warm* same-type nodes, scored by the summed similarity of the neighbors
    that hold the partner. `partners` is (n_nodes x n_opposite), observed pairs only.
    """
    import torch

    warm = np.flatnonzero(np.diff(partners.indptr) > 0)
    if warm.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float32)
    k = min(k_neighbors, warm.size)
    xw = torch.as_tensor(x[warm], dtype=torch.float32, device=device)
    pos_in_warm = np.full(x.shape[0], -1, dtype=np.int64)
    pos_in_warm[warm] = np.arange(warm.size)
    warm_partners = partners[warm]
    out_r, out_c, out_v = [], [], []
    for s in range(0, x.shape[0], chunk):
        idx = np.arange(s, min(s + chunk, x.shape[0]))
        sim = torch.as_tensor(x[idx], dtype=torch.float32, device=device) @ xw.T
        own = pos_in_warm[idx]
        has = own >= 0                      # a warm node must not be its own neighbor
        if has.any():
            sim[torch.as_tensor(np.flatnonzero(has), device=device),
                torch.as_tensor(own[has], device=device)] = -1.0
        top = torch.topk(sim, k, dim=1)
        val, nb = top.values.cpu().numpy(), top.indices.cpu().numpy()
        val = np.where(val >= min_similarity, val, 0.0)
        weights = sparse.csr_matrix(
            (val.ravel(), (np.repeat(np.arange(len(idx)), k), nb.ravel())),
            shape=(len(idx), warm.size),
        )
        weights.eliminate_zeros()
        votes = (weights @ warm_partners).tocoo()
        r, c, v = _topk_per_row(votes.row, votes.col, votes.data, top_m)
        out_r.append(r + s); out_c.append(c); out_v.append(v)
    return (np.concatenate(out_r).astype(np.int64), np.concatenate(out_c).astype(np.int64),
            np.concatenate(out_v).astype(np.float32))


def inherited_edges(
    project_x: np.ndarray,
    company_x: np.ndarray,
    observed: np.ndarray,
    k_neighbors: int,
    top_m: int,
    min_similarity: float = 0.0,
    exclude_keys: np.ndarray | None = None,
    device: str = "cpu",
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges inherited from similar same-type nodes (embedding inner product).

    project side: project p -> companies linked to the projects most similar to p
    company side: company c -> projects linked to the companies most similar to c
    `observed` (2, E) are the *training* pairs to inherit from; held-out pairs must
    not be in it. Pairs in `exclude_keys` are removed from the result.
    Returns (edge_index (2, E) int64, score (E,)), the union of both sides.
    """
    n_p, n_c = project_x.shape[0], company_x.shape[0]
    ones = np.ones(observed.shape[1], dtype=np.float32)
    p2c = sparse.csr_matrix((ones, (observed[0], observed[1])), shape=(n_p, n_c))
    p2c.data[:] = 1.0
    c2p = p2c.T.tocsr()
    p1, c1, v1 = _inherit_one_side(project_x, p2c, k_neighbors, top_m, min_similarity,
                                   device, chunk)
    c2, p2, v2 = _inherit_one_side(company_x, c2p, k_neighbors, top_m, min_similarity,
                                   device, chunk)
    edge_index, score = _dedup(np.concatenate([p1, p2]), np.concatenate([c1, c2]),
                               np.concatenate([v1, v2]), n_c)
    if exclude_keys is not None and edge_index.shape[1]:
        keep = ~np.isin(edge_index[0] * n_c + edge_index[1], exclude_keys)
        edge_index, score = edge_index[:, keep], score[keep]
    return edge_index, score


def drop_pairs(edge_index: np.ndarray, n_company: int,
               exclude_keys: Iterable[int] | np.ndarray) -> tuple[np.ndarray, int]:
    """Remove pairs whose key is in `exclude_keys`. Returns (edge_index, n_removed)."""
    keys = edge_index[0].astype(np.int64) * n_company + edge_index[1].astype(np.int64)
    keep = ~np.isin(keys, np.asarray(exclude_keys, dtype=np.int64))
    return edge_index[:, keep], int((~keep).sum())

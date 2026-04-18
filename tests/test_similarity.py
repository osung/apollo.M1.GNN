import numpy as np
import pytest

from src.graph.similarity import (
    build_similarity_edges,
    dedup_edges,
    drop_edges_in,
    edges_from_topk,
)


def _rand_norm(n, d, rng):
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v)


def test_edges_from_topk_p2c():
    I = np.array([[10, 11], [20, 21]], dtype=np.int64)  # 2 projects, k=2 companies
    edges = edges_from_topk(I, "p2c")
    assert edges.shape == (2, 4)
    assert set(map(tuple, edges.T.tolist())) == {(0, 10), (0, 11), (1, 20), (1, 21)}


def test_edges_from_topk_c2p_swaps_orientation():
    I = np.array([[7, 8]], dtype=np.int64)  # 1 company, k=2 projects
    edges = edges_from_topk(I, "c2p")
    assert edges.shape == (2, 2)
    # first row is project, second is company
    assert set(map(tuple, edges.T.tolist())) == {(7, 0), (8, 0)}


def test_dedup_removes_duplicates():
    edges = np.array([[0, 0, 1], [5, 5, 6]], dtype=np.int64)
    d = dedup_edges(edges, n_project=10, n_company=20)
    assert d.shape[1] == 2
    assert set(map(tuple, d.T.tolist())) == {(0, 5), (1, 6)}


def test_drop_edges_in_removes_matches():
    sim = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.int64)
    known = np.array([[1], [11]], dtype=np.int64)
    filtered, n_removed = drop_edges_in(sim, known, n_project=10, n_company=20)
    assert n_removed == 1
    assert filtered.shape[1] == 2
    assert (1, 11) not in set(map(tuple, filtered.T.tolist()))


def test_drop_edges_in_handles_empty_known():
    sim = np.array([[0, 1], [5, 6]], dtype=np.int64)
    empty = np.zeros((2, 0), dtype=np.int64)
    filtered, n_removed = drop_edges_in(sim, empty, 10, 20)
    assert n_removed == 0
    assert filtered.shape == sim.shape


def test_build_similarity_edges_shapes():
    rng = np.random.default_rng(0)
    d = 16
    px = _rand_norm(20, d, rng)
    cx = _rand_norm(25, d, rng)
    result = build_similarity_edges(
        project_x=px, company_x=cx, topk=5,
        direction="p2c", drop_overlap=False,
    )
    # 20 projects × 5 neighbors, dedup unlikely to remove many for random data
    assert result.edge_index.shape[0] == 2
    assert 0 < result.edge_index.shape[1] <= 20 * 5
    assert (result.edge_index[0] < 20).all()
    assert (result.edge_index[1] < 25).all()


def test_build_similarity_both_directions_merges():
    rng = np.random.default_rng(1)
    d = 16
    px = _rand_norm(20, d, rng)
    cx = _rand_norm(25, d, rng)

    only_p2c = build_similarity_edges(px, cx, topk=5, direction="p2c", drop_overlap=False)
    both = build_similarity_edges(px, cx, topk=5, direction="both", drop_overlap=False)

    # union ≥ each side alone (equality only if full overlap)
    assert both.n_final >= only_p2c.n_final


def test_build_similarity_drop_overlap_removes_known():
    rng = np.random.default_rng(2)
    d = 16
    px = _rand_norm(10, d, rng)
    cx = _rand_norm(10, d, rng)

    base = build_similarity_edges(px, cx, topk=3, direction="p2c", drop_overlap=False)
    # Mark every edge in `base` as "known" — drop_overlap=True should leave zero.
    filtered = build_similarity_edges(
        px, cx, topk=3, direction="p2c",
        known_edges=base.edge_index, drop_overlap=True,
    )
    assert filtered.n_final == 0


def test_build_similarity_invalid_direction():
    rng = np.random.default_rng(3)
    px = _rand_norm(5, 8, rng)
    cx = _rand_norm(5, 8, rng)
    with pytest.raises(ValueError):
        build_similarity_edges(px, cx, topk=2, direction="wrong")

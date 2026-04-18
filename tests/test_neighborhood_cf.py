import numpy as np
import pytest

from src.baselines.neighborhood_cf import CFConfig, NeighborhoodCF


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    d = 16
    n_p, n_c = 20, 25

    def rand_norm(n):
        v = rng.standard_normal((n, d)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    px = rand_norm(n_p)
    cx = rand_norm(n_c)

    edges_per_relation = {
        "royalty": np.array([[1, 2, 3], [5, 6, 7]], dtype=np.int64),
        "commercial": np.array([[4, 4, 5], [10, 11, 12]], dtype=np.int64),
        "performance": np.array([[1, 6, 7, 8], [5, 13, 14, 15]], dtype=np.int64),
    }
    weights = {"royalty": 1.0, "commercial": 0.75, "performance": 0.5}
    return px, cx, edges_per_relation, weights, (n_p, n_c, d)


def test_cf_shapes(toy):
    px, cx, edges, w, (n_p, n_c, _) = toy
    cf = NeighborhoodCF(px, cx, edges, w, CFConfig(k_neighbors=5, topk=7))
    idx, sc = cf.recommend_companies(np.array([0, 1, 2]))
    assert idx.shape == (3, 7)
    assert sc.shape == (3, 7)
    assert (idx >= 0).all() and (idx < n_c).all()


def test_cf_reverse_direction(toy):
    px, cx, edges, w, (n_p, n_c, _) = toy
    cf = NeighborhoodCF(px, cx, edges, w, CFConfig(k_neighbors=5, topk=7))
    idx, sc = cf.recommend_projects(np.array([5, 10]))
    assert idx.shape == (2, 7)
    assert (idx >= 0).all() and (idx < n_p).all()


def test_cf_connected_query_returns_known_neighbor(toy):
    """A project with an interaction should surface its own company
    as a candidate when we probe its self-neighborhood via include_self."""
    px, cx, edges, w, _ = toy
    cf = NeighborhoodCF(
        px, cx, edges, w, CFConfig(k_neighbors=1, topk=5, include_self=True)
    )
    idx, _ = cf.recommend_companies(np.array([1]))
    assert 5 in idx[0]


def test_cf_fallback_when_no_interaction(toy):
    """Isolated project (no neighbors with interactions) should still return
    topk via two-tower fallback."""
    px, cx, edges, w, (n_p, n_c, _) = toy
    cf = NeighborhoodCF(
        px, cx, edges, w, CFConfig(k_neighbors=3, topk=5, fallback_two_tower=True)
    )
    isolated = np.array([0])
    idx, sc = cf.recommend_companies(isolated)
    assert idx.shape == (1, 5)
    assert (idx >= 0).all()


def test_cf_deterministic(toy):
    px, cx, edges, w, _ = toy
    cfg = CFConfig(k_neighbors=5, topk=7)
    a = NeighborhoodCF(px, cx, edges, w, cfg).recommend_companies(np.arange(5))
    b = NeighborhoodCF(px, cx, edges, w, cfg).recommend_companies(np.arange(5))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_cf_relation_weight_affects_score(toy):
    px, cx, edges, _, _ = toy
    cfg = CFConfig(k_neighbors=5, topk=5, include_self=True)
    hi = NeighborhoodCF(px, cx, edges, {"royalty": 10.0, "commercial": 0.0, "performance": 0.0}, cfg)
    lo = NeighborhoodCF(px, cx, edges, {"royalty": 0.1, "commercial": 0.0, "performance": 0.0}, cfg)
    _, hi_s = hi.recommend_companies(np.array([1]))
    _, lo_s = lo.recommend_companies(np.array([1]))
    assert hi_s[0].max() > lo_s[0].max()

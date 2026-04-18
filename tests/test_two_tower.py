import numpy as np

from src.baselines.two_tower import TwoTowerBaseline


def _rand_norm(n, d, rng):
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_two_tower_shapes():
    rng = np.random.default_rng(0)
    px = _rand_norm(20, 16, rng)
    cx = _rand_norm(25, 16, rng)
    tt = TwoTowerBaseline(px, cx)
    idx, sc = tt.recommend_companies(np.array([0, 1, 2]), topk=7)
    assert idx.shape == (3, 7)
    assert sc.shape == (3, 7)
    assert (idx >= 0).all() and (idx < 25).all()


def test_two_tower_reverse():
    rng = np.random.default_rng(1)
    px = _rand_norm(20, 16, rng)
    cx = _rand_norm(25, 16, rng)
    tt = TwoTowerBaseline(px, cx)
    idx, sc = tt.recommend_projects(np.array([0, 5]), topk=4)
    assert idx.shape == (2, 4)
    assert (idx >= 0).all() and (idx < 20).all()


def test_two_tower_query_identical_company_is_top1():
    """If we plant a company vector that exactly matches a project vector,
    that company must be the top-1 match."""
    rng = np.random.default_rng(2)
    px = _rand_norm(10, 16, rng)
    cx = _rand_norm(12, 16, rng)
    cx[7] = px[3]
    tt = TwoTowerBaseline(px, cx)
    idx, sc = tt.recommend_companies(np.array([3]), topk=3)
    assert idx[0, 0] == 7
    np.testing.assert_allclose(sc[0, 0], 1.0, atol=1e-5)


def test_two_tower_dim_mismatch_raises():
    import pytest

    rng = np.random.default_rng(3)
    px = _rand_norm(5, 8, rng)
    cx = _rand_norm(5, 16, rng)
    with pytest.raises(ValueError):
        TwoTowerBaseline(px, cx)

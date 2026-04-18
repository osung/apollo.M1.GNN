import numpy as np
import pytest

from src.graph.split import split_held_out


def _edges(n):
    return np.stack([np.arange(n), np.arange(n) + 1000], axis=0).astype(np.int64)


def test_split_ratio():
    ei = _edges(100)
    tr, ho = split_held_out(ei, ratio=0.2, seed=0)
    assert tr.shape[1] == 80
    assert ho.shape[1] == 20


def test_split_deterministic():
    ei = _edges(200)
    a_tr, a_ho = split_held_out(ei, ratio=0.1, seed=42)
    b_tr, b_ho = split_held_out(ei, ratio=0.1, seed=42)
    np.testing.assert_array_equal(a_tr, b_tr)
    np.testing.assert_array_equal(a_ho, b_ho)


def test_split_seed_changes_partition():
    ei = _edges(200)
    _, a_ho = split_held_out(ei, ratio=0.1, seed=1)
    _, b_ho = split_held_out(ei, ratio=0.1, seed=2)
    assert not np.array_equal(a_ho, b_ho)


def test_split_disjoint_union():
    ei = _edges(123)
    tr, ho = split_held_out(ei, ratio=0.25, seed=7)
    assert tr.shape[1] + ho.shape[1] == ei.shape[1]
    tr_set = {tuple(x) for x in tr.T.tolist()}
    ho_set = {tuple(x) for x in ho.T.tolist()}
    assert tr_set.isdisjoint(ho_set)
    assert tr_set | ho_set == {tuple(x) for x in ei.T.tolist()}


def test_split_empty_held_out():
    ei = _edges(50)
    tr, ho = split_held_out(ei, ratio=0.0, seed=0)
    assert tr.shape[1] == 50
    assert ho.shape[1] == 0


def test_split_rejects_bad_shape():
    with pytest.raises(ValueError):
        split_held_out(np.arange(10), ratio=0.1, seed=0)


def test_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        split_held_out(_edges(10), ratio=1.0, seed=0)

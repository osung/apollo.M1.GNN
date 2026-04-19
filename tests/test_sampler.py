import numpy as np

from src.training.sampler import EdgeSampler


def _sampler(num_neg=3, batch_size=5, seed=0):
    edges = {
        "royalty": np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int64),
        "commercial": np.array([[4, 5], [14, 15]], dtype=np.int64),
        "performance": np.zeros((2, 0), dtype=np.int64),  # empty
    }
    weights = {"royalty": 1.0, "commercial": 0.5, "performance": 0.25}
    return EdgeSampler(
        edges_per_relation=edges,
        relation_weights=weights,
        n_dst=100,
        num_neg=num_neg,
        batch_size=batch_size,
        seed=seed,
    )


def test_sampler_counts():
    s = _sampler()
    assert s.n_edges == 6  # 4 + 2, performance empty dropped
    assert set(s.relation_names) == {"royalty", "commercial"}


def test_sampler_batch_shapes():
    s = _sampler(num_neg=4, batch_size=3)
    batches = list(s.iter_epoch())
    assert sum(b.pos_src.shape[0] for b in batches) == 6
    for b in batches:
        B = b.pos_src.shape[0]
        assert b.pos_dst.shape == (B,)
        assert b.neg_dst.shape == (B, 4)
        assert b.weights.shape == (B,)
        assert b.relation.shape == (B,)


def test_sampler_negs_in_range():
    s = _sampler()
    for b in s.iter_epoch():
        assert int(b.neg_dst.min()) >= 0
        assert int(b.neg_dst.max()) < 100


def test_sampler_weights_match_relation():
    s = _sampler()
    for b in s.iter_epoch():
        for i in range(b.pos_src.shape[0]):
            rel_name = s.relation_names[int(b.relation[i])]
            if rel_name == "royalty":
                assert float(b.weights[i]) == 1.0
            elif rel_name == "commercial":
                assert float(b.weights[i]) == 0.5


def test_sampler_deterministic():
    a = list(_sampler(seed=7).iter_epoch())
    b = list(_sampler(seed=7).iter_epoch())
    for ba, bb in zip(a, b):
        assert (ba.pos_src == bb.pos_src).all()
        assert (ba.neg_dst == bb.neg_dst).all()


def test_sampler_hard_negatives_drawn_from_pool():
    """With hard_ratio=1.0 every negative must come from the provided pool."""
    edges = {"royalty": np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int64)}
    weights = {"royalty": 1.0}
    # Small, disjoint pools per source so we can verify provenance.
    hard_map = {
        0: np.array([50, 51], dtype=np.int64),
        1: np.array([60, 61], dtype=np.int64),
        2: np.array([70, 71], dtype=np.int64),
        3: np.array([80, 81], dtype=np.int64),
    }
    s = EdgeSampler(
        edges_per_relation=edges,
        relation_weights=weights,
        n_dst=100,
        num_neg=4,
        batch_size=4,
        seed=0,
        hard_neg_map=hard_map,
        hard_ratio=1.0,
    )
    assert s.n_hard == 4 and s.n_random == 0
    batch = next(s.iter_epoch())
    for i in range(batch.pos_src.shape[0]):
        src = int(batch.pos_src[i])
        expected_pool = set(hard_map[src].tolist())
        actual = set(batch.neg_dst[i].tolist())
        assert actual.issubset(expected_pool), (
            f"src={src} drew negatives outside its pool: {actual - expected_pool}"
        )


def test_sampler_hybrid_hard_plus_random():
    """hard_ratio=0.5 with num_neg=4 splits 2 hard + 2 random per row."""
    edges = {"royalty": np.array([[0, 1], [10, 11]], dtype=np.int64)}
    hard_map = {
        0: np.array([500], dtype=np.int64),  # single-entry pool; sampled with replacement
        1: np.array([600], dtype=np.int64),
    }
    s = EdgeSampler(
        edges_per_relation=edges,
        relation_weights={"royalty": 1.0},
        n_dst=1000,
        num_neg=4,
        batch_size=2,
        seed=0,
        hard_neg_map=hard_map,
        hard_ratio=0.5,
    )
    assert s.n_hard == 2 and s.n_random == 2
    batch = next(s.iter_epoch())
    for i in range(batch.pos_src.shape[0]):
        src = int(batch.pos_src[i])
        # First n_hard columns must be the pool value
        hard_part = batch.neg_dst[i, :2].tolist()
        expected = hard_map[src][0]
        assert all(x == expected for x in hard_part)


def test_sampler_missing_src_in_hard_map_falls_back_to_random():
    edges = {"royalty": np.array([[7], [17]], dtype=np.int64)}
    s = EdgeSampler(
        edges_per_relation=edges,
        relation_weights={"royalty": 1.0},
        n_dst=100,
        num_neg=2,
        batch_size=1,
        seed=0,
        hard_neg_map={99: np.array([50])},  # doesn't cover src=7
        hard_ratio=1.0,
    )
    batch = next(s.iter_epoch())
    # Should not crash; negatives are random in [0, 100)
    assert batch.neg_dst.shape == (1, 2)
    assert (batch.neg_dst >= 0).all() and (batch.neg_dst < 100).all()

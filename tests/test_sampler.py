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

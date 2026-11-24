import torch
from torch_geometric.nn import HGTConv

from src.models.encoder import FastHGTConv, GNNEncoder
from src.training.neighbor_sampler import NeighborSampler

P, C = "project", "company"
FWD, REV = (P, "performance", C), (C, "rev_performance", P)


def _graph(n_p: int = 30, n_c: int = 40, n_e: int = 300, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.unique(
        torch.stack([torch.randint(n_p, (n_e,), generator=g),
                     torch.randint(n_c, (n_e,), generator=g)]), dim=1)
    return {FWD: pairs, REV: pairs.flip(0)}, {P: n_p, C: n_c}


def _global_edges(sub, layer: int, et):
    ei = sub.layers[layer][et]
    return torch.stack([sub.nodes[et[0]][ei[0]], sub.nodes[et[2]][ei[1]]])


def test_sampled_edges_are_real_bounded_and_distinct():
    edges, num_nodes = _graph()
    sampler = NeighborSampler(edges, num_nodes, fanouts=[3, 2], seed=1)
    sampler.reshuffle()
    seeds = {P: torch.tensor([4, 0, 7]), C: torch.tensor([5, 9])}
    sub = sampler.sample(seeds)
    assert sub.nodes[P][:3].tolist() == [4, 0, 7] and sub.nodes[C][:2].tolist() == [5, 9]
    assert len(sub.layers) == 2
    for layer, fanout in ((1, 3), (0, 2)):          # layers[-1] is the hop next to the seeds
        for et, full in edges.items():
            got = _global_edges(sub, layer, et)
            real = {tuple(e) for e in full.t().tolist()}
            listed = [tuple(e) for e in got.t().tolist()]
            assert set(listed) <= real
            assert len(listed) == len(set(listed))
            if got.numel():
                assert torch.bincount(got[1]).max() <= fanout
    # the last layer only feeds the seeds
    assert set(_global_edges(sub, 1, FWD)[1].tolist()) <= {5, 9}
    assert set(_global_edges(sub, 1, REV)[1].tolist()) <= {4, 0, 7}
    sampler.release(sub)
    assert all((v == -1).all() for v in sampler._local.values())


def test_large_fanout_keeps_every_in_edge_of_the_seeds():
    edges, num_nodes = _graph()
    sampler = NeighborSampler(edges, num_nodes, fanouts=[1000])
    seeds = {P: torch.arange(num_nodes[P]), C: torch.arange(num_nodes[C])}
    sub = sampler.sample(seeds)
    for et, full in edges.items():
        got = {tuple(e) for e in _global_edges(sub, 0, et).t().tolist()}
        assert got == {tuple(e) for e in full.t().tolist()}


def test_batch_pairs_are_dropped_in_both_directions():
    edges, num_nodes = _graph()
    n_c = num_nodes[C]
    p, c = edges[FWD][:, 0].tolist()
    drop = {et: torch.tensor([p * n_c + c]) for et in edges}
    sampler = NeighborSampler(edges, num_nodes, fanouts=[1000, 1000])
    sub = sampler.sample({P: torch.tensor([p]), C: torch.tensor([c])}, drop, pair_base=n_c)
    for layer in range(2):
        assert (p, c) not in {tuple(e) for e in _global_edges(sub, layer, FWD).t().tolist()}
        assert (c, p) not in {tuple(e) for e in _global_edges(sub, layer, REV).t().tolist()}


def test_minibatch_and_chunked_encoding_match_the_full_graph_pass():
    edges, num_nodes = _graph()
    torch.manual_seed(0)
    x = {P: torch.randn(num_nodes[P], 8), C: torch.randn(num_nodes[C], 8)}
    model = GNNEncoder(input_dim=8, hidden_dim=16, output_dim=16, num_layers=2,
                       metadata=([P, C], list(edges)), layer_type="hgt", num_heads=2)
    full = model.encode_all(x, edges)
    chunked = model.encode_all(x, edges, chunk_nodes=7)
    for nt in full:
        assert torch.allclose(full[nt], chunked[nt], atol=1e-5)
    # with every in-edge kept, the sampled subgraph reproduces the seeds' embeddings
    sampler = NeighborSampler(edges, num_nodes, fanouts=[1000, 1000])
    seeds = {P: torch.tensor([3, 11]), C: torch.tensor([2, 8, 20])}
    sub = sampler.sample(seeds)
    with torch.no_grad():
        z = model({nt: x[nt][ids] for nt, ids in sub.nodes.items()}, sub.layers)
    for nt, ids in seeds.items():
        assert torch.allclose(z[nt][: len(ids)], full[nt][ids], atol=1e-5)


def test_fast_hgt_conv_matches_pyg_hgt_conv():
    edges, num_nodes = _graph()
    torch.manual_seed(0)
    meta = ([P, C], list(edges))
    ref, fast = HGTConv(16, 16, meta, 4), FastHGTConv(16, 16, meta, 4)
    fast.load_state_dict(ref.state_dict())
    x = {P: torch.randn(num_nodes[P], 16), C: torch.randn(num_nodes[C], 16)}
    out_ref, out_fast = ref(x, edges), fast(x, edges)
    for nt in out_ref:
        assert torch.allclose(out_ref[nt], out_fast[nt], atol=1e-6)


def test_ranked_sampler_keeps_the_top_scored_in_edges():
    edges, num_nodes = _graph()
    g = torch.Generator().manual_seed(3)
    scores = {et: torch.rand(ei.shape[1], generator=g) for et, ei in edges.items()}
    sampler = NeighborSampler(edges, num_nodes, fanouts=[3, 2], seed=1, edge_score_dict=scores)
    seeds = {P: torch.arange(num_nodes[P]), C: torch.arange(num_nodes[C])}
    first = sampler.sample(seeds)
    got = {et: _global_edges(first, 1, et) for et in edges}
    sampler.release(first)
    sampler.reshuffle()                               # no effect on a ranked sampler
    again = sampler.sample(seeds)
    for et, full in edges.items():
        assert torch.equal(got[et], _global_edges(again, 1, et))
        kept = {tuple(e) for e in got[et].t().tolist()}
        for t in range(num_nodes[et[2]]):
            mine = (full[1] == t).nonzero().flatten()
            best = mine[torch.argsort(scores[et][mine], descending=True)[:3]]
            assert {tuple(e) for e in full[:, best].t().tolist()} == {e for e in kept if e[1] == t}


def test_encode_nodes_matches_per_batch_encoding():
    from src.training.neighbor_sampler import cosine_edge_scores, encode_nodes

    edges, num_nodes = _graph()
    torch.manual_seed(0)
    x = {P: torch.randn(num_nodes[P], 8), C: torch.randn(num_nodes[C], 8)}
    scores = {et: cosine_edge_scores(x[et[0]], x[et[2]], ei) for et, ei in edges.items()}
    model = GNNEncoder(input_dim=8, hidden_dim=16, output_dim=16, num_layers=2,
                       metadata=([P, C], list(edges)), layer_type="hgt", num_heads=2)
    sampler = NeighborSampler(edges, num_nodes, fanouts=[3, 2], edge_score_dict=scores)
    z_all = encode_nodes(model, x, sampler, seed_nodes=7)
    seeds = {P: torch.tensor([4, 0, 29]), C: torch.tensor([5, 39])}
    sub = sampler.sample(seeds)
    with torch.no_grad():
        z = model({nt: x[nt][ids] for nt, ids in sub.nodes.items()}, sub.layers)
    for nt, ids in seeds.items():
        assert torch.allclose(z_all[nt][ids], z[nt][: ids.shape[0]], atol=1e-5)

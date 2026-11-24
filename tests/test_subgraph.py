import torch
from torch_geometric.data import HeteroData

from src.graph.subgraph import induced_subgraph

P, C = "project", "company"


def _data(n_p: int, n_c: int, edges: dict[str, list[tuple[int, int]]], x: bool) -> HeteroData:
    data = HeteroData()
    data[P].num_nodes, data[C].num_nodes = n_p, n_c
    if x:
        data[P].x = torch.arange(n_p, dtype=torch.float32).unsqueeze(1)
        data[C].x = torch.arange(n_c, dtype=torch.float32).unsqueeze(1)
    for rel, pairs in edges.items():
        ei = torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1)
        data[(P, rel, C)].edge_index = ei
        if x:
            data[(P, rel, C)].edge_weight = torch.arange(ei.shape[1], dtype=torch.float32)
            data[(C, f"rev_{rel}", P)].edge_index = ei.flip(0)
    return data


def test_induced_subgraph_keeps_nodes_of_the_given_relations():
    graph = _data(6, 5, {"royalty": [(0, 1), (2, 1)], "patent": [(2, 4), (5, 1), (0, 0)]}, x=True)
    held_out = _data(6, 5, {"royalty": [(4, 3)]}, x=False)
    res = induced_subgraph(graph, held_out, ("royalty",))

    # nodes: training royalty endpoints plus the held-out endpoints (project 4, company 3)
    assert res.project_index.tolist() == [0, 2, 4] and res.company_index.tolist() == [1, 3]
    assert res.graph[P].x.squeeze(1).tolist() == [0.0, 2.0, 4.0]
    assert res.graph[C].x.squeeze(1).tolist() == [1.0, 3.0]
    assert res.graph[P].num_nodes == 3 and res.held_out[C].num_nodes == 2

    def original(data, edge_type, src_index, dst_index):
        ei = data[edge_type].edge_index.numpy()
        return sorted(zip(src_index[ei[0]].tolist(), dst_index[ei[1]].tolist(), strict=True))

    assert original(res.graph, (P, "royalty", C), res.project_index, res.company_index) == [
        (0, 1), (2, 1)]
    # patent edges survive only when both endpoints are kept: none of the three here
    assert res.graph[(P, "patent", C)].edge_index.shape[1] == 0
    assert res.graph[(P, "patent", C)].edge_weight.numel() == 0
    assert original(res.graph, (C, "rev_royalty", P), res.company_index, res.project_index) == [
        (1, 0), (1, 2)]
    assert original(res.held_out, (P, "royalty", C), res.project_index, res.company_index) == [
        (4, 3)]


def test_induced_subgraph_keeps_edge_attributes_aligned():
    graph = _data(4, 4, {"royalty": [(0, 0)], "patent": [(3, 3), (0, 0), (1, 0)]}, x=True)
    held_out = _data(4, 4, {"royalty": []}, x=False)
    res = induced_subgraph(graph, held_out, ("royalty",))
    assert res.graph[(P, "patent", C)].edge_index.tolist() == [[0], [0]]
    assert res.graph[(P, "patent", C)].edge_weight.tolist() == [1.0]

from src.graph.schema import EDGE_TYPES, NodeMap


def test_edge_types():
    assert EDGE_TYPES == ("royalty", "commercial", "performance", "similarity")


def test_node_map_roundtrip():
    m = NodeMap.from_ids(["a", "b", "c"])
    assert m.id_to_idx["b"] == 1
    assert m.idx_to_id[2] == "c"
    assert len(m) == 3

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.models.encoder import GNNEncoder


def _toy_graph(n_p=8, n_c=10, d=16, seed=0):
    g = HeteroData()
    rng = np.random.default_rng(seed)
    px = rng.standard_normal((n_p, d)).astype(np.float32)
    cx = rng.standard_normal((n_c, d)).astype(np.float32)
    g["project"].x = torch.from_numpy(px)
    g["company"].x = torch.from_numpy(cx)
    g["project"].num_nodes = n_p
    g["company"].num_nodes = n_c

    for et in ("royalty", "commercial", "performance"):
        p = torch.randint(0, n_p, (15,))
        c = torch.randint(0, n_c, (15,))
        g["project", et, "company"].edge_index = torch.stack([p, c])
        g["company", f"rev_{et}", "project"].edge_index = torch.stack([c, p])
    return g


@pytest.mark.parametrize("layer_type", ["sage", "gcn", "gat", "hgt"])
def test_encoder_forward_shapes(layer_type):
    g = _toy_graph()
    enc = GNNEncoder(
        input_dim=16,
        hidden_dim=32,
        output_dim=8,
        num_layers=2,
        metadata=g.metadata(),
        layer_type=layer_type,
        num_heads=2,
    )
    z = enc(g.x_dict, g.edge_index_dict)
    assert z["project"].shape == (g["project"].num_nodes, 8)
    assert z["company"].shape == (g["company"].num_nodes, 8)


def test_encoder_output_is_l2_normalized():
    g = _toy_graph()
    enc = GNNEncoder(
        input_dim=16, hidden_dim=32, output_dim=8, num_layers=2,
        metadata=g.metadata(), layer_type="sage",
    )
    z = enc(g.x_dict, g.edge_index_dict)
    for v in z.values():
        norms = v.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encoder_unknown_layer_type_raises():
    g = _toy_graph()
    with pytest.raises(ValueError):
        GNNEncoder(
            input_dim=16, hidden_dim=32, output_dim=8, num_layers=1,
            metadata=g.metadata(), layer_type="mlp",
        )


def test_encoder_trains_gradient_flows():
    g = _toy_graph()
    enc = GNNEncoder(
        input_dim=16, hidden_dim=16, output_dim=8, num_layers=1,
        metadata=g.metadata(), layer_type="sage",
    )
    z = enc(g.x_dict, g.edge_index_dict)
    loss = z["project"].sum() + z["company"].sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)

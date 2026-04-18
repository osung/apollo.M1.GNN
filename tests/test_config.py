from pathlib import Path

from src.utils import load_yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_all_configs_load():
    for name in ("paths.yaml", "graph.yaml", "model.yaml", "train.yaml"):
        assert load_yaml(CONFIG / name) is not None


def test_edge_priority_order():
    g = load_yaml(CONFIG / "graph.yaml")
    prio = {k: v["priority"] for k, v in g["edge_types"].items()}
    assert prio["royalty"] < prio["commercial"] < prio["performance"] < prio["similarity"]

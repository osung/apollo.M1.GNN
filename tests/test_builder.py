"""Integration test for src.graph.builder on synthetic pickle files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.graph.builder import build_graph, summarize
from src.graph.schema import (
    EDGE_COMMERCIAL,
    EDGE_PERFORMANCE,
    EDGE_ROYALTY,
    NODE_TYPE_COMPANY,
    NODE_TYPE_PROJECT,
)


@pytest.fixture
def synthetic_data(tmp_path: Path):
    rng = np.random.default_rng(0)
    n_p, n_c, d = 20, 15, 8

    p_ids = [f"P{i:03d}" for i in range(n_p)]
    c_ids = [f"C{i:03d}" for i in range(n_c)]

    def rand_emb(n):
        v = rng.standard_normal((n, d)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return [v[i] for i in range(n)]

    projects = pd.DataFrame({"과제고유번호": p_ids, "norm_embed": rand_emb(n_p)})
    companies = pd.DataFrame({"사업자번호": c_ids, "norm_embed": rand_emb(n_c)})

    def sample_edges(project_col, company_col, k, also_unknown=0):
        rows = []
        for _ in range(k):
            rows.append(
                {
                    project_col: rng.choice(p_ids),
                    company_col: rng.choice(c_ids),
                }
            )
        for i in range(also_unknown):
            rows.append({project_col: f"P_UNKNOWN_{i}", company_col: c_ids[i % len(c_ids)]})
        return pd.DataFrame(rows)

    royalty = sample_edges("과제고유번호", "기술실시대상기관_사업자번호", 30)
    commercial = sample_edges("과제고유번호", "사업화주체_사업자등록번호", 40, also_unknown=3)
    performance = sample_edges("과제고유번호", "사업자번호", 50)

    paths = {
        "projects": tmp_path / "projects.pkl",
        "companies": tmp_path / "companies.pkl",
        "royalty": tmp_path / "royalty.pkl",
        "commercial": tmp_path / "commercial.pkl",
        "performance": tmp_path / "performance.pkl",
    }
    projects.to_pickle(paths["projects"])
    companies.to_pickle(paths["companies"])
    royalty.to_pickle(paths["royalty"])
    commercial.to_pickle(paths["commercial"])
    performance.to_pickle(paths["performance"])

    path_cfg = {
        "raw": {
            "projects": str(paths["projects"]),
            "companies": str(paths["companies"]),
            "edges_royalty": str(paths["royalty"]),
            "edges_commercial": str(paths["commercial"]),
            "edges_performance": str(paths["performance"]),
        },
        "id_columns": {"project": "과제고유번호", "company": "사업자번호"},
        "edge_columns": {
            EDGE_ROYALTY: {"project": "과제고유번호", "company": "기술실시대상기관_사업자번호"},
            EDGE_COMMERCIAL: {"project": "과제고유번호", "company": "사업화주체_사업자등록번호"},
            EDGE_PERFORMANCE: {"project": "과제고유번호", "company": "사업자번호"},
        },
    }
    graph_cfg = {
        "edge_types": {
            EDGE_ROYALTY: {"priority": 1, "weight": 1.0},
            EDGE_COMMERCIAL: {"priority": 2, "weight": 0.75},
            EDGE_PERFORMANCE: {"priority": 3, "weight": 0.5},
        },
        "reverse_edges": {"enabled": True},
        "edge_weights_as_attr": True,
        "held_out": {
            "enabled": True,
            "ratio": 0.2,
            "seed": 42,
            "apply_to": [EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE],
        },
    }
    return path_cfg, graph_cfg, (n_p, n_c, d)


def test_builder_shapes(synthetic_data):
    path_cfg, graph_cfg, (n_p, n_c, d) = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    g = result.graph

    assert g[NODE_TYPE_PROJECT].num_nodes == n_p
    assert g[NODE_TYPE_COMPANY].num_nodes == n_c
    assert g[NODE_TYPE_PROJECT].x.shape == (n_p, d)
    assert g[NODE_TYPE_COMPANY].x.shape == (n_c, d)


def test_builder_reverse_edges_present(synthetic_data):
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):
        assert (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY) in result.graph.edge_types
        assert (NODE_TYPE_COMPANY, f"rev_{et}", NODE_TYPE_PROJECT) in result.graph.edge_types


def test_builder_edge_weights_as_attr(synthetic_data):
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    rel = (NODE_TYPE_PROJECT, EDGE_ROYALTY, NODE_TYPE_COMPANY)
    ew = result.graph[rel].edge_weight
    assert ew.shape[0] == result.graph[rel].edge_index.shape[1]
    assert float(ew.unique().item()) == 1.0


def test_builder_bipartite_invariant(synthetic_data):
    """Only project<->company edges exist. No project-project or company-company."""
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    for src, _, dst in result.graph.edge_types:
        assert {src, dst} == {NODE_TYPE_PROJECT, NODE_TYPE_COMPANY}


def test_builder_edge_indices_in_range(synthetic_data):
    path_cfg, graph_cfg, (n_p, n_c, _) = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    for src, et, dst in result.graph.edge_types:
        ei = result.graph[src, et, dst].edge_index
        if ei.shape[1] == 0:
            continue
        max_src = n_p if src == NODE_TYPE_PROJECT else n_c
        max_dst = n_p if dst == NODE_TYPE_PROJECT else n_c
        assert int(ei[0].max()) < max_src
        assert int(ei[1].max()) < max_dst
        assert int(ei.min()) >= 0


def test_builder_held_out_disjoint_from_train(synthetic_data):
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        tr = result.graph[rel].edge_index.numpy()
        ho = result.held_out[rel].edge_index.numpy()
        tr_set = {tuple(x) for x in tr.T.tolist()}
        ho_set = {tuple(x) for x in ho.T.tolist()}
        assert tr_set.isdisjoint(ho_set)
        assert ho.shape[1] > 0  # some edges were held out


def test_builder_drops_unknown_ids(synthetic_data):
    """commercial fixture has 3 rows with 'P_UNKNOWN' — they must be dropped."""
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    assert result.drop_counts[EDGE_COMMERCIAL] >= 3


def test_builder_deterministic_same_seed(synthetic_data):
    path_cfg, graph_cfg, _ = synthetic_data
    a = build_graph(path_cfg, graph_cfg)
    b = build_graph(path_cfg, graph_cfg)
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        np.testing.assert_array_equal(
            a.graph[rel].edge_index.numpy(), b.graph[rel].edge_index.numpy()
        )
        np.testing.assert_array_equal(
            a.held_out[rel].edge_index.numpy(), b.held_out[rel].edge_index.numpy()
        )


def test_builder_summarize_runs(synthetic_data):
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    text = summarize(result)
    assert "nodes:" in text
    assert "edges (train):" in text

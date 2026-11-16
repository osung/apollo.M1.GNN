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


def test_load_nodes_uses_configured_embedding_and_drops_missing(tmp_path):
    import pytest

    from src.data.loader import load_nodes

    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "norm_embed": [np.ones(3, dtype=np.float32)] * 4,
        "bge": [np.full(5, 0.5, dtype=np.float16), None, np.zeros(5, dtype=np.float32),
                np.full(5, 0.25, dtype=np.float32)],
    })
    path = tmp_path / "nodes.pkl"
    df.to_pickle(path)

    out = load_nodes(path, "id", embed_col="bge", drop_missing=True)
    assert out["id"].tolist() == ["a", "d"]          # missing and all-zero vectors dropped
    assert "bge" not in out.columns
    assert out["norm_embed"].iloc[0].shape == (5,)
    assert out["norm_embed"].iloc[0].dtype == np.float32
    with pytest.raises(ValueError):
        load_nodes(path, "id", embed_col="bge", drop_missing=False)
    assert len(load_nodes(path, "id")) == 4          # default: original norm_embed, nothing dropped


def test_held_out_pairs_absent_from_every_training_relation(synthetic_data):
    """A pair held out in one relation must not survive as a train edge of another."""
    path_cfg, graph_cfg, _ = synthetic_data
    result = build_graph(path_cfg, graph_cfg)
    n_c = result.graph[NODE_TYPE_COMPANY].num_nodes
    held: set[int] = set()
    train: set[int] = set()
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ho = result.held_out[rel].edge_index.numpy()
        tr = result.graph[rel].edge_index.numpy()
        held.update((ho[0] * n_c + ho[1]).tolist())
        train.update((tr[0] * n_c + tr[1]).tolist())
    assert held and not (held & train)


def test_patent_relation_is_trained_on_but_never_held_out(synthetic_data, tmp_path):
    """Patent pairs join the training graph; pairs equal to a held-out pair are dropped
    and the held-out set itself is unchanged by adding the relation."""
    path_cfg, graph_cfg, _ = synthetic_data
    base = build_graph(path_cfg, graph_cfg)
    held = set()
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):
        ei = base.held_out[(NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)].edge_index.numpy()
        held.update(zip(ei[0].tolist(), ei[1].tolist()))
    p_ids, c_ids = base.project_map.idx_to_id, base.company_map.idx_to_id
    leaked = sorted(held)[:3]                       # patent rows that repeat held-out pairs
    fresh = [(0, 1), (2, 3), (4, 5)]
    rows = [{"과제고유번호": p_ids[p], "사업자번호": c_ids[c]} for p, c in leaked + fresh]
    patent_path = tmp_path / "patent.pkl"
    pd.DataFrame(rows).to_pickle(patent_path)

    path_cfg = {**path_cfg, "raw": {**path_cfg["raw"], "edges_patent": str(patent_path)},
                "edge_columns": {**path_cfg["edge_columns"],
                                 "patent": {"project": "과제고유번호", "company": "사업자번호"}}}
    graph_cfg = {**graph_cfg, "edge_types": {**graph_cfg["edge_types"],
                                             "patent": {"priority": 3, "weight": 0.5}}}
    result = build_graph(path_cfg, graph_cfg)

    rel = (NODE_TYPE_PROJECT, "patent", NODE_TYPE_COMPANY)
    pat = result.graph[rel].edge_index.numpy()
    pat_pairs = set(zip(pat[0].tolist(), pat[1].tolist()))
    assert not (pat_pairs & held)                               # no held-out pair leaks in
    assert pat_pairs == set(fresh) - held
    assert float(result.graph[rel].edge_weight[0]) == 0.5
    assert (NODE_TYPE_COMPANY, "rev_patent", NODE_TYPE_PROJECT) in result.graph.edge_types
    assert rel not in result.held_out.edge_types                # never evaluated
    for et in (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE):  # same held-out set as before
        r = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        assert (result.held_out[r].edge_index == base.held_out[r].edge_index).all()

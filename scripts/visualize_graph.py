"""Visualize 2-hop neighborhoods of the top-scoring project / company seeds.

Selection:
- Top 10 projects and top 10 companies ranked by `유망성점수` (promising score).
- 20 seeds total → 20 PDF pages.

Per seed:
- Include ALL real-relation neighbors (royalty / commercial / performance).
- Include at most `TOP_SIM_K` similarity neighbors (cosine similarity, recomputed
  on the fly via FAISS — the sim_edges.npz cache does not store scores).
- Expand 2 hops (seed + 1-hop + 2-hop).
- Both real and similarity edges are pruned at every node by the same rule
  (all real, top-5 sim).

Output: a single multi-page PDF with one seed per page.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

faiss.omp_set_num_threads(4)

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.schema import NODE_TYPE_COMPANY, NODE_TYPE_PROJECT
from src.utils import load_yaml

REAL_RELATIONS = ("royalty", "commercial", "performance")
EDGE_COLOR = {
    "royalty": "#2ca02c",
    "commercial": "#ff7f0e",
    "performance": "#9467bd",
    "similarity": "#808080",
}
EDGE_WIDTH = {"royalty": 2.0, "commercial": 2.0, "performance": 2.0, "similarity": 0.6}
EDGE_STYLE = {
    "royalty": "solid",
    "commercial": "solid",
    "performance": "solid",
    "similarity": "dashed",
}


def _setup_korean_font() -> None:
    """Pick a Korean-capable font and force PDF to embed it as TrueType.

    matplotlib's default `pdf.fonttype=3` (Type 3) renders CJK glyphs as
    paths without proper subsetting — the result looks like boxes or
    mojibake in PDF viewers. fonttype=42 embeds the TTF, which displays
    Korean correctly.
    """
    candidates = (
        "AppleGothic",
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
    )
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is None:
        warnings.warn("No Korean-capable font found; Hangul may render as boxes.")
        chosen = matplotlib.rcParams["font.family"][0]
    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["axes.unicode_minus"] = False
    print(f"[viz] font: {chosen}")


def _build_real_adjacency(graph):
    p_adj: dict[int, dict[str, set[int]]] = {}
    c_adj: dict[int, dict[str, set[int]]] = {}
    for et in REAL_RELATIONS:
        rel = (NODE_TYPE_PROJECT, et, NODE_TYPE_COMPANY)
        ei = graph[rel].edge_index.numpy()
        srcs, dsts = ei[0].tolist(), ei[1].tolist()
        for p_idx, c_idx in zip(srcs, dsts):
            p_adj.setdefault(p_idx, {}).setdefault(et, set()).add(c_idx)
            c_adj.setdefault(c_idx, {}).setdefault(et, set()).add(p_idx)
    return p_adj, c_adj


class SimLookup:
    """Cached top-k cosine similarity lookup per (node_type, index)."""

    def __init__(self, project_x: np.ndarray, company_x: np.ndarray, k: int):
        self.k = k
        self.project_x = np.ascontiguousarray(project_x.astype(np.float32))
        self.company_x = np.ascontiguousarray(company_x.astype(np.float32))
        d = self.project_x.shape[1]
        self.idx_c = faiss.IndexFlatIP(d)
        self.idx_c.add(self.company_x)
        self.idx_p = faiss.IndexFlatIP(d)
        self.idx_p.add(self.project_x)
        self._cache: dict[tuple[str, int], list[tuple[int, float]]] = {}

    def top_k(self, node_type: str, idx: int) -> list[tuple[int, float]]:
        key = (node_type, idx)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if node_type == NODE_TYPE_PROJECT:
            q = self.project_x[idx : idx + 1]
            D, I = self.idx_c.search(q, self.k)
        else:
            q = self.company_x[idx : idx + 1]
            D, I = self.idx_p.search(q, self.k)
        result = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        self._cache[key] = result
        return result


def build_neighborhood(
    seed_idx: int,
    seed_type: str,
    p_adj: dict,
    c_adj: dict,
    sim: SimLookup,
    hops: int = 2,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    seed_key = (seed_type, seed_idx)
    G.add_node(seed_key, type=seed_type, hop=0)

    frontier = [seed_key]
    for hop in range(hops):
        next_frontier = []
        for node in frontier:
            nt, idx = node
            if nt == NODE_TYPE_PROJECT:
                other_type = NODE_TYPE_COMPANY
                real = p_adj.get(idx, {})
            else:
                other_type = NODE_TYPE_PROJECT
                real = c_adj.get(idx, {})

            for et, nbrs in real.items():
                for nbr_idx in nbrs:
                    tgt = (other_type, nbr_idx)
                    if tgt not in G.nodes:
                        G.add_node(tgt, type=other_type, hop=hop + 1)
                        next_frontier.append(tgt)
                    if not G.has_edge(node, tgt, key=et):
                        G.add_edge(node, tgt, key=et, rel=et)

            for nbr_idx, score in sim.top_k(nt, idx):
                tgt = (other_type, nbr_idx)
                if tgt not in G.nodes:
                    G.add_node(tgt, type=other_type, hop=hop + 1)
                    next_frontier.append(tgt)
                if not G.has_edge(node, tgt, key="similarity"):
                    G.add_edge(node, tgt, key="similarity", rel="similarity", weight=score)

        frontier = next_frontier
    return G


def _label(node_type: str, idx: int, project_ids, company_ids, proj_df, comp_df) -> str:
    if node_type == NODE_TYPE_PROJECT:
        pid = project_ids[idx]
        if pid in proj_df.index:
            name = str(proj_df.at[pid, "과제명"])
            return (name[:20] + "…") if len(name) > 20 else name
        return str(pid)[:14]
    cid = company_ids[idx]
    if cid in comp_df.index:
        name = str(comp_df.at[cid, "한글업체명"])
        return (name[:14] + "…") if len(name) > 14 else name
    return str(cid)[:14]


def draw(G: nx.MultiDiGraph, seed_key, ax, project_ids, company_ids, proj_df, comp_df):
    if G.number_of_nodes() == 1:
        pos = {seed_key: (0, 0)}
    else:
        try:
            pos = nx.spring_layout(G, seed=42, k=0.6, iterations=60)
        except Exception:
            pos = nx.random_layout(G, seed=42)

    for rel in ("similarity", "performance", "commercial", "royalty"):
        edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("rel") == rel
        ]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=EDGE_COLOR[rel], width=EDGE_WIDTH[rel],
            style=EDGE_STYLE[rel], alpha=0.55, arrows=True,
            connectionstyle="arc3,rad=0.08",
        )

    for node, attrs in G.nodes(data=True):
        nt = attrs["type"]
        hop = attrs.get("hop", 2)
        color = "#9ecae1" if nt == NODE_TYPE_PROJECT else "#fcae91"
        if node == seed_key:
            size, ec, lw = 1400, "#d62728", 3
        elif hop == 1:
            size, ec, lw = 550, "black", 1.2
        else:
            size, ec, lw = 170, "#888888", 0.6
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], ax=ax,
            node_color=color, node_size=size,
            edgecolors=ec, linewidths=lw,
        )

    labels = {
        node: _label(node[0], node[1], project_ids, company_ids, proj_df, comp_df)
        for node, attrs in G.nodes(data=True)
        if attrs.get("hop", 2) <= 1
    }
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#9ecae1", markersize=10, label="Project"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#fcae91", markersize=10, label="Company"),
        Line2D([0], [0], color=EDGE_COLOR["royalty"], lw=2, label="royalty"),
        Line2D([0], [0], color=EDGE_COLOR["commercial"], lw=2, label="commercial"),
        Line2D([0], [0], color=EDGE_COLOR["performance"], lw=2, label="performance"),
        Line2D([0], [0], color=EDGE_COLOR["similarity"], lw=0.8,
               linestyle="--", label="similarity (top-5)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output", default="data/processed/graph_viz.pdf")
    parser.add_argument("--top-n", type=int, default=10,
                        help="top-N seeds per side (default 10 → 20 pages)")
    parser.add_argument("--sim-k", type=int, default=5,
                        help="max similarity edges retained per node")
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()

    _setup_korean_font()
    warnings.filterwarnings("ignore", category=UserWarning)

    paths = load_yaml(args.paths)
    graph_path = args.graph_path or paths["processed"]["graph"]
    node_maps_path = Path(graph_path).with_name("node_maps.pkl")

    print(f"[viz] loading graph {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    for nt in graph.node_types:
        if graph[nt].x.dtype != torch.float32:
            graph[nt].x = graph[nt].x.float()

    print(f"[viz] loading node maps {node_maps_path}")
    with open(node_maps_path, "rb") as f:
        nm = pickle.load(f)
    project_ids: list = nm["project_ids"]
    company_ids: list = nm["company_ids"]

    print(f"[viz] loading raw project/company pickles for names & scores")
    proj_id_col = paths["id_columns"]["project"]
    comp_id_col = paths["id_columns"]["company"]
    proj_df = pd.read_pickle(paths["raw"]["projects"]).set_index(proj_id_col)
    comp_df = pd.read_pickle(paths["raw"]["companies"]).set_index(comp_id_col)

    top_projects = proj_df.nlargest(args.top_n, "유망성점수")
    top_companies = comp_df.nlargest(args.top_n, "유망성점수")
    print(f"[viz] top {args.top_n} projects (score): "
          f"{top_projects['유망성점수'].iloc[0]:.3f} .. {top_projects['유망성점수'].iloc[-1]:.3f}")
    print(f"[viz] top {args.top_n} companies (score): "
          f"{top_companies['유망성점수'].iloc[0]:.3f} .. {top_companies['유망성점수'].iloc[-1]:.3f}")

    proj_id_to_idx = {pid: i for i, pid in enumerate(project_ids)}
    comp_id_to_idx = {cid: i for i, cid in enumerate(company_ids)}

    print(f"[viz] building real adjacency (3 relations)")
    p_adj, c_adj = _build_real_adjacency(graph)

    print(f"[viz] building FAISS similarity lookup (top-{args.sim_k})")
    project_x = graph[NODE_TYPE_PROJECT].x.numpy()
    company_x = graph[NODE_TYPE_COMPANY].x.numpy()
    sim = SimLookup(project_x, company_x, args.sim_k)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[viz] rendering {args.top_n * 2} pages -> {output}")

    with PdfPages(output) as pdf:
        page = 0
        for pid, row in top_projects.iterrows():
            if pid not in proj_id_to_idx:
                print(f"  skipping project {pid} (not in graph)")
                continue
            idx = proj_id_to_idx[pid]
            page += 1
            lbl = _label(NODE_TYPE_PROJECT, idx, project_ids, company_ids, proj_df, comp_df)
            print(f"  [{page:2d}] project {lbl} (score={row['유망성점수']:.3f})")
            G = build_neighborhood(idx, NODE_TYPE_PROJECT, p_adj, c_adj, sim, hops=args.hops)
            fig, ax = plt.subplots(figsize=(14, 10))
            draw(G, (NODE_TYPE_PROJECT, idx), ax, project_ids, company_ids, proj_df, comp_df)
            ax.set_title(
                f"Project seed: {lbl}  |  score={row['유망성점수']:.3f}  "
                f"|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}",
                fontsize=11,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for cid, row in top_companies.iterrows():
            if cid not in comp_id_to_idx:
                print(f"  skipping company {cid} (not in graph)")
                continue
            idx = comp_id_to_idx[cid]
            page += 1
            lbl = _label(NODE_TYPE_COMPANY, idx, project_ids, company_ids, proj_df, comp_df)
            print(f"  [{page:2d}] company {lbl} (score={row['유망성점수']:.3f})")
            G = build_neighborhood(idx, NODE_TYPE_COMPANY, p_adj, c_adj, sim, hops=args.hops)
            fig, ax = plt.subplots(figsize=(14, 10))
            draw(G, (NODE_TYPE_COMPANY, idx), ax, project_ids, company_ids, proj_df, comp_df)
            ax.set_title(
                f"Company seed: {lbl}  |  score={row['유망성점수']:.3f}  "
                f"|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}",
                fontsize=11,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[viz] done: {page} pages saved to {output}")


if __name__ == "__main__":
    main()

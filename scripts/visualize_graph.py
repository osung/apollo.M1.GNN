"""Visualize 2-hop neighborhoods of the top-scoring project / company seeds.

Layout per page:
    ┌───────────────┬───────────────┐
    │               │  Node table   │
    │   Graph       │───────────────┤
    │   (symbols)   │  Edge table   │
    └───────────────┴───────────────┘

- The graph is drawn on the left half; node labels are short symbols
  (P0 is the seed project, C0 the seed company; other projects are
  P1, P2, … and companies C1, C2, …).
- The right half has two tables that let the reader decode the symbols:
  a Nodes table (Symbol | Type | Name) and an Edges table
  (Project | Company | Relation) in canonical (project → company)
  orientation.

Similarity edges are capped per node (default top-3) so the picture
stays readable; real relations are kept in full.
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
import textwrap
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
from matplotlib.gridspec import GridSpec
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
EDGE_WIDTH = {"royalty": 1.8, "commercial": 1.8, "performance": 1.8, "similarity": 0.5}
EDGE_STYLE = {
    "royalty": "solid",
    "commercial": "solid",
    "performance": "solid",
    "similarity": "dashed",
}
REL_ORDER = {"royalty": 0, "commercial": 1, "performance": 2, "similarity": 3}


def _setup_korean_font() -> None:
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
        for p_idx, c_idx in zip(ei[0].tolist(), ei[1].tolist()):
            p_adj.setdefault(p_idx, {}).setdefault(et, set()).add(c_idx)
            c_adj.setdefault(c_idx, {}).setdefault(et, set()).add(p_idx)
    return p_adj, c_adj


class SimLookup:
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
) -> nx.MultiGraph:
    """Undirected MultiGraph with unique (u, v, rel) triples."""
    G = nx.MultiGraph()
    seed_key = (seed_type, seed_idx)
    G.add_node(seed_key, type=seed_type, hop=0)
    seen_edges: set[tuple[frozenset, str]] = set()

    def add_edge_once(u, v, rel, **data):
        key = (frozenset({u, v}), rel)
        if key in seen_edges:
            return
        seen_edges.add(key)
        G.add_edge(u, v, rel=rel, **data)

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
                    add_edge_once(node, tgt, et)

            for nbr_idx, score in sim.top_k(nt, idx):
                tgt = (other_type, nbr_idx)
                if tgt not in G.nodes:
                    G.add_node(tgt, type=other_type, hop=hop + 1)
                    next_frontier.append(tgt)
                add_edge_once(node, tgt, "similarity", weight=score)

        frontier = next_frontier
    return G


def assign_symbols(G: nx.MultiGraph, seed_key) -> dict[tuple, str]:
    """Stable short symbols: seed=P0/C0; others P1/C1/... by (hop, type, idx)."""
    symbols: dict[tuple, str] = {}
    symbols[seed_key] = "P0" if seed_key[0] == NODE_TYPE_PROJECT else "C0"

    others = sorted(
        (n for n in G.nodes if n != seed_key),
        key=lambda n: (G.nodes[n]["hop"], n[0], n[1]),
    )
    p_n, c_n = 1, 1
    for node in others:
        if node[0] == NODE_TYPE_PROJECT:
            symbols[node] = f"P{p_n}"
            p_n += 1
        else:
            symbols[node] = f"C{c_n}"
            c_n += 1
    return symbols


def _label(node_type: str, idx: int, project_ids, company_ids, proj_df, comp_df) -> str:
    if node_type == NODE_TYPE_PROJECT:
        pid = project_ids[idx]
        if pid in proj_df.index:
            return str(proj_df.at[pid, "과제명"])
        return str(pid)
    cid = company_ids[idx]
    if cid in comp_df.index:
        return str(comp_df.at[cid, "한글업체명"])
    return str(cid)


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _wrap_name(text: str, width: int) -> str:
    """Wrap long names to multiple lines so they aren't truncated."""
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ").strip()
    lines = textwrap.wrap(
        text, width=width, break_long_words=True, break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else text


def draw_graph(G: nx.MultiGraph, seed_key, symbols: dict, ax) -> None:
    if G.number_of_nodes() == 1:
        pos = {seed_key: (0, 0)}
    else:
        try:
            pos = nx.spring_layout(G, seed=42, k=0.9, iterations=80)
        except Exception:
            pos = nx.random_layout(G, seed=42)

    for rel in ("similarity", "performance", "commercial", "royalty"):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") == rel]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=EDGE_COLOR[rel], width=EDGE_WIDTH[rel],
            style=EDGE_STYLE[rel], alpha=0.6,
        )

    for node, attrs in G.nodes(data=True):
        nt = attrs["type"]
        hop = attrs.get("hop", 2)
        color = "#9ecae1" if nt == NODE_TYPE_PROJECT else "#fcae91"
        if node == seed_key:
            size, ec, lw = 700, "#d62728", 2.2
        elif hop == 1:
            size, ec, lw = 320, "black", 1.0
        else:
            size, ec, lw = 150, "#888888", 0.5
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], ax=ax,
            node_color=color, node_size=size,
            edgecolors=ec, linewidths=lw,
        )

    nx.draw_networkx_labels(
        G, pos, {n: symbols[n] for n in G.nodes}, ax=ax, font_size=5.5,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#9ecae1", markersize=8, label="Project"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#fcae91", markersize=8, label="Company"),
        Line2D([0], [0], color=EDGE_COLOR["royalty"], lw=1.8, label="royalty"),
        Line2D([0], [0], color=EDGE_COLOR["commercial"], lw=1.8, label="commercial"),
        Line2D([0], [0], color=EDGE_COLOR["performance"], lw=1.8, label="performance"),
        Line2D([0], [0], color=EDGE_COLOR["similarity"], lw=0.8,
               linestyle="--", label="similarity"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=6.5, frameon=True)
    ax.set_axis_off()


def _node_table_font_size(n_rows: int) -> float:
    return float(max(4.0, min(9.0, 22.0 * 18 / max(n_rows, 1) ** 0.85)))


def _node_table_row_height(n_rows: int) -> float:
    return float(max(0.6, min(1.5, 50.0 / max(n_rows, 1))))


def draw_node_table(
    G: nx.MultiGraph, symbols: dict, seed_key, ax,
    project_ids, company_ids, proj_df, comp_df,
    wrap_width: int = 34,
) -> None:
    ax.set_axis_off()
    rows = []
    ordered = [seed_key] + sorted(
        (n for n in G.nodes if n != seed_key),
        key=lambda n: (int(symbols[n][1:]), n[0]),
    )
    for node in ordered:
        sym = symbols[node]
        if node == seed_key:
            sym = f"★ {sym}"
        nt_display = "Project" if node[0] == NODE_TYPE_PROJECT else "Company"
        name = _label(node[0], node[1], project_ids, company_ids, proj_df, comp_df)
        rows.append([sym, nt_display, _wrap_name(name, wrap_width)])

    if not rows:
        return

    max_lines = max((r[2].count("\n") + 1) for r in rows)
    table = ax.table(
        cellText=rows,
        colLabels=["Symbol", "Type", "Name"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.10, 0.15, 0.75],
    )
    table.auto_set_font_size(False)
    fs = _node_table_font_size(len(rows) + 1)
    table.set_fontsize(fs)

    base_h = _node_table_row_height(len(rows) + 1)
    table.scale(1.0, base_h)
    for i, row in enumerate(rows, start=1):
        lines = max(1, row[2].count("\n") + 1)
        if lines > 1:
            for j in range(3):
                cell = table[(i, j)]
                cell.set_height(cell.get_height() * lines)

    for j in range(3):
        cell = table[(0, j)]
        cell.set_facecolor("#eaeaea")
        cell.set_text_props(weight="bold")
    for i, node in enumerate(ordered, start=1):
        if node == seed_key:
            for j in range(3):
                table[(i, j)].set_facecolor("#fff2cc")

    ax.set_title(f"Nodes ({len(rows)})", fontsize=10, loc="left", pad=4)


def render_page(
    pdf, G, seed_key, seed_label: str, score: float,
    project_ids, company_ids, proj_df, comp_df,
) -> None:
    symbols = assign_symbols(G, seed_key)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.05)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_nodes = fig.add_subplot(gs[0, 1])

    draw_graph(G, seed_key, symbols, ax_graph)
    draw_node_table(G, symbols, seed_key, ax_nodes,
                    project_ids, company_ids, proj_df, comp_df)

    seed_sym = symbols[seed_key]
    nt_display = "Project" if seed_key[0] == NODE_TYPE_PROJECT else "Company"
    fig.suptitle(
        f"{nt_display} seed [{seed_sym}]: {_truncate(seed_label, 50)}  "
        f"|  score={score:.3f}  |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}",
        fontsize=11, y=0.995,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output", default="data/processed/graph_viz.pdf")
    parser.add_argument("--top-n", type=int, default=10,
                        help="top-N seeds per side (default 10 → 20 pages)")
    parser.add_argument("--sim-k", type=int, default=3,
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
                continue
            idx = proj_id_to_idx[pid]
            page += 1
            lbl = _label(NODE_TYPE_PROJECT, idx, project_ids, company_ids, proj_df, comp_df)
            print(f"  [{page:2d}] project {_truncate(lbl, 30)} (score={row['유망성점수']:.3f})")
            G = build_neighborhood(idx, NODE_TYPE_PROJECT, p_adj, c_adj, sim, hops=args.hops)
            render_page(pdf, G, (NODE_TYPE_PROJECT, idx), lbl, float(row["유망성점수"]),
                        project_ids, company_ids, proj_df, comp_df)

        for cid, row in top_companies.iterrows():
            if cid not in comp_id_to_idx:
                continue
            idx = comp_id_to_idx[cid]
            page += 1
            lbl = _label(NODE_TYPE_COMPANY, idx, project_ids, company_ids, proj_df, comp_df)
            print(f"  [{page:2d}] company {_truncate(lbl, 30)} (score={row['유망성점수']:.3f})")
            G = build_neighborhood(idx, NODE_TYPE_COMPANY, p_adj, c_adj, sim, hops=args.hops)
            render_page(pdf, G, (NODE_TYPE_COMPANY, idx), lbl, float(row["유망성점수"]),
                        project_ids, company_ids, proj_df, comp_df)

    print(f"[viz] done: {page} pages saved to {output}")


if __name__ == "__main__":
    main()

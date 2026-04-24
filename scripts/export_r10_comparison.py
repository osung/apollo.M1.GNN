"""Generate a Recall@10 / NDCG@10 comparison table.

Separate from export_results.py (which tracks R@100) so we can quickly
eyeball top-of-list retrieval quality, which matters more for UI/UX
than deep-list recall.

All numbers are taken from the experiment logs recorded during the
study. Cells marked '-' were not logged for that configuration.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path("results/r10_comparison.xlsx")
OUT.parent.mkdir(parents=True, exist_ok=True)


# R@10 comparison rows
r10_rows = [
    # --- baselines ---
    {"method": "Two-Tower (zero-train)", "type": "baseline",
     "p2c_roy_R10": 0.0094, "p2c_com_R10": 0.0067, "p2c_per_R10": 0.0100,
     "c2p_roy_R10": 0.0042, "c2p_com_R10": 0.0032, "c2p_per_R10": 0.0044,
     "notes": "No training. norm_embed + FAISS"},
    {"method": "LightGCN hd=256 ep200", "type": "baseline",
     "p2c_roy_R10": 0.0460, "p2c_com_R10": 0.0389, "p2c_per_R10": 0.0553,
     "c2p_roy_R10": 0.0131, "c2p_com_R10": 0.0241, "c2p_per_R10": 0.0009,
     "notes": "No W matrix; saturated after ep120"},
    # --- LightFM per-direction specialists ---
    {"method": "LightFM (project-train, p2c native)", "type": "specialist-p2c",
     "p2c_roy_R10": 0.4041, "p2c_com_R10": 0.3343, "p2c_per_R10": 0.3428,
     "c2p_roy_R10": 0.0489, "c2p_com_R10": 0.0354, "c2p_per_R10": 0.0450,
     "notes": "Strong in native p2c; weak reversed c2p"},
    {"method": "LightFM (company-train, c2p native)", "type": "specialist-c2p",
     "p2c_roy_R10": 0.0873, "p2c_com_R10": 0.0799, "p2c_per_R10": 0.0760,
     "c2p_roy_R10": 0.1835, "c2p_com_R10": 0.1358, "c2p_per_R10": 0.1617,
     "notes": "Strong in native c2p; weak reversed p2c"},
    # --- our GNN families (best balanced checkpoints) ---
    {"method": "SAGE hd=384 ep180 (clean)", "type": "gnn-unified",
     "p2c_roy_R10": 0.3394, "p2c_com_R10": 0.2883, "p2c_per_R10": 0.2639,
     "c2p_roy_R10": 0.1556, "c2p_com_R10": 0.0864, "c2p_per_R10": 0.0978,
     "notes": "SAGE clean hd=384 extended peak"},
    {"method": "GCN hd=256 ep185 (clean)", "type": "gnn-unified",
     "p2c_roy_R10": 0.3771, "p2c_com_R10": 0.3308, "p2c_per_R10": 0.2875,
     "c2p_roy_R10": 0.1588, "c2p_com_R10": 0.1132, "c2p_per_R10": 0.0961,
     "notes": "Our balanced champion (recommended main model)"},
    {"method": "GCN hd=256 ep135 (c2p peak)", "type": "gnn-unified",
     "p2c_roy_R10": 0.3264, "p2c_com_R10": 0.3164, "p2c_per_R10": 0.2766,
     "c2p_roy_R10": 0.1518, "c2p_com_R10": 0.1098, "c2p_per_R10": 0.1245,
     "notes": "GCN hd=256 ep135 peak on c2p commercial/performance"},
]

# NDCG@10 comparison rows (same setups, different metric)
ndcg10_rows = [
    {"method": "Two-Tower (zero-train)", "type": "baseline",
     "p2c_roy_N10": 0.0055, "p2c_com_N10": 0.0033, "p2c_per_N10": 0.0055,
     "c2p_roy_N10": 0.0022, "c2p_com_N10": 0.0018, "c2p_per_N10": 0.0026,
     "notes": ""},
    {"method": "LightGCN hd=256 ep200", "type": "baseline",
     "p2c_roy_N10": 0.0230, "p2c_com_N10": 0.0185, "p2c_per_N10": 0.0285,
     "c2p_roy_N10": 0.0058, "c2p_com_N10": 0.0170, "c2p_per_N10": 0.0003,
     "notes": ""},
    {"method": "LightFM (project-train, p2c native)", "type": "specialist-p2c",
     "p2c_roy_N10": 0.2535, "p2c_com_N10": 0.2083, "p2c_per_N10": 0.2254,
     "c2p_roy_N10": 0.0250, "c2p_com_N10": 0.0186, "c2p_per_N10": 0.0220,
     "notes": ""},
    {"method": "LightFM (company-train, c2p native)", "type": "specialist-c2p",
     "p2c_roy_N10": 0.0415, "p2c_com_N10": 0.0407, "p2c_per_N10": 0.0389,
     "c2p_roy_N10": 0.0937, "c2p_com_N10": 0.0688, "c2p_per_N10": 0.0855,
     "notes": ""},
    {"method": "SAGE hd=384 ep180 (clean)", "type": "gnn-unified",
     "p2c_roy_N10": 0.2300, "p2c_com_N10": 0.1766, "p2c_per_N10": 0.1750,
     "c2p_roy_N10": 0.0859, "c2p_com_N10": 0.0472, "c2p_per_N10": 0.0511,
     "notes": ""},
    {"method": "GCN hd=256 ep185 (clean)", "type": "gnn-unified",
     "p2c_roy_N10": 0.2598, "p2c_com_N10": 0.2206, "p2c_per_N10": 0.1955,
     "c2p_roy_N10": 0.0858, "c2p_com_N10": 0.0590, "c2p_per_N10": 0.0522,
     "notes": ""},
]

# Head-to-head: LightFM dual vs GCN
h2h_rows = [
    {"metric": "p2c_royalty R@10",       "LightFM_best": 0.4041, "GCN_h256_ep185": 0.3771, "gap_pct": "+7.2% (LightFM)"},
    {"metric": "p2c_commercial R@10",    "LightFM_best": 0.3343, "GCN_h256_ep185": 0.3308, "gap_pct": "+1.1% (tie)"},
    {"metric": "p2c_performance R@10",   "LightFM_best": 0.3428, "GCN_h256_ep185": 0.2875, "gap_pct": "+19.2% (LightFM)"},
    {"metric": "c2p_royalty R@10",       "LightFM_best": 0.1835, "GCN_h256_ep185": 0.1588, "gap_pct": "+15.5% (LightFM)"},
    {"metric": "c2p_commercial R@10",    "LightFM_best": 0.1358, "GCN_h256_ep185": 0.1132, "gap_pct": "+20.0% (LightFM)"},
    {"metric": "c2p_performance R@10",   "LightFM_best": 0.1617, "GCN_h256_ep185": 0.0961, "gap_pct": "+68.3% (LightFM)"},
    # NDCG@10: two surprising GCN wins on p2c royalty/commercial
    {"metric": "p2c_royalty NDCG@10",    "LightFM_best": 0.2535, "GCN_h256_ep185": 0.2598, "gap_pct": "+2.5% (GCN WIN)"},
    {"metric": "p2c_commercial NDCG@10", "LightFM_best": 0.2083, "GCN_h256_ep185": 0.2206, "gap_pct": "+5.9% (GCN WIN)"},
    {"metric": "p2c_performance NDCG@10","LightFM_best": 0.2254, "GCN_h256_ep185": 0.1955, "gap_pct": "+15.3% (LightFM)"},
    {"metric": "c2p_royalty NDCG@10",    "LightFM_best": 0.0937, "GCN_h256_ep185": 0.0858, "gap_pct": "+9.2% (LightFM)"},
    {"metric": "c2p_commercial NDCG@10", "LightFM_best": 0.0688, "GCN_h256_ep185": 0.0590, "gap_pct": "+16.6% (LightFM)"},
    {"metric": "c2p_performance NDCG@10","LightFM_best": 0.0855, "GCN_h256_ep185": 0.0522, "gap_pct": "+63.8% (LightFM)"},
]


def main() -> None:
    df_r10 = pd.DataFrame(r10_rows)
    df_ndcg10 = pd.DataFrame(ndcg10_rows)
    df_h2h = pd.DataFrame(h2h_rows)

    # Add avg columns
    df_r10["p2c_avg"] = df_r10[
        ["p2c_roy_R10", "p2c_com_R10", "p2c_per_R10"]
    ].mean(axis=1).round(4)
    df_r10["c2p_avg"] = df_r10[
        ["c2p_roy_R10", "c2p_com_R10", "c2p_per_R10"]
    ].mean(axis=1).round(4)

    df_ndcg10["p2c_avg"] = df_ndcg10[
        ["p2c_roy_N10", "p2c_com_N10", "p2c_per_N10"]
    ].mean(axis=1).round(4)
    df_ndcg10["c2p_avg"] = df_ndcg10[
        ["c2p_roy_N10", "c2p_com_N10", "c2p_per_N10"]
    ].mean(axis=1).round(4)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        df_r10.to_excel(w, sheet_name="recall@10", index=False)
        df_ndcg10.to_excel(w, sheet_name="ndcg@10", index=False)
        df_h2h.to_excel(w, sheet_name="head2head_LightFM_vs_GCN", index=False)

    print(f"[r10] wrote {OUT}")
    print(f"  recall@10               : {len(df_r10)} methods")
    print(f"  ndcg@10                 : {len(df_ndcg10)} methods")
    print(f"  head2head_LightFM_vs_GCN: {len(df_h2h)} metrics")


if __name__ == "__main__":
    main()

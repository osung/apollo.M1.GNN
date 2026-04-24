"""Export all experiment results to Excel.

Consolidates SAGE/HGT/GAT/GCN runs from this study into a single workbook
with multiple sheets:
  - main_results: all completed experiments at peak/final checkpoints
  - sage_h256_clean_trajectory: full epoch-by-epoch for the clean run
  - sage_h384_clean_trajectory: full epoch-by-epoch for the clean run
  - champion_by_metric: peak per-metric across all setups
  - failed_runs: abandoned / unusable experiments with reasons
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("results/all_experiments.xlsx")
OUT.parent.mkdir(parents=True, exist_ok=True)


# Columns shared across runs. R@100 and NDCG@100 reported for each relation.
COLS = [
    "arch", "hidden_dim", "num_layers", "num_heads",
    "hard_ratio", "sim_topk", "c2p_weight", "hard_neg",
    "amp", "epochs_total", "ep_reported", "ep_type",
    "p2c_royalty_R100", "p2c_commercial_R100", "p2c_performance_R100",
    "c2p_royalty_R100", "c2p_commercial_R100", "c2p_performance_R100",
    "p2c_avg_R100", "c2p_avg_R100",
    "time_per_epoch_s", "params", "notes",
]


def _row(arch, hd, L, H, hr, sim, c2p, hn, amp, ep_total, ep, ep_type,
         p2c_r, p2c_c, p2c_p, c2p_r, c2p_c, c2p_p,
         time_s, params, notes=""):
    p2c_avg = round((p2c_r + p2c_c + p2c_p) / 3, 4) if p2c_r is not None else None
    c2p_avg = round((c2p_r + c2p_c + c2p_p) / 3, 4) if c2p_r is not None else None
    return {
        "arch": arch, "hidden_dim": hd, "num_layers": L, "num_heads": H,
        "hard_ratio": hr, "sim_topk": sim, "c2p_weight": c2p, "hard_neg": hn,
        "amp": amp, "epochs_total": ep_total, "ep_reported": ep, "ep_type": ep_type,
        "p2c_royalty_R100": p2c_r, "p2c_commercial_R100": p2c_c, "p2c_performance_R100": p2c_p,
        "c2p_royalty_R100": c2p_r, "c2p_commercial_R100": c2p_c, "c2p_performance_R100": c2p_p,
        "p2c_avg_R100": p2c_avg, "c2p_avg_R100": c2p_avg,
        "time_per_epoch_s": time_s, "params": params, "notes": notes,
    }


main_rows = [
    # ---- Baselines (no training / CF only) ----
    _row("Two-Tower", "-", "-", "-", "-", "-", "-", "none", "-", 0, 0, "zero_train",
         0.0324, 0.0298, 0.0441, 0.0184, 0.0221, 0.0212, 0.0, 0,
         "Zero-training baseline: norm_embed + FAISS IndexFlatIP. "
         "No bipartite learning, no relation type distinction"),
    _row("LightFM", 128, "-", "-", "-", "-", "-", "warm-proj", "-", 20, 20, "final",
         0.5854, 0.5094, 0.5236, 0.1928, 0.1425, 0.1757, 14.2, 0,
         "LightFM trained project-as-user (WARP, warm identity). "
         "Native p2c: STRONG (beats GNN). Reversed c2p: WEAK"),
    _row("LightFM", 128, "-", "-", "-", "-", "-", "warm-comp", "-", 20, 20, "final",
         0.3079, 0.2565, 0.2418, 0.4595, 0.3789, 0.3507, 15.3, 0,
         "LightFM trained company-as-user (WARP, warm identity). "
         "Native c2p: STRONG (beats GNN). Reversed p2c: WEAK. "
         "Two separate models required for bidirectional service"),
    # ---- SAGE hidden_dim scaling (all hr=0.8, c2p=1.0, sim_top10 unless noted) ----
    _row("SAGE", 64, 2, "-", 0.7, 10, 1.0, "dirty", "fp32", 100, 85, "peak",
         0.380, 0.334, 0.296, 0.245, 0.137, 0.174, 7.6, 247168,
         "first hr sweep"),
    _row("SAGE", 64, 2, "-", 0.7, 10, 1.0, "dirty", "fp32", 100, 100, "final",
         0.383, 0.346, 0.281, 0.215, 0.132, 0.139, 7.6, 247168, ""),
    _row("SAGE", 64, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 100, 100, "final",
         0.387, 0.346, 0.291, 0.229, 0.138, 0.171, 7.6, 247168,
         "hr=0.8 slightly better than hr=0.7"),
    _row("SAGE", 64, 2, "-", 0.8, 20, 1.0, "dirty", "fp32", 120, 120, "final",
         0.395, 0.357, 0.306, 0.229, 0.144, 0.168, 11.1, 247168,
         "sim20 marginal p2c +3%, c2p tied; +46% time"),

    _row("SAGE", 128, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 120, 105, "peak_c2p",
         0.458, 0.402, 0.353, 0.304, 0.234, 0.243, 11.4, 756224, ""),
    _row("SAGE", 128, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 120, 115, "peak_p2c",
         0.469, 0.417, 0.360, 0.304, 0.228, 0.259, 11.4, 756224,
         "hd=128 champion (dirty)"),
    _row("SAGE", 128, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 120, 120, "final",
         0.447, 0.346, 0.310, 0.305, 0.219, 0.248, 11.4, 756224,
         "post-peak decline"),

    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 120, 120, "final",
         0.483, 0.435, 0.367, 0.358, 0.278, 0.262, 25.0, 2560768,
         "dirty champion; best p2c royalty overall"),
    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "dirty", "fp32", 120, 105, "mid",
         0.466, 0.426, 0.362, 0.347, 0.289, 0.265, 25.0, 2560768, ""),

    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 100, "mid",
         0.463, 0.426, 0.364, 0.322, 0.256, 0.247, 25.0, 2560768, ""),
    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 105, "mid",
         0.466, 0.426, 0.362, 0.347, 0.289, 0.265, 25.0, 2560768, ""),
    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 130, "peak_p2c",
         0.473, 0.429, 0.371, 0.345, 0.270, 0.259, 25.0, 2560768,
         "clean p2c peak"),
    _row("SAGE", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 150, "final",
         0.463, 0.428, 0.370, 0.354, 0.302, 0.270, 25.0, 2560768,
         "clean c2p peak (commercial)"),

    _row("SAGE", 384, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 105, "peak_p2c",
         0.474, 0.426, 0.369, 0.362, 0.302, 0.282, 41.8, 5413888, ""),
    _row("SAGE", 384, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 150, "final",
         0.466, 0.430, 0.372, 0.362, 0.337, 0.285, 41.8, 5413888,
         "clean hd=384 c2p commercial peak 0.337"),
    _row("SAGE", 384, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 180, 180, "final",
         0.470, 0.428, 0.369, 0.398, 0.293, 0.299, 41.8, 5413888,
         "extended to ep180; c2p royalty peak 0.398"),

    # ---- HGT ----
    _row("HGT", 64, 2, 4, 0.8, 10, 1.0, "dirty", "fp32", 100, 100, "final",
         0.381, 0.343, 0.297, 0.239, 0.154, 0.186, 36.0, 214468, ""),
    _row("HGT", 64, 2, 4, 0.8, 10, 1.0, "dirty", "fp32", 120, 105, "peak",
         0.414, 0.376, 0.315, 0.257, 0.184, 0.193, 36.0, 214468,
         "HGT champion (hd=64)"),

    # ---- GCN (clean only; dirty was broken due to aggr='add' bug) ----
    _row("GCN", 64, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 145, "peak_p2c",
         0.434, 0.376, 0.339, 0.282, 0.169, 0.181, 7.6, 247168,
         "GCN peak p2c (clean + mean aggr fix)"),
    _row("GCN", 64, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 150, "final",
         0.429, 0.380, 0.330, 0.301, 0.167, 0.209, 7.6, 247168,
         "GCN final; stable monotonic convergence, no dips"),
    _row("GCN", 128, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 105, "peak_p2c",
         0.451, 0.393, 0.355, 0.274, 0.192, 0.194, 11.4, 756224,
         "GCN hd=128 early peak p2c royalty"),
    _row("GCN", 128, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 150, "final",
         0.444, 0.405, 0.349, 0.326, 0.230, 0.247, 11.4, 756224,
         "GCN hd=128 ep150 final"),
    _row("GCN", 128, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 195, "peak_extended",
         0.466, 0.430, 0.366, 0.363, 0.254, 0.286, 11.4, 756224,
         "GCN hd=128 ep195 peak (extended); matches SAGE hd=384 ep150 on 5/6 metrics"),
    _row("GCN", 128, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 200, "final_extended",
         0.458, 0.421, 0.356, 0.330, 0.218, 0.254, 11.4, 756224,
         "GCN hd=128 ep200 final (extended)"),
    _row("GCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 135, "peak_c2p",
         0.471, 0.430, 0.368, 0.367, 0.317, 0.297, 25.0, 2560768,
         "GCN hd=256 ep135 peak c2p metrics (best c2p commercial so far)"),
    _row("GCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 150, 150, "final",
         0.470, 0.435, 0.371, 0.367, 0.300, 0.284, 25.0, 2560768,
         "GCN hd=256 ep150 final; new champion candidate — 5/6 metrics tie or beat SAGE hd=384"),
    _row("GCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 185, "peak_balanced",
         0.482, 0.437, 0.373, 0.378, 0.332, 0.270, 25.0, 2560768,
         "GCN hd=256 ep185 (extended); new records on 3 p2c metrics, balanced across directions"),
    _row("GCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 170, "peak_c2p_perf",
         0.478, 0.424, 0.368, 0.370, 0.278, 0.316, 25.0, 2560768,
         "GCN hd=256 ep170 (extended); new c2p performance record 0.316, beats SAGE hd=384"),
    _row("GCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 200, "final_extended",
         0.475, 0.433, 0.370, 0.340, 0.264, 0.243, 25.0, 2560768,
         "GCN hd=256 ep200 final (extended)"),
    # ---- LightGCN: norm_embed adaptation (weak baseline due to no W matrix) ----
    _row("LightGCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 120, "midrun",
         0.200, 0.157, 0.191, 0.048, 0.039, 0.018, 17.2, 393728,
         "LightGCN hd=256 ep120 (still climbing slowly but plateauing); "
         "adapted with norm_embed input_proj instead of ID embedding. "
         "Without W matrix and nonlinearity, 2-8x below GCN/SAGE at same hd"),
    _row("LightGCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 190, "peak",
         0.202, 0.168, 0.199, 0.014, 0.040, 0.020, 17.2, 393728,
         "LightGCN hd=256 ep190 peak; no W/no nonlinearity limits expressiveness"),
    _row("LightGCN", 256, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 200, "final",
         0.200, 0.166, 0.198, 0.048, 0.041, 0.021, 17.2, 393728,
         "LightGCN hd=256 ep200 final; saturated after ep120; "
         "2.4x p2c, 9.7x c2p gap vs GCN hd=256 confirms W matrix necessity"),
    # ---- GCN hd=384 (capacity scaling saturation confirmed) ----
    _row("GCN", 384, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 125, "peak_balanced",
         0.477, 0.434, 0.373, 0.376, 0.309, 0.297, 42.0, 5427840,
         "GCN hd=384 ep125 balanced peak; matches hd=256 champion but 2.1x params. "
         "Capacity saturation confirmed beyond hd=256."),
    _row("GCN", 384, 2, "-", 0.8, 10, 1.0, "clean", "fp32", 200, 200, "final",
         0.464, 0.431, 0.365, 0.342, 0.287, 0.243, 42.0, 5427840,
         "GCN hd=384 ep200 final; post-peak oscillation"),
    # ---- hr=0.9 ablation: too aggressive, hurts all 6 metrics ----
    _row("GCN", 384, 2, "-", 0.9, 10, 1.0, "clean", "fp32", 200, 180, "peak_hr09",
         0.467, 0.416, 0.361, 0.323, 0.260, 0.224, 41.9, 5427840,
         "GCN hd=384 hr=0.9 balanced peak at ep180; WORSE than hr=0.8 on all 6 metrics"),
    _row("GCN", 384, 2, "-", 0.9, 10, 1.0, "clean", "fp32", 200, 195, "peak_hr09_p2c",
         0.464, 0.429, 0.365, 0.296, 0.251, 0.235, 41.9, 5427840,
         "GCN hd=384 hr=0.9 p2c peak at ep195; shows 10% oscillation (vs 5% at hr=0.8)"),
    _row("GCN", 384, 2, "-", 0.9, 10, 1.0, "clean", "fp32", 200, 200, "final_hr09",
         0.464, 0.415, 0.347, 0.287, 0.260, 0.212, 41.9, 5427840,
         "GCN hd=384 hr=0.9 ep200 final; ablation confirming hr=0.8 is sweet spot"),
    # ---- SeHGNN: new p2c royalty champion despite only 140K params ----
    _row("SeHGNN", 64, 2, 4, 0.8, 5, 1.0, "clean", "fp32", 200, 175, "peak_p2c",
         0.503, 0.426, 0.367, 0.308, 0.262, 0.212, 20.9, 140416,
         "SeHGNN hd=64 ep175 — NEW RECORD p2c royalty 0.503; beats GCN hd=256 (2.56M params) with just 140K"),
    _row("SeHGNN", 64, 2, 4, 0.8, 5, 1.0, "clean", "fp32", 200, 170, "peak_c2p",
         0.494, 0.422, 0.355, 0.325, 0.277, 0.223, 20.9, 140416,
         "SeHGNN hd=64 ep170 — peak c2p royalty for SeHGNN (0.325)"),
]

failed_rows = [
    {
        "arch": "GCN", "hidden_dim": 64, "num_layers": 2, "amp": "fp32",
        "status": "failed -> fixed", "reason":
            "GraphConv default aggr='add' + 12M sim edges -> activation "
            "explosion with --no-normalize. Fixed via aggr='mean' (commit 6448b14). "
            "After fix, GCN clean achieves p2c royalty 0.434 at ep145.",
    },
    {
        "arch": "GCN", "hidden_dim": 64, "num_layers": 2, "amp": "fp32",
        "status": "failed -> fixed", "reason":
            "First clean run accidentally omitted --no-normalize flag; loss stuck "
            "at BPR plateau ~0.27 (L2 bottleneck). Rerun with --no-normalize "
            "produced the working result noted above.",
    },
    {
        "arch": "GAT", "hidden_dim": 64, "num_layers": 2, "num_heads": 4, "amp": "fp32",
        "status": "OOM",
        "reason":
            "GAT via HeteroConv wrapper materializes per-edge "
            "(E, H, C) intermediates per relation. With 12M sim edges "
            "this is 11.6 GiB per relation, exceeding A100 80GB.",
    },
    {
        "arch": "HGT", "hidden_dim": 128, "num_heads": 4, "amp": "bf16",
        "status": "OOM", "reason":
            "Even with bf16, sim_top10 (24M edges inc. reverse) plus "
            "full-graph forward exhausts A100 80GB during encode_all "
            "fp32 spike at checkpoint.",
    },
    {
        "arch": "HGT", "hidden_dim": 96, "num_layers": 1, "num_heads": 2, "amp": "bf16",
        "status": "abandoned", "reason":
            "L=1 insufficient message-passing depth: p2c royalty ~0.10 "
            "at ep50 vs HGT hd=64 L=2 fp32 achieved 0.34 at ep50.",
    },
    {
        "arch": "HGT", "hidden_dim": 96, "num_layers": 2, "num_heads": 2, "amp": "bf16",
        "status": "abandoned", "reason":
            "150s/epoch (4.2x slower than hd=64 fp32). PyG HGTConv "
            "autocast path inefficient. Projected peak ~0.45 doesn't "
            "beat SAGE hd=256 (0.483) at 1/3 the time.",
    },
    {
        "arch": "SAGE", "hidden_dim": 128, "num_layers": 2, "sim_topk": 20, "amp": "fp32",
        "status": "abandoned_ep45", "reason":
            "sim20 at hd=128 trails sim10 at all checkpoints. Additional "
            "similarity edges add noise once capacity is sufficient.",
    },
]

champion_rows = [
    {"metric": "p2c_royalty_R100",    "best_value": 0.483, "setup": "SAGE hd=256 dirty ep120",     "context": "stochastic peak; multi-seed needed"},
    {"metric": "p2c_commercial_R100", "best_value": 0.437, "setup": "GCN hd=256 clean ep185",      "context": "NEW RECORD via GCN extended training"},
    {"metric": "p2c_performance_R100","best_value": 0.375, "setup": "GCN hd=256 clean ep190",      "context": "NEW RECORD via GCN extended training"},
    {"metric": "c2p_royalty_R100",    "best_value": 0.398, "setup": "SAGE hd=384 clean ep180",     "context": "extended training breakthrough"},
    {"metric": "c2p_commercial_R100", "best_value": 0.337, "setup": "SAGE hd=384 clean ep150",     "context": "clean + capacity combined"},
    {"metric": "c2p_performance_R100","best_value": 0.316, "setup": "GCN hd=256 clean ep170",      "context": "NEW RECORD via GCN extended training"},
    # --- LightFM baseline champions (unidirectional) ---
    {"metric": "p2c_royalty_baseline_R100", "best_value": 0.585, "setup": "LightFM warm WARP ep20", "context": "LightFM dominates p2c; +21% over our GNN (unidirectional advantage)"},
    {"metric": "p2c_commercial_baseline_R100","best_value": 0.509, "setup": "LightFM warm WARP ep20", "context": "LightFM p2c commercial; +16% over our GNN"},
    {"metric": "p2c_performance_baseline_R100","best_value": 0.524, "setup": "LightFM warm WARP ep20", "context": "LightFM p2c performance; +40% over our GNN"},
    # --- efficient alternatives (simpler + longer training) ---
    {"metric": "p2c_royalty_efficient","best_value": 0.466, "setup": "GCN hd=128 clean ep195",    "context": "7x fewer params than SAGE hd=384; ties on p2c"},
    {"metric": "c2p_royalty_efficient","best_value": 0.363, "setup": "GCN hd=128 clean ep195",    "context": "ties SAGE hd=384 with 7x fewer params"},
    # --- GCN hd=256 (new balanced champion candidate) ---
    {"metric": "p2c_royalty_GCN",     "best_value": 0.471, "setup": "GCN hd=256 clean ep130/145","context": "GCN hd=256 p2c peak; 2x fewer params than SAGE hd=384"},
    {"metric": "c2p_royalty_GCN",     "best_value": 0.367, "setup": "GCN hd=256 clean ep135/150","context": "GCN hd=256 c2p peak; beats SAGE hd=256"},
    {"metric": "c2p_commercial_GCN",  "best_value": 0.317, "setup": "GCN hd=256 clean ep135",    "context": "GCN hd=256 c2p commercial peak; beats SAGE hd=256 (0.302)"},
]


sage_h256_clean_trajectory = [
    # (epoch, p2c_r, p2c_c, p2c_p, c2p_r, c2p_c, c2p_p)
    (70,  0.4503, 0.4094, 0.3559, 0.3023, 0.2302, 0.2322),
    (75,  0.4524, 0.4082, 0.3516, 0.3088, 0.2604, 0.2185),
    (80,  0.4550, 0.4158, 0.3542, 0.3431, 0.2515, 0.2419),
    (85,  0.4540, 0.4079, 0.3533, 0.3121, 0.2019, 0.2538),
    (90,  0.4571, 0.4060, 0.3452, 0.3121, 0.2628, 0.2369),
    (95,  0.4540, 0.4151, 0.3602, 0.3097, 0.2251, 0.2312),
    (100, 0.4629, 0.4255, 0.3635, 0.3218, 0.2556, 0.2468),
    (105, 0.4655, 0.4262, 0.3621, 0.3465, 0.2888, 0.2652),
    (110, 0.3985, 0.3864, 0.2828, 0.2142, 0.1963, 0.1573),  # major dip
    (115, 0.4189, 0.3770, 0.3302, 0.2166, 0.1831, 0.1397),
    (120, 0.4388, 0.4108, 0.3499, 0.2732, 0.1901, 0.2299),
    (125, 0.4477, 0.4208, 0.3442, 0.2885, 0.2694, 0.2001),
    (130, 0.4733, 0.4294, 0.3714, 0.3453, 0.2696, 0.2594),  # peak avg
    (135, 0.4618, 0.4266, 0.3645, 0.3254, 0.2802, 0.2422),
    (140, 0.4676, 0.4233, 0.3645, 0.3202, 0.2653, 0.2201),
    (145, 0.4650, 0.4305, 0.3666, 0.3246, 0.2895, 0.2475),
    (150, 0.4629, 0.4279, 0.3697, 0.3538, 0.3015, 0.2698),  # final
]


sage_h384_clean_trajectory = [
    (10,  0.1736, 0.1495, 0.1189, 0.0967, 0.0679, 0.0503),
    (15,  0.3149, 0.2529, 0.2192, 0.1509, 0.0972, 0.0953),
    (20,  0.3421, 0.2821, 0.2361, 0.1909, 0.1036, 0.1201),
    (25,  0.3954, 0.3671, 0.2732, 0.2059, 0.1554, 0.1561),
    (30,  0.4210, 0.3750, 0.2899, 0.2606, 0.1617, 0.1846),
    (35,  0.4467, 0.3941, 0.3399, 0.2699, 0.1804, 0.1989),
    (40,  0.4508, 0.4069, 0.3385, 0.3016, 0.2316, 0.2219),
    (45,  0.4508, 0.3990, 0.3485, 0.2892, 0.2224, 0.2293),
    (50,  0.4409, 0.3973, 0.3371, 0.2753, 0.2272, 0.1953),
    (55,  0.4561, 0.4191, 0.3576, 0.3137, 0.2578, 0.2534),
    (60,  0.4498, 0.4115, 0.3487, 0.2978, 0.2274, 0.2040),
    (65,  0.4519, 0.4119, 0.3509, 0.2918, 0.2210, 0.2280),
    (70,  0.4582, 0.4157, 0.3499, 0.3153, 0.2385, 0.2553),
    (75,  0.4514, 0.4206, 0.3595, 0.3396, 0.2864, 0.2566),
    (80,  0.4587, 0.4228, 0.3585, 0.3097, 0.2332, 0.2613),
    (85,  0.4446, 0.3964, 0.3497, 0.3239, 0.2099, 0.2569),
    (90,  0.4555, 0.4090, 0.3585, 0.3209, 0.2439, 0.2589),
    (95,  0.4671, 0.4150, 0.3626, 0.3368, 0.2082, 0.2705),
    (100, 0.4571, 0.4230, 0.3595, 0.3314, 0.2687, 0.2439),
    (105, 0.4738, 0.4263, 0.3690, 0.3623, 0.3017, 0.2816),  # peak p2c
    (110, 0.4618, 0.4205, 0.3635, 0.3063, 0.2421, 0.2235),
    (115, 0.4634, 0.4220, 0.3668, 0.3342, 0.2728, 0.2466),
    (120, 0.4592, 0.4180, 0.3630, 0.3079, 0.2044, 0.2302),
    (125, 0.4655, 0.4216, 0.3673, 0.3237, 0.2509, 0.2167),
    (130, 0.4597, 0.4277, 0.3647, 0.3519, 0.2558, 0.2466),
    (135, 0.4691, 0.4281, 0.3657, 0.3163, 0.2772, 0.2430),
    (140, 0.4603, 0.4289, 0.3661, 0.3134, 0.3065, 0.2259),
    (145, 0.4555, 0.4182, 0.3542, 0.3263, 0.2732, 0.2526),
    (150, 0.4660, 0.4300, 0.3716, 0.3617, 0.3365, 0.2847),  # original final
    # --- extended run (ep146-180 with different sampler rng) ---
    (155, 0.4629, 0.4277, 0.3618, 0.3703, 0.3082, 0.2686),
    (160, 0.4712, 0.4222, 0.3633, 0.3298, 0.2443, 0.2504),
    (165, 0.4691, 0.4296, 0.3690, 0.3503, 0.2814, 0.2727),
    (170, 0.4576, 0.4267, 0.3566, 0.3659, 0.3089, 0.2530),
    (175, 0.4508, 0.4267, 0.3652, 0.3516, 0.2823, 0.2608),
    (180, 0.4697, 0.4275, 0.3690, 0.3982, 0.2926, 0.2994),  # extended final
]

TRAJ_COLS = ["epoch", "p2c_royalty", "p2c_commercial", "p2c_performance",
             "c2p_royalty", "c2p_commercial", "c2p_performance"]


def main():
    df_main = pd.DataFrame(main_rows, columns=COLS)
    df_failed = pd.DataFrame(failed_rows)
    df_champ = pd.DataFrame(champion_rows)
    df_t256 = pd.DataFrame(sage_h256_clean_trajectory, columns=TRAJ_COLS)
    df_t384 = pd.DataFrame(sage_h384_clean_trajectory, columns=TRAJ_COLS)

    for df in (df_t256, df_t384):
        df["p2c_avg"] = df[["p2c_royalty", "p2c_commercial", "p2c_performance"]].mean(axis=1).round(4)
        df["c2p_avg"] = df[["c2p_royalty", "c2p_commercial", "c2p_performance"]].mean(axis=1).round(4)

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        df_main.to_excel(writer, sheet_name="main_results", index=False)
        df_champ.to_excel(writer, sheet_name="champion_by_metric", index=False)
        df_t256.to_excel(writer, sheet_name="sage_h256_clean_traj", index=False)
        df_t384.to_excel(writer, sheet_name="sage_h384_clean_traj", index=False)
        df_failed.to_excel(writer, sheet_name="failed_runs", index=False)

    print(f"[export] wrote {OUT}")
    print(f"  main_results:              {len(df_main)} runs")
    print(f"  champion_by_metric:        {len(df_champ)} metrics")
    print(f"  sage_h256_clean_traj:      {len(df_t256)} epochs")
    print(f"  sage_h384_clean_traj:      {len(df_t384)} epochs")
    print(f"  failed_runs:               {len(df_failed)} entries")


if __name__ == "__main__":
    main()

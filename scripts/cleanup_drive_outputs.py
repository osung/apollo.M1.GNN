"""Cleanup script for Drive outputs/ — delete intermediate epoch files,
keep only peaks + final + tag-level files.

Run this in Colab (or any environment with write access to the Drive
folder). It mirrors the KEEP list we used when copying peak files to
the local laptop, so executing this script after the local copy leaves
Drive trim with the same set of survivors as local outputs/.

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')

    # Then either run this file directly:
    !python scripts/cleanup_drive_outputs.py --dry-run
    # Inspect the plan, then:
    !python scripts/cleanup_drive_outputs.py --execute

    # Or paste the body into a notebook cell and toggle DRY_RUN.

Files deleted go to Drive trash (30-day recovery window).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Default Drive path inside Colab. Override with --outputs-dir if mounted
# elsewhere or if running outside Colab.
DEFAULT_OUTPUTS = "/content/drive/MyDrive/apollo.M1.GNN/outputs"


# Per-experiment epochs to KEEP. Everything else with a matching tag gets
# deleted. The list mirrors export_results.py rows + documented peaks.
KEEP: dict[str, list[int]] = {
    # ---- GFM family (current SOTA family) ----
    "gfm_h128_l2_hr00_c2p100_sim10_nonorm": [200, 235, 260, 265, 300],
    "gfm_h128_l3_hr00_c2p100_sim5_nonorm":  [195, 270, 285, 295, 300],
    "gfm_h256_l2_hr00_c2p100_sim3_nonorm":  [190, 200, 225, 275],
    # ---- GCN family ----
    "gcn_h128_l2_hr80_c2p100_sim10_nonorm": [185, 200],
    "gcn_h256_l2_hr80_c2p100_sim10_nonorm": [135, 185, 200],
    "gcn_h384_l2_hr80_c2p100_sim10_nonorm": [125, 200],
    "gcn_h384_l2_hr90_c2p100_sim10_nonorm": [180, 195, 200],
    "gcn_h64_l2_hr80_c2p100_sim10":          [145],
    "gcn_h64_l2_hr80_c2p100_sim10_nonorm":   [145],
    "gcn_h64_l2_sim10":                      [],   # crashed run, delete entirely
    # ---- SAGE family ----
    "sage_h256_l2_hr80_c2p100_sim10_nonorm": [145],
    "sage_h384_l2_hr80_c2p100_sim10_nonorm": [175],
    # ---- LightGCN family ----
    "lightgcn_h256_l2_hr80_c2p100_sim10_nonorm": [190, 200],
    "lightgcn_h384_l2_hr80_c2p100_sim10_nonorm": [200],
    # ---- SeHGNN ----
    "sehgnn_h64_l2_hr80_c2p100_sim5_nonorm": [170, 175, 200],
}


PATTERNS = [
    re.compile(r"^model_(.+)_epoch(\d+)\.pt$"),
    re.compile(r"^project_z_(.+)_epoch(\d+)\.npy$"),
    re.compile(r"^company_z_(.+)_epoch(\d+)\.npy$"),
    re.compile(r"^metrics_(.+)_epoch(\d+)\.json$"),
]
TAG_LEVEL = re.compile(r"^(history|metrics_final)_(.+)\.json$")
CLEAN_SUMMARY = re.compile(r"^metrics_.+_clean\.json$")


def classify(name: str) -> tuple[str, str | None, int | None]:
    """Return (verdict, tag, epoch). verdict ∈ {KEEP, DELETE, UNTOUCHED}."""
    for rx in PATTERNS:
        m = rx.match(name)
        if not m:
            continue
        tag, ep = m.group(1), int(m.group(2))
        if tag in KEEP and ep in KEEP[tag]:
            return "KEEP", tag, ep
        if tag in KEEP:
            # Tag is known, epoch isn't on the keep list -> delete
            return "DELETE", tag, ep
        # Unknown tag — leave alone to avoid surprises
        return "UNTOUCHED", tag, ep
    if TAG_LEVEL.match(name) or CLEAN_SUMMARY.match(name):
        return "KEEP", None, None
    return "UNTOUCHED", None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir", default=DEFAULT_OUTPUTS,
        help="path to Drive's outputs/ (default: %(default)s)",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--dry-run", action="store_true",
        help="report the plan without deleting anything",
    )
    grp.add_argument(
        "--execute", action="store_true",
        help="actually delete intermediate files (goes to Drive trash)",
    )
    args = parser.parse_args()

    root = Path(args.outputs_dir)
    if not root.is_dir():
        raise SystemExit(f"outputs dir not found: {root}")

    keep_files, delete_files, untouched_files = [], [], []
    for f in root.iterdir():
        if not f.is_file():
            continue
        verdict, _, _ = classify(f.name)
        if verdict == "KEEP":
            keep_files.append(f)
        elif verdict == "DELETE":
            delete_files.append(f)
        else:
            untouched_files.append(f)

    keep_gb = sum(f.stat().st_size for f in keep_files) / 1e9
    del_gb = sum(f.stat().st_size for f in delete_files) / 1e9
    unt_gb = sum(f.stat().st_size for f in untouched_files) / 1e9

    mode = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(f"=== Drive outputs/ cleanup ({mode}) ===")
    print(f"path: {root}")
    print(f"  KEEP      : {len(keep_files):>5} files  {keep_gb:>7.2f} GB")
    print(f"  DELETE    : {len(delete_files):>5} files  {del_gb:>7.2f} GB")
    print(f"  UNTOUCHED : {len(untouched_files):>5} files  {unt_gb:>7.2f} GB")
    print()

    if untouched_files and len(untouched_files) <= 20:
        print("UNTOUCHED files (review and delete manually if not needed):")
        for f in untouched_files:
            sz = f.stat().st_size / 1e6
            print(f"  {sz:>7.2f} MB  {f.name}")
        print()

    if args.dry_run:
        print("Add --execute to actually delete the DELETE list above.")
        return

    print(f"Deleting {len(delete_files)} files...")
    deleted, errors = 0, []
    for i, f in enumerate(delete_files, 1):
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            errors.append((f.name, str(e)))
        if i % 50 == 0 or i == len(delete_files):
            print(f"  progress: {i}/{len(delete_files)}  ({deleted} deleted, {len(errors)} errors)")

    print(f"\n=== DONE ===")
    print(f"  deleted: {deleted}")
    print(f"  errors:  {len(errors)}")
    if errors:
        print("First 5 errors:")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

"""Cleanup script for Drive outputs/ or sweep/ subfolders — delete
intermediate epoch checkpoints, keep only peaks + final + tag-level files.

Two modes:

  --keep-mode explicit   (default): use the KEEP dict below — peak
      epochs documented in export_results.py for the long single-cell
      runs (gfm_h128_l2_..._epoch{200,235,260,265,300}, etc.). Best
      for the curated outputs/ folder.

  --keep-mode last-epoch: per <tag>, keep only the highest-epoch ckpt.
      Best for sweep/ output where each cell ran ep1..ep_total and the
      intermediate ep20/40/.../180 don't carry independent value beyond
      diagnostics. Tag-level files (history, metrics_final, summary
      json) are always kept regardless.

Run from any environment with delete permission on the target folder
(typically Colab; the laptop Stream-mode mount lacks delete perm).

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')

    # outputs/ — explicit KEEP list
    !python scripts/cleanup_drive_outputs.py --dry-run
    !python scripts/cleanup_drive_outputs.py --execute

    # sweep/<model>/ — last-epoch rule, recursive
    !python scripts/cleanup_drive_outputs.py \\
        --outputs-dir /content/drive/MyDrive/apollo.M1.GNN/sweep/gfm \\
        --keep-mode last-epoch --dry-run
    !python scripts/cleanup_drive_outputs.py \\
        --outputs-dir /content/drive/MyDrive/apollo.M1.GNN/sweep/gfm \\
        --keep-mode last-epoch --execute

    # sweep/ root (recurses into all model subdirs)
    !python scripts/cleanup_drive_outputs.py \\
        --outputs-dir /content/drive/MyDrive/apollo.M1.GNN/sweep \\
        --keep-mode last-epoch --recursive --dry-run

Files deleted go to Drive trash (30-day recovery window).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


# Default Drive path inside Colab. Override with --outputs-dir if mounted
# elsewhere or if running outside Colab.
DEFAULT_OUTPUTS = "/content/drive/MyDrive/apollo.M1.GNN/outputs"


# Per-experiment epochs to KEEP in --keep-mode explicit. Everything else
# with a matching tag gets deleted. Mirrors export_results.py rows.
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
TAG_LEVEL = re.compile(r"^(history|metrics_final|metrics)_(.+)\.json$")
CLEAN_SUMMARY = re.compile(r"^metrics_.+_clean\.json$")


def parse_epoch_file(name: str) -> tuple[str, int] | None:
    """Return (tag, epoch) if name matches one of PATTERNS, else None."""
    for rx in PATTERNS:
        m = rx.match(name)
        if m:
            return m.group(1), int(m.group(2))
    return None


def is_tag_level(name: str) -> bool:
    """True if name is a per-tag summary file (history, metrics_*) we
    always want to keep regardless of mode."""
    return bool(TAG_LEVEL.match(name) or CLEAN_SUMMARY.match(name))


def classify_explicit(name: str) -> str:
    """Verdict for --keep-mode explicit. ∈ {KEEP, DELETE, UNTOUCHED}."""
    parsed = parse_epoch_file(name)
    if parsed is not None:
        tag, ep = parsed
        if tag in KEEP and ep in KEEP[tag]:
            return "KEEP"
        if tag in KEEP:
            return "DELETE"
        return "UNTOUCHED"  # unknown tag — defensive
    if is_tag_level(name):
        return "KEEP"
    return "UNTOUCHED"


def collect_files(root: Path, recursive: bool) -> list[Path]:
    """Iterate files under root (one level or recursive)."""
    if recursive:
        return [p for p in root.rglob("*") if p.is_file()]
    return [p for p in root.iterdir() if p.is_file()]


def classify_all_last_epoch(files: list[Path]) -> dict[Path, str]:
    """Verdict per file for --keep-mode last-epoch. Group epoch files
    by tag, keep highest epoch per tag, delete the rest. Tag-level
    summary files always KEEP. Unknown -> UNTOUCHED."""
    # tag -> list of (epoch, file_path) (across all PATTERNS)
    tag_files: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    other: list[Path] = []
    for f in files:
        parsed = parse_epoch_file(f.name)
        if parsed is not None:
            tag, ep = parsed
            tag_files[tag].append((ep, f))
        else:
            other.append(f)

    verdict: dict[Path, str] = {}
    for tag, items in tag_files.items():
        max_ep = max(ep for ep, _ in items)
        for ep, f in items:
            verdict[f] = "KEEP" if ep == max_ep else "DELETE"
    for f in other:
        verdict[f] = "KEEP" if is_tag_level(f.name) else "UNTOUCHED"
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir", default=DEFAULT_OUTPUTS,
        help="path to the folder to clean up (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-mode",
        choices=["explicit", "last-epoch"],
        default="explicit",
        help="explicit: use the curated KEEP dict (best for outputs/). "
             "last-epoch: per tag, keep only highest epoch (best for sweep/).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="recurse into subdirectories. Useful for sweep/ root which "
             "contains per-model subfolders.",
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

    files = collect_files(root, args.recursive)

    keep_files, delete_files, untouched_files = [], [], []
    if args.keep_mode == "explicit":
        for f in files:
            v = classify_explicit(f.name)
            if v == "KEEP":
                keep_files.append(f)
            elif v == "DELETE":
                delete_files.append(f)
            else:
                untouched_files.append(f)
    else:  # last-epoch
        verdicts = classify_all_last_epoch(files)
        for f in files:
            v = verdicts[f]
            if v == "KEEP":
                keep_files.append(f)
            elif v == "DELETE":
                delete_files.append(f)
            else:
                untouched_files.append(f)

    keep_gb = sum(f.stat().st_size for f in keep_files) / 1e9
    del_gb = sum(f.stat().st_size for f in delete_files) / 1e9
    unt_gb = sum(f.stat().st_size for f in untouched_files) / 1e9

    action = "DRY-RUN" if args.dry_run else "EXECUTE"
    print(f"=== cleanup ({action}, mode={args.keep_mode}, recursive={args.recursive}) ===")
    print(f"path: {root}")
    print(f"  KEEP      : {len(keep_files):>5} files  {keep_gb:>7.2f} GB")
    print(f"  DELETE    : {len(delete_files):>5} files  {del_gb:>7.2f} GB")
    print(f"  UNTOUCHED : {len(untouched_files):>5} files  {unt_gb:>7.2f} GB")
    print()

    if args.keep_mode == "last-epoch":
        # Show what's being kept per tag for transparency
        by_tag: dict[str, list[int]] = defaultdict(list)
        for f in keep_files:
            parsed = parse_epoch_file(f.name)
            if parsed is not None:
                by_tag[parsed[0]].append(parsed[1])
        if by_tag:
            print("KEEP (per tag, highest epoch only):")
            for tag in sorted(by_tag):
                eps = sorted(set(by_tag[tag]))
                print(f"  {tag:<55} keep epoch={eps[-1]}")
            print()

    if untouched_files and len(untouched_files) <= 30:
        print("UNTOUCHED files (review manually):")
        for f in untouched_files:
            sz = f.stat().st_size / 1e6
            try:
                rel = f.relative_to(root)
            except ValueError:
                rel = f.name
            print(f"  {sz:>7.2f} MB  {rel}")
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

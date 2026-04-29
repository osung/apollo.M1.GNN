"""Run match_top10.py for every (project_z, company_z) pair in outputs/.

Iterates over all model checkpoints synced into outputs/ and produces a
top-10 match DataFrame (.pkl + .xlsx) per checkpoint per direction. Skips
runs whose output files already exist, so the script is safe to re-run
after a partial completion.

Output filenames embed the checkpoint tag and top-percent:
    results/match_top10_p2c__<tag>__top10pct.pkl
    results/match_top10_c2p__<tag>__top10pct.pkl

Usage from project root:
    python scripts/match_top10_all_models.py
    python scripts/match_top10_all_models.py --top-percent 0.05 --topk 20
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def _scan_pairs(outputs_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (tag, project_z_path, company_z_path) — only pairs
    where both files exist for the same tag."""
    proj_re = re.compile(r"^project_z_(.+)\.npy$")
    pairs: list[tuple[str, Path, Path]] = []
    for p in sorted(outputs_dir.iterdir()):
        if not p.is_file():
            continue
        m = proj_re.match(p.name)
        if not m:
            continue
        tag = m.group(1)
        c = outputs_dir / f"company_z_{tag}.npy"
        if c.exists():
            pairs.append((tag, p, c))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--top-percent", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--direction", default="both", choices=["p2c", "c2p", "both"],
    )
    parser.add_argument(
        "--out-prefix", default="match_top10",
    )
    parser.add_argument(
        "--match-script", default="scripts/match_top10.py",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the planned commands without running them",
    )
    parser.add_argument(
        "--include", default=None,
        help="optional substring filter; only run tags containing this",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs = _scan_pairs(outputs_dir)
    if args.include:
        pairs = [p for p in pairs if args.include in p[0]]
    print(f"[match_all] found {len(pairs)} (proj, comp) z pairs")
    if not pairs:
        return

    pct_tag = f"top{int(round(args.top_percent * 100))}pct"

    plan: list[tuple[str, list[str], list[Path]]] = []
    for tag, p_path, c_path in pairs:
        # The directions match_top10.py will produce
        directions = (
            ["p2c"] if args.direction == "p2c"
            else ["c2p"] if args.direction == "c2p"
            else ["p2c", "c2p"]
        )
        expected_outs = [
            results_dir / f"{args.out_prefix}_{d}__{tag}__{pct_tag}.{ext}"
            for d in directions for ext in ("pkl", "xlsx")
        ]
        if all(o.exists() for o in expected_outs):
            print(f"  SKIP  {tag}  (all {len(expected_outs)} output files already exist)")
            continue
        cmd = [
            sys.executable, args.match_script,
            "--project-z", str(p_path),
            "--company-z", str(c_path),
            "--top-percent", str(args.top_percent),
            "--topk", str(args.topk),
            "--direction", args.direction,
            "--output-dir", str(results_dir),
            "--out-prefix", args.out_prefix,
        ]
        plan.append((tag, cmd, expected_outs))

    print(f"[match_all] {len(plan)} pair(s) to run")
    for tag, cmd, _ in plan:
        print(f"  RUN   {tag}")
    if args.dry_run:
        print("\n--dry-run: not executing")
        return

    # Execute one at a time (each run is full-batch and uses CPU/RAM heavily).
    t_total = time.time()
    failures = 0
    for i, (tag, cmd, _) in enumerate(plan, 1):
        print(f"\n=== [{i}/{len(plan)}] running for {tag} ===")
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            print(f"  OK ({time.time()-t0:.0f}s)")
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"  FAILED rc={e.returncode}: {tag}")
        except KeyboardInterrupt:
            print("\n[match_all] interrupted by user")
            raise

    print(f"\n[match_all] DONE  total={time.time()-t_total:.0f}s  "
          f"runs={len(plan)}  failures={failures}")


if __name__ == "__main__":
    main()

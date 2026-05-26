"""
Select the best N runs from one or two sources and create a folder of symlinks.

Sources:
  top-runs-may-ok  – ranked by distances.txt (lowest objective_value = best)
  optuna-ec2       – best_* dirs (already filtered by the pipeline)

Omit --top-runs or --optuna to select from one source only.
Constraint: no repeated iterXYZ_runNMB across the combined selection.

Each symlink points to the trial-level directory (parent of 'outputs'),
so the result is directly usable with --synth-root in compare_stats_synth_vs_real.py.

Usage (both sources):
    python od_scripts/select_best_runs.py \\
        --top-runs  /path/to/top-runs-may-ok \\
        --optuna    /path/to/optuna-ec2 \\
        --output    /path/to/best-combined-10 \\
        --n-each    5

Usage (single source):
    python od_scripts/select_best_runs.py \\
        --top-runs  /path/to/top-runs-may-ok \\
        --n          10 \\
        --output    /path/to/best-top10

    python od_scripts/select_best_runs.py \\
        --optuna    /path/to/optuna-ec2 \\
        --n          7 \\
        --output    /path/to/best-ec2-7
"""

import argparse
from pathlib import Path


def select_from_top_runs(top_runs_dir: Path, n: int, exclude: set[str] | None = None) -> list[dict]:
    exclude = exclude or set()
    dist_file = top_runs_dir / "distances.txt"
    entries = []
    with open(dist_file) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            rank, workspace, run_id, obj_val, trial_num = (
                int(parts[0]), parts[1], parts[2], float(parts[3]), int(parts[4])
            )
            entries.append((rank, workspace, run_id, obj_val, trial_num))

    seen = set(exclude)
    selected = []
    for rank, workspace, run_id, obj_val, trial_num in sorted(entries, key=lambda x: x[0]):
        if run_id in seen:
            continue
        trial_dir = top_runs_dir / workspace / f"trial_{trial_num}"
        if not (trial_dir / "outputs").is_dir():
            continue
        seen.add(run_id)
        selected.append({
            "source": "top-runs-may-ok",
            "run_id": run_id,
            "obj_val": obj_val,
            "trial_dir": trial_dir,
        })
        if len(selected) >= n:
            break
    return selected


def select_from_optuna(optuna_dir: Path, n: int, exclude: set[str] | None = None) -> list[dict]:
    seen = set(exclude or set())
    selected = []
    for best_dir in sorted(optuna_dir.glob("**/best_*")):
        if not best_dir.is_dir():
            continue
        outputs = best_dir / "outputs"
        if not outputs.is_dir():
            continue
        run_dirs = [d for d in outputs.iterdir() if d.is_dir()]
        if not run_dirs:
            continue
        run_id = run_dirs[0].name
        if run_id in seen:
            continue
        seen.add(run_id)
        selected.append({
            "source": "optuna-ec2",
            "run_id": run_id,
            "trial_dir": best_dir,
        })
        if len(selected) >= n:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Symlink best runs from one or two sources into one folder")
    parser.add_argument("--top-runs", default=None,
                        help="top-runs-may-ok directory (omit to skip this source)")
    parser.add_argument("--optuna",   default=None,
                        help="optuna-ec2 directory (omit to skip this source)")
    parser.add_argument("--output",   default="/Users/orram/Tensorleap/data/warehouse/best-combined-10")
    parser.add_argument("--n-each",   type=int, default=None,
                        help="Runs to take from each source (used when both sources are given)")
    parser.add_argument("--n",        type=int, default=None,
                        help="Total runs to take (used when only one source is given)")
    args = parser.parse_args()

    if not args.top_runs and not args.optuna:
        parser.error("Provide at least one of --top-runs or --optuna")

    both = bool(args.top_runs) and bool(args.optuna)

    if both:
        n_each = args.n_each or 5
    else:
        n_each = args.n or args.n_each or 5

    output_dir = Path(args.output)
    all_runs   = []

    if args.top_runs:
        may_ok = select_from_top_runs(Path(args.top_runs), n_each)
        print(f"Selected {len(may_ok)} from top-runs-may-ok:")
        for r in may_ok:
            print(f"  {r['run_id']:25s}  obj={r['obj_val']:.6f}  {r['trial_dir']}")
        all_runs.extend(may_ok)

    if args.optuna:
        exclude = {r["run_id"] for r in all_runs}
        ec2 = select_from_optuna(Path(args.optuna), n_each, exclude=exclude)
        print(f"\nSelected {len(ec2)} from optuna-ec2:")
        for r in ec2:
            print(f"  {r['run_id']:25s}  {r['trial_dir']}")
        all_runs.extend(ec2)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating symlinks in {output_dir} ...")
    for r in all_runs:
        prefix  = "may" if r["source"] == "top-runs-may-ok" else "ec2"
        link    = output_dir / f"{prefix}_{r['run_id']}"
        target  = r["trial_dir"].resolve()
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        print(f"  {link.name} -> {target}")

    print(f"\nDone. Use with:")
    print(f"  python od_scripts/compare_stats_synth_vs_real.py --synth-root {output_dir} ...")


if __name__ == "__main__":
    main()

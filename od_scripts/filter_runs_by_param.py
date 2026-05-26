"""
Filter runs in a top-runs directory by a YAML parameter value.

Dot-notation key traverses nested YAML (e.g. "environment.name", "camera.fov_mean").

Matching rules:
  - String value  → exact equality
  - Float value   → |yaml_val - target| <= tol  (default tol=0, i.e. exact)
                    Pass --tol for absolute tolerance, or --tol-pct for % tolerance.
  - Range value   → pass as "min:max" (e.g. "60:80"); matched if min <= yaml_val <= max

Output: tab-separated table of matching runs.
Optionally symlink results into --output dir (same layout as select_best_runs.py).

Usage:
    python od_scripts/filter_runs_by_param.py \\
        --root /path/to/top-runs-may-ok \\
        --key  environment.name \\
        --val  full_warehouse

    python od_scripts/filter_runs_by_param.py \\
        --root /path/to/top-runs-may-ok \\
        --key  camera.fov_mean \\
        --val  70 --tol 5

    python od_scripts/filter_runs_by_param.py \\
        --root /path/to/top-runs-may-ok \\
        --key  camera.fov_mean \\
        --val  60:80 \\
        --output /path/to/output-dir
"""

import argparse
import os
from pathlib import Path

import yaml


def get_nested(d: dict, dotkey: str):
    """Traverse nested dict with dot-separated key. Returns None if missing."""
    keys = dotkey.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def parse_value(val_str: str):
    """Return (kind, value) where kind is 'str', 'float', or 'range'."""
    if ":" in val_str:
        lo, hi = val_str.split(":", 1)
        return "range", (float(lo), float(hi))
    try:
        return "float", float(val_str)
    except ValueError:
        return "str", val_str


def matches(yaml_val, kind: str, target, tol: float, tol_pct: float) -> bool:
    if yaml_val is None:
        return False
    if kind == "str":
        return str(yaml_val) == target
    if kind == "range":
        lo, hi = target
        try:
            v = float(yaml_val)
        except (TypeError, ValueError):
            return False
        return lo <= v <= hi
    # float
    try:
        v = float(yaml_val)
    except (TypeError, ValueError):
        return False
    effective_tol = tol
    if tol_pct > 0:
        effective_tol = max(effective_tol, abs(target) * tol_pct / 100.0)
    return abs(v - target) <= effective_tol


def find_yamls(root: Path) -> list[tuple[Path, Path]]:
    """Return list of (trial_dir, yaml_path) for all trials."""
    results = []
    for ws_dir in sorted(root.iterdir()):
        if not ws_dir.is_dir():
            continue
        for trial_dir in sorted(ws_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            yamls_dir = trial_dir / "yamls"
            if not yamls_dir.is_dir():
                continue
            for yf in sorted(yamls_dir.glob("*.yaml")):
                results.append((trial_dir, yf))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    default="/Users/orram/Tensorleap/data/warehouse/top-runs-may-ok")
    parser.add_argument("--key",     required=True, help="Dot-notation YAML key, e.g. environment.name")
    parser.add_argument("--val",     required=True, help="Target value. Float range: 'min:max'")
    parser.add_argument("--tol",     type=float, default=0.0, help="Absolute tolerance for float match")
    parser.add_argument("--tol-pct", type=float, default=0.0, dest="tol_pct",
                        help="Percentage tolerance for float match (e.g. 10 = ±10%%)")
    parser.add_argument("--output",  default=None,
                        help="If set, symlink matching trial dirs here (usable as --synth-root)")
    args = parser.parse_args()

    root = Path(args.root)
    kind, target = parse_value(args.val)

    print(f"Scanning {root}")
    print(f"Filter: {args.key} = {args.val}  (kind={kind})", end="")
    if kind == "float":
        print(f"  tol={args.tol}" + (f"  tol_pct={args.tol_pct}%" if args.tol_pct else ""), end="")
    print()

    hits = []
    for trial_dir, yaml_path in find_yamls(root):
        cfg = yaml.safe_load(yaml_path.read_text())
        yaml_val = get_nested(cfg, args.key)
        if matches(yaml_val, kind, target, args.tol, args.tol_pct):
            run_id = yaml_path.stem
            ws = trial_dir.parent.name
            hits.append({
                "workspace": ws,
                "trial": trial_dir.name,
                "run_id": run_id,
                "yaml_val": yaml_val,
                "trial_dir": trial_dir,
                "yaml_path": yaml_path,
            })

    print(f"\nFound {len(hits)} matching runs:\n")
    print(f"{'workspace':<45}  {'trial':<10}  {'run_id':<22}  {args.key}")
    print("-" * 100)
    for h in hits:
        print(f"{h['workspace']:<45}  {h['trial']:<10}  {h['run_id']:<22}  {h['yaml_val']}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\nSymlinking into {out} ...")
        for h in hits:
            link = out / f"{h['workspace']}__{h['run_id']}"
            target_path = h["trial_dir"].resolve()
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target_path)
            print(f"  {link.name}")
        print(f"\nUse with:")
        print(f"  python od_scripts/compare_stats_synth_vs_real.py --synth-root {out} ...")


if __name__ == "__main__":
    main()

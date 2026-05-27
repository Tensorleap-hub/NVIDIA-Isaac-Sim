"""Apply parameter bounds to all YAMLs in a directory.

Reads a bounds YAML produced by plot_top_k_bounds.py --save-bounds, then for
every *.yaml in input_dir writes a modified copy to output_dir:

  - Numeric params:     clamp value to [min, max]
  - Categorical params: replace value with the first keep entry if not already
                        in the keep list

output_dir must differ from input_dir. Existing files in output_dir are
overwritten. Params absent from a YAML are silently skipped.

Usage:
    python apply_bounds.py <input_dir> <output_dir> <bounds_yaml> [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# ── nested YAML path helpers ──────────────────────────────────────────────────

def _get_nested(cfg: dict, dotpath: str):
    """Navigate a dotted path like 'camera.dataset_noise.jpeg_quality_mean'."""
    node = cfg
    for key in dotpath.split("."):
        if not isinstance(node, dict) or key not in node:
            return None, False
        node = node[key]
    return node, True


def _set_nested(cfg: dict, dotpath: str, value) -> bool:
    """Set a value at a dotted path. Returns False if the path doesn't exist."""
    keys = dotpath.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    leaf = keys[-1]
    if not isinstance(node, dict) or leaf not in node:
        return False
    node[leaf] = value
    return True


# ── bound application ─────────────────────────────────────────────────────────

def _values_match(current, keep_list: list) -> bool:
    """Check if current value is in the keep list (handles list-valued params)."""
    for kept in keep_list:
        if current == kept:
            return True
        # compare string representations for list-valued params like visibility_choices
        if str(current) == str(kept):
            return True
    return False


def apply_bounds_to_cfg(cfg: dict, bounds: dict) -> list[str]:
    """Mutate cfg in-place. Returns list of human-readable change descriptions."""
    changes = []
    for param, spec in bounds.items():
        current, found = _get_nested(cfg, param)
        if not found:
            continue

        kind = spec["type"]

        if kind == "numeric":
            lo, hi = spec["min"], spec["max"]
            try:
                val = float(current)
            except (TypeError, ValueError):
                continue
            clamped = max(lo, min(hi, val))
            if abs(clamped - val) > 1e-9:
                _set_nested(cfg, param, round(clamped, 8))
                direction = "↑" if clamped > val else "↓"
                changes.append(f"  {param}: {val:.4g} → {clamped:.4g}  {direction} (clamp [{lo:.4g}, {hi:.4g}])")

        elif kind == "categorical":
            keep = spec["keep"]
            if not _values_match(current, keep):
                replacement = keep[0]
                _set_nested(cfg, param, replacement)
                changes.append(f"  {param}: {current!r} → {replacement!r}  (not in keep list)")

    return changes


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply parameter bounds to a directory of YAMLs")
    parser.add_argument("input_dir",  help="Directory containing source *.yaml files")
    parser.add_argument("output_dir", help="Directory to write modified YAMLs (must differ from input_dir)")
    parser.add_argument("bounds_yaml", help="Bounds YAML produced by plot_top_k_bounds.py --save-bounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing any files")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    bounds_path = Path(args.bounds_yaml).expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"input_dir does not exist: {input_dir}")
    if input_dir == output_dir:
        sys.exit("output_dir must differ from input_dir")
    if not bounds_path.exists():
        sys.exit(f"bounds_yaml not found: {bounds_path}")

    doc = yaml.safe_load(bounds_path.read_text())
    bounds = doc.get("params", {})
    if not bounds:
        sys.exit("No params found in bounds YAML")

    meta = doc.get("meta", {})
    print(f"Bounds: {len(bounds)} params  (top {meta.get('top_pct','?')}%,  "
          f"p ≤ {meta.get('pvalue_threshold','?')},  "
          f"MMD threshold {meta.get('mmd_threshold','?')})")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("(dry-run — no files written)\n")

    yaml_files = sorted(input_dir.glob("*.yaml"))
    if not yaml_files:
        sys.exit(f"No *.yaml files found in {input_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_changed = 0
    total_files_modified = 0

    for yaml_path in yaml_files:
        cfg = yaml.safe_load(yaml_path.read_text())
        if not isinstance(cfg, dict):
            print(f"[skip] {yaml_path.name} — not a YAML dict")
            continue

        changes = apply_bounds_to_cfg(cfg, bounds)

        if changes:
            total_changed += len(changes)
            total_files_modified += 1
            print(f"\n{yaml_path.name}  ({len(changes)} change{'s' if len(changes) != 1 else ''}):")
            for c in changes:
                print(c)
        else:
            print(f"{yaml_path.name}  — no changes needed")

        if not args.dry_run:
            out_path = output_dir / yaml_path.name
            out_path.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False))

    print(f"\n{'─'*60}")
    print(f"Files processed: {len(yaml_files)}")
    print(f"Files modified:  {total_files_modified}")
    print(f"Total changes:   {total_changed}")
    if not args.dry_run:
        print(f"Written to:      {output_dir}")


if __name__ == "__main__":
    main()

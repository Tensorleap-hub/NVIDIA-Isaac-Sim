"""Aggregate param_importances from all state.json files in a directory.

Usage:
    python analyze_param_importances.py <state_dir> [--top N] [--out <csv_path>]

Reads every *.json file in <state_dir> that contains a param_importances key,
writes a CSV with one column per theme and one row per parameter, then prints
the top-N parameters ranked by mean importance across all themes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def load_importances(state_dir: Path) -> dict[str, dict[str, float]]:
    """Return {theme_name: {param: importance}} for every state file found."""
    results: dict[str, dict[str, float]] = {}
    for path in sorted(state_dir.glob("*.json")):
        data = json.loads(path.read_text())
        pi = data.get("param_importances")
        if not pi:
            print(f"[skip] {path.name} — no param_importances", file=sys.stderr)
            continue
        theme = re.sub(r"_state\.json$", "", path.stem)
        results[theme] = pi
    return results


def build_table(
    importances: dict[str, dict[str, float]],
) -> tuple[list[str], list[str], list[list]]:
    """Return (themes, params, rows) where rows[i] is one row per param."""
    themes = list(importances)
    all_params: set[str] = set()
    for pi in importances.values():
        all_params.update(pi)

    # Sort params by mean importance descending, skip shape_logit internals.
    # Mean is computed over themes where the param actually exists (non-empty),
    # not over all themes — params absent from a theme should not dilute the mean.
    def mean_importance(param: str) -> float:
        vals = [importances[t][param] for t in themes if param in importances[t]]
        return sum(vals) / len(vals) if vals else 0.0

    params = sorted(
        [p for p in all_params if not p.startswith("shape_logit")],
        key=mean_importance,
        reverse=True,
    )

    rows = []
    for param in params:
        row = [param] + [importances[t].get(param, "") for t in themes]
        mean = mean_importance(param)
        row.append(mean)
        rows.append(row)

    return themes, params, rows


def save_csv(path: Path, themes: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter"] + themes + ["mean"])
        writer.writerows(rows)
    print(f"Saved: {path}")


def print_top(themes: list[str], rows: list[list], top_n: int) -> None:
    col_w = max(len(p) for p, *_ in rows[:top_n]) + 2
    theme_w = 10
    header = f"{'parameter':<{col_w}}" + "".join(f"{t:>{theme_w}}" for t in themes) + f"{'mean':>{theme_w}}"
    print(f"\nTop {top_n} parameters by mean importance across {len(themes)} theme(s):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows[:top_n]:
        param = row[0]
        values = row[1:-1]
        mean = row[-1]
        line = f"{param:<{col_w}}"
        for v in values:
            line += f"{v:>{theme_w}.4f}" if isinstance(v, float) else f"{'—':>{theme_w}}"
        line += f"{mean:>{theme_w}.4f}"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate param importances from theme state files")
    parser.add_argument("state_dir", help="Directory containing *_state.json files")
    parser.add_argument("--top", type=int, default=10, metavar="N", help="Number of top params to print (default: 10)")
    parser.add_argument("--out", default=None, metavar="CSV", help="Output CSV path (default: <state_dir>/param_importances.csv)")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).expanduser().resolve()
    if not state_dir.is_dir():
        sys.exit(f"Not a directory: {state_dir}")

    importances = load_importances(state_dir)
    if not importances:
        sys.exit("No state files with param_importances found.")

    themes, params, rows = build_table(importances)

    out_path = Path(args.out) if args.out else state_dir / "param_importances.csv"
    save_csv(out_path, themes, rows)
    print_top(themes, rows, min(args.top, len(params)))


if __name__ == "__main__":
    main()

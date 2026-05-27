"""
Add a new theme to SEARCH_SPACE_THEMES in config.py from a param-importance CSV.

The CSV must have a `parameter` column and a `mean` column (average importance
across runs).  Indexed params like `palletjacks.position_std[0]` are collapsed
to their base path (`palletjacks.position_std`) before counting, so --n refers
to unique base paths.

Usage:
    python add_theme_from_importance.py \\
        --csv more_points_param_importance.csv \\
        --n 10 \\
        [--theme-name my_theme] \\
        [--config simulation_calibration_loop/config.py] \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_INDEX_RE = re.compile(r"\[\d+\]$")
_THEMES_START_RE = re.compile(r"^SEARCH_SPACE_THEMES\s*[:=]")


def _strip_index(path: str) -> str:
    return _INDEX_RE.sub("", path)


def top_params(csv_path: Path, n: int) -> list[str]:
    df = pd.read_csv(csv_path)
    if "parameter" not in df.columns or "mean" not in df.columns:
        sys.exit(f"CSV must have 'parameter' and 'mean' columns — found: {list(df.columns)}")

    df = df.dropna(subset=["mean"]).sort_values("mean", ascending=False)

    seen: set[str] = set()
    params: list[str] = []
    for raw in df["parameter"]:
        base = _strip_index(str(raw))
        if base not in seen:
            seen.add(base)
            params.append(base)
        if len(params) == n:
            break

    if len(params) < n:
        print(f"Warning: only {len(params)} unique params available (requested {n})")
    return params


def _build_theme_block(name: str, params: list[str]) -> str:
    lines = [f'    "{name}": [']
    for p in params:
        lines.append(f'        "{p}",')
    lines.append("    ],")
    return "\n".join(lines)


def insert_theme(config_path: Path, name: str, params: list[str], overwrite: bool) -> None:
    text = config_path.read_text()

    # Locate the SEARCH_SPACE_THEMES dict boundaries
    themes_start = None
    for i, line in enumerate(text.splitlines()):
        if _THEMES_START_RE.match(line):
            themes_start = i
            break
    if themes_start is None:
        sys.exit("Could not find SEARCH_SPACE_THEMES in config file.")

    # Check for existing theme
    existing_re = re.compile(rf'^\s*"{re.escape(name)}"\s*:\s*\[', re.MULTILINE)
    if existing_re.search(text):
        if not overwrite:
            sys.exit(
                f"Theme '{name}' already exists. Use --overwrite to replace it."
            )
        # Remove the existing theme block: from its key line to the closing ],
        text = re.sub(
            rf'    "{re.escape(name)}": \[.*?\n    \],\n',
            "",
            text,
            flags=re.DOTALL,
        )
        print(f"Replaced existing theme '{name}'.")

    # Find the closing } of SEARCH_SPACE_THEMES (first } at column 0 after start)
    lines = text.splitlines(keepends=True)
    insert_before = None
    inside = False
    for i, line in enumerate(lines):
        if i < themes_start:
            continue
        if _THEMES_START_RE.match(line):
            inside = True
            continue
        if inside and line.rstrip() == "}":
            insert_before = i
            break

    if insert_before is None:
        sys.exit("Could not find closing '}' of SEARCH_SPACE_THEMES.")

    block = _build_theme_block(name, params) + "\n"
    lines.insert(insert_before, block)
    config_path.write_text("".join(lines))
    print(f"Added theme '{name}' with {len(params)} params to {config_path}.")
    for p in params:
        print(f"  {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a theme to SEARCH_SPACE_THEMES from a param-importance CSV.")
    parser.add_argument("--csv", required=True, help="Path to param-importance CSV")
    parser.add_argument("--n", required=True, type=int, help="Number of top params to include")
    parser.add_argument("--theme-name", default=None, help="Theme name (default: CSV stem)")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.py"),
        help="Path to config.py (default: same dir as this script)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace theme if it already exists")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")

    config_path = Path(args.config)
    if not config_path.is_file():
        sys.exit(f"Config not found: {config_path}")

    theme_name = args.theme_name or csv_path.stem
    params = top_params(csv_path, args.n)
    insert_theme(config_path, theme_name, params, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

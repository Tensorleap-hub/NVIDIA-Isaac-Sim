"""Summarize parameter statistics across experiment YAML files.

Default behavior scans YAML files recursively under the provided root and prints
a slim, human-readable summary per parameter.

Examples:
    python summarize_yaml_stats.py
    python summarize_yaml_stats.py --root palletjack_sdg/experiments/experiment_second_order
    python summarize_yaml_stats.py --format json
    python summarize_yaml_stats.py --format csv --output stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml


NUMERIC_EPS = 1e-12


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def iter_yaml_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def flatten_config(
    value: Any,
    prefix: str,
    scalars: dict[str, Any],
    list_lengths: dict[str, int],
    list_items: dict[str, list[Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_config(child, child_prefix, scalars, list_lengths, list_items)
        return

    if isinstance(value, list):
        list_lengths[f"{prefix}.__len__"] = len(value)
        if all(not isinstance(item, (dict, list)) for item in value):
            list_items[f"{prefix}.__items__"] = list(value)
            numeric_items = [item for item in value if is_number(item)]
            if len(numeric_items) == len(value):
                for idx, item in enumerate(value):
                    scalars[f"{prefix}[{idx}]"] = item
        else:
            for idx, child in enumerate(value):
                flatten_config(child, f"{prefix}[{idx}]", scalars, list_lengths, list_items)
        return

    scalars[prefix] = value


def numeric_summary(values: list[float]) -> dict[str, Any]:
    result = {
        "type": "numeric",
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }
    if len(values) > 1:
        result["iqr"] = percentile(values, 0.75) - percentile(values, 0.25)
    else:
        result["iqr"] = 0.0
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile() requires at least one value")
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def categorical_summary(values: list[Any], top_k: int) -> dict[str, Any]:
    rendered = [stable_json(value) for value in values]
    counts = Counter(rendered)
    return {
        "type": "categorical",
        "count": len(values),
        "unique": len(counts),
        "top": counts.most_common(top_k),
    }


def list_items_summary(lists: list[list[Any]], top_k: int) -> dict[str, Any]:
    lengths = [len(items) for items in lists]
    flattened = [stable_json(item) for items in lists for item in items]
    item_counts = Counter(flattened)
    result = {
        "type": "list",
        "count": len(lists),
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
        "len_mean": mean(lengths) if lengths else 0.0,
        "unique_items": len(item_counts),
        "top_items": item_counts.most_common(top_k),
    }
    return result


def build_stats(yaml_paths: list[Path], top_k: int) -> tuple[list[dict[str, Any]], int]:
    scalar_values: dict[str, list[Any]] = defaultdict(list)
    scalar_presence: Counter[str] = Counter()
    list_lengths: dict[str, list[int]] = defaultdict(list)
    list_presence: Counter[str] = Counter()
    list_item_values: dict[str, list[list[Any]]] = defaultdict(list)

    parsed_files = 0
    for path in yaml_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            continue

        parsed_files += 1
        scalars: dict[str, Any] = {}
        lengths: dict[str, int] = {}
        items: dict[str, list[Any]] = {}
        flatten_config(data, "", scalars, lengths, items)

        for key, value in scalars.items():
            scalar_presence[key] += 1
            if value is not None:
                scalar_values[key].append(value)

        for key, value in lengths.items():
            list_presence[key] += 1
            list_lengths[key].append(value)

        for key, value in items.items():
            list_item_values[key].append(value)

    rows: list[dict[str, Any]] = []
    seen_keys = sorted(set(scalar_presence) | set(list_presence))
    for key in seen_keys:
        if key.endswith(".__items__"):
            continue

        if key.endswith(".__len__"):
            base_key = key[: -len(".__len__")]
            values = [float(length) for length in list_lengths[key]]
            row = {
                "parameter": f"{base_key}.__len__",
                "present": list_presence[key],
                "missing": parsed_files - list_presence[key],
            }
            row.update(numeric_summary(values))
            rows.append(row)

            item_key = f"{base_key}.__items__"
            if item_key in list_item_values:
                item_row = {
                    "parameter": base_key,
                    "present": len(list_item_values[item_key]),
                    "missing": parsed_files - len(list_item_values[item_key]),
                }
                item_row.update(list_items_summary(list_item_values[item_key], top_k))
                rows.append(item_row)
            continue

        values = scalar_values.get(key, [])
        if not values:
            continue

        row = {
            "parameter": key,
            "present": scalar_presence[key],
            "missing": parsed_files - scalar_presence[key],
        }

        if all(is_number(value) for value in values):
            row.update(numeric_summary([float(value) for value in values]))
        else:
            row.update(categorical_summary(values, top_k))
        rows.append(row)

    return rows, parsed_files


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value) and abs(value) < NUMERIC_EPS:
            value = 0.0
        return f"{value:.6g}"
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    header = [
        "parameter",
        "type",
        "present",
        "missing",
        "min",
        "max",
        "mean",
        "std",
        "median",
        "iqr",
        "unique",
        "top",
        "top_items",
    ]

    lines = ["\t".join(header)]
    for row in rows:
        line = []
        for key in header:
            value = row.get(key, "")
            if isinstance(value, list):
                value = json.dumps(value, ensure_ascii=True)
            line.append(format_number(value))
        lines.append("\t".join(line))
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fieldnames = [
        "parameter",
        "type",
        "present",
        "missing",
        "count",
        "min",
        "max",
        "mean",
        "std",
        "median",
        "iqr",
        "unique",
        "len_min",
        "len_max",
        "len_mean",
        "unique_items",
        "top",
        "top_items",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = row.copy()
            for key in ("top", "top_items"):
                if key in serializable:
                    serializable[key] = json.dumps(serializable[key], ensure_ascii=True)
            writer.writerow(serializable)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory to scan recursively for YAML files.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. Defaults to stdout for table/json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of top categorical/list values to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    yaml_paths = iter_yaml_files(root)
    if not yaml_paths:
        raise SystemExit(f"No YAML files found under {root}")

    rows, parsed_files = build_stats(yaml_paths, args.top_k)
    rows.sort(key=lambda row: row["parameter"])

    metadata = {
        "root": str(root),
        "yaml_files": len(yaml_paths),
        "parsed_files": parsed_files,
        "parameters": len(rows),
    }

    if args.format == "json":
        payload = {"summary": metadata, "stats": rows}
        text = json.dumps(payload, indent=2, ensure_ascii=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return

    if args.format == "csv":
        if args.output is None:
            raise SystemExit("--output is required when --format csv")
        write_csv(rows, args.output)
        print(
            f"Wrote {len(rows)} parameter rows from {parsed_files} YAML files to {args.output}"
        )
        return

    text = "\n".join(
        [
            f"root:\t{metadata['root']}",
            f"yaml_files:\t{metadata['yaml_files']}",
            f"parsed_files:\t{metadata['parsed_files']}",
            f"parameters:\t{metadata['parameters']}",
            "",
            render_table(rows),
        ]
    )
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

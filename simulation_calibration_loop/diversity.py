"""Shared helpers for diverse top-k trial selection.

Used by the controller's automatic best_top{k}_diverse exports and by the
standalone export_diverse_thresholded.py re-export script, so both produce
identical selections for the same pool.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

# Keys that vary per run without describing the rendered scene.
NOISE_KEY_PREFIXES = ("run.",)
NOISE_KEY_SUFFIXES = (".seed", "seed")


def flatten_yaml(node, prefix: str = "") -> dict:
    """Flatten a materialized config into dot-keyed scalars.

    Numeric lists flatten per index so Gower normalization applies element-wise
    (bounds, std vectors, resolutions); any other list is compared as one
    categorical value. Run-noise keys (per-run paths, seeds) are dropped.
    """
    flat: dict = {}
    if isinstance(node, dict):
        for key, value in node.items():
            flat.update(flatten_yaml(value, f"{prefix}{key}."))
        return flat
    key = prefix[:-1]
    if key.startswith(NOISE_KEY_PREFIXES) or key.endswith(NOISE_KEY_SUFFIXES):
        return flat
    if isinstance(node, list):
        if node and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in node
        ):
            for index, value in enumerate(node):
                flat[f"{key}[{index}]"] = value
        else:
            flat[key] = json.dumps(node, sort_keys=True, default=str)
    else:
        flat[key] = node
    return flat


def full_config_flat(yaml_path: str | Path, fallback_params: dict, log=print) -> dict:
    """Flattened full materialized config; falls back to the searched params."""
    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        return flatten_yaml(yaml.safe_load(yaml_path.read_text()))
    log(f"WARNING: {yaml_path} missing — falling back to searched params")
    return dict(fallback_params)


def build_gower_distance(params_by_id: dict[str, dict], log=print):
    """Gower-style distance between two ids' flattened params (0..1 mean).

    Numeric values normalize by the range observed across the pool, everything
    else contributes 0/1 mismatch. Keys constant across the pool are dropped —
    they add nothing but dilute the mean. Comparing full configs (not just each
    study's searched params) keeps cross-study comparisons fair: every key
    exists on both sides, so absent-vs-present never counts as a mismatch.
    """
    all_keys = sorted({key for flat in params_by_id.values() for key in flat})
    keys = [
        key for key in all_keys
        if len({json.dumps(flat.get(key), default=str) for flat in params_by_id.values()}) > 1
    ]
    log(f"param distance over {len(keys)} varying keys (of {len(all_keys)} in the full configs)")
    ranges: dict[str, float] = {}
    for key in keys:
        values = [
            flat.get(key)
            for flat in params_by_id.values()
            if isinstance(flat.get(key), (int, float))
            and not isinstance(flat.get(key), bool)
        ]
        if values:
            ranges[key] = float(max(values) - min(values))

    def distance(left_id: str, right_id: str) -> float:
        left_flat = params_by_id[left_id]
        right_flat = params_by_id[right_id]
        total = 0.0
        for key in keys:
            left = left_flat.get(key)
            right = right_flat.get(key)
            if key in ranges and isinstance(left, (int, float)) and isinstance(right, (int, float)) \
                    and not isinstance(left, bool) and not isinstance(right, bool):
                span = ranges[key]
                total += abs(float(left) - float(right)) / span if span > 0 else 0.0
            else:
                total += 0.0 if left == right else 1.0
        return total / len(keys) if keys else 0.0

    return distance

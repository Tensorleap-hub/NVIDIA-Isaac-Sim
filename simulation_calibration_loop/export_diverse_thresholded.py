#!/usr/bin/env python3
"""Export objective-thresholded diverse top-k trial lists from a loop workspace.

The controller's built-in best_top{k}_diverse[_latent].yaml exports draw
candidates from the top `diverse_candidate_pool` (30) trials by objective, so
the greedy max-min diversity selection can pick configs whose objective MMD is
well above the plain top-k — diversity at the cost of quality.

This script rebuilds both diversity lists with a hard quality gate instead:
only trials whose objective value (MMD vs real) is <= --threshold are
candidates. Within the gated pool the selection is identical to the
controller's: greedy max-min (farthest-point), seeded with the best trial,
under (a) normalized Gower-style parameter distance and (b) pairwise RBF-MMD
between the runs' cached embedding sets with a shared median-heuristic gamma.

Reads the workspace's state.json ledger; safe to run while the loop is live
(read-only on the workspace, writes only the two new YAMLs to --out-dir).

Usage:
  python export_diverse_thresholded.py \
      --state rounds_ws_20260712_top_important/workspace_trajectory_top_important_r01/state.json \
      --out-dir promoted_baseline_trajectory_top_important \
      --threshold 0.44 --k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from calibration_optuna.metrics import DistributionMetrics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversity  # noqa: E402


def collect_unique_ranked(states: list[tuple[str, dict]]) -> list[dict]:
    """Best-scoring run per fingerprint across all states, ranked by objective.

    run_id and optuna trial numbers restart per workspace, so when merging
    multiple ledgers each artifact is tagged with its source workspace and its
    run_id is prefixed to stay unique (the latent embedding cache is keyed on
    run_id).
    """
    tag_prefix = len(states) > 1
    best_by_fingerprint: dict[str, dict] = {}
    for tag, state in states:
        for iteration in state["iterations"]:
            for item in iteration["artifacts"]:
                if item.get("objective_value") is None:
                    continue
                item = dict(item)
                item["source"] = tag
                if tag_prefix:
                    item["run_id"] = f"{tag}/{item['run_id']}"
                fingerprint = item.get("run_fingerprint", "legacy")
                if tag_prefix and fingerprint == "legacy":
                    fingerprint = item["run_id"]
                existing = best_by_fingerprint.get(fingerprint)
                if existing is None or item["objective_value"] < existing["objective_value"]:
                    best_by_fingerprint[fingerprint] = item
    return sorted(best_by_fingerprint.values(), key=lambda item: item["objective_value"])


def param_distance_fn(pool: list[dict]):
    """Gower-style distance over the full flattened configs (0..1 mean)."""
    params = {
        item["run_id"]: diversity.full_config_flat(item["yaml_path"], item["flattened_params"])
        for item in pool
    }
    id_distance = diversity.build_gower_distance(params)

    def distance(a: dict, b: dict) -> float:
        return id_distance(a["run_id"], b["run_id"])

    return distance


def latent_distance_fn(pool: list[dict], seed: int):
    """Pairwise RBF-MMD over cached run embeddings, shared median-heuristic gamma."""
    embeddings = {item["run_id"]: np.load(item["embedding_path"]) for item in pool}
    gamma = None
    if len(pool) >= 2:
        rng = np.random.default_rng(seed)
        stacked = np.vstack(list(embeddings.values()))
        if stacked.shape[0] > 2000:
            stacked = stacked[rng.choice(stacked.shape[0], 2000, replace=False)]
        half = stacked.shape[0] // 2
        gamma = DistributionMetrics._compute_gamma_median_heuristic(
            stacked[:half], stacked[half:]
        )
    cache: dict[tuple[str, str], float] = {}

    def distance(a: dict, b: dict) -> float:
        key = (a["run_id"], b["run_id"]) if a["run_id"] <= b["run_id"] else (b["run_id"], a["run_id"])
        if key not in cache:
            cache[key] = DistributionMetrics.mmd(
                embeddings[a["run_id"]], embeddings[b["run_id"]], kernel="rbf", gamma=gamma
            )
        return cache[key]

    return distance


def select_diverse(pool: list[dict], k: int, distance) -> tuple[list[dict], list[float | None]]:
    """Greedy max-min (farthest-point) selection, seeded with the best trial."""
    if not pool:
        return [], []
    selected = [pool[0]]
    min_distances: list[float | None] = [None]
    remaining = list(pool[1:])
    while remaining and len(selected) < k:
        scored = [
            (min(distance(candidate, chosen) for chosen in selected), candidate)
            for candidate in remaining
        ]
        best_score = max(score for score, _ in scored)
        picked = next(candidate for score, candidate in scored if score == best_score)
        selected.append(picked)
        min_distances.append(best_score)
        remaining.remove(picked)
    return selected, min_distances


def trial_label(item: dict) -> str:
    base = (
        f"trial_{item['optuna_trial_number']}"
        if item.get("optuna_trial_number") is not None
        else item["run_id"]
    )
    source = item.get("source")
    return f"{source}:{base}" if source and "/" in item["run_id"] else base


def trial_entry(rank: int, item: dict) -> dict:
    entry = {
        "rank": rank,
        "trial_id": trial_label(item),
        "source_workspace": item.get("source"),
        "run_id": item["run_id"],
        "objective_value": item["objective_value"],
        "yaml_path": item["yaml_path"],
        "embedding_path": item["embedding_path"],
        "params": item["flattened_params"],
    }
    yaml_path = Path(item["yaml_path"])
    if yaml_path.exists():
        entry["config"] = yaml.safe_load(yaml_path.read_text())
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True, nargs="+",
                        help="One or more workspace state.json paths (merged when several)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the exported YAMLs")
    parser.add_argument("--threshold", type=float, default=0.44,
                        help="Max objective MMD for a trial to be a diversity candidate (default 0.44)")
    parser.add_argument("--k", type=int, default=10, help="List size (default 10)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the gamma subsample RNG")
    args = parser.parse_args()

    states = [
        (path.resolve().parent.name.removeprefix("workspace_"), json.loads(path.read_text()))
        for path in args.state
    ]
    first_state = states[0][1]
    project_name = (
        first_state.get("project_name")
        or first_state.get("config", {}).get("project_name", "unknown")
    )
    ranked = collect_unique_ranked(states)
    pool = [item for item in ranked if item["objective_value"] <= args.threshold]
    print(f"{len(ranked)} unique completed trials; {len(pool)} pass objective <= {args.threshold}")
    if not pool:
        sys.exit("no trials under the threshold — nothing to export")
    if len(pool) < args.k:
        print(f"WARNING: only {len(pool)} candidates for k={args.k}; lists will be short")

    diverse, diverse_dists = select_diverse(pool, args.k, param_distance_fn(pool))

    latent_pool = [item for item in pool if Path(item["embedding_path"]).exists()]
    if len(latent_pool) < len(pool):
        print(f"latent list: {len(pool) - len(latent_pool)} candidates dropped (embedding cache missing)")
    latent, latent_dists = select_diverse(latent_pool, args.k, latent_distance_fn(latent_pool, args.seed))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"objmax{args.threshold}"
    param_path = args.out_dir / f"best_top{args.k}_diverse_{suffix}.yaml"
    latent_path = args.out_dir / f"best_top{args.k}_diverse_latent_{suffix}.yaml"

    param_path.write_text(yaml.safe_dump({
        "project_name": project_name,
        "selection": "objective+diversity",
        "diversity_metric": "normalized_param_distance_full_config",
        "objective_threshold": args.threshold,
        "candidate_pool_size": len(pool),
        "trials": [
            {**trial_entry(rank, item), "min_param_distance_to_selected": dist}
            for rank, (item, dist) in enumerate(zip(diverse, diverse_dists), start=1)
        ],
    }, sort_keys=False))
    latent_path.write_text(yaml.safe_dump({
        "project_name": project_name,
        "selection": "objective+diversity",
        "diversity_metric": "embedding_mmd_rbf",
        "objective_threshold": args.threshold,
        "candidate_pool_size": len(latent_pool),
        "trials": [
            {**trial_entry(rank, item), "min_latent_mmd_to_selected": dist}
            for rank, (item, dist) in enumerate(zip(latent, latent_dists), start=1)
        ],
    }, sort_keys=False))

    for name, selected, dists in (
        ("param-diverse", diverse, diverse_dists),
        ("latent-diverse", latent, latent_dists),
    ):
        print(f"\n{name}:")
        for rank, (item, dist) in enumerate(zip(selected, dists), start=1):
            dist_str = f"{dist:.4f}" if dist is not None else "  seed"
            print(f"  {rank:2d}. {trial_label(item):<40} obj={item['objective_value']:.4f}  min_dist={dist_str}")
    print(f"\nwrote {param_path}\nwrote {latent_path}")


if __name__ == "__main__":
    main()

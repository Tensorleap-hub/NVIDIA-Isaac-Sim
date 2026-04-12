"""Sequentially run themed calibration configs in repeated rounds.

Each config should point at the same `promoted_baseline_dir`. The calibration
workflow updates that shared directory with the best completed YAML after each
run, so later themed runs automatically inherit the latest promoted baseline.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run themed calibration configs in rounds")
    parser.add_argument(
        "--round-config",
        type=str,
        help="YAML file with `rounds` and ordered `configs` entries.",
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        default=[],
        help="Workflow config path. Pass multiple times in the order they should run.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many times to cycle through the provided configs.",
    )
    args = parser.parse_args()
    if args.round_config is None and not args.configs:
        parser.error("Pass either --round-config or at least one --config")
    return args


def load_round_config(path: Path) -> tuple[int, list[Path]]:
    """Load the round-run YAML and resolve listed configs relative to it."""
    payload = yaml.safe_load(path.read_text()) or {}
    rounds = int(payload.get("rounds", 1))
    config_items = payload.get("configs", [])
    if not config_items:
        raise ValueError(f"Round config has no configs: {path}")
    config_paths = [(path.parent / item).resolve() for item in config_items]
    return rounds, config_paths


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    retry_script = repo_root / "simulation_calibration_loop" / "run_main_loop_with_retry.sh"
    if args.round_config is not None:
        rounds, config_paths = load_round_config(Path(args.round_config).resolve())
    else:
        rounds = args.rounds
        config_paths = [Path(item).resolve() for item in args.configs]

    for round_index in range(rounds):
        for config_path in config_paths:
            print(
                f"[theme-rounds] round {round_index + 1}/{rounds} "
                f"running {config_path}"
            )
            subprocess.run(
                [str(retry_script), "--config", str(config_path)],
                cwd=repo_root,
                check=True,
            )


if __name__ == "__main__":
    main()

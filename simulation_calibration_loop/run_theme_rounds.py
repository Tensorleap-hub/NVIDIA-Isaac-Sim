"""Sequentially run themed calibration configs in repeated rounds.

Each config should point at the same `promoted_baseline_dir`. The calibration
workflow updates that shared directory with the best completed YAML after each
run, so later themed runs automatically inherit the latest promoted baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run themed calibration configs in rounds")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        required=True,
        help="Workflow config path. Pass multiple times in the order they should run.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="How many times to cycle through the provided configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    retry_script = repo_root / "simulation_calibration_loop" / "run_main_loop_with_retry.sh"
    config_paths = [Path(item).resolve() for item in args.configs]

    for round_index in range(args.rounds):
        for config_index, config_path in enumerate(config_paths):
            theme_name = config_path.stem.replace("project_config_", "")
            meta_label = (
                f"theme={theme_name} "
                f"theme_step={config_index + 1}/{len(config_paths)} "
                f"theme_round={round_index + 1}/{args.rounds}"
            )
            print(
                f"[theme-rounds] {meta_label} config={config_path}"
            )
            env = os.environ.copy()
            env["SIM_CAL_LOOP_META_LABEL"] = meta_label
            subprocess.run(
                [str(retry_script), "--config", str(config_path)],
                cwd=repo_root,
                check=True,
                env=env,
            )


if __name__ == "__main__":
    main()

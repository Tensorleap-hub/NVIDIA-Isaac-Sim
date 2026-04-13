"""Sequentially run themed calibration configs in repeated rounds.

Each config should point at the same `promoted_baseline_dir`. The calibration
workflow updates that shared directory with the best completed YAML after each
run, so later themed runs automatically inherit the latest promoted baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
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
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Optional root directory for all derived round workspaces.",
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
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else None
    if workspace_root is not None:
        workspace_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir = Path(
        tempfile.mkdtemp(prefix="simulation_calibration_loop_rounds_", dir=repo_root / "simulation_calibration_loop")
    )

    for round_index in range(rounds):
        first_round_workspace = _derive_round_workspace_dir(
            config_paths[0],
            round_index=round_index,
            workspace_root=workspace_root,
        )
        for config_index, config_path in enumerate(config_paths):
            theme_name = config_path.stem.replace("project_config_", "")
            meta_label = (
                f"theme={theme_name} "
                f"theme_step={config_index + 1}/{len(config_paths)} "
                f"theme_round={round_index + 1}/{rounds}"
            )
            derived_config_path = _write_round_config(
                source_config_path=config_path,
                destination_dir=generated_config_dir,
                round_index=round_index,
                workspace_root=workspace_root,
                synthetic_rgb_base_dir=first_round_workspace / "iteration_000" / "outputs",
            )
            print(
                f"[theme-rounds] {meta_label} config={derived_config_path}"
            )
            env = os.environ.copy()
            env["SIM_CAL_LOOP_META_LABEL"] = meta_label
            subprocess.run(
                [str(retry_script), "--config", str(derived_config_path)],
                cwd=repo_root,
                check=True,
                env=env,
            )


def _write_round_config(
    source_config_path: Path,
    destination_dir: Path,
    round_index: int,
    workspace_root: Path | None,
    synthetic_rgb_base_dir: Path | None,
) -> Path:
    """Write a derived config with a round-specific workspace and project name."""
    raw = yaml.safe_load(source_config_path.read_text()) or {}
    round_suffix = f"_r{round_index + 1:02d}"
    source_config_dir = source_config_path.parent

    workspace_path = _derive_round_workspace_dir(
        source_config_path,
        round_index=round_index,
        workspace_root=workspace_root,
    )
    raw["workspace_dir"] = str(workspace_path)

    project_name = str(raw["project_name"])
    raw["project_name"] = f"{project_name}{round_suffix}"
    if synthetic_rgb_base_dir is not None:
        raw["synthetic_rgb_base_dir"] = str(synthetic_rgb_base_dir)
    _absolutize_workflow_paths(raw, source_config_dir)

    base_pool = raw.get("base_pool") or {}
    if base_pool.get("enabled") and not base_pool.get("state_path"):
        promoted_baseline_dir = raw.get("promoted_baseline_dir")
        if promoted_baseline_dir:
            base_pool["state_path"] = str(Path(promoted_baseline_dir) / "base_pool.json")
        elif workspace_root is not None:
            base_pool["state_path"] = str((workspace_root / "base_pool.json").resolve())
        raw["base_pool"] = base_pool

    destination_path = destination_dir / f"{source_config_path.stem}{round_suffix}.yaml"
    destination_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return destination_path


def _derive_round_workspace_dir(
    source_config_path: Path,
    round_index: int,
    workspace_root: Path | None,
) -> Path:
    """Compute the derived workspace directory for one config/round pair."""
    raw = yaml.safe_load(source_config_path.read_text()) or {}
    workspace_dir = Path(str(raw["workspace_dir"]))
    round_suffix = f"_r{round_index + 1:02d}"
    workspace_name = f"{workspace_dir.name}{round_suffix}"
    if workspace_root is None:
        if workspace_dir.is_absolute():
            return Path(f"{workspace_dir}{round_suffix}")
        return (source_config_path.parent / f"{workspace_dir}{round_suffix}").resolve()
    return (workspace_root / workspace_name).resolve()


def _absolutize_workflow_paths(raw: dict, source_config_dir: Path) -> None:
    """Freeze path-like config fields so derived configs resolve identically."""
    for key in (
        "promoted_baseline_dir",
        "baseline_state_path",
        "synthetic_rgb_base_dir",
        "seed_config_dir",
        "real_dataset_root",
        "real_annotations_file",
    ):
        value = raw.get(key)
        if value:
            raw[key] = str(_resolve_config_path(value, source_config_dir))

    isaac_cfg = raw.get("isaac")
    if isinstance(isaac_cfg, dict) and isaac_cfg.get("script_path"):
        isaac_cfg["script_path"] = str(_resolve_config_path(isaac_cfg["script_path"], source_config_dir))


def _resolve_config_path(value: str, source_config_dir: Path) -> Path:
    """Resolve one possibly-relative workflow path against the original config dir."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (source_config_dir / path).resolve()


if __name__ == "__main__":
    main()

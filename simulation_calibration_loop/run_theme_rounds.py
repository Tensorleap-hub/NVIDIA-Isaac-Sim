"""Sequentially run themed calibration configs in repeated rounds.

Each config should point at the same `promoted_baseline_dir`. The calibration
workflow updates that shared directory with the best completed YAML after each
run, so later themed runs automatically inherit the latest promoted baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser.add_argument(
        "--s3-dir",
        type=str,
        default=None,
        help="S3 prefix (s3://bucket/path) to upload best runs after each theme config completes.",
    )
    parser.add_argument(
        "--s3-top-n",
        type=int,
        default=3,
        help="Number of best runs to upload per theme config (default: 3).",
    )
    args = parser.parse_args()
    if args.round_config is None and not args.configs:
        parser.error("Pass either --round-config or at least one --config")
    return args


def load_round_config(path: Path) -> tuple[int, list[Path], dict]:
    """Load the round-run YAML and resolve listed configs relative to it."""
    payload = yaml.safe_load(path.read_text()) or {}
    rounds = int(payload.get("rounds", 1))
    config_items = payload.get("configs", [])
    if not config_items:
        raise ValueError(f"Round config has no configs: {path}")
    config_paths = [(path.parent / item).resolve() for item in config_items]
    common = dict(payload.get("common") or {})
    if common.get("promoted_baseline_dir"):
        common["promoted_baseline_dir"] = str(
            _resolve_config_path(common["promoted_baseline_dir"], path.parent)
        )
    return rounds, config_paths, common


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    retry_script = repo_root / "simulation_calibration_loop" / "run_main_loop_with_retry.sh"
    if args.round_config is not None:
        rounds, config_paths, common_overrides = load_round_config(Path(args.round_config).resolve())
    else:
        rounds = args.rounds
        config_paths = [Path(item).resolve() for item in args.configs]
        common_overrides = {}
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else None
    if workspace_root is not None:
        workspace_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir = Path(
        tempfile.mkdtemp(prefix="simulation_calibration_loop_rounds_", dir=repo_root / "simulation_calibration_loop")
    )

    s3_dir: str | None = args.s3_dir
    s3_top_n: int = args.s3_top_n
    if s3_dir is not None:
        _validate_s3_connection(s3_dir)

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
                common_overrides=common_overrides,
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

            if s3_dir is not None:
                workspace_dir = _derive_round_workspace_dir(config_path, round_index=round_index, workspace_root=workspace_root)
                s3_prefix = f"{s3_dir.rstrip('/')}/{theme_name}/round_{round_index + 1:02d}/"
                print(f"[theme-rounds] uploading top-{s3_top_n} runs to {s3_prefix}")
                _upload_theme_best_runs(workspace_dir, s3_prefix, top_n=s3_top_n)


def _validate_s3_connection(s3_dir: str) -> None:
    """Verify S3 connectivity at startup by writing a marker object. Crashes on failure."""
    if shutil.which("aws") is None:
        raise RuntimeError("AWS CLI not found on PATH — cannot upload to S3")
    marker = s3_dir.rstrip("/") + "/.run_theme_rounds_init"
    result = subprocess.run(
        ["aws", "s3", "cp", "-", marker],
        input=b"run_theme_rounds S3 connectivity check\n",
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"S3 connection check failed for {s3_dir!r}:\n{result.stderr.decode().strip()}"
        )
    print(f"[theme-rounds] S3 connection verified: {s3_dir}")


def _upload_theme_best_runs(workspace_dir: Path, s3_prefix: str, top_n: int) -> None:
    """Read the workspace state, stage top_n runs + distances.txt, and sync to S3."""
    state_path = workspace_dir / "state.json"
    if not state_path.exists():
        print(f"[theme-rounds] no state.json found at {workspace_dir}, skipping S3 upload")
        return

    state = json.loads(state_path.read_text())
    artifacts = [
        item
        for iteration in state.get("iterations", [])
        for item in iteration.get("artifacts", [])
        if item.get("objective_value") is not None
    ]
    if not artifacts:
        print("[theme-rounds] no scored artifacts found, skipping S3 upload")
        return

    artifacts.sort(key=lambda x: float(x["objective_value"]))

    distances_lines = ["run_id\tobjective_value\tdist_id\tyaml_path"]
    for item in artifacts:
        distances_lines.append(
            f"{item['run_id']}\t"
            f"{item['objective_value']:.6f}\t"
            f"{item.get('dist_id', '')}\t"
            f"{item.get('yaml_path', '')}"
        )
    distances_text = "\n".join(distances_lines) + "\n"

    with tempfile.TemporaryDirectory(prefix="theme_rounds_s3_") as tmp:
        stage = Path(tmp)
        (stage / "distances.txt").write_text(distances_text)

        for item in artifacts[:top_n]:
            run_id = item["run_id"]
            dist_id = item.get("dist_id")
            trial_id = dist_id if dist_id is not None else run_id
            trial_dir = stage / trial_id

            output_dir = Path(item["output_dir"])
            if output_dir.exists():
                dest = trial_dir / "outputs" / output_dir.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(output_dir, dest)

            embedding_path = Path(item.get("embedding_path", ""))
            if embedding_path.exists():
                cache_dir = trial_dir / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(embedding_path, cache_dir / embedding_path.name)
                manifest = embedding_path.with_suffix(".manifest.json")
                if manifest.exists():
                    shutil.copy2(manifest, cache_dir / manifest.name)

            yaml_path = Path(item.get("yaml_path", ""))
            if yaml_path.exists():
                yamls_dir = trial_dir / "yamls"
                yamls_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(yaml_path, yamls_dir / yaml_path.name)

        subprocess.run(
            ["aws", "s3", "sync", str(stage) + "/", s3_prefix],
            check=True,
        )
        print(f"[theme-rounds] uploaded {min(top_n, len(artifacts))} runs to {s3_prefix}")


def _apply_common_overrides(raw: dict, common: dict) -> None:
    """Merge common section from theme_rounds.yaml into a per-theme config dict."""
    for key in (
        "promoted_baseline_dir",
        "max_iterations",
        "iteration_batch_size",
        "embedder_backend",
        "top_k_export",
        "diverse_candidate_pool",
        "diverse_objective_threshold",
    ):
        if key in common:
            raw[key] = common[key]
    if "base_pool" in common:
        existing = raw.get("base_pool") or {}
        existing.update(common["base_pool"])
        raw["base_pool"] = existing
    if "rfdetr_embedder" in common:
        existing = raw.get("rfdetr_embedder") or {}
        existing.update(common["rfdetr_embedder"])
        raw["rfdetr_embedder"] = existing
    if "isaac" in common:
        # Shallow-merge isaac knobs (e.g. episode_mode/capture_mode) so a
        # rounds file can switch the render pipeline for every theme at once.
        existing = raw.get("isaac") or {}
        existing.update(common["isaac"])
        raw["isaac"] = existing
    if "sample_number" in common:
        isaac_cfg = raw.setdefault("isaac", {})
        isaac_cfg["num_frames_override"] = int(common["sample_number"])
    if "eval_seeds" in common:
        isaac_cfg = raw.setdefault("isaac", {})
        isaac_cfg["eval_seeds"] = [int(s) for s in common["eval_seeds"]]


def _write_round_config(
    source_config_path: Path,
    destination_dir: Path,
    round_index: int,
    workspace_root: Path | None,
    synthetic_rgb_base_dir: Path | None,
    common_overrides: dict | None = None,
) -> Path:
    """Write a derived config with a round-specific workspace and project name."""
    raw = yaml.safe_load(source_config_path.read_text()) or {}
    round_suffix = f"_r{round_index + 1:02d}"
    source_config_dir = source_config_path.parent

    _apply_common_overrides(raw, common_overrides or {})

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

    rfdetr_cfg = raw.get("rfdetr_embedder")
    if isinstance(rfdetr_cfg, dict) and rfdetr_cfg.get("checkpoint_path"):
        rfdetr_cfg["checkpoint_path"] = str(_resolve_config_path(rfdetr_cfg["checkpoint_path"], source_config_dir))


def _resolve_config_path(value: str, source_config_dir: Path) -> Path:
    """Resolve one possibly-relative workflow path against the original config dir."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (source_config_dir / path).resolve()


if __name__ == "__main__":
    main()

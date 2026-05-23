"""
Upload the global top-N runs across all workspace_* directories to S3.

Reads every state.json under --workspaces-root, ranks all artifacts by
objective_value (ascending = better), stages the top N preserving the
workspace folder, then syncs to S3.

Staged layout:
    <tmp>/
        distances.txt                       ← global ranking
        <workspace_name>/
            <run_id>/
                outputs/<output_dir_name>/  ← rgb images + isaac.log
                cache/<embedding>.npy       ← DINOv2 embedding + manifest
                yamls/<run_id>.yaml         ← Isaac config

Usage:
    python upload_top_runs_to_s3.py \\
        --workspaces-root /home/ubuntu/NVIDIA-Isaac-Sim/simulation_calibration_loop/may_rounds_ok \\
        --s3-prefix s3://nvidia-isaac-bucket/top-runs/ \\
        [--n 40] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _collect_artifacts(workspaces_root: Path) -> list[dict]:
    """Collect all scored artifacts from every workspace_* state.json."""
    workspace_dirs = sorted(
        p for p in workspaces_root.iterdir()
        if p.is_dir() and (p / "state.json").exists()
    )
    if not workspace_dirs:
        sys.exit(f"No workspace dirs with state.json found under {workspaces_root}")

    artifacts = []
    for ws_dir in workspace_dirs:
        state = json.loads((ws_dir / "state.json").read_text())
        for iteration in state.get("iterations", []):
            for item in iteration.get("artifacts", []):
                if item.get("objective_value") is not None:
                    item = dict(item)
                    item["_workspace_name"] = ws_dir.name
                    artifacts.append(item)
        print(f"  {ws_dir.name}: {sum(1 for a in artifacts if a['_workspace_name'] == ws_dir.name)} scored artifacts")

    return artifacts


def _stage_run(item: dict, stage: Path) -> None:
    """Copy one artifact's outputs/cache/yaml into the staging directory."""
    ws_name = item["_workspace_name"]
    run_id = item["run_id"]
    trial_num = item.get("optuna_trial_number")
    trial_id = f"trial_{trial_num}" if trial_num is not None else run_id
    run_dir = stage / ws_name / trial_id

    output_dir = Path(item["output_dir"])
    if output_dir.exists():
        dest = run_dir / "outputs" / output_dir.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_dir, dest)
    else:
        print(f"    [warn] output_dir not found: {output_dir}")

    embedding_path = Path(item.get("embedding_path", ""))
    if embedding_path.exists():
        cache_dir = run_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(embedding_path, cache_dir / embedding_path.name)
        manifest = embedding_path.with_suffix(".manifest.json")
        if manifest.exists():
            shutil.copy2(manifest, cache_dir / manifest.name)
    else:
        print(f"    [warn] embedding not found: {embedding_path}")

    yaml_path = Path(item.get("yaml_path", ""))
    if yaml_path.exists():
        yamls_dir = run_dir / "yamls"
        yamls_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_path, yamls_dir / yaml_path.name)
    else:
        print(f"    [warn] yaml not found: {yaml_path}")


def _build_distances_txt(top_artifacts: list[dict], all_artifacts: list[dict]) -> str:
    lines = [
        "rank\tworkspace\trun_id\tobjective_value\toptuna_trial_number\tyaml_path",
    ]
    for rank, item in enumerate(top_artifacts, start=1):
        lines.append(
            f"{rank}\t"
            f"{item['_workspace_name']}\t"
            f"{item['run_id']}\t"
            f"{item['objective_value']:.6f}\t"
            f"{item.get('optuna_trial_number', '')}\t"
            f"{item.get('yaml_path', '')}"
        )
    lines.append("")
    lines.append(f"# Total scored artifacts across all workspaces: {len(all_artifacts)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload global top-N runs across all workspaces to S3."
    )
    parser.add_argument(
        "--workspaces-root",
        required=True,
        help="Directory containing workspace_* subdirs with state.json files",
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 destination prefix, e.g. s3://my-bucket/top-runs/",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40,
        help="Number of top runs to upload (default: 40)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files locally and print what would be synced, but skip the aws s3 sync",
    )
    args = parser.parse_args()

    workspaces_root = Path(args.workspaces_root)
    if not workspaces_root.is_dir():
        sys.exit(f"workspaces-root not found: {workspaces_root}")

    s3_prefix = args.s3_prefix.rstrip("/") + "/"

    print(f"Collecting artifacts from {workspaces_root} ...")
    all_artifacts = _collect_artifacts(workspaces_root)
    all_artifacts.sort(key=lambda x: float(x["objective_value"]))
    print(f"\nTotal scored artifacts: {len(all_artifacts)}")

    top = all_artifacts[: args.n]
    print(f"Top {len(top)} runs by objective_value:")
    for rank, item in enumerate(top, start=1):
        print(f"  [{rank:02d}] {item['_workspace_name']}/{item['run_id']}  obj={item['objective_value']:.4f}")

    with tempfile.TemporaryDirectory(prefix="upload_top_runs_") as tmp:
        stage = Path(tmp)

        distances_txt = _build_distances_txt(top, all_artifacts)
        (stage / "distances.txt").write_text(distances_txt)

        print(f"\nStaging {len(top)} runs ...")
        for item in top:
            print(f"  staging {item['_workspace_name']}/{item['run_id']} ...")
            _stage_run(item, stage)

        if args.dry_run:
            print(f"\n[dry-run] would sync {stage}/ → {s3_prefix}")
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(stage)
                    print(f"  {rel}")
        else:
            print(f"\nSyncing to {s3_prefix} ...")
            subprocess.run(
                ["aws", "s3", "sync", str(stage) + "/", s3_prefix],
                check=True,
            )
            print("Done.")


if __name__ == "__main__":
    main()

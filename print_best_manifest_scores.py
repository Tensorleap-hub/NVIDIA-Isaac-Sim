from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path("s3_best_runs_manifests")
DEFAULT_TOP_N = 3
DEFAULT_REPORT_PATH = DEFAULT_ROOT / "best_runs_summary.txt"
DEFAULT_DOWNLOAD_SPEC_PATH = DEFAULT_ROOT / "best_worst_trial_folders.yaml"
DEFAULT_DOWNLOAD_ROOT = Path("selected_trial_downloads")
TIMESTAMP_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


ManifestEntry = tuple[int, str, str, Path]
DownloadItem = dict[str, Any]


def iter_manifest_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/best_runs_manifest.json"))


def extract_category_and_timestamp(path: Path) -> tuple[str, str]:
    timestamp = path.parent.name
    category = path.parent.parent.name

    if not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValueError(f"{path} parent folder is not a timestamp run folder")

    return category, timestamp


def order_manifests_by_cycle(paths: list[Path]) -> list[ManifestEntry]:
    paths_by_category: dict[str, list[tuple[str, Path]]] = {}

    for path in paths:
        category, timestamp = extract_category_and_timestamp(path)
        paths_by_category.setdefault(category, []).append((timestamp, path))

    ordered_entries: list[ManifestEntry] = []
    for category, timestamp_paths in paths_by_category.items():
        for cycle_index, (timestamp, path) in enumerate(sorted(timestamp_paths), start=1):
            ordered_entries.append((cycle_index, category, timestamp, path))

    return sorted(ordered_entries, key=lambda entry: (entry[0], entry[1], entry[2]))


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as manifest_file:
        data = json.load(manifest_file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def format_trial(rank_label: str, trial: dict[str, Any]) -> str:
    score = trial.get("objective_value")
    trial_id = trial.get("trial_id", "unknown_trial")
    run_id = trial.get("run_id", "unknown_run")
    iteration_index = trial.get("iteration_index", "unknown_iteration")

    return (
        f"  {rank_label}. score={score} "
        f"trial_id={trial_id} run_id={run_id} iteration={iteration_index}"
    )


def build_manifest_block(
    path: Path,
    manifest: dict[str, Any],
    top_n: int,
    category: str,
    timestamp: str,
    cycle_index: int,
) -> tuple[list[str], list[DownloadItem]]:
    best_trials = manifest.get("best_trials", [])
    if not isinstance(best_trials, list):
        raise ValueError(f"{path} field 'best_trials' must be a list")

    project_name = manifest.get("project_name", "unknown_project")
    s3_prefix = manifest.get("s3_prefix", "unknown_s3_prefix")

    lines = [
        f"cycle {cycle_index}: {category} ({timestamp})",
        f"path: {path}",
        f"project: {project_name}",
        f"s3_prefix: {s3_prefix}",
        "top saved trials:",
    ]

    for rank, trial in enumerate(best_trials[:top_n], start=1):
        if not isinstance(trial, dict):
            raise ValueError(f"{path} best_trials[{rank - 1}] must be an object")

        lines.append(format_trial(str(rank), trial))

    download_items: list[DownloadItem] = []
    if best_trials:
        post_seed_trials = [
            t for t in best_trials
            if isinstance(t, dict) and int(t.get("iteration_index", 0)) > 0
        ]
        top_bests = post_seed_trials[:top_n]
        worst_trial = best_trials[-1]

        selected_trials: list[tuple[str, dict[str, Any]]] = [
            (f"best_{rank}", trial) for rank, trial in enumerate(top_bests, start=1)
        ]
        if isinstance(worst_trial, dict):
            selected_trials.append(("worst", worst_trial))

        seen_source_prefixes: set[str] = set()

        for kind, trial in selected_trials:
            if not isinstance(trial, dict):
                raise ValueError(f"{path} selected trial must be an object")

            trial_id = str(trial.get("trial_id", "unknown_trial"))
            source_s3_prefix = f"{str(s3_prefix).rstrip('/')}/{trial_id}/"

            if source_s3_prefix in seen_source_prefixes:
                continue
            seen_source_prefixes.add(source_s3_prefix)

            destination_dir = (
                DEFAULT_DOWNLOAD_ROOT
                / "optuna-ec2"
                / category
                / f"cycle_{cycle_index:02d}_{timestamp}"
                / f"{kind}_{trial_id}"
            )

            download_items.append(
                {
                    "kind": kind,
                    "category": category,
                    "cycle_index": cycle_index,
                    "timestamp": timestamp,
                    "manifest_path": str(path),
                    "project_name": project_name,
                    "trial_id": trial_id,
                    "run_id": str(trial.get("run_id", "unknown_run")),
                    "score": trial.get("objective_value"),
                    "source_s3_prefix": source_s3_prefix,
                    "destination_dir": str(destination_dir),
                }
            )

    lines.append("worst saved trial:")
    if best_trials:
        worst_trial = best_trials[-1]
        if not isinstance(worst_trial, dict):
            raise ValueError(f"{path} best_trials[-1] must be an object")

        lines.append(format_trial("worst", worst_trial))
    else:
        lines.append("  worst. <no saved trials>")

    return lines, download_items


def build_download_spec(items: list[DownloadItem]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket": "nvidia-isaac-bucket",
        "source_prefix_root": "s3://nvidia-isaac-bucket/optuna-ec2",
        "download_root": str(DEFAULT_DOWNLOAD_ROOT),
        "items": items,
    }


def write_text_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_yaml_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the top trials and objective scores from downloaded best_runs_manifest.json files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of trials to print from each manifest. Default: {DEFAULT_TOP_N}",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Write the text report here. Default: {DEFAULT_REPORT_PATH}",
    )
    parser.add_argument(
        "--download-spec-path",
        type=Path,
        default=DEFAULT_DOWNLOAD_SPEC_PATH,
        help=f"Write the YAML download spec here. Default: {DEFAULT_DOWNLOAD_SPEC_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_paths = iter_manifest_paths(args.root)

    if not manifest_paths:
        raise FileNotFoundError(f"No best_runs_manifest.json files found under {args.root}")

    report_lines = [f"Found {len(manifest_paths)} manifest files under {args.root}"]
    download_items: list[DownloadItem] = []

    for position, (cycle_index, category, timestamp, path) in enumerate(
        order_manifests_by_cycle(manifest_paths)
    ):
        if position > 0:
            report_lines.append("")

        block_lines, block_download_items = build_manifest_block(
            path=path,
            manifest=load_manifest(path),
            top_n=args.top,
            category=category,
            timestamp=timestamp,
            cycle_index=cycle_index,
        )
        report_lines.extend(block_lines)
        download_items.extend(block_download_items)

    report_text = "\n".join(report_lines).rstrip() + "\n"
    print(report_text, end="")
    write_text_report(args.report_path, report_lines)
    write_yaml_spec(args.download_spec_path, build_download_spec(download_items))


if __name__ == "__main__":
    main()

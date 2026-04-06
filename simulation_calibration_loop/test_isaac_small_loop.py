from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from calibration_optuna import DEFAULT_CONFIG
from calibration_optuna.data_utils import infer_bounds_and_types_from_metadata
from calibration_optuna.experiment_runner import ExperimentRunner
from simulation_calibration_loop.config import load_workflow_config
from simulation_calibration_loop.data import (
    DINOv2Embedder,
    discover_generated_images,
    prepare_output_dir,
    run_isaac_generation,
    select_real_image_paths,
)
from simulation_calibration_loop.parameter_schema import (
    filter_parameter_specs,
    flatten_config,
    infer_parameter_schema,
    load_yaml_configs,
    materialize_config,
    save_yaml_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run a small end-to-end Isaac + DINOv2 + Optuna loop")
    parser.add_argument(
        "--config",
        type=str,
        default="simulation_calibration_loop/test_isaac_small_loop.yaml",
        help="Path to the small-loop config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the DINOv2 device from the workflow config",
    )
    return parser.parse_args()


def load_small_loop_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text()) or {}
    config["workflow_config"] = str((config_path.parent / config["workflow_config"]).resolve())
    config["workspace_dir"] = str((config_path.parent / config["workspace_dir"]).resolve())
    config["initial_yaml"] = str((config_path.parent / config["initial_yaml"]).resolve())
    config["bounds_seed_dir"] = str((config_path.parent / config["bounds_seed_dir"]).resolve())
    return config


def build_schema(workflow_config, bounds_seed_dir: Path):
    inferred = infer_parameter_schema([item[1] for item in load_yaml_configs(bounds_seed_dir)])
    schema = filter_parameter_specs(
        inferred,
        include=workflow_config.search_space.include,
        exclude=workflow_config.search_space.exclude,
    )
    if not schema:
        raise ValueError("The configured search space produced an empty Isaac parameter schema")
    return schema


def build_param_bounds(schema, bounds_seed_dir: Path) -> tuple[dict, dict]:
    rows = []
    for distribution_id, (_, config) in enumerate(load_yaml_configs(bounds_seed_dir)):
        flattened = flatten_config(config, schema)
        row = {"distribution_id": distribution_id, "shape_logit_simulation_1": 0.0}
        for key, value in flattened.items():
            row[f"simulation_1__{key}"] = value
        rows.append(row)
    metadata_df = pd.DataFrame(rows)
    return infer_bounds_and_types_from_metadata(metadata_df, ["simulation_1"])


def sample_real_reference_paths(dataset_root: Path, annotations_file: Path, sample_count: int, seed: int) -> list[Path]:
    all_paths = select_real_image_paths(dataset_root, annotations_file)
    if len(all_paths) < sample_count:
        raise ValueError(f"Requested {sample_count} real samples but only found {len(all_paths)}")
    rng = random.Random(seed)
    candidates = list(all_paths)
    rng.shuffle(candidates)
    return sorted(candidates[:sample_count])


def embed_paths(embedder: DINOv2Embedder, paths: list[Path], batch_size: int, cache_path: Path, manifest: dict) -> np.ndarray:
    return embedder.embed_paths(
        paths,
        batch_size=batch_size,
        cache_path=cache_path,
        manifest=manifest,
    )


def main() -> None:
    args = parse_args()
    config = load_small_loop_config(args.config)
    workflow_config = load_workflow_config(config["workflow_config"])
    if args.device is not None:
        workflow_config.dino.device = args.device

    workspace_dir = Path(config["workspace_dir"])
    workspace_dir.mkdir(parents=True, exist_ok=True)
    initial_yaml_path = Path(config["initial_yaml"])
    bounds_seed_dir = Path(config["bounds_seed_dir"])

    schema = build_schema(workflow_config, bounds_seed_dir)
    param_bounds, param_type = build_param_bounds(schema, bounds_seed_dir)

    optimizer_config = deepcopy(DEFAULT_CONFIG)
    optimizer_config["experiment_name"] = config["experiment_name"]
    optimizer_config["experiments_base_dir"] = str(workspace_dir / "optuna")
    optimizer_config["iteration_batch_size"] = 1
    optimizer_config["top_n_best_trials"] = 1
    optimizer_config["mmd_max_samples"] = int(config.get("mmd_max_samples", 100))
    optimizer_config["random_seed"] = int(config.get("seed", 42))

    runner = ExperimentRunner(
        config=optimizer_config,
        param_bounds=param_bounds,
        param_type=param_type,
    )
    embedder = DINOv2Embedder(
        repo=workflow_config.dino.repo,
        model_name=workflow_config.dino.model_name,
        device=workflow_config.dino.device,
        image_size=workflow_config.dino.image_size,
        resize_size=workflow_config.dino.resize_size,
    )

    real_paths = sample_real_reference_paths(
        Path(workflow_config.real_dataset_root),
        Path(workflow_config.real_annotations_file),
        sample_count=int(config.get("real_samples", 16)),
        seed=int(config.get("seed", 42)),
    )
    real_embeddings = embed_paths(
        embedder,
        real_paths,
        batch_size=workflow_config.dino.batch_size,
        cache_path=workspace_dir / "cache" / "real.npy",
        manifest={
            "kind": "real",
            "model_name": workflow_config.dino.model_name,
            "paths": [str(path) for path in real_paths],
        },
    )
    runner.set_real_embeddings(real_embeddings)

    resolved_initial_config = None
    for candidate_path, candidate_config in load_yaml_configs(initial_yaml_path.parent):
        if candidate_path.resolve() == initial_yaml_path:
            resolved_initial_config = candidate_config
            break
    if resolved_initial_config is None:
        raise ValueError(f"Failed to resolve initial YAML config {initial_yaml_path}")

    initial_config = resolved_initial_config
    current_row = flatten_config(initial_config, schema)

    print("Small-loop configuration:")
    print(json.dumps(
        {
            "iterations": int(config["iterations"]),
            "num_frames_override": int(config.get("num_frames_override", 10)),
            "real_samples": len(real_paths),
            "search_space_paths": [spec.path for spec in schema],
        },
        indent=2,
        sort_keys=True,
    ))
    print("")

    for iteration_index in range(int(config["iterations"])):
        iteration_dir = workspace_dir / f"iteration_{iteration_index:03d}"
        yaml_dir = iteration_dir / "yamls"
        output_dir = iteration_dir / "output"
        cache_dir = iteration_dir / "cache"
        prepare_output_dir(yaml_dir, clean=False)
        prepare_output_dir(output_dir, clean=False)
        prepare_output_dir(cache_dir, clean=False)

        run_id = f"iter{iteration_index:03d}"
        yaml_path = yaml_dir / f"{run_id}.yaml"
        log_path = output_dir / "isaac.log"
        embedding_path = cache_dir / "synthetic.npy"

        config_to_run = materialize_config(initial_config, current_row, schema)
        save_yaml_config(yaml_path, config_to_run)

        run_isaac_generation(
            isaac_sim_path=Path(workflow_config.isaac.isaac_sim_path),
            script_path=Path(workflow_config.isaac.script_path),
            yaml_path=yaml_path,
            output_dir=output_dir,
            log_path=log_path,
            headless=workflow_config.isaac.headless,
            num_frames_override=int(config.get("num_frames_override", 10)),
            log_callback=lambda line: print(line),
        )

        image_paths = discover_generated_images(output_dir / "Camera" / "rgb")
        if not image_paths:
            raise ValueError(f"No generated images discovered under {output_dir}")

        synthetic_embeddings = embed_paths(
            embedder,
            image_paths,
            batch_size=workflow_config.dino.batch_size,
            cache_path=embedding_path,
            manifest={
                "kind": "synthetic",
                "model_name": workflow_config.dino.model_name,
                "paths": [str(path) for path in image_paths],
                "yaml_path": str(yaml_path),
            },
        )

        current_distributions = [
            (
                run_id,
                {
                    "shape_logit_simulation_1": 0.0,
                    **{f"simulation_1__{key}": value for key, value in current_row.items()},
                },
            )
        ]
        embeddings_by_shape = [synthetic_embeddings]
        embeddings_indices_by_dist = {0: [(0, np.arange(len(synthetic_embeddings)))]}
        suggestions = runner.run_iteration(
            current_distributions=current_distributions,
            embeddings_by_shape=embeddings_by_shape,
            embeddings_indices_by_dist=embeddings_indices_by_dist,
        )
        best_trials = runner.get_best_trials(top_n=1)

        suggestion_id, suggestion_params = suggestions[0]
        next_row = {}
        for key, value in suggestion_params.items():
            if key.startswith("simulation_1__"):
                next_row[key[len("simulation_1__"):]] = value

        print("")
        print(f"Iteration {iteration_index + 1}/{int(config['iterations'])}")
        print(f"  YAML: {yaml_path}")
        print(f"  Images generated: {len(image_paths)}")
        print(f"  Embedding shape: {tuple(synthetic_embeddings.shape)}")
        print(f"  Best trial: {best_trials[0][0]}")
        print(f"  Next suggestion id: {suggestion_id}")
        print("  Next suggestion params:")
        for key in sorted(next_row):
            print(f"    {key}: {next_row[key]}")
        print("")

        current_row = next_row


if __name__ == "__main__":
    main()

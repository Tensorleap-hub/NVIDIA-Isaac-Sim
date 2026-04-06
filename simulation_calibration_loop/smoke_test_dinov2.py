from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml

from calibration_optuna.run_optimizer import run_optimizer_iteration
from simulation_calibration_loop.data import DINOv2Embedder, discover_generated_images, select_real_image_paths
from simulation_calibration_loop.config import load_workflow_config
from simulation_calibration_loop.parameter_schema import (
    filter_parameter_specs,
    flatten_config,
    infer_parameter_schema,
    load_yaml_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DINOv2 smoke test for real vs synthetic palletjack images")
    parser.add_argument(
        "--real-dataset-root",
        type=str,
        default="/Users/orram/Tensorleap/data/warehouse/dataset",
        help="Root of the LOCO-style warehouse dataset",
    )
    parser.add_argument(
        "--real-annotations-file",
        type=str,
        default="/Users/orram/Tensorleap/data/warehouse/dataset/labels/loco-sub3-v1-train.json",
        help="Annotation file used to select the real subset",
    )
    parser.add_argument(
        "--synthetic-root",
        type=str,
        default="/Users/orram/Tensorleap/data/warehouse/palletjack_run_0",
        help="Root directory containing synthetic experiment outputs",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=8,
        help="Number of real and synthetic images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vitb14_reg",
        help="Torch Hub DINOv2 model name",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="facebookresearch/dinov2",
        help="Torch Hub repo for DINOv2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to use, for example cuda, mps, or cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/tmp/dinov2_smoke_test_cache",
        help="Directory for temporary embedding cache files",
    )
    parser.add_argument(
        "--seed-config-dir",
        type=str,
        default="palletjack_sdg/experiments/ec2-loop/base_v2",
        help="Directory of the base YAML configs used to infer the Optuna parameter schema",
    )
    parser.add_argument(
        "--workflow-config",
        type=str,
        default="simulation_calibration_loop/project_config.yaml",
        help="Optional workflow config used to reuse search_space include/exclude filters",
    )
    parser.add_argument(
        "--synthetic-experiments",
        type=int,
        default=4,
        help="How many synthetic experiment folders to sample for the Optuna smoke test",
    )
    parser.add_argument(
        "--synthetic-images-per-experiment",
        type=int,
        default=1,
        help="How many RGB images to sample from each selected synthetic experiment",
    )
    return parser.parse_args()


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError(f"Requested {count} samples but only found {len(paths)} images")
    rng = random.Random(seed)
    chosen = list(paths)
    rng.shuffle(chosen)
    return sorted(chosen[:count])


def collect_synthetic_paths(synthetic_root: Path) -> list[Path]:
    all_paths: list[Path] = []
    for experiment_dir in sorted(synthetic_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        rgb_dir = experiment_dir / "Camera" / "rgb"
        if rgb_dir.exists():
            all_paths.extend(discover_generated_images(rgb_dir))
    return sorted(all_paths)


def collect_synthetic_experiment_map(synthetic_root: Path) -> dict[str, list[Path]]:
    experiments: dict[str, list[Path]] = {}
    for experiment_dir in sorted(synthetic_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        rgb_dir = experiment_dir / "Camera" / "rgb"
        if rgb_dir.exists():
            images = discover_generated_images(rgb_dir)
            if images:
                experiments[experiment_dir.name] = images
    return experiments


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
    squared = np.maximum(a_sq + b_sq - 2 * (a @ b.T), 0.0)
    return np.sqrt(squared)


def summarize_distances(name: str, distances: np.ndarray) -> dict[str, float]:
    return {
        f"{name}_min": float(np.min(distances)),
        f"{name}_mean": float(np.mean(distances)),
        f"{name}_median": float(np.median(distances)),
        f"{name}_max": float(np.max(distances)),
    }


def off_diagonal(values: np.ndarray) -> np.ndarray:
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def sample_synthetic_runs(
    synthetic_root: Path,
    *,
    seed_config_dir: Path,
    workflow_config_path: Path,
    seed: int,
    synthetic_experiments: int,
    synthetic_images_per_experiment: int,
) -> tuple[list[Path], pd.DataFrame]:
    experiment_map = collect_synthetic_experiment_map(synthetic_root)
    experiment_names = sorted(experiment_map)
    if len(experiment_names) < synthetic_experiments:
        raise ValueError(
            f"Requested {synthetic_experiments} synthetic experiments but only found {len(experiment_names)}"
        )

    workflow_config = load_workflow_config(workflow_config_path)
    schema = filter_parameter_specs(
        infer_parameter_schema([item[1] for item in load_yaml_configs(seed_config_dir)]),
        include=workflow_config.search_space.include,
        exclude=workflow_config.search_space.exclude,
    )
    if not schema:
        raise ValueError("Search-space filtering removed all Isaac parameters for the smoke test")
    rng = random.Random(seed)
    chosen_names = list(experiment_names)
    rng.shuffle(chosen_names)
    chosen_names = sorted(chosen_names[:synthetic_experiments])

    sampled_paths: list[Path] = []
    metadata_rows: list[dict[str, object]] = []
    for experiment_name in chosen_names:
        image_candidates = experiment_map[experiment_name]
        if len(image_candidates) < synthetic_images_per_experiment:
            raise ValueError(
                f"Experiment {experiment_name} has only {len(image_candidates)} RGB images, "
                f"cannot sample {synthetic_images_per_experiment}"
            )
        local_rng = random.Random(f"{seed}:{experiment_name}")
        local_candidates = list(image_candidates)
        local_rng.shuffle(local_candidates)
        selected_images = sorted(local_candidates[:synthetic_images_per_experiment])

        run_config_path = synthetic_root / experiment_name / "run_config.yaml"
        run_config = yaml.safe_load(run_config_path.read_text())
        flattened = flatten_config(run_config, schema)

        sampled_paths.extend(selected_images)
        metadata_rows.extend([flattened] * len(selected_images))

    return sampled_paths, pd.DataFrame(metadata_rows)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    real_paths = select_real_image_paths(args.real_dataset_root, args.real_annotations_file)
    synth_paths = collect_synthetic_paths(Path(args.synthetic_root))

    sampled_real = sample_paths(real_paths, args.samples_per_domain, args.seed)
    sampled_synth_for_stats = sample_paths(synth_paths, args.samples_per_domain, args.seed + 1)
    sampled_synth_for_optuna, synthetic_metadata = sample_synthetic_runs(
        Path(args.synthetic_root),
        seed_config_dir=Path(args.seed_config_dir),
        workflow_config_path=Path(args.workflow_config),
        seed=args.seed + 7,
        synthetic_experiments=args.synthetic_experiments,
        synthetic_images_per_experiment=args.synthetic_images_per_experiment,
    )

    embedder = DINOv2Embedder(
        repo=args.repo,
        model_name=args.model_name,
        device=args.device,
        image_size=224,
        resize_size=256,
    )

    real_embeddings = embedder.embed_paths(
        sampled_real,
        batch_size=args.batch_size,
        cache_path=cache_dir / "real.npy",
        manifest={
            "kind": "real",
            "model_name": args.model_name,
            "paths": [str(path) for path in sampled_real],
        },
    )
    synth_embeddings = embedder.embed_paths(
        sampled_synth_for_stats,
        batch_size=args.batch_size,
        cache_path=cache_dir / "synthetic.npy",
        manifest={
            "kind": "synthetic",
            "model_name": args.model_name,
            "paths": [str(path) for path in sampled_synth_for_stats],
        },
    )
    optuna_synth_embeddings = embedder.embed_paths(
        sampled_synth_for_optuna,
        batch_size=args.batch_size,
        cache_path=cache_dir / "synthetic_optuna.npy",
        manifest={
            "kind": "synthetic_optuna",
            "model_name": args.model_name,
            "paths": [str(path) for path in sampled_synth_for_optuna],
        },
    )

    rr = pairwise_distances(real_embeddings, real_embeddings)
    ss = pairwise_distances(synth_embeddings, synth_embeddings)
    rs = pairwise_distances(real_embeddings, synth_embeddings)

    summary = {
        "model_name": args.model_name,
        "device": args.device,
        "real_count": len(sampled_real),
        "synthetic_count": len(sampled_synth_for_stats),
        "synthetic_optuna_count": len(sampled_synth_for_optuna),
        "embedding_dim": int(real_embeddings.shape[1]),
        "real_norm_mean": float(np.linalg.norm(real_embeddings, axis=1).mean()),
        "synthetic_norm_mean": float(np.linalg.norm(synth_embeddings, axis=1).mean()),
        **summarize_distances("real_real", off_diagonal(rr)),
        **summarize_distances("synthetic_synthetic", off_diagonal(ss)),
        **summarize_distances("real_synthetic", rs),
    }

    suggestions_df, best_trials_df = run_optimizer_iteration(
        real_embeddings,
        [optuna_synth_embeddings],
        [synthetic_metadata],
    )

    print("Real samples:")
    for path in sampled_real:
        print(path)
    print("")
    print("Synthetic samples:")
    for path in sampled_synth_for_stats:
        print(path)
    print("")
    print("Synthetic samples for Optuna:")
    for path in sampled_synth_for_optuna:
        print(path)
    print("")
    print("Summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("")
    print("Best trials:")
    print(best_trials_df.to_string(index=False))
    print("")
    print("Optuna suggestions:")
    print(suggestions_df.to_string(index=False))


if __name__ == "__main__":
    main()

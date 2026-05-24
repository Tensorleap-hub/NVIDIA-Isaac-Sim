"""Build a DINOv2 PCA population view for real, base, and selected loop runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_calibration_loop.config import DINOv2Config, load_workflow_config
from simulation_calibration_loop.data import DINOv2Embedder, make_cache_key, select_real_image_paths


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_BASE_ROOT = Path("/Users/orram/Tensorleap/data/warehouse/palletjack_run_0")
DEFAULT_OUTPUT_DIR = Path("simulation_calibration_loop/population_view")
DEFAULT_CONFIG_PATH = Path("simulation_calibration_loop/project_config.yaml")
DEFAULT_SELECTED_ROOT = Path("selected_trial_downloads")
DEFAULT_SCORE_MANIFEST_ROOT = Path("s3_best_runs_manifests")
CYCLE_PATTERN = re.compile(r"^cycle_(?P<cycle_index>\d+)_(?P<timestamp>.+)$")


@dataclass
class ScoreRecord:
    objective_value: float | None
    manifest_rank: int | None
    project_name: str
    trial_id: str
    iteration_index: int | None


@dataclass
class SelectedRun:
    cache_path: Path
    label: str
    category: str
    cycle_index: int | None
    timestamp: str
    selection_kind: str
    trial_id: str
    run_id: str
    run_fingerprint: str
    manifest_image_count: int | None
    yaml_path: str
    score: ScoreRecord | None
    embeddings: np.ndarray


@dataclass
class PopulationGroup:
    label: str
    display_name: str
    group_type: str
    embeddings: np.ndarray
    color: str
    source_path: str
    selected_run: SelectedRun | None = None
    base_part_name: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a DINOv2 PCA population view for LOCO real data, "
            "the initial palletjack base run, and selected calibration-loop runs."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--real-dataset-root", type=Path)
    parser.add_argument("--real-annotations-file", type=Path)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--score-manifest-root", type=Path, default=DEFAULT_SCORE_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="population_pca", help="Stem for output file names.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=None)
    parser.add_argument("--metric-max-samples", type=int, default=1000)
    parser.add_argument("--plot-max-points-per-group", type=int, default=None)
    parser.add_argument("--max-real-images", type=int, default=None)
    parser.add_argument("--max-base-images", type=int, default=None)
    parser.add_argument("--max-selected-runs", type=int, default=None)
    parser.add_argument(
        "--reuse-real-base-only",
        action="store_true",
        help="Load existing real/base embedding caches and fail if they are missing.",
    )
    parser.add_argument(
        "--split-base-experiments",
        action="store_true",
        help=(
            "Show each subdirectory of --base-root as a separate group instead of "
            "one merged base group. Useful for comparing individual base experiments."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered input counts. Does not load DINO or write outputs.",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        help="Skip t-SNE computation (can be slow on large datasets).",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP computation.",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> tuple[Path, Path, DINOv2Config]:
    workflow = load_workflow_config(args.config)
    dino = workflow.dino

    if args.model_name is not None:
        dino.model_name = args.model_name
    if args.repo is not None:
        dino.repo = args.repo
    if args.batch_size is not None:
        dino.batch_size = args.batch_size
    if args.image_size is not None:
        dino.image_size = args.image_size
    if args.resize_size is not None:
        dino.resize_size = args.resize_size
    if args.device != "auto":
        dino.device = args.device
    else:
        dino.device = choose_device()

    real_dataset_root = args.real_dataset_root or Path(workflow.real_dataset_root)
    real_annotations_file = args.real_annotations_file or Path(workflow.real_annotations_file)
    return real_dataset_root.resolve(), real_annotations_file.resolve(), dino


def choose_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def limited_paths(paths: list[Path], limit: int | None) -> list[Path]:
    if limit is None:
        return paths
    if limit <= 0:
        raise ValueError("Image limits must be positive")
    return paths[:limit]


def discover_base_rgb_images(base_root: Path) -> list[Path]:
    rgb_dirs = sorted(path for path in base_root.rglob("rgb") if path.is_dir() and path.parent.name == "Camera")
    if rgb_dirs:
        image_paths: list[Path] = []
        for rgb_dir in rgb_dirs:
            image_paths.extend(
                sorted(
                    path
                    for path in rgb_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
            )
    else:
        # flat layout: images directly inside subdirectories (no Camera/rgb nesting)
        image_paths = sorted(
            path
            for path in base_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not image_paths:
        raise ValueError(f"No RGB image files found under {base_root}")
    return image_paths


def discover_base_parts(base_root: Path) -> list[tuple[str, list[Path]]]:
    parts: list[tuple[str, list[Path]]] = []
    for child in sorted(base_root.iterdir()):
        if not child.is_dir():
            continue
        part_paths = sorted(
            path
            for path in child.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if part_paths:
            parts.append((child.name, part_paths))
    if not parts:
        raise ValueError(f"No base parts with RGB images found under {base_root}")
    return parts


def filter_base_parts_for_paths(
    *,
    base_root: Path,
    image_paths: list[Path],
) -> list[tuple[str, list[Path]]]:
    parts = []
    selected = {path.resolve() for path in image_paths}
    for part_name, part_paths in discover_base_parts(base_root):
        filtered_paths = [path for path in part_paths if path.resolve() in selected]
        if filtered_paths:
            parts.append((part_name, filtered_paths))
    if not parts:
        raise ValueError("No base part paths matched the selected image list")
    return parts


def build_embedding_manifest(
    *,
    label: str,
    image_paths: list[Path],
    dino: DINOv2Config,
) -> dict[str, Any]:
    return {
        "label": label,
        "model_name": dino.model_name,
        "repo": dino.repo,
        "image_size": dino.image_size,
        "resize_size": dino.resize_size,
        "image_paths": [str(path) for path in image_paths],
    }


def cache_path_for(output_dir: Path, label: str, image_paths: list[Path], dino: DINOv2Config) -> Path:
    cache_key = make_cache_key(
        [
            label,
            dino.model_name,
            dino.repo,
            str(dino.image_size),
            str(dino.resize_size),
            *(str(path) for path in image_paths),
        ]
    )[:16]
    safe_label = label.replace("-", "_")
    return output_dir / "cache" / f"{safe_label}_{cache_key}_{dino.model_name}.npy"


def partition_indices_by_base_part(base_root: Path, image_paths: list[Path]) -> dict[str, list[int]]:
    part_indices: dict[str, list[int]] = {}
    for index, path in enumerate(image_paths):
        rel_path = path.relative_to(base_root)
        if len(rel_path.parts) == 0:
            continue
        part_name = rel_path.parts[0]
        part_indices.setdefault(part_name, []).append(index)
    return part_indices


def make_base_shade(index: int, total: int) -> str:
    if total <= 1:
        return "hsl(214, 52%, 43%)"
    start_lightness = 28.0
    end_lightness = 60.0
    lightness = start_lightness + (end_lightness - start_lightness) * (index / (total - 1))
    return f"hsl(214, 52%, {lightness:.1f}%)"


def load_matching_cache(cache_path: Path, manifest: dict[str, Any]) -> np.ndarray | None:
    manifest_path = cache_path.with_suffix(".manifest.json")
    if not cache_path.exists() or not manifest_path.exists():
        return None
    cached_manifest = json.loads(manifest_path.read_text())
    if cached_manifest != manifest:
        return None
    return np.load(cache_path)


def load_or_compute_image_embeddings(
    *,
    label: str,
    image_paths: list[Path],
    cache_path: Path,
    manifest: dict[str, Any],
    embedder: DINOv2Embedder | None,
    batch_size: int,
    reuse_only: bool,
) -> np.ndarray:
    cached = load_matching_cache(cache_path, manifest)
    if cached is not None:
        return validate_embeddings(label, cached)
    if reuse_only:
        raise ValueError(f"Missing reusable cache for {label}: {cache_path}")
    if embedder is None:
        raise ValueError(f"Cannot compute {label} embeddings without a DINOv2 embedder")
    embeddings = embedder.embed_paths(
        image_paths,
        batch_size=batch_size,
        cache_path=cache_path,
        manifest=manifest,
    )
    return validate_embeddings(label, embeddings)


def split_embeddings_by_part(
    *,
    base_root: Path,
    image_paths: list[Path],
    embeddings: np.ndarray,
    base_parts: list[tuple[str, list[Path]]],
) -> list[tuple[str, np.ndarray]]:
    part_indices = partition_indices_by_base_part(base_root, image_paths)
    by_part: list[tuple[str, np.ndarray]] = []
    for part_name, part_paths in base_parts:
        indices = part_indices.get(part_name, [])
        if len(indices) != len(part_paths):
            raise ValueError(
                f"Base part '{part_name}' has {len(part_paths)} images but {len(indices)} cached embeddings"
            )
        by_part.append((part_name, embeddings[indices]))
    return by_part


def validate_embeddings(label: str, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"{label} embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError(f"{label} embeddings are empty")
    if embeddings.shape[1] < 2:
        raise ValueError(f"{label} embeddings must have at least 2 dimensions")
    return embeddings.astype(np.float32, copy=False)


def load_score_index(score_manifest_root: Path) -> dict[tuple[str, str, str], ScoreRecord]:
    score_index: dict[tuple[str, str, str], ScoreRecord] = {}
    for manifest_path in sorted(score_manifest_root.glob("**/best_runs_manifest.json")):
        payload = json.loads(manifest_path.read_text())
        category = manifest_path.parent.parent.name
        timestamp = manifest_path.parent.name
        project_name = str(payload.get("project_name", ""))
        for rank, trial in enumerate(payload.get("best_trials", []), start=1):
            source_embedding_path = trial.get("source_embedding_path")
            if not source_embedding_path:
                continue
            key = (category, timestamp, Path(str(source_embedding_path)).name)
            objective_value = trial.get("objective_value")
            iteration_index = trial.get("iteration_index")
            score_index[key] = ScoreRecord(
                objective_value=float(objective_value) if objective_value is not None else None,
                manifest_rank=rank,
                project_name=project_name,
                trial_id=str(trial.get("trial_id", "")),
                iteration_index=int(iteration_index) if iteration_index is not None else None,
            )
    return score_index


def discover_selected_runs(
    *,
    selected_root: Path,
    dino: DINOv2Config,
    score_index: dict[tuple[str, str, str], ScoreRecord],
    max_selected_runs: int | None,
) -> list[SelectedRun]:
    cache_paths = sorted(selected_root.glob(f"**/cache/*{dino.model_name}.npy"))
    if max_selected_runs is not None:
        if max_selected_runs <= 0:
            raise ValueError("--max-selected-runs must be positive")
        cache_paths = cache_paths[:max_selected_runs]
    if not cache_paths:
        raise ValueError(f"No selected DINO cache files found under {selected_root}")

    selected_runs: list[SelectedRun] = []
    for index, cache_path in enumerate(cache_paths, start=1):
        selected_runs.append(
            load_selected_run(
                cache_path=cache_path,
                selected_root=selected_root,
                label=f"run_{index:02d}",
                score_index=score_index,
            )
        )
    return selected_runs


def load_selected_run(
    *,
    cache_path: Path,
    selected_root: Path,
    label: str,
    score_index: dict[tuple[str, str, str], ScoreRecord],
) -> SelectedRun:
    rel_parts = cache_path.relative_to(selected_root).parts
    if len(rel_parts) >= 6:
        # layout: {any}/{category}/{cycle_folder}/{selected_folder}/cache/{file}
        category = rel_parts[1]
        cycle_folder = rel_parts[2]
        selected_folder = rel_parts[3]
        cycle_match = CYCLE_PATTERN.fullmatch(cycle_folder)
        cycle_index = int(cycle_match.group("cycle_index")) if cycle_match else None
        timestamp = cycle_match.group("timestamp") if cycle_match else cycle_folder
        selection_kind, trial_id = parse_selected_folder(selected_folder)
    elif len(rel_parts) == 4:
        # flat layout: {category}/{trial_id}/cache/{file}
        category = rel_parts[0]
        cycle_index = None
        timestamp = ""
        selection_kind = ""
        trial_id = rel_parts[1]
    else:
        raise ValueError(f"Unexpected selected cache layout: {cache_path}")

    manifest_path = cache_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    embeddings = validate_embeddings(label, np.load(cache_path))
    run_id = str(manifest.get("run_id") or parse_run_id_from_cache_name(cache_path.name))
    run_fingerprint = str(manifest.get("run_fingerprint") or "")
    image_paths = manifest.get("image_paths") or []
    yaml_path = str(manifest.get("yaml_path") or "")
    score = score_index.get((category, timestamp, cache_path.name))

    return SelectedRun(
        cache_path=cache_path,
        label=label,
        category=category,
        cycle_index=cycle_index,
        timestamp=timestamp,
        selection_kind=selection_kind,
        trial_id=trial_id,
        run_id=run_id,
        run_fingerprint=run_fingerprint,
        manifest_image_count=len(image_paths) if image_paths else None,
        yaml_path=yaml_path,
        score=score,
        embeddings=embeddings,
    )


def get_uncached_output_run_specs(
    selected_root: Path,
    dino: DINOv2Config,
) -> list[tuple[Path, list[Path], dict[str, Any]]]:
    """Find runs under {category}/{trial}/outputs/{run} that have images but no DINO cache."""
    specs: list[tuple[Path, list[Path], dict[str, Any]]] = []
    for outputs_subdir in sorted(selected_root.glob("*/*/outputs/*")):
        if not outputs_subdir.is_dir():
            continue
        image_paths = sorted(
            p for p in outputs_subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            continue
        trial_dir = outputs_subdir.parent.parent
        cache_dir = trial_dir / "cache"
        run_name = outputs_subdir.name
        if list(cache_dir.glob(f"*{run_name}*{dino.model_name}.npy")):
            continue
        label = run_name
        manifest = build_embedding_manifest(label=label, image_paths=image_paths, dino=dino)
        cache_key = make_cache_key(
            [label, dino.model_name, dino.repo, str(dino.image_size), str(dino.resize_size), *(str(p) for p in image_paths)]
        )[:16]
        safe_label = label.replace("-", "_")
        cache_path = cache_dir / f"{safe_label}__{cache_key}_{dino.model_name}.npy"
        specs.append((cache_path, image_paths, manifest))
    return specs


def parse_selected_folder(folder_name: str) -> tuple[str, str]:
    parts = folder_name.split("_", 1)
    if len(parts) == 1:
        return folder_name, ""
    return parts[0], parts[1]


def parse_run_id_from_cache_name(file_name: str) -> str:
    stem = file_name.removesuffix(".npy")
    return stem.split("__", 1)[0]


def validate_consistent_dimensions(groups: list[PopulationGroup]) -> int:
    dimensions = {group.embeddings.shape[1] for group in groups}
    if len(dimensions) != 1:
        details = ", ".join(f"{group.label}:{group.embeddings.shape}" for group in groups)
        raise ValueError(f"Embedding dimensions do not match: {details}")
    return dimensions.pop()


def fit_pca_2d(groups: list[PopulationGroup]) -> tuple[np.ndarray, np.ndarray]:
    validate_consistent_dimensions(groups)
    matrix = np.concatenate([group.embeddings for group in groups], axis=0).astype(np.float64, copy=False)
    if matrix.shape[0] < 2:
        raise ValueError("At least two embeddings are required for PCA")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, components_t = np.linalg.svd(centered, full_matrices=False)
    if components_t.shape[0] < 2:
        raise ValueError("PCA could not produce two components")
    coords = centered @ components_t[:2].T
    variances = singular_values**2
    explained = variances[:2] / variances.sum()
    return coords.astype(np.float32), explained.astype(np.float64)


def fit_tsne_2d(groups: list[PopulationGroup]) -> np.ndarray:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.manifold import TSNE
    validate_consistent_dimensions(groups)
    matrix = np.concatenate([group.embeddings for group in groups], axis=0).astype(np.float64, copy=False)
    n_pre = min(50, matrix.shape[1], matrix.shape[0] - 1)
    if n_pre < matrix.shape[1]:
        matrix = SklearnPCA(n_components=n_pre, random_state=42).fit_transform(matrix)
    coords = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(matrix)
    return coords.astype(np.float32)


def fit_umap_2d(groups: list[PopulationGroup]) -> np.ndarray:
    import umap as umap_lib
    validate_consistent_dimensions(groups)
    matrix = np.concatenate([group.embeddings for group in groups], axis=0).astype(np.float64, copy=False)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    return reducer.fit_transform(matrix).astype(np.float32)


def group_coordinate_slices(groups: list[PopulationGroup], coords: np.ndarray) -> dict[str, np.ndarray]:
    slices: dict[str, np.ndarray] = {}
    start = 0
    for group in groups:
        end = start + len(group.embeddings)
        slices[group.label] = coords[start:end]
        start = end
    return slices


def deterministic_sample_indices(count: int, max_count: int | None, seed: int) -> np.ndarray:
    if max_count is None or count <= max_count:
        return np.arange(count)
    if max_count <= 0:
        raise ValueError("--plot-max-points-per-group must be positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def pairwise_sq_dists(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    left_norm = np.sum(left64 * left64, axis=1, keepdims=True)
    right_norm = np.sum(right64 * right64, axis=1, keepdims=True).T
    distances = left_norm + right_norm - 2.0 * (left64 @ right64.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def sample_embeddings(embeddings: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if len(embeddings) <= max_samples:
        return embeddings
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(embeddings), size=max_samples, replace=False))
    return embeddings[indices]


def compute_rbf_gamma(real_embeddings: np.ndarray) -> float:
    distances_sq = pairwise_sq_dists(real_embeddings, real_embeddings)
    positive = distances_sq > 0.0
    if not np.any(positive):
        raise ValueError("Real embeddings do not contain positive pairwise distances")
    sigma = float(np.median(np.sqrt(distances_sq[positive])))
    if sigma <= 0.0:
        raise ValueError("Real embedding median pairwise distance must be positive")
    return 1.0 / (2.0 * sigma**2)


def compute_mmd_rbf(left: np.ndarray, right: np.ndarray, gamma: float) -> float:
    left_left = np.exp(-gamma * pairwise_sq_dists(left, left)).mean()
    right_right = np.exp(-gamma * pairwise_sq_dists(right, right)).mean()
    left_right = np.exp(-gamma * pairwise_sq_dists(left, right)).mean()
    value = left_left + right_right - 2.0 * left_right
    return float(math.sqrt(max(value, 0.0)))


def nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    target64 = target.astype(np.float64, copy=False)
    target_norm = np.sum(target64 * target64, axis=1, keepdims=True).T
    chunks: list[np.ndarray] = []
    for start in range(0, len(source), chunk_size):
        source_chunk = source[start:start + chunk_size].astype(np.float64, copy=False)
        source_norm = np.sum(source_chunk * source_chunk, axis=1, keepdims=True)
        distances_sq = source_norm + target_norm - 2.0 * (source_chunk @ target64.T)
        np.maximum(distances_sq, 0.0, out=distances_sq)
        chunks.append(np.sqrt(distances_sq.min(axis=1)))
    return np.concatenate(chunks, axis=0)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(left @ right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        raise ValueError("Cannot compute cosine distance for a zero-norm centroid")
    return 1.0 - numerator / denominator


def compute_performance_rows(
    *,
    groups: list[PopulationGroup],
    coords_by_label: dict[str, np.ndarray],
    metric_max_samples: int,
) -> list[dict[str, Any]]:
    real_group = next(group for group in groups if group.label == "real-data")
    real_embeddings = real_group.embeddings
    real_centroid = real_embeddings.mean(axis=0)
    real_coords = coords_by_label["real-data"]
    real_pca_centroid = real_coords.mean(axis=0)
    real_metric_sample = sample_embeddings(real_embeddings, metric_max_samples, seed=42)
    gamma = compute_rbf_gamma(real_metric_sample)

    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if group.label == "real-data":
            continue
        metric_sample = sample_embeddings(group.embeddings, metric_max_samples, seed=1000 + index)
        centroid = group.embeddings.mean(axis=0)
        group_coords = coords_by_label[group.label]
        pca_centroid = group_coords.mean(axis=0)
        syn_to_real = nearest_distances(group.embeddings, real_embeddings)
        real_to_syn = nearest_distances(real_embeddings, group.embeddings)
        selected = group.selected_run
        score = selected.score if selected is not None else None
        rows.append(
            {
                "label": group.label,
                "display_name": group.display_name,
                "group_type": group.group_type,
                "image_count": len(group.embeddings),
                "centroid_l2_to_real": float(np.linalg.norm(centroid - real_centroid)),
                "centroid_cosine_to_real": cosine_distance(centroid, real_centroid),
                "pca_centroid_l2_to_real": float(np.linalg.norm(pca_centroid - real_pca_centroid)),
                "mmd_rbf_to_real": compute_mmd_rbf(metric_sample, real_metric_sample, gamma),
                "syn_to_real_nn_mean": float(syn_to_real.mean()),
                "syn_to_real_nn_median": float(np.median(syn_to_real)),
                "real_to_syn_nn_mean": float(real_to_syn.mean()),
                "source_objective_value": score.objective_value if score else None,
                "source_manifest_rank": score.manifest_rank if score else None,
                "project_name": score.project_name if score else "",
                "category": selected.category if selected else "",
                "cycle_index": selected.cycle_index if selected else None,
                "timestamp": selected.timestamp if selected else "",
                "selection_kind": selected.selection_kind if selected else "",
                "trial_id": selected.trial_id if selected else "",
                "run_id": selected.run_id if selected else "",
                "iteration_index": score.iteration_index if score else None,
                "cache_path": str(selected.cache_path) if selected else group.source_path,
            }
        )

    return sorted(rows, key=lambda row: row["centroid_l2_to_real"])


def format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_performance_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "display_name",
        "group_type",
        "image_count",
        "centroid_l2_to_real",
        "centroid_cosine_to_real",
        "pca_centroid_l2_to_real",
        "mmd_rbf_to_real",
        "syn_to_real_nn_mean",
        "syn_to_real_nn_median",
        "real_to_syn_nn_mean",
        "source_objective_value",
        "source_manifest_rank",
        "project_name",
        "category",
        "cycle_index",
        "timestamp",
        "selection_kind",
        "trial_id",
        "run_id",
        "iteration_index",
        "cache_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_run_color(index: int) -> str:
    hue = (index * 137.508) % 360
    return f"hsl({hue:.1f}, 68%, 43%)"


def build_population_groups(
    *,
    real_embeddings: np.ndarray,
    base_parts: list[tuple[str, np.ndarray]],
    base_root: Path,
    selected_runs: list[SelectedRun],
) -> list[PopulationGroup]:
    groups = [
        PopulationGroup(
            label="real-data",
            display_name="real-data",
            group_type="real",
            embeddings=real_embeddings,
            color="#111827",
            source_path="loco_dataset/subset-3",
        ),
    ]
    total_base_parts = len(base_parts)
    for index, (part_name, embeddings) in enumerate(base_parts):
        groups.append(
            PopulationGroup(
                label=f"base:{part_name}",
                display_name=part_name,
                group_type="base",
                embeddings=embeddings,
                color=make_base_shade(index, total_base_parts),
                source_path=str(base_root / part_name),
                base_part_name=part_name,
            )
        )
    for index, selected in enumerate(selected_runs, start=1):
        groups.append(
            PopulationGroup(
                label=selected.label,
                display_name=f"{selected.label} {selected.category} {selected.selection_kind}",
                group_type="selected",
                embeddings=selected.embeddings,
                color=make_run_color(index),
                source_path=str(selected.cache_path),
                selected_run=selected,
            )
        )
    return groups


def hover_key_for_group(group: PopulationGroup) -> str:
    if group.selected_run is not None:
        return str(group.selected_run.cache_path)
    if group.group_type == "base":
        return group.source_path
    return group.label


def svg_star_points(cx: float, cy: float, outer_radius: float, inner_radius: float) -> str:
    points: list[str] = []
    for step in range(10):
        angle = -math.pi / 2 + step * math.pi / 5
        radius = outer_radius if step % 2 == 0 else inner_radius
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def scale_coordinates(coords: np.ndarray, width: int, height: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    if x_min == x_max or y_min == y_max:
        raise ValueError("PCA coordinates collapsed onto one axis")
    x_scale = (width - 2 * margin) / (x_max - x_min)
    y_scale = (height - 2 * margin) / (y_max - y_min)
    scaled_x = margin + (coords[:, 0] - x_min) * x_scale
    scaled_y = height - margin - (coords[:, 1] - y_min) * y_scale
    return scaled_x, scaled_y


def svg_circle_rows(
    *,
    groups: list[PopulationGroup],
    coords: np.ndarray,
    width: int,
    height: int,
    margin: int,
    max_points_per_group: int | None,
    best_selected_keys: set[str],
    best_base_keys: set[str],
) -> list[str]:
    scaled_x, scaled_y = scale_coordinates(coords, width, height, margin)
    rows: list[str] = []
    start = 0
    for group_index, group in enumerate(groups):
        end = start + len(group.embeddings)
        indices = deterministic_sample_indices(end - start, max_points_per_group, seed=group_index + 1)
        opacity = "0.55" if group.group_type == "selected" else "0.38"
        radius = "3.0" if group.group_type == "selected" else "2.4"
        if group.group_type == "real":
            opacity = "0.44"
        hover_key = hover_key_for_group(group)
        escaped_hover_key = html.escape(hover_key)
        for local_index in indices:
            point_index = start + int(local_index)
            title = html.escape(f"{group.label} point {int(local_index)}")
            point_class = "point point-selected" if group.group_type == "selected" else "point point-static"
            if hover_key in best_selected_keys:
                points = svg_star_points(scaled_x[point_index], scaled_y[point_index], outer_radius=4.4, inner_radius=1.9)
                rows.append(
                    f'<polygon class="{point_class} point-best" data-hover-key="{escaped_hover_key}" '
                    f'points="{points}" fill="{group.color}" fill-opacity="{opacity}" stroke="none">'
                    f"<title>{title}</title></polygon>"
                )
                continue
            if hover_key in best_base_keys:
                diamond_points = [
                    (scaled_x[point_index], scaled_y[point_index] - 4.4),
                    (scaled_x[point_index] + 4.4, scaled_y[point_index]),
                    (scaled_x[point_index], scaled_y[point_index] + 4.4),
                    (scaled_x[point_index] - 4.4, scaled_y[point_index]),
                ]
                points = " ".join(f"{x:.2f},{y:.2f}" for x, y in diamond_points)
                rows.append(
                    f'<polygon class="{point_class} point-best-base" data-hover-key="{escaped_hover_key}" '
                    f'points="{points}" fill="{group.color}" fill-opacity="{opacity}" stroke="none">'
                    f"<title>{title}</title></polygon>"
                )
                continue
            if group.group_type == "base":
                rows.append(
                    f'<rect class="{point_class} point-base" data-hover-key="{escaped_hover_key}" '
                    f'x="{scaled_x[point_index] - 3.4:.2f}" y="{scaled_y[point_index] - 3.4:.2f}" width="6.8" height="6.8" '
                    f'rx="1.0" ry="1.0" fill="{group.color}" fill-opacity="{opacity}" stroke="none">'
                    f"<title>{title}</title></rect>"
                )
                continue
            rows.append(
                f'<circle class="{point_class}" data-hover-key="{escaped_hover_key}" '
                f'cx="{scaled_x[point_index]:.2f}" cy="{scaled_y[point_index]:.2f}" '
                f'r="{radius}" fill="{group.color}" fill-opacity="{opacity}" stroke="none">'
                f"<title>{title}</title></circle>"
            )
        start = end
    return rows


def build_legend(groups: list[PopulationGroup]) -> str:
    items = []
    for group in groups:
        hover_key = hover_key_for_group(group)
        items.append(
            f'<span class="legend-item hover-target" data-hover-key="{html.escape(hover_key)}">'
            f'<span class="swatch" style="background:{group.color}"></span>'
            f"{html.escape(group.label)}</span>"
        )
    return "\n".join(items)


def build_performance_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "label",
        "group_type",
        "image_count",
        "centroid_l2_to_real",
        "mmd_rbf_to_real",
        "syn_to_real_nn_mean",
        "real_to_syn_nn_mean",
        "source_objective_value",
        "category",
        "selection_kind",
        "trial_id",
        "run_id",
    ]
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        hover_key = str(row.get("cache_path") or row.get("label") or "")
        row_class = "hover-row hover-target"
        visible_run_id = html.escape(format_float(row.get("run_id")))
        body_rows.append(
            f'<tr class="{row_class}" data-hover-key="{html.escape(hover_key)}">'
            + "".join(
                (
                    f'<td class="hover-target run-id-cell" data-hover-key="{html.escape(hover_key)}">'
                    f"{visible_run_id}"
                    "</td>"
                    if header == "run_id"
                    else f"<td>{html.escape(format_float(row.get(header)))}</td>"
                )
                for header in headers
            )
            + "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_population_html(
    *,
    path: Path,
    groups: list[PopulationGroup],
    coords_by_method: dict[str, tuple[np.ndarray, str]],
    performance_rows: list[dict[str, Any]],
    max_points_per_group: int | None,
) -> None:
    width = 1200
    height = 760
    margin = 72
    selected_rows = [row for row in performance_rows if row.get("group_type") == "selected" and row.get("cache_path")]
    base_rows = [row for row in performance_rows if row.get("group_type") == "base" and row.get("cache_path")]
    best_selected_keys: set[str] = set()
    best_base_keys: set[str] = set()
    if selected_rows:
        best_mmd = min(float(row["mmd_rbf_to_real"]) for row in selected_rows)
        for row in selected_rows:
            if float(row["mmd_rbf_to_real"]) == best_mmd:
                best_selected_keys.add(str(row["cache_path"]))
    if base_rows:
        best_mmd = min(float(row["mmd_rbf_to_real"]) for row in base_rows)
        for row in base_rows:
            if float(row["mmd_rbf_to_real"]) == best_mmd:
                best_base_keys.add(str(row["cache_path"]))

    method_names = list(coords_by_method.keys())
    _METHOD_LABELS: dict[str, str] = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}
    svg_groups_html = ""
    for i, (method_name, (method_coords, _)) in enumerate(coords_by_method.items()):
        circles = "\n".join(svg_circle_rows(
            groups=groups,
            coords=method_coords,
            width=width,
            height=height,
            margin=margin,
            max_points_per_group=max_points_per_group,
            best_selected_keys=best_selected_keys,
            best_base_keys=best_base_keys,
        ))
        active_class = " active" if i == 0 else ""
        svg_groups_html += f'      <g id="view-{method_name}" class="view-group{active_class}">\n{circles}\n      </g>\n'

    legend = build_legend(groups)
    table = build_performance_table(performance_rows)
    total_points = sum(len(group.embeddings) for group in groups)
    generated_at = datetime.now(timezone.utc).isoformat()

    toggle_html = ""
    if len(method_names) > 1:
        btns = "".join(
            f'<button class="view-btn{" active" if i == 0 else ""}" data-view="{name}">'
            f'{html.escape(_METHOD_LABELS.get(name, name.upper()))}</button>'
            for i, name in enumerate(method_names)
        )
        toggle_html = f'  <div class="view-toggle">{btns}</div>\n'

    meta_parts = []
    for i, (name, (_, meta_text)) in enumerate(coords_by_method.items()):
        hide = ' style="display:none"' if i > 0 else ''
        meta_parts.append(
            f'  <p class="meta meta-view" id="meta-{name}"{hide}>'
            f'Generated {html.escape(generated_at)} from {len(groups)} populations and {total_points} embeddings. '
            f'{html.escape(meta_text)}</p>'
        )
    meta_html = "\n".join(meta_parts)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DINOv2 Population Embedding</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0 0 18px;
      color: #475569;
      font-size: 14px;
      line-height: 1.45;
    }}
    .view-toggle {{
      display: flex;
      gap: 8px;
      margin: 0 0 14px;
    }}
    .view-btn {{
      padding: 6px 18px;
      border: 1px solid #94a3b8;
      border-radius: 6px;
      background: #fff;
      color: #334155;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: background 120ms, color 120ms, border-color 120ms;
    }}
    .view-btn:hover {{
      background: #f1f5f9;
    }}
    .view-btn.active {{
      background: #1e40af;
      color: #fff;
      border-color: #1e40af;
    }}
    .plot {{
      background: #ffffff;
      border: 1px solid #dbe3ea;
      border-radius: 8px;
      overflow: auto;
    }}
    svg {{
      display: block;
      width: 100%;
      min-width: 920px;
      height: auto;
    }}
    .view-group {{
      display: none;
    }}
    .view-group.active {{
      display: inline;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 18px 0 26px;
      font-size: 13px;
      color: #334155;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .swatch {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      border: 1px solid rgba(15, 23, 42, 0.22);
      display: inline-block;
    }}
    .point {{
      transition: opacity 120ms ease, transform 120ms ease, filter 120ms ease;
      transform-box: fill-box;
      transform-origin: center;
    }}
    .point.dimmed {{
      opacity: 0.08;
      filter: grayscale(0.15);
    }}
    .point.highlighted {{
      opacity: 1;
      stroke: #0f172a;
      stroke-width: 1.2;
      filter: drop-shadow(0 0 3px rgba(15, 23, 42, 0.18));
    }}
    .hover-target {{
      cursor: pointer;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #dbe3ea;
      border-radius: 8px;
      overflow: hidden;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 8px;
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eaf0f6;
      color: #0f172a;
      font-weight: 700;
      position: sticky;
      top: 0;
    }}
    tbody tr.hovered {{
      background: #eef6ff;
    }}
    tbody tr.hovered td {{
      color: #0f172a;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .table-wrap {{
      overflow: auto;
      margin-top: 12px;
    }}
    .run-id-cell {{
      font-weight: 700;
      color: #0f172a;
    }}
  </style>
</head>
<body>
<main>
  <h1>DINOv2 Population Embedding</h1>
{toggle_html}{meta_html}
  <div class="plot">
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="DINOv2 embedding scatter plot">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
      <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1"/>
      <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1"/>
      <text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="13" fill="#334155">Dim 1</text>
      <text x="22" y="{height / 2:.1f}" text-anchor="middle" font-size="13" fill="#334155" transform="rotate(-90 22 {height / 2:.1f})">Dim 2</text>
{svg_groups_html}    </svg>
  </div>
  <div class="legend">{legend}</div>
  <h2>Run Performance</h2>
  <p class="meta">Rows are sorted by original DINO centroid L2 distance to the real-data centroid.</p>
  <div class="table-wrap">{table}</div>
</main>
<script>
(() => {{
  const scope = document.body;
  const rows = Array.from(scope.querySelectorAll('tbody tr[data-hover-key]'));

  const getActivePoints = () => Array.from(scope.querySelectorAll('.view-group.active [data-hover-key]'));

  const clear = () => {{
    scope.querySelectorAll('[data-hover-key]').forEach(node => node.classList.remove('dimmed', 'highlighted'));
    rows.forEach(row => row.classList.remove('hovered'));
  }};
  const apply = (key) => {{
    if (!key) {{ clear(); return; }}
    getActivePoints().forEach(node => {{
      const match = node.getAttribute('data-hover-key') === key;
      node.classList.toggle('highlighted', match);
      node.classList.toggle('dimmed', !match);
    }});
    rows.forEach(row => {{
      row.classList.toggle('hovered', row.getAttribute('data-hover-key') === key);
    }});
  }};
  const hoverables = Array.from(scope.querySelectorAll('.hover-target'));
  hoverables.forEach(node => {{
    node.addEventListener('mouseenter', () => apply(node.getAttribute('data-hover-key')));
    node.addEventListener('mouseleave', clear);
  }});
  scope.addEventListener('mouseleave', clear);

  const viewBtns = Array.from(scope.querySelectorAll('.view-btn'));
  viewBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      clear();
      const view = btn.dataset.view;
      viewBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      scope.querySelectorAll('.view-group').forEach(g => g.classList.remove('active'));
      document.getElementById('view-' + view).classList.add('active');
      scope.querySelectorAll('.meta-view').forEach(m => {{ m.style.display = 'none'; }});
      document.getElementById('meta-' + view).style.display = '';
    }});
  }});
}})();
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def write_points_csv(path: Path, groups: list[PopulationGroup], coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["label", "group_type", "point_index", "pc1", "pc2"])
        writer.writeheader()
        start = 0
        for group in groups:
            end = start + len(group.embeddings)
            group_coords = coords[start:end]
            for point_index, point in enumerate(group_coords):
                writer.writerow(
                    {
                        "label": group.label,
                        "group_type": group.group_type,
                        "point_index": point_index,
                        "pc1": float(point[0]),
                        "pc2": float(point[1]),
                    }
                )
            start = end


def print_discovery_summary(
    *,
    real_paths: list[Path],
    base_paths: list[Path],
    selected_runs: list[SelectedRun],
    dino: DINOv2Config,
    split_base: bool = False,
) -> None:
    selected_count = sum(len(run.embeddings) for run in selected_runs)
    print(f"DINO model: {dino.repo}:{dino.model_name}")
    print(f"Device: {dino.device}")
    print(f"Real images: {len(real_paths)}")
    base_mode = "split by experiment" if split_base else "merged"
    print(f"Base images: {len(base_paths)} ({base_mode})")
    print(f"Selected runs: {len(selected_runs)}")
    print(f"Selected cached embeddings: {selected_count}")


def main() -> int:
    args = parse_args()
    real_dataset_root, real_annotations_file, dino = resolve_runtime_config(args)
    output_dir = args.output_dir.resolve()
    selected_root = args.selected_root.resolve()
    base_root = args.base_root.resolve()
    score_manifest_root = args.score_manifest_root.resolve()

    real_paths = limited_paths(select_real_image_paths(real_dataset_root, real_annotations_file), args.max_real_images)
    if not real_paths:
        raise ValueError(f"No real images resolved from {real_annotations_file}")
    base_paths = limited_paths(discover_base_rgb_images(base_root), args.max_base_images)

    uncached_run_specs = get_uncached_output_run_specs(selected_root, dino)

    score_index = load_score_index(score_manifest_root)
    selected_runs = discover_selected_runs(
        selected_root=selected_root,
        dino=dino,
        score_index=score_index,
        max_selected_runs=args.max_selected_runs,
    )
    print_discovery_summary(
        real_paths=real_paths,
        base_paths=base_paths,
        selected_runs=selected_runs,
        dino=dino,
        split_base=args.split_base_experiments,
    )
    if uncached_run_specs:
        print(f"Uncached output runs (will embed): {len(uncached_run_specs)}")
    if args.dry_run:
        return 0

    real_manifest = build_embedding_manifest(label="real-data", image_paths=real_paths, dino=dino)
    real_cache_path = cache_path_for(output_dir, "real-data", real_paths, dino)

    if args.split_base_experiments:
        base_parts_meta = filter_base_parts_for_paths(base_root=base_root, image_paths=base_paths)
        base_part_cache_specs = [
            (
                part_name,
                part_paths,
                build_embedding_manifest(label=f"base:{part_name}", image_paths=part_paths, dino=dino),
                cache_path_for(output_dir, f"base:{part_name}", part_paths, dino),
            )
            for part_name, part_paths in base_parts_meta
        ]
        missing_base_caches = [
            cache_path
            for _, _, part_manifest, cache_path in base_part_cache_specs
            if load_matching_cache(cache_path, part_manifest) is None
        ]
        if missing_base_caches:
            merged_base_manifest = build_embedding_manifest(label="base", image_paths=base_paths, dino=dino)
            merged_base_cache_path = cache_path_for(output_dir, "base", base_paths, dino)
            merged_embeddings = load_matching_cache(merged_base_cache_path, merged_base_manifest)
            if merged_embeddings is not None:
                print("Deriving per-experiment caches from merged base embeddings...")
                by_part = split_embeddings_by_part(
                    base_root=base_root,
                    image_paths=base_paths,
                    embeddings=merged_embeddings,
                    base_parts=base_parts_meta,
                )
                for (_, _, part_manifest, part_cache_path), (_, part_embeddings) in zip(base_part_cache_specs, by_part):
                    part_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(part_cache_path, part_embeddings)
                    part_cache_path.with_suffix(".manifest.json").write_text(
                        json.dumps(part_manifest, indent=2)
                    )
                missing_base_caches = []
    else:
        base_manifest = build_embedding_manifest(label="base", image_paths=base_paths, dino=dino)
        base_cache_path = cache_path_for(output_dir, "base", base_paths, dino)
        missing_base_caches = (
            [base_cache_path] if load_matching_cache(base_cache_path, base_manifest) is None else []
        )

    missing_caches = (
        ([real_cache_path] if load_matching_cache(real_cache_path, real_manifest) is None else [])
        + missing_base_caches
    )
    embedder = None
    if missing_caches or uncached_run_specs:
        if args.reuse_real_base_only and missing_caches:
            raise ValueError(
                "Missing real/base embedding cache(s): " + ", ".join(str(p) for p in missing_caches)
            )
        embedder = DINOv2Embedder(
            repo=dino.repo,
            model_name=dino.model_name,
            device=dino.device,
            image_size=dino.image_size,
            resize_size=dino.resize_size,
        )

    if uncached_run_specs and embedder is not None:
        print(f"Computing DINO embeddings for {len(uncached_run_specs)} uncached output runs...")
        for spec_cache_path, spec_image_paths, spec_manifest in uncached_run_specs:
            spec_cache_path.parent.mkdir(parents=True, exist_ok=True)
            embedder.embed_paths(spec_image_paths, batch_size=dino.batch_size, cache_path=spec_cache_path, manifest=spec_manifest)
        selected_runs = discover_selected_runs(
            selected_root=selected_root,
            dino=dino,
            score_index=score_index,
            max_selected_runs=args.max_selected_runs,
        )

    real_embeddings = load_or_compute_image_embeddings(
        label="real-data",
        image_paths=real_paths,
        cache_path=real_cache_path,
        manifest=real_manifest,
        embedder=embedder,
        batch_size=dino.batch_size,
        reuse_only=args.reuse_real_base_only,
    )

    if args.split_base_experiments:
        base_part_embeddings = [
            (
                part_name,
                load_or_compute_image_embeddings(
                    label=f"base:{part_name}",
                    image_paths=part_paths,
                    cache_path=part_cache_path,
                    manifest=part_manifest,
                    embedder=embedder,
                    batch_size=dino.batch_size,
                    reuse_only=args.reuse_real_base_only,
                ),
            )
            for part_name, part_paths, part_manifest, part_cache_path in base_part_cache_specs
        ]
    else:
        base_embeddings = load_or_compute_image_embeddings(
            label="base",
            image_paths=base_paths,
            cache_path=base_cache_path,
            manifest=base_manifest,
            embedder=embedder,
            batch_size=dino.batch_size,
            reuse_only=args.reuse_real_base_only,
        )
        base_part_embeddings = [("base", base_embeddings)]

    groups = build_population_groups(
        real_embeddings=real_embeddings,
        base_parts=base_part_embeddings,
        base_root=base_root,
        selected_runs=selected_runs,
    )
    print("Computing PCA...")
    coords_pca, explained_pca = fit_pca_2d(groups)
    pca_meta = f"PCA variance: PC1 {explained_pca[0] * 100:.2f}%, PC2 {explained_pca[1] * 100:.2f}%."
    coords_by_method: dict[str, tuple[np.ndarray, str]] = {"pca": (coords_pca, pca_meta)}

    if not args.no_tsne:
        print("Computing t-SNE...")
        coords_by_method["tsne"] = (fit_tsne_2d(groups), "t-SNE (perplexity=30, random_state=42).")

    if not args.no_umap:
        print("Computing UMAP...")
        coords_by_method["umap"] = (fit_umap_2d(groups), "UMAP (n_neighbors=15, min_dist=0.1, random_state=42).")

    coords_by_label = group_coordinate_slices(groups, coords_pca)
    performance_rows = compute_performance_rows(
        groups=groups,
        coords_by_label=coords_by_label,
        metric_max_samples=args.metric_max_samples,
    )

    name = args.output_name
    html_path = output_dir / f"{name}.html"
    csv_path = output_dir / f"{name}_performance.csv"
    points_path = output_dir / f"{name}_points.csv"
    write_population_html(
        path=html_path,
        groups=groups,
        coords_by_method=coords_by_method,
        performance_rows=performance_rows,
        max_points_per_group=args.plot_max_points_per_group,
    )
    write_performance_csv(csv_path, performance_rows)
    write_points_csv(points_path, groups, coords_pca)

    print(f"Wrote {html_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {points_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

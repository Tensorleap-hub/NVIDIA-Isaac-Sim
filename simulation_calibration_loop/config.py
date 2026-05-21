"""Configuration loading for the simulation calibration workflow.

This module keeps the user-facing YAML config small and typed while also
expanding higher-level concepts such as search-space themes into the explicit
Isaac parameter paths consumed by the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class DINOv2Config:
    """Runtime settings for the DINOv2 feature extractor."""

    model_name: str = "dinov2_vitb14_reg"
    repo: str = "facebookresearch/dinov2"
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 224
    resize_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IsaacConfig:
    """Settings for launching the external Isaac Sim generator."""

    isaac_sim_path: str = "/opt/IsaacSim"
    script_path: str = "palletjack_sdg/standalone_palletjack_sdg_mean_std.py"
    headless: bool = True
    num_frames_override: int | None = None


@dataclass
class SearchSpaceConfig:
    """Controls which flattened Isaac parameters are exposed to Optuna."""

    themes: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class BasePoolConfig:
    """Persistent candidate-pool settings for staged theme optimization."""

    enabled: bool = False
    state_path: str | None = None
    max_size: int = 60
    elite_size: int = 10
    recent_size: int = 10
    score_weight: float = 0.6
    diversity_weight: float = 0.3
    recency_weight: float = 0.1
    near_duplicate_threshold: float = 0.08
    pin_seeds: bool = True


SEARCH_SPACE_THEMES: dict[str, list[str]] = {
    "environment": [
        "environment.name",
    ],
    "camera": [
        "camera.camera_height_mean",
        "camera.camera_height_std",
        "camera.camera_tilt_mean",
        "camera.camera_tilt_std",
        "camera.camera_yaw_mean",
        "camera.camera_yaw_std",
        "camera.camera_roll_mean",
        "camera.camera_roll_std",
        "camera.fov_mean",
        "camera.fov_std",
    ],
    "camera-color": [
        "image_augmentation.enabled",
        "image_augmentation.brightness_gain_mean",
        "image_augmentation.brightness_gain_std",
        "image_augmentation.contrast_gain_mean",
        "image_augmentation.contrast_gain_std",
        "image_augmentation.gamma_mean",
        "image_augmentation.gamma_std",
        "image_augmentation.color_gain_mean",
        "image_augmentation.color_gain_std",
    ],
    "noise": [
        "camera.motion_blur_strength_mean",
        "camera.motion_blur_strength_std",
        "camera.dataset_noise.mode",
        "camera.dataset_noise.sigma_mean",
        "camera.dataset_noise.sigma_std",
        "camera.dataset_noise.shot_scale_mean",
        "camera.dataset_noise.shot_scale_std",
        "camera.dataset_noise.jpeg_quality_mean",
        "camera.dataset_noise.jpeg_quality_std",
    ],
    "objects": [
        "distractors.clutter_level",
        "distractors.groups.BarelPlastic.occurrence",
        "distractors.groups.BottlePlastic.occurrence",
        "distractors.groups.Bucket.occurrence",
        "distractors.groups.CardBox.occurrence",
        "distractors.groups.CratePlastic.occurrence",
        "distractors.groups.PushCart.occurrence",
        "distractors.groups.RackPile.occurrence",
        "distractors.groups.TrafficSigns.occurrence",
    ],
    "diversity": [
        "distractors.groups.BarelPlastic.diversity",
        "distractors.groups.BottlePlastic.diversity",
        "distractors.groups.Bucket.diversity",
        "distractors.groups.CardBox.diversity",
        "distractors.groups.CratePlastic.diversity",
        "distractors.groups.PushCart.diversity",
        "distractors.groups.RackPile.diversity",
        "distractors.groups.TrafficSigns.diversity",
    ],
    "scene-objects": [
        "palletjacks.count_per_model",
        "palletjacks.position_std",
        "palletjacks.rotation_std",
        "forklifts.count_per_model",
        "forklifts.position_std",
        "forklifts.rotation_std",
        "pallets.count_per_model",
        "pallets.position_std",
        "pallets.rotation_std",
        "pallet_stacks.enabled",
        "pallet_stacks.num_stacks",
        "pallet_stacks.layers",
        "pallet_stacks.position_std",
        "pallet_stacks.rotation_std",
    ],
    "lighting": [
        "lighting.intensity_mean",
        "lighting.intensity_std",
        "lighting.visibility_choices",
    ],
    "materials": [
        "materials.textures",
        "materials.roughness_mean",
        "materials.roughness_std",
        "materials.emissive_intensity_mean",
        "materials.emissive_intensity_std",
    ],
}


@dataclass
class WorkflowConfig:
    """Top-level workflow configuration loaded from `project_config.yaml`."""

    project_name: str
    workspace_dir: str
    s3_best_runs_prefix: str | None
    promoted_baseline_dir: str | None
    baseline_state_path: str | None
    seed_config_dir: str
    real_dataset_root: str
    real_annotations_file: str
    max_iterations: int
    iteration_batch_size: int
    random_seed: int = 42
    top_n_best_trials: int = 3
    mmd_max_samples: int = 1000
    synthetic_rgb_base_dir: str | None = None
    dino: DINOv2Config = field(default_factory=DINOv2Config)
    isaac: IsaacConfig = field(default_factory=IsaacConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    base_pool: BasePoolConfig = field(default_factory=BasePoolConfig)

    def resolve_path(self, candidate: str, *, relative_to_config: Path) -> Path:
        """Resolve a config path relative to the YAML file when needed."""
        path = Path(candidate)
        if path.is_absolute():
            return path
        return (relative_to_config.parent / path).resolve()


def _load_section(data: dict[str, Any] | None, cls: type[Any]) -> Any:
    """Instantiate a dataclass-backed subsection with defaults."""
    section_data = data or {}
    return cls(**section_data)


def _expand_search_space(search_space: SearchSpaceConfig) -> SearchSpaceConfig:
    """Expand theme names into explicit parameter paths and deduplicate them."""
    expanded_include = list(search_space.include)
    for theme in search_space.themes:
        if theme not in SEARCH_SPACE_THEMES:
            valid = ", ".join(sorted(SEARCH_SPACE_THEMES))
            raise ValueError(f"Unknown search-space theme '{theme}'. Valid themes: {valid}")
        expanded_include.extend(SEARCH_SPACE_THEMES[theme])

    deduped_include = list(dict.fromkeys(expanded_include))
    deduped_exclude = list(dict.fromkeys(search_space.exclude))
    return SearchSpaceConfig(
        themes=list(search_space.themes),
        include=deduped_include,
        exclude=deduped_exclude,
    )


def load_workflow_config(config_path: str | Path) -> WorkflowConfig:
    """Load, normalize, and path-resolve the workflow configuration YAML."""
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}

    workflow = WorkflowConfig(
        project_name=raw["project_name"],
        workspace_dir=str(Path(raw["workspace_dir"]).expanduser()),
        s3_best_runs_prefix=str(raw["s3_best_runs_prefix"]).rstrip("/") if raw.get("s3_best_runs_prefix") else None,
        promoted_baseline_dir=str(Path(raw["promoted_baseline_dir"]).expanduser()) if raw.get("promoted_baseline_dir") else None,
        baseline_state_path=str(Path(raw["baseline_state_path"]).expanduser()) if raw.get("baseline_state_path") else None,
        synthetic_rgb_base_dir=str(Path(raw["synthetic_rgb_base_dir"]).expanduser()) if raw.get("synthetic_rgb_base_dir") else None,
        seed_config_dir=str(Path(raw["seed_config_dir"]).expanduser()),
        real_dataset_root=str(Path(raw["real_dataset_root"]).expanduser()),
        real_annotations_file=str(Path(raw["real_annotations_file"]).expanduser()),
        max_iterations=int(raw["max_iterations"]),
        iteration_batch_size=int(raw["iteration_batch_size"]),
        random_seed=int(raw.get("random_seed", 42)),
        top_n_best_trials=int(raw.get("top_n_best_trials", 3)),
        mmd_max_samples=int(raw.get("mmd_max_samples", 1000)),
        dino=_load_section(raw.get("dino"), DINOv2Config),
        isaac=_load_section(raw.get("isaac"), IsaacConfig),
        search_space=_load_section(raw.get("search_space"), SearchSpaceConfig),
        base_pool=_load_section(raw.get("base_pool"), BasePoolConfig),
    )
    workflow.search_space = _expand_search_space(workflow.search_space)

    workflow.workspace_dir = str(workflow.resolve_path(workflow.workspace_dir, relative_to_config=config_path))
    if workflow.promoted_baseline_dir is not None:
        workflow.promoted_baseline_dir = str(
            workflow.resolve_path(workflow.promoted_baseline_dir, relative_to_config=config_path)
        )
    if workflow.synthetic_rgb_base_dir is not None:
        workflow.synthetic_rgb_base_dir = str(
            workflow.resolve_path(workflow.synthetic_rgb_base_dir, relative_to_config=config_path)
        )
    if workflow.baseline_state_path is not None:
        workflow.baseline_state_path = str(
            workflow.resolve_path(workflow.baseline_state_path, relative_to_config=config_path)
        )
    if workflow.base_pool.state_path is not None:
        workflow.base_pool.state_path = str(
            workflow.resolve_path(workflow.base_pool.state_path, relative_to_config=config_path)
        )
    workflow.seed_config_dir = str(workflow.resolve_path(workflow.seed_config_dir, relative_to_config=config_path))
    workflow.real_dataset_root = str(workflow.resolve_path(workflow.real_dataset_root, relative_to_config=config_path))
    workflow.real_annotations_file = str(workflow.resolve_path(workflow.real_annotations_file, relative_to_config=config_path))
    workflow.isaac.script_path = str(
        workflow.resolve_path(workflow.isaac.script_path, relative_to_config=config_path)
    )
    return workflow

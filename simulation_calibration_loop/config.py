from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class DINOv2Config:
    model_name: str = "dinov2_vitb14_reg"
    repo: str = "facebookresearch/dinov2"
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 224
    resize_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IsaacConfig:
    isaac_sim_path: str = "/opt/IsaacSim"
    script_path: str = "palletjack_sdg/standalone_palletjack_sdg_mean_std.py"
    headless: bool = True
    num_frames_override: int | None = None


@dataclass
class SearchSpaceConfig:
    themes: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


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
    "noise": [
        "camera.motion_blur_strength_mean",
        "camera.motion_blur_strength_std",
        "camera.dataset_noise.mode",
        "camera.dataset_noise.sigma_mean",
        "camera.dataset_noise.sigma_std",
        "camera.dataset_noise.jpeg_quality_mean",
        "camera.dataset_noise.jpeg_quality_std",
    ],
    "objects": [
        "distractors.clutter_level",
        "palletjacks.count_per_model",
        "palletjacks.position_std",
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
    project_name: str
    workspace_dir: str
    seed_config_dir: str
    real_dataset_root: str
    real_annotations_file: str
    max_iterations: int
    iteration_batch_size: int
    random_seed: int = 42
    top_n_best_trials: int = 3
    mmd_max_samples: int = 1000
    dino: DINOv2Config = field(default_factory=DINOv2Config)
    isaac: IsaacConfig = field(default_factory=IsaacConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)

    def resolve_path(self, candidate: str, *, relative_to_config: Path) -> Path:
        path = Path(candidate)
        if path.is_absolute():
            return path
        return (relative_to_config.parent / path).resolve()


def _load_section(data: dict[str, Any] | None, cls: type[Any]) -> Any:
    section_data = data or {}
    return cls(**section_data)


def _expand_search_space(search_space: SearchSpaceConfig) -> SearchSpaceConfig:
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
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}

    workflow = WorkflowConfig(
        project_name=raw["project_name"],
        workspace_dir=str(Path(raw["workspace_dir"]).expanduser()),
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
    )
    workflow.search_space = _expand_search_space(workflow.search_space)

    workflow.workspace_dir = str(workflow.resolve_path(workflow.workspace_dir, relative_to_config=config_path))
    workflow.seed_config_dir = str(workflow.resolve_path(workflow.seed_config_dir, relative_to_config=config_path))
    workflow.real_dataset_root = str(workflow.resolve_path(workflow.real_dataset_root, relative_to_config=config_path))
    workflow.real_annotations_file = str(workflow.resolve_path(workflow.real_annotations_file, relative_to_config=config_path))
    workflow.isaac.script_path = str(
        workflow.resolve_path(workflow.isaac.script_path, relative_to_config=config_path)
    )
    return workflow

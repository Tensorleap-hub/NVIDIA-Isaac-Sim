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
class RFDETREmbedderConfig:
    """Settings for using a fine-tuned RF-DETR backbone as the feature extractor."""

    checkpoint_path: str = ""
    num_classes: int = 3
    layer_index: int = 3
    batch_size: int = 16
    image_size: int = 224
    resize_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class IsaacConfig:
    """Settings for launching the external Isaac Sim generator."""

    isaac_sim_path: str = "/opt/IsaacSim"
    script_path: str = "palletjack_sdg/standalone_palletjack_trajectory_sdg.py"
    headless: bool = True
    # Per-seed frame count. Each trial is generated once per entry in
    # `eval_seeds`, each Isaac run using this many frames, and the resulting
    # images are pooled before embedding — so total frames = num_frames_override
    # * len(eval_seeds).
    num_frames_override: int | None = None
    # Seeds used to evaluate a single candidate config. Running the same YAML
    # across several seeds re-rolls the scene layout + trajectory each time, so
    # the pooled embedding samples the config's DISTRIBUTION (many layouts)
    # rather than one lucky/unlucky realization. A FIXED seed set across all
    # trials makes trial-to-trial MMD differences reflect the config, not seed
    # luck. Default [0] reproduces the historical single-run behavior.
    eval_seeds: list[int] = field(default_factory=lambda: [0])
    # Episode mode: render ALL eval_seeds in ONE Isaac session via the SDG's
    # --seeds/--out_root flags (the scene re-rolls in-process per seed and the
    # SDG does its own seed+k*1000 layout retries). Cuts a trial's render cost
    # from len(eval_seeds) Isaac boots to one. Per-seed outputs land at
    # <output_dir>/<yaml-stem>_seed<S>/ instead of <output_dir>/seed_<S>/.
    episode_mode: bool = False
    # Capture mode forwarded to the SDG (episode mode only). None keeps the
    # trial config's own default (trajectory); "random" gives every frame an
    # independent freespace pose — with num_frames_override=1 each image is a
    # fresh scene re-roll (maximally decorrelated stills for the MMD objective).
    capture_mode: str | None = None


@dataclass
class TPESamplerConfig:
    """Optuna TPE sampler settings exposed to per-theme project configs."""

    multivariate: bool = True
    group: bool = False
    constant_liar: bool = True
    # Trials sampled randomly before TPE activates. The optimizer default is
    # max(50, 3*params) — with 10-iteration themes (40 trials) that leaves the
    # whole theme on random search. Pool priming injects seed anchors +
    # prior-theme trials as completed observations, so a small value is safe.
    # None keeps the optimizer default.
    n_startup_trials: int | None = None


@dataclass
class SearchSpaceConfig:
    """Controls which flattened Isaac parameters are exposed to Optuna.

    `bounds` declares the explicit Optuna search range per flattened parameter
    path. Numeric paths map to `[min, max]`; categorical paths map to a list of
    allowed values. Every path that survives include/exclude filtering must have
    an entry here — the controller raises if any are missing.
    """

    themes: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    bounds: dict[str, list[Any]] = field(default_factory=dict)


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


# Themes for the trajectory-SDG search space. Paths must match those emitted by
# infer_parameter_schema() over the base_v1 seeds. Deferred themes
# (per-distractor occurrence/diversity, placement std, environment.name, motion
# blur, image augmentation, dataset noise) are intentionally absent — see
# optuna_search_trajectory.md §3.5 for the reintroduction order.
SEARCH_SPACE_THEMES: dict[str, list[str]] = {
    # Tune FIRST in a round: the environment is the biggest single domain-gap
    # lever, and the promoted-baseline flow means whatever is tuned first becomes
    # the substrate every later theme conditions on. base_v4 already exercises
    # all four warehouses in trajectory mode, so switching between them is
    # validated (the per-env roam bounds differ by <=1m, which the occupancy
    # planner tolerates).
    "traj-environment": [
        "environment.name",
    ],
    "traj-camera-intrinsics": [
        "cameras.ego.fov_mean",
        "cameras.ego.fov_std",
        "cameras.ego.f_stop",
        "cameras.ego.focus_distance_m",
    ],
    "traj-camera-mount": [
        "cameras.ego.height_m",
        "cameras.ego.pitch_deg",
        "cameras.ego.roll_deg",
    ],
    "traj-camera-jitter": [
        "cameras.ego.pitch_jitter.amp_deg",
        "cameras.ego.pitch_jitter.hz",
        "cameras.ego.roll_jitter.amp_deg",
        "cameras.ego.roll_jitter.hz",
        "cameras.ego.yaw_jitter.amp_deg",
        "cameras.ego.yaw_jitter.hz",
        "cameras.ego.lateral_jitter.amp_m",
        "cameras.ego.lateral_jitter.hz",
        "cameras.ego.vertical_jitter.amp_m",
        "cameras.ego.vertical_jitter.hz",
    ],
    "traj-agent": [
        "agent.speed_mps",
        "agent.turn_rate_dps",
    ],
    "traj-scene": [
        "palletjacks.count_per_model",
        # Body-color repaint (restored random-frame knob): mean = the color,
        # std = allowed per-episode randomness. Indexed lists — bounds are
        # declared per component (color_mean[0..2] / color_std[0..2]).
        "palletjacks.color_mean",
        "palletjacks.color_std",
        "forklifts.count_per_model",
        "pallets.count_per_model",
        "distractors.clutter_level",
        "lighting.intensity_mean",
        "lighting.intensity_std",
        # Light COLOR (per-channel RGB, like palletjacks.color_*). This is the
        # dominant v4b<->mixedbase_50 domain-gap lever (colorfulness KS D=0.27):
        # mb50's neon/colored halls. mean = the light tint, std = allowed
        # per-episode color randomness. Indexed lists — bounds declared per
        # component (color_mean[0..2] / color_std[0..2]). Seeds carry
        # [0.5,0.5,0.5] / [0.288675 x3] (base_v1) plus the exp23-32 hand-set
        # colored-lighting configs, all within the declared bounds.
        "lighting.color_mean",
        "lighting.color_std",
        "materials.roughness_mean",
        "materials.roughness_std",
        "materials.textures",
    ],
    "traj-characters": [
        "characters.enabled",
        "characters.count",
    ],
    # Per-distractor-group frequency. `occurrence` directly shapes what the OD
    # model learns to ignore / suppress false positives on, so it's the first
    # object theme to reintroduce (occurrence=0 disables a group, subsuming the
    # old `.use` boolean). The 8 groups match those present in every base_v4
    # seed (schema inference intersects across seeds).
    "traj-distractor-occurrence": [
        "distractors.groups.BarelPlastic.occurrence",
        "distractors.groups.BottlePlastic.occurrence",
        "distractors.groups.Bucket.occurrence",
        "distractors.groups.CardBox.occurrence",
        "distractors.groups.CratePlastic.occurrence",
        "distractors.groups.PushCart.occurrence",
        "distractors.groups.RackPile.occurrence",
        "distractors.groups.TrafficSigns.occurrence",
    ],
    # Asset variety within each distractor group. Layer in AFTER occurrence has
    # converged — it's naturally downstream of how often each group appears.
    "traj-distractor-diversity": [
        "distractors.groups.BarelPlastic.diversity",
        "distractors.groups.BottlePlastic.diversity",
        "distractors.groups.Bucket.diversity",
        "distractors.groups.CardBox.diversity",
        "distractors.groups.CratePlastic.diversity",
        "distractors.groups.PushCart.diversity",
        "distractors.groups.RackPile.diversity",
        "distractors.groups.TrafficSigns.diversity",
    ],
    # Stage 7: exploration-boundary optimization. Searches WHERE the camera is
    # allowed to roam via a constrained center+extent reparameterization
    # (fractions of the env envelope), NOT the four raw trajectory.bounds_xy
    # floats (which admit invalid boxes and confound size with position). The
    # Isaac script's apply_roam_bounds() derives trajectory.bounds_xy from these
    # before occupancy planning — clamped inside the envelope, env-relative, with
    # a min_path_m feasibility floor so a small box can't wedge the planner. Add
    # AFTER traj-environment has promoted a baseline env (the envelope is that
    # env's tuned bounds_xy). See optuna_search_trajectory.md Stage 7.
    "traj-exploration-bounds": [
        "trajectory.roam.center_x_frac",
        "trajectory.roam.center_y_frac",
        "trajectory.roam.width_frac",
        "trajectory.roam.height_frac",
    ],
    # Cross-cutting shortlist of the highest-fANOVA-importance knobs distilled
    # from the first full theme-rounds run (DINOv2, 2026-07-09→11; per-study
    # `param_importances` in each workspace state.json). Recreates the old
    # `top20_important` idea for the trajectory pipeline: instead of cycling
    # single-axis themes, search the ~18 knobs that actually moved MMD across
    # camera / scene / occurrence / agent in ONE joint space, seeded from the
    # weekend run's promoted best (0.32604) + reused 60-obs pool. Selection =
    # "broad union": the params that both ranked high in importance AND drove
    # the three promotions, plus the next tier (2nd agent knob, roll/fov_std,
    # roughness_std, two more distractor groups). `environment.name` is NOT
    # here — it is added via the `traj-environment` theme in the project config
    # so it is searched jointly (it dominates fANOVA in every study).
    "traj-top-important": [
        # camera (drove the round-1 promotion)
        "cameras.ego.fov_mean",
        "cameras.ego.fov_std",
        "cameras.ego.f_stop",
        "cameras.ego.focus_distance_m",
        "cameras.ego.height_m",
        "cameras.ego.pitch_deg",
        "cameras.ego.roll_deg",
        # scene (drove the round-1 scene promotion)
        "lighting.intensity_mean",
        "materials.roughness_mean",
        "materials.roughness_std",
        "distractors.clutter_level",
        # distractor occurrence / composition (drove the round-3 promotion)
        "distractors.groups.BottlePlastic.occurrence",
        "distractors.groups.TrafficSigns.occurrence",
        "distractors.groups.RackPile.occurrence",
        "distractors.groups.CratePlastic.occurrence",
        "distractors.groups.BarelPlastic.occurrence",
        # agent (top fANOVA sensitivity; never promoted, kept per broad-union)
        "agent.turn_rate_dps",
        "agent.speed_mps",
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
    # Size of the best_top{k}.yaml / best_top{k}_diverse.yaml exports written
    # next to the promoted best.yaml after every iteration.
    top_k_export: int = 10
    # How many of the best trials are eligible when picking the diverse top-k.
    # The diverse set trades a little objective quality for parameter spread,
    # so it draws from a pool larger than k itself. Only used when
    # diverse_objective_threshold is null.
    diverse_candidate_pool: int = 30
    # Quality gate for the diverse top-k candidate pool: every unique trial
    # with objective MMD <= this value is a candidate (the unconditional
    # top-30 pool let greedy max-min pick trials with objectives up to 0.72
    # on the 20260710 weekend run). Set to null to restore the old
    # pool-size-based behavior. 0.44 was chosen against the DINOv2
    # vitb14_reg objective scale (best runs ~0.33-0.40).
    diverse_objective_threshold: float | None = 0.44
    mmd_max_samples: int = 1000
    synthetic_rgb_base_dir: str | None = None
    embedder_backend: str = "dinov2"
    dino: DINOv2Config = field(default_factory=DINOv2Config)
    rfdetr_embedder: RFDETREmbedderConfig = field(default_factory=RFDETREmbedderConfig)
    isaac: IsaacConfig = field(default_factory=IsaacConfig)
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    base_pool: BasePoolConfig = field(default_factory=BasePoolConfig)
    tpe_sampler: TPESamplerConfig = field(default_factory=TPESamplerConfig)

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
        bounds=dict(search_space.bounds),
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
        top_k_export=int(raw.get("top_k_export", 10)),
        diverse_candidate_pool=int(raw.get("diverse_candidate_pool", 30)),
        diverse_objective_threshold=(
            float(raw["diverse_objective_threshold"])
            if raw.get("diverse_objective_threshold") is not None
            else (None if "diverse_objective_threshold" in raw else 0.44)
        ),
        mmd_max_samples=int(raw.get("mmd_max_samples", 1000)),
        embedder_backend=str(raw.get("embedder_backend", "dinov2")),
        dino=_load_section(raw.get("dino"), DINOv2Config),
        rfdetr_embedder=_load_section(raw.get("rfdetr_embedder"), RFDETREmbedderConfig),
        isaac=_load_section(raw.get("isaac"), IsaacConfig),
        search_space=_load_section(raw.get("search_space"), SearchSpaceConfig),
        base_pool=_load_section(raw.get("base_pool"), BasePoolConfig),
        tpe_sampler=_load_section(raw.get("tpe_sampler"), TPESamplerConfig),
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
    if workflow.rfdetr_embedder.checkpoint_path:
        workflow.rfdetr_embedder.checkpoint_path = str(
            workflow.resolve_path(workflow.rfdetr_embedder.checkpoint_path, relative_to_config=config_path)
        )
    return workflow

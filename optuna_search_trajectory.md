# Optuna Search Over Trajectory SDG

## Purpose

The `trajectory` branch is dedicated to episode-based trajectory generation
(`standalone_palletjack_trajectory_sdg.py`). The random-frame generator lives
on other branches and does not need to coexist here. This document scopes the
work needed to run `simulation_calibration_loop` (Isaac → DINOv2 → Optuna)
against the trajectory pipeline, starting with a hard purge of every
random-frame artifact that would otherwise leak into schema inference, waste
Optuna trials on dead knobs, or fill the disk.

## Readiness Snapshot

### What already works

- **Trajectory script CLI is loop-compatible** — `--config`, `--headless`,
  `--data_dir`, `--num_frames` match `data.run_isaac_generation`, so the
  loop launches trajectory runs by changing only `isaac.script_path`.
- **Image discovery contract is honored** — trajectory writes `Camera/rgb/*.png`;
  the `Camera_chase/` leak was fixed in `controller.py:652` at stage 3.
- **Bounds are now declared in the project YAML** (commit `4dc56fe`), so
  ranges for trajectory paths can be added without touching schema inference.
- **Loop plumbing is generic** — `controller.py`, `parameter_schema.py`,
  `data.py`, and the `SearchSpaceConfig` dataclass in `config.py` are
  parameter-agnostic. Only `SEARCH_SPACE_THEMES` (config.py:95–237) hardcodes
  the old flat schema.

### Gaps blocking Optuna on trajectory

1. **`SEARCH_SPACE_THEMES` has zero trajectory paths** — every theme still
   references `camera.*`, `distractors.*`, `materials.*` etc.
2. **No trajectory seed-config directory** — `controller.py:69` requires
   ≥2 YAMLs whose intersection defines the flattened schema. `sdg_config_stage6.yaml`
   is a lone file and inherits `sdg_config_mean_std.yaml` full of dead knobs.
3. **No `project_config_trajectory.yaml`** — needs `isaac.script_path` →
   trajectory script, `seed_config_dir` → new dir, and explicit
   `search_space.bounds` for every included path (controller raises at
   `controller.py:772–777` on missing bounds).
4. **Legacy random-frame params leak through inheritance.**
   `sdg_config_stage6.yaml extends sdg_config_mean_std.yaml`, so schema
   inference will surface `camera.camera_height_mean/std`,
   `camera.position_mean/std`, `camera.motion_blur_strength_*`, and many
   others that the trajectory script mostly ignores. Any of these left in
   the search space burns budget on no-ops.
5. **Knowingly no-op knobs must be excluded** until later stages land:
   - `cameras.ego.shutter_close_fraction` — wired but RTX doesn't integrate
     camera-xform time samples (stage 6.1 outcome #5). Optimizing before
     stage 7.5 is meaningless.
   - `cameras.ego.fisheye.*` — post-render only; DINOv2 embeds `Camera/rgb/`
     which is untouched.
   - `capture.video` — CosmosWriter adds ~5 MP4s per trial; irrelevant for
     scoring.
   - Chase-camera fields — chase output lives under `Camera_chase/` and is
     ignored by discovery.
6. **Trial cost is higher than random-frame.** Trajectory steps physics +
   Replicator per frame. Plan `num_frames_override` and
   `iteration_batch_size` accordingly.

## Embedder: use the trained RF-DETR backbone, not stock DINOv2

The loop already supports two embedder backends via `WorkflowConfig.embedder_backend`
(`config.py:258`): stock `dinov2` and `rfdetr`. RF-DETR's backbone is
`WindowedDinov2WithRegistersBackbone` (`data.py:150–151`) — it *is* a DINOv2,
just fine-tuned on the palletjack/forklift/pallet OD task. Using the trained
detector's backbone as the scoring embedder makes the whole optimization loop
coherent:

- **Signal alignment.** MMD between synthetic and real embeddings measures
  distance in the feature space the OD model actually reads from. Stock DINOv2
  distances reflect general visual similarity, which is only a proxy.
- **Story continuity.** SDG → OD training → OD backbone → SDG-scoring closes
  the loop end-to-end. Improvements ranked by this embedder should transfer
  to OD accuracy more directly than DINOv2-ranked improvements.
- **Wiring already exists.** `RFDETREmbedder` (`data.py:119–170`) loads a
  checkpoint, unwraps to the backbone, and GAPs the selected scale to a
  384-dim embedding. Set `embedder_backend: rfdetr` and
  `rfdetr_embedder.checkpoint_path: <path>` in the project YAML.

Action:

- Pick the RF-DETR checkpoint we want to canonize for this branch (the one
  trained on the current real+synthetic mix). Copy it into the repo or
  reference by absolute path from the project YAML.
- Set `embedder_backend: rfdetr` in `project_config_trajectory.yaml` at
  Stage 3.
- Keep the DINOv2 path installed as a fallback for debugging, but do not
  optimize against it.

## Implementation Stages

### Stage 1 — Purge the branch of random-frame artifacts

**Goal:** leave this branch with a single, trajectory-only SDG surface and a
loop that has no lingering old themes / project configs / workspaces. Every
delete below is reversible via `git` on other branches; local run data is
already backed up on S3.

The purge is *careful* because the trajectory script still inherits from
`sdg_config_mean_std.yaml`. Step 1.1 breaks that inheritance before removing
the base file.

#### 1.1 Consolidate the trajectory config

- Inline every field the trajectory script actually reads from
  `sdg_config_mean_std.yaml` into `sdg_config_stage6.yaml` (or a renamed
  `sdg_config_trajectory.yaml`) so `extends:` can be dropped. Audit against
  `standalone_palletjack_trajectory_sdg.py` — keys used include
  `render.width/height`, `environment_urls.*`, `palletjacks.*`, `forklifts.*`,
  `pallets.*`, `pallet_stacks.*`, `distractors.*`, `materials.*`, `lighting.*`,
  `characters.*`, and the `cameras.ego.*`/`agent.*`/`trajectory.*`/`capture.*`
  blocks already stated explicitly.
- Rename the consolidated file to `palletjack_sdg/sdg_config_trajectory.yaml`
  (the current file with that name is the stale stage-5 version — overwrite it).
- Confirm the trajectory script runs from the standalone config with
  `--num_frames 3`.

#### 1.2 Delete random-frame SDG files under `palletjack_sdg/`

Scripts:
- `standalone_palletjack_sdg.py`
- `standalone_palletjack_sdg_mean_std.py`
- `palletjack_datagen.sh`
- `list_isaac_props.sh`
- `run_experiments.sh`
- `run_experiments_mean_std.sh`

Stale trajectory shells (superseded by `run_trajectory_stage6.sh`):
- `run_trajectory_stage1.sh`
- `run_trajectory_stage3.sh`
- `run_trajectory_stage5.sh`

SDG configs (blur A/B, panoramic, DoF/focal/fisheye/posjit stage-6 sweeps,
per-seed stage-6 variants, ad-hoc tests):
- `sdg_config.yaml`
- `sdg_config_mean_std.yaml`
- `sdg_config_blur_ab.yaml`, `sdg_config_blur_ab_OFF.yaml`, `sdg_config_blur_ab_ON.yaml`
- `sdg_config_blur_clean.yaml`, `sdg_config_blur_clean_OFF.yaml`, `sdg_config_blur_clean_ON.yaml`
- `sdg_config_panoramic.yaml`
- `sdg_config_stage6_blur_off.yaml`, `sdg_config_stage6_blur_on.yaml`
- `sdg_config_stage6_dof.yaml`, `sdg_config_stage6_dof2.yaml`
- `sdg_config_stage6_fish.yaml`, `sdg_config_stage6_focal.yaml`, `sdg_config_stage6_posjit.yaml`
- `sdg_config_stage6_seed_1.yaml` … `sdg_config_stage6_seed_5.yaml`
- `test_ego_only.yaml`, `test_fov_seed42.yaml`, `test_yaw_fix.yaml`

Keep on the branch:
- `standalone_palletjack_trajectory_sdg.py`
- `sdg_config_trajectory.yaml` (new consolidated form)
- `sdg_config_stage6.yaml` if we choose to keep it as the trajectory default
  instead of renaming — pick one.
- `run_trajectory_stage6.sh`
- `utils/`
- `requirements.txt`

#### 1.3 Prune `palletjack_sdg/experiments/`

Delete:
- `ec2-loop/`
- `experiment_first_order/`
- `experiment_second_order/`
- `test_exp/`
- `generate_configs.py`
- `generate_configs_mean_std.py`
- `summarize_yaml_stats.py`

Keep temporarily as reference for authoring the trajectory seed configs,
then delete once stage 2 lands:
- `experiment_mean_std/base_v2/`

Everything else under `experiment_mean_std/` (all `iter-*`, `theme`,
`first_order`, `base_v2_bounded`, `base_v3/v4`, `test_*`, `README.md`,
`1st_optuna_rounds_best`) deletes with the parent purge.

#### 1.4 Prune `simulation_calibration_loop/`

Delete every project YAML tied to the old flat schema (all reference deleted
seed dirs / themes):
- `project_config.yaml`
- `project_config_all.yaml`
- `project_config_camera.yaml`
- `project_config_camera_color.yaml`
- `project_config_camera_color_objects.yaml`
- `project_config_camera_noise_light.yaml`
- `project_config_diversity.yaml`
- `project_config_environment.yaml`
- `project_config_lighting.yaml`
- `project_config_materials.yaml`
- `project_config_materials_objects.yaml`
- `project_config_noise.yaml`
- `project_config_noise_lighting.yaml`
- `project_config_objects.yaml`
- `project_config_run2.yaml`
- `project_config_scene_object.yaml`
- `project_config_second_order.yaml`
- `project_config_second_order_groups.yaml`
- `project_config_to_30.yaml`
- `test_isaac_small_loop.yaml`
- `theme_rounds.yaml`

Delete the runner that depends on `theme_rounds.yaml`:
- `run_theme_rounds.py`

Delete the smoke tests that hardcode the old `include/exclude` semantics
against old seed dirs (rewrite in stage 4 if we want a trajectory smoke test):
- `test_isaac_small_loop.py`
- `smoke_test_dinov2.py`

Keep and audit in step 1.5:
- `__init__.py`, `base_pool.py`, `config.py`, `controller.py`, `data.py`,
  `parameter_schema.py`, `ui.py`
- `collect_top_runs.py`, `upload_top_runs_to_s3.py`, `visualize_population.py`
  — utilities, not path-specific; keep unless audit shows dependence on
  removed configs.

#### 1.5 Deep audit of the loop code for random-frame assumptions

Read top-to-bottom with an eye for hardcoded old paths, discovery
assumptions, or theme names:

- `config.py:95–237` — **replace `SEARCH_SPACE_THEMES` entirely** with
  trajectory themes (stage 3 below). Every existing entry references a
  deleted knob.
- `data.py` — confirm `discover_generated_images` still points at
  `Camera/rgb`, confirm `run_isaac_generation` builds the correct Isaac CLI
  for the trajectory script, confirm any KITTI-style layout expectations are
  dropped.
- `controller.py` — confirm no hardcoded references to old themes, no
  filename patterns like `standalone_palletjack_sdg_mean_std.py`.
- `parameter_schema.py` — generic; expected clean.
- `base_pool.py` — check for hardcoded param paths used for de-duplication.
- `ui.py` — check the log/status strings.
- Utility scripts (`collect_top_runs.py`, `upload_top_runs_to_s3.py`,
  `visualize_population.py`) — flag any hardcoded workspace dirs, S3
  prefixes, or theme names; leave them as TODOs to fix when we use them.

Track findings inline with edits or with `# TODO(trajectory-optuna): ...`
comments if the fix is bigger than the audit pass.

#### 1.6 Delete local run data (backed up on S3)

`palletjack_sdg/palletjack_data/` (~28 GB) and the loop workspaces
(~225 GB total):
- `simulation_calibration_loop/may_rounds_freshstart/`
- `simulation_calibration_loop/may_rounds_insh/`
- `simulation_calibration_loop/may_rounds_ok/`
- `simulation_calibration_loop/may_rounds_rfdetr/`
- `simulation_calibration_loop/promoted_baseline_rfdetr/`
- `simulation_calibration_loop/promoted_baseline_theme_rounds/`
- `simulation_calibration_loop/promoted_baseline_theme_rounds_2/`
- `simulation_calibration_loop/promoted_saved/`
- `simulation_calibration_loop/promoted_saved_v2/`
- `simulation_calibration_loop/round_workspaces/`
- `simulation_calibration_loop/round_workspaces_run2/`
- `simulation_calibration_loop/small_loop_workspace/`
- `simulation_calibration_loop/workspace_top_30_groups/`

Confirm each is either present on S3 (per user) or intentionally ephemeral
before `rm -rf`.

#### 1.7 Commit and push the purge

Single commit titled something like `trajectory-branch: purge random-frame
SDG + old loop artifacts`. Describe scope in the body so the diff is
auditable. The commit should compile-check (import the loop package, load
the surviving trajectory YAML, dry-run schema inference against a single
placeholder seed) even though schema inference will refuse to run until
stage 2 adds ≥2 seed YAMLs.

**Exit criteria for stage 1:**
- Trajectory YAML runs standalone with no `extends:`.
- No file in the branch references `standalone_palletjack_sdg*` or
  `sdg_config_mean_std.yaml`.
- `simulation_calibration_loop/` contains only generic code + this doc-set;
  no project YAMLs, no workspaces.
- `du -sh palletjack_sdg/palletjack_data simulation_calibration_loop/` is
  under a few hundred MB.

### Stage 2 — Trajectory seed configs

**Goal:** author 5–10 seed YAMLs under
`palletjack_sdg/experiments/trajectory/base_v1/` that together cover the
trajectory knobs we want Optuna to search. `experiment_mean_std/base_v2/`
is the structural template for how much per-seed variation is enough;
delete it once these are in place.

Design principles:
- Every seed uses the same top-level structure (no `extends:`).
- Every path we intend to include in the search space must appear in
  every seed (schema inference intersects across YAMLs).
- Seeds should differ meaningfully in the values of each optimizable path
  so Optuna's initial pool has variance.
- `capture.video: false`, `cameras.chase.enabled: false`, `fisheye.enabled:
  false`, `shutter_close_fraction: 0.0` across all seeds (constant → not
  in the search space unless we add per-seed variance later).

Suggested initial optimizable knobs (all present in `sdg_config_stage6.yaml`):
- **Camera intrinsics:** `cameras.ego.fov_mean`, `cameras.ego.fov_std`,
  `cameras.ego.focal_length_mm` (or `fov_mean` only, pick one),
  `cameras.ego.f_stop`, `cameras.ego.focus_distance_m`
- **Camera mount:** `cameras.ego.height_m`, `cameras.ego.pitch_deg`,
  `cameras.ego.roll_deg`
- **Camera jitter:** `cameras.ego.pitch_jitter.amp_deg`,
  `cameras.ego.pitch_jitter.hz`, same for `roll_jitter`, `lateral_jitter`,
  `vertical_jitter`
- **Agent:** `agent.speed_mps`, `agent.turn_rate_dps`
- **Scene composition (retained from mean_std, still consumed):**
  `palletjacks.count_per_model`, `forklifts.count_per_model`,
  `pallets.count_per_model`, `distractors.clutter_level`,
  `lighting.intensity_mean`, `lighting.intensity_std`, `materials.roughness_mean`,
  `materials.roughness_std`, `materials.textures`
- **Characters (stage 7):** `characters.count`, `characters.enabled` — decide
  whether to search or pin.

**Exit criteria:** `parameter_schema.load_yaml_configs("…/base_v1/")` +
`infer_parameter_schema` returns the expected paths and value kinds;
`experiment_mean_std/base_v2/` is deleted.

### Stage 3 — Search-space themes + bounds

**Goal:** rewrite `SEARCH_SPACE_THEMES` in `config.py` and author
`project_config_trajectory.yaml` with explicit bounds for every included path.

- Replace `SEARCH_SPACE_THEMES` with a small, curated set:
  - `traj-camera-intrinsics`: fov, focal, f_stop, focus_distance
  - `traj-camera-mount`: height, pitch, roll
  - `traj-camera-jitter`: all four `*_jitter.*` blocks
  - `traj-agent`: speed_mps, turn_rate_dps
  - `traj-scene`: palletjack/forklift/pallet counts, clutter_level,
    lighting mean/std, materials roughness mean/std, materials.textures
  - `traj-characters`: count, enabled (optional; can be off for first run)
- Author `project_config_trajectory.yaml`:
  - `isaac.script_path: ../palletjack_sdg/standalone_palletjack_trajectory_sdg.py`
  - `seed_config_dir: ../palletjack_sdg/experiments/trajectory/base_v1`
  - `search_space.themes: [traj-camera-intrinsics, traj-camera-mount, ...]`
  - `search_space.exclude: [run.data_dir, run.num_frames, run.headless]`
    plus every knowingly no-op knob from the readiness gaps list
  - `search_space.bounds: {...}` — one entry per surviving path, min/max
    for numeric, list for categorical (materials.textures)
  - `embedder_backend: rfdetr` +
    `rfdetr_embedder.checkpoint_path: <path-to-trained-checkpoint>` so
    scoring uses the trained OD backbone (see the Embedder section above).

**Exit criteria:** `controller._build_param_bounds_from_config()` succeeds
without raising for any surviving path.

### Stage 4 — Smoke test

**Goal:** end-to-end dry run with tiny numbers.

- `iteration_batch_size: 1`, `num_frames_override: 5`, `max_iterations: 2`.
- Confirm: Isaac launches from the trajectory script, `Camera/rgb/`
  populates, embeddings compute, state store writes, second iteration
  suggests different params than the first.
- Inspect the generated per-trial YAMLs — every trajectory path should be
  populated with a value inside its declared bounds.

**Exit criteria:** two clean iterations, no schema/bounds errors, workspace
under 1 GB.

### Stage 5 — First real optimization round

**Goal:** the smallest meaningful Optuna run to validate signal.

- Start with a single theme (recommend `traj-camera-intrinsics` because it
  has the fewest parameters and the tightest link to visual embeddings).
- `num_frames_override: 30`, `iteration_batch_size: 4`, `max_iterations: 10`.
- Save top-N configs, back up to S3, record observations in
  `trajectory_plan.md` progress log.

Follow-up rounds can layer additional themes (mount, jitter, agent, scene)
once the first round demonstrates the loop scores trajectory embeddings
sensibly.

## Risks & Mitigations

- **`sdg_config_mean_std.yaml` inline may miss a key.** Diff the resolved
  merged config before/after step 1.1 by loading it via
  `parameter_schema._load_yaml_with_extends` and dumping — anything that
  changes value or key set is a bug.
- **Deleting `experiment_mean_std/base_v2/` too early.** It's the only
  structural reference for authoring diverse seed YAMLs. Delete only in
  stage 2 exit.
- **`SEARCH_SPACE_THEMES` replacement breaks importers.** `run_theme_rounds.py`
  is deleted in stage 1.4, but grep for any surviving references to the
  old theme names before pushing stage 3.
- **Trial cost.** Trajectory frames are physics-stepped; budget accordingly.
  A 60-frame trial ≈ N seconds of Isaac wall-clock — measure once in the
  smoke test and extrapolate.

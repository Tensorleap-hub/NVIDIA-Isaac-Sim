# Optuna Search Over Trajectory SDG

## Purpose

The `trajectory` branch is dedicated to episode-based trajectory generation
(`standalone_palletjack_trajectory_sdg.py`). The random-frame generator lives
on other branches and does not need to coexist here. This document scopes the
work needed to run `simulation_calibration_loop` (Isaac → DINOv2 → Optuna)
against the trajectory pipeline, starting with a hard purge of every
random-frame artifact that would otherwise leak into schema inference, waste
Optuna trials on dead knobs, or fill the disk.

## Progress log

- **Stage 1 — Purge random-frame + old loop artifacts.** Landed 2026-07-01
  in commit `0bdab27`. Trajectory YAML runs standalone; `simulation_calibration_loop/`
  contains only generic code + this doc-set.
- **Stage 2 — Trajectory seed configs.** Landed 2026-07-01 (uncommitted at
  time of writing). `palletjack_sdg/experiments/trajectory/base_v1/` holds
  **20** standalone flat seeds (no `extends:`). `parameter_schema.load_yaml_configs`
  + `infer_parameter_schema` returns **144 params**; `validate_configs_against_schema`
  OK. `experiment_mean_std/base_v2/` deleted. Only exp03 has
  `cameras.ego.fisheye.enabled: true`. Pinned constants held across all 20:
  `capture.video=false`, `cameras.chase.enabled=false`,
  `cameras.ego.shutter_close_fraction=0.0`, `environment.name=full_warehouse`.
  5-frame smoke pass: **20/20 seeds** after two config fixes discovered by
  the smoke: exp02 `buffer_m 1.4 → 1.0` and exp14 `z_slice_m 1.4 → 1.7`
  (both fixed occupancy-path-planner freespace collapses in `full_warehouse`).
- **Stage 3 — Search-space themes, per-group configs, round orchestrator.**
  Landed 2026-07-02 (uncommitted at time of writing). `SEARCH_SPACE_THEMES`
  rewritten (`config.py:100–142`) with the 6 trajectory groups; old-theme
  names raise a clear `ValueError`. 6 per-group project configs
  (`simulation_calibration_loop/project_config_trajectory_{camera_intrinsics,
  camera_mount, camera_jitter, agent, scene, characters}.yaml`) validate with
  0 missing bounds. Orchestrator `theme_rounds_trajectory.yaml` written with
  `rounds: 2` and a `common:` block carrying the shared knobs (max_iterations,
  iteration_batch_size, sample_number, promoted_baseline_dir, embedder_backend,
  base_pool, rfdetr_embedder). `materials.textures` categorical uses the
  19 unique JSON-encoded texture bundles harvested from base_v1 so seed rows
  replay through Optuna's categorical without collision.
  **Blocking TBDs:** `rfdetr_embedder.checkpoint_path` (currently
  `TBD_rfdetr_checkpoint_path` in the orchestrator's `common:`) and
  `real_dataset_root` / `real_annotations_file` (currently `TBD_*` in every
  per-group YAML — required by `load_workflow_config`).
- **Stage 3 blockers resolved + embedder rebuilt.** Landed 2026-07-03
  (uncommitted at time of writing).
  - All `TBD_*` placeholders filled across the 6 per-group configs +
    `theme_rounds_trajectory.yaml`: `real_dataset_root: /home/ubuntu/loco_dataset`
    (LOCO-nested source — the flat Roboflow `valid/` folder resolves 0 images
    because annotations carry `/dataset/subset-3/...` paths),
    `real_annotations_file: /home/ubuntu/warehouse3cls_traj_v2/valid/_annotations.coco.json`
    (858 real images resolve), and a **SMOKE-ONLY** placeholder checkpoint
    `warehouse3cls_traj_v2/.../checkpoint_best_ema.pth` (tagged in-line;
    swap to the fresh model for the full run).
  - **Embedder blocker found + fixed.** `RFDETREmbedder` used to `import rfdetr`,
    but rfdetr 1.7.0 (the checkpoint's version) needs transformers 5.x → torch≥2.4,
    which collides with the loop venv's pinned torch 2.1.2. The torch pin is *not*
    an Isaac constraint (Isaac runs as a subprocess via its own `./python.sh`,
    `data.py:294`). Rather than upgrade torch, we exploit that the RF-DETR backbone
    **is** a DINOv2 ViT-S/14 whose checkpoint stores encoder weights under
    `backbone.0.encoder.encoder.*` with exact HuggingFace `Dinov2Model` key naming
    (223/223 keys match, 0 missing/unexpected). `RFDETREmbedder` now builds a
    `Dinov2Model` and loads just those weights — same constructor + `embed_paths`
    contract, so `controller.py`/`config.py` are untouched. Added
    `transformers==4.46.3` to `local_requirements.txt` (torch-2.1.2 compatible).
    Verified end-to-end: 384-dim finite embeddings, cache round-trips.
  - `theme_rounds_trajectory_smoke.yaml` authored for Stage 4 (single group
    `camera_intrinsics`, 5 frames/trial, 1/batch, 2 iterations).
  - **Verified non-Isaac plumbing:** `load_workflow_config` OK; 32 seeds →
    144-param schema; the 4 intrinsics bounds paths present + matched. Only the
    Isaac launch itself (Stage 4) remains untested.
- **Stage 4 — Smoke test PASSED.** Ran 2026-07-03 via
  `run_theme_rounds.py --round-config theme_rounds_trajectory_smoke.yaml`
  (single group `camera_intrinsics`, 5 frames/trial, 1/batch, 2 iterations,
  `rounds: 1`). All exit criteria met:
  - Isaac launched from the trajectory script; `Camera/rgb/` populated (5 PNGs/run).
  - Real embeddings computed via the reworked DINOv2-encoder embedder: `(858, 384)`.
  - Synthetic embeddings + MMD scored: iter-1 `best_obj=0.574626` (trial_9);
    iteration 2 suggested a new point and scored it (`iter_best=0.624434`).
  - 33 trials total (32 pinned seeds + 1 iter-2 trial); workspace 142 MB (< 1 GB);
    no schema/bounds errors.
  - **Transient CDN flake observed + auto-recovered.** Two individual Isaac runs
    died with exit 1 on "Maximum loading time reached while waiting for assets to
    load" (Omniverse S3 asset streaming), not param/schema errors. The retry
    wrapper (`run_main_loop_with_retry.sh`) restarts the *whole* workflow on any
    single Isaac failure, so it re-ran the seed batch and succeeded on attempt 3.
    Note for the full run: at 60 frames × more trials, a late flake is an
    expensive restart (the Optuna study DB persists, so scored trials aren't fully
    lost, but generation re-runs). Consider a per-run retry instead of whole-workflow.
  - **Thematic-rounds: partially validated.** The orchestrator harness works —
    it loaded the round-config, applied the `common:` overrides
    (`sample_number → num_frames_override`, max_iterations, iteration_batch_size,
    embedder, base_pool), derived a round workspace, drove the config through the
    retry wrapper, and **promoted a baseline** (`promoted_baseline_trajectory_smoke/`
    now has `best.json`, `best.yaml`, `base_pool.json`). NOT yet exercised because
    the smoke used 1 theme × 1 round: cross-theme handoff (group B inheriting group
    A's promoted baseline) and multi-round re-optimization. Those are the next
    thing to confirm on the full run (or a 2-theme × 2-round mini-run).
- **Stragglers noted (non-blocking).** Old-theme references remain in
  `README.md:146` (`camera-color` mention), `tensorleap_intgration_code/data_preprocess.py:1020–1021`
  (parses legacy output-folder names), and `tests/test_base_pool.py`
  (synthetic test data using `camera.camera_height_mean` — tests still pass
  because they don't hit the real schema). `sdg_config_mean_std.yaml` still
  present as a Stage 1 leftover.

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
  parameter-agnostic. `SEARCH_SPACE_THEMES` (originally `config.py:95–237`,
  now `config.py:100–142`) was rewritten at Stage 3.1 to reference only
  trajectory paths.

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
- `theme_rounds.yaml` — **DELETE the content but keep the filename slot; will
  be rewritten in Stage 3 as a trajectory round orchestrator**

Delete the smoke tests that hardcode the old `include/exclude` semantics
against old seed dirs (rewrite in stage 4 if we want a trajectory smoke test):
- `test_isaac_small_loop.py`
- `smoke_test_dinov2.py`

Keep and audit in step 1.5:
- `__init__.py`, `base_pool.py`, `config.py`, `controller.py`, `data.py`,
  `parameter_schema.py`, `ui.py`
- `run_theme_rounds.py` — **KEEP**. The iterate-by-group strategy is
  preserved on this branch; this runner cycles per-group project configs
  N rounds against a shared `promoted_baseline_dir`. Audit it for
  hardcoded theme names in step 1.5.
- `collect_top_runs.py`, `upload_top_runs_to_s3.py`, `visualize_population.py`
  — utilities, not path-specific; keep unless audit shows dependence on
  removed configs.

#### 1.5 Deep audit of the loop code for random-frame assumptions

Read top-to-bottom with an eye for hardcoded old paths, discovery
assumptions, or theme names:

- `config.py:100–142` — **`SEARCH_SPACE_THEMES` rewritten at Stage 3.1**
  with trajectory themes. Every prior entry referenced a deleted knob.
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
trajectory knobs we want Optuna to search. Delete
`experiment_mean_std/base_v2/` once these are in place.

Design principles:
- **Structural template = `palletjack_sdg/experiments/experiment_mean_std/base_v2/`.**
  Model the new `base_v1/` on its shape: one flat YAML per seed, no
  `extends:`, same keys across every file with values chosen to span the
  intended search ranges. Only that directory is preserved through the
  Stage 1 purge specifically to serve as this reference; delete it at the
  Stage 2 exit.
- Every seed uses the same top-level structure (no `extends:`).
- Every path we intend to include in the search space must appear in
  every seed (schema inference intersects across YAMLs).
- Seeds should differ meaningfully in the values of each optimizable path
  so Optuna's initial pool has variance.
- **At least one seed sets `cameras.ego.fisheye.enabled: true`** with
  sensible `k1..k4` (start from the stage-6 config's defaults, e.g.
  `k1: -0.20, k2: 0.04`). Fisheye is not in the search space — it writes
  to `Camera/rgb_fisheye/` which the RF-DETR embedder doesn't score — but
  keeping it live in one seed exercises the OpenCV post-render pipeline
  every round and reserves the option to score fisheye later without
  reintroducing the code path from scratch.
- Other knobs pinned constant across all seeds: `capture.video: false`,
  `cameras.chase.enabled: false`, `cameras.ego.shutter_close_fraction: 0.0`.

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

### Stage 3 — Search-space themes, per-group project configs, and round orchestrator

**Goal:** replace `SEARCH_SPACE_THEMES` in `config.py` with trajectory themes,
author **one project config per group** (iterate-by-group strategy preserved
from the old workflow), and write a new `theme_rounds_trajectory.yaml`
orchestrator that `run_theme_rounds.py` cycles through.

**Why one config per group instead of one big config?** The loop's
`run_theme_rounds.py` runs each group as its own Optuna study against a shared
`promoted_baseline_dir`, so the best YAML from group A becomes the baseline
that group B optimizes on top of. This keeps each study's search space narrow
(better TPE efficiency), makes rounds visible/auditable, and lets us pause
and inspect between groups. The alternative — one giant study — is harder
to interpret and less compute-efficient.

#### 3.1 Replace `SEARCH_SPACE_THEMES`

Curated trajectory groups (delivered at `config.py:100–142`):

- `traj-camera-intrinsics`: fov, focal, f_stop, focus_distance
- `traj-camera-mount`: height, pitch, roll
- `traj-camera-jitter`: all four `*_jitter.*` blocks
- `traj-agent`: speed_mps, turn_rate_dps
- `traj-scene`: palletjack / forklift / pallet counts, clutter_level,
  lighting mean/std, materials roughness mean/std, materials.textures
- `traj-characters`: count, enabled (optional; can be off for first run)

Grep `SEARCH_SPACE_THEMES` after replacement to confirm no old theme names
survive elsewhere in the codebase.

#### 3.2 Author per-group project configs

One YAML per group under `simulation_calibration_loop/`:

- `project_config_trajectory_camera_intrinsics.yaml`
- `project_config_trajectory_camera_mount.yaml`
- `project_config_trajectory_camera_jitter.yaml`
- `project_config_trajectory_agent.yaml`
- `project_config_trajectory_scene.yaml`
- `project_config_trajectory_characters.yaml` (optional; skip if pinned off)

Every per-group config sets:

- `isaac.script_path: ../palletjack_sdg/standalone_palletjack_trajectory_sdg.py`
- `seed_config_dir: ../palletjack_sdg/experiments/trajectory/base_v1`
- `search_space.themes: [<single group name>]`
- `search_space.exclude: [run.data_dir, run.num_frames, run.headless]`
  plus every knowingly no-op knob from the readiness gaps list
- `search_space.bounds: {...}` — only the paths for that group's theme,
  min/max for numeric, list for categorical (materials.textures)
- `embedder_backend: rfdetr` +
  `rfdetr_embedder.checkpoint_path: <path-to-trained-checkpoint>`
  (see the Embedder section)

Shared settings (`iteration_batch_size`, `max_iterations`, `sample_number`,
`base_pool`, and the RF-DETR checkpoint) can be kept once in the round
orchestrator's `common:` block and omitted from the per-group configs — the
orchestrator applies them before each run.

#### 3.3 Write the trajectory round orchestrator

New `simulation_calibration_loop/theme_rounds_trajectory.yaml`:

```yaml
rounds: 2

common:
  promoted_baseline_dir: ./promoted_baseline_trajectory
  max_iterations: 15
  iteration_batch_size: 4
  sample_number: 64          # → isaac.num_frames_override in each config
  base_pool:
    enabled: true
    max_size: 60
    pin_seeds: true

configs:
  - project_config_trajectory_camera_intrinsics.yaml
  - project_config_trajectory_camera_mount.yaml
  - project_config_trajectory_camera_jitter.yaml
  - project_config_trajectory_agent.yaml
  - project_config_trajectory_scene.yaml
#  - project_config_trajectory_characters.yaml

rfdetr_embedder:
  checkpoint_path: <path-to-trained-rfdetr-checkpoint>
  num_classes: 3
  layer_index: 3
  batch_size: 16
  resize_size: 256
  image_size: 224

embedder_backend: rfdetr
```

#### 3.4 Migration reference: old themes → new

The old `SEARCH_SPACE_THEMES` had 9 core themes + 2 "top-N important"
shortcuts, ~60 knobs total. The new set has 6 themes, ~20 knobs total.
Fewer knobs and single-axis groups so the first rounds converge quickly
and each group's contribution is legible.

| Old theme | Old knobs (representative) | New home |
|---|---|---|
| `environment` | `environment.name` | **Dropped.** Pinned to `full_warehouse` in seeds. Re-add later. |
| `camera` | `camera_height_mean/std`, `camera_tilt_mean/std`, `camera_yaw_mean/std`, `camera_roll_mean/std`, `fov_mean/std` | **Reshaped, split across three new themes** (see below). |
| `camera-color` | `image_augmentation.*` | **Dropped.** Not wired through the trajectory script; RF-DETR embeds raw RGB. |
| `noise` | `motion_blur_strength_*`, `dataset_noise.*` | **Dropped.** Motion blur is stage-7.5-parked; dataset noise not exercised on the trajectory path. |
| `objects` | `distractors.clutter_level` + per-group `.occurrence` | **Partial.** Only `clutter_level` survives (in `traj-scene`). Per-group deferred. |
| `diversity` | per-group `.diversity` | **Dropped from initial themes.** Deferred. |
| `scene-objects` | palletjack/forklift/pallet `count_per_model` + `position_std` + `rotation_std` + `pallet_stacks.*` | **Partial.** Only counts survive (in `traj-scene`). Placement variance deferred. |
| `lighting` | `intensity_mean/std`, `visibility_choices` | **Partial.** `intensity_mean/std` in `traj-scene`. `visibility_choices` dropped. |
| `materials` | `textures`, `roughness_mean/std`, `emissive_intensity_mean/std` | **Partial.** `textures` + `roughness_mean/std` in `traj-scene`. Emissive dropped. |
| `top20_important` / `top30_important` | cross-cutting shortlists | **Dropped.** No trajectory equivalent yet. |

**How the old `camera` theme fragments.** Random-frame sampled a full 6-DOF
camera pose distribution every frame. Trajectory has a static mount plus
per-frame sinusoidal jitter, so the same intent lives across three new themes:

- **`traj-camera-mount`** ← static portion of old `camera`: `height_m`,
  `pitch_deg`, `roll_deg`. Old `camera_height_mean/std`,
  `camera_tilt_mean/std`, `camera_roll_mean/std` collapse here. No `_std` —
  mount is fixed per-episode, not sampled per-frame.
- **`traj-camera-jitter`** ← *new axis*. Old schema had no continuous-jitter
  concept (`_std` sampled independent poses, not smooth oscillation).
- **`traj-camera-intrinsics`** ← the lens portion. Old `fov_mean/std` maps
  in directly. `focal_length_mm`, `f_stop`, `focus_distance_m` are new
  (DoF and calibrated-lens overrides only exist in the trajectory pipeline).

Old `camera_yaw_mean/std` has no new home — yaw is fully determined by the
planned path in trajectory mode.

Genuinely new (no old counterpart): `traj-agent`, `traj-characters`.

#### 3.5 Deferred themes — layer in after the baseline is set

Anything not in the six starting themes is intentionally deferred. Add
them one at a time to `theme_rounds_trajectory.yaml`'s `configs:` list
once the initial rounds have produced a stable `promoted_baseline_dir`.
Don't stack multiple new themes in a single round — cascading noise
across groups makes attribution hard.

Order of intended reintroduction, tightest OD signal first:

1. **`traj-distractor-groups`** — per-group `distractors.groups.<X>.occurrence`
   (CardBox, BarelPlastic, Bucket, CratePlastic, BottlePlastic, PushCart,
   RackPile, TrafficSigns). Reintroduce first because per-class distractor
   frequency directly shapes what the OD model learns to ignore.
2. **`traj-distractor-diversity`** — per-group `.diversity`. Adds asset
   variety within each distractor group. Follows occurrence naturally.
3. **`traj-scene-placement`** — palletjack/forklift/pallet `position_std`
   + `rotation_std` + `pallet_stacks.*`. Placement variance affects OD
   generalization but adds many knobs, so wait until per-class distractor
   tuning is in place.
4. **`traj-materials-emissive`** — `materials.emissive_intensity_mean/std`.
   Matters for edge cases (LED sign visibility) but low leverage until
   base materials tuning has converged.
5. **`traj-environment`** — `environment.name` categorical over
   `full_warehouse` / `warehouse` / `warehouse_multiple_shelves` /
   `warehouse_with_forklifts`. Large axis; do last because switching
   environment invalidates any placement/lighting/mount tuning specific
   to the previous env.
6. **`traj-camera-motion-blur`** — `shutter_close_fraction`. Blocked
   on Stage 7.5 landing the physics-driven camera. Optimizing before
   then is a no-op (Stage 6.1 outcome #5).
7. **`traj-camera-color-aug`** — image_augmentation.\* if we decide the
   OD model benefits from color-space augmentation at SDG time. Currently
   assumed handled downstream of SDG, so lowest priority.
8. **`traj-camera-noise`** — dataset noise (shot_scale, jpeg_quality,
   sigma) once we've confirmed the trajectory script exercises the noise
   post-processing path (audit needed).

**Exit criteria:**

- `controller._build_param_bounds_from_config()` succeeds without raising
  for every per-group project config.
- `run_theme_rounds.py --round-config theme_rounds_trajectory.yaml --rounds 0`
  (or an equivalent dry-run flag if one exists) loads all listed configs
  without errors.

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

**Goal:** the smallest meaningful Optuna run to validate signal, using the
round-orchestrator entry point that Stage 3 delivered.

- Start with a **single group** by commenting out all but one entry in
  `theme_rounds_trajectory.yaml`'s `configs:` list. Recommend
  `project_config_trajectory_camera_intrinsics.yaml` because it has the
  fewest parameters and the tightest link to visual embeddings.
- Common overrides for this first run:
  `sample_number: 30`, `iteration_batch_size: 4`, `max_iterations: 10`,
  `rounds: 1`.
- Invoke via `run_theme_rounds.py --round-config theme_rounds_trajectory.yaml`.
- Save top-N configs, back up to S3 via `upload_top_runs_to_s3.py`, record
  observations in `trajectory_plan.md` progress log.

Follow-up rounds re-enable additional group configs in the orchestrator's
`configs:` list. The shared `promoted_baseline_dir` means each group builds
on the previous group's best result. Bump `rounds:` to 2+ once multiple
groups are stable so later rounds re-optimize each group against the
new baseline.

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

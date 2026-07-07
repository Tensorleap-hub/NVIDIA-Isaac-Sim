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
  - **Thematic-rounds: FULLY validated (2026-07-03).** A 2-theme × 2-round mini
    run (`theme_rounds_trajectory_handoff.yaml`: camera_intrinsics + camera_mount,
    `--workspace-root handoff_ws` to avoid stale-workspace reuse) ran all four
    theme-rounds cleanly. Behavioral proof of cross-theme handoff: after
    intrinsics-r1 promoted `best.yaml` (fov_mean 68→77.89, f_stop 0→7.61,
    focus_distance 5→7.86), the camera_mount theme's iteration-1 suggestion YAML
    carried exactly those tuned intrinsics values (not the seed defaults) while
    independently searching its own mount params (height/pitch/roll). Round 2
    inherited the same. Confirms: baseline promotion, cross-theme inheritance,
    per-group search isolation, multi-round sequencing.
    NOTE: `_load_base_template` runs at controller `__init__` and merges
    `promoted_baseline_dir/best.yaml` into the base template; its `[baseline]`
    log line goes through `ui.append_log`, which is IN-MEMORY ONLY (never written
    to `main_loop_screen.log`) — so verify handoff by inspecting materialized
    YAMLs, not by grepping logs.
    GOTCHA: re-running a theme whose `workspace_<name>_r01/` already holds a
    completed Optuna study short-circuits (`runs=0/0`, replays old best). Always
    launch rounds with a fresh `--workspace-root`.
- **Stage 5 — First real optimization run (2026-07-03 → ongoing).** Swapped the
  SMOKE-ONLY placeholder for the freshly-trained
  `warehouse3cls_mixedbase_50/output/rfdetr_mixedbase50_base/checkpoint_best_ema.pth`
  (loads 223/223 into the Dinov2 encoder) across the orchestrator `common:` block
  and all per-group configs. Committed `1bb639c`.
  - **First launch (6 themes × 2 rounds, 60 frames) STALLED on `characters`.**
    Themes 1–5 of round 1 (intrinsics, mount, jitter, agent, scene) completed fine
    with only ~11 total flake-retries and MMD tracking ~0.58–0.60. Theme 6
    `characters` then failed **1493×** — its Isaac runs die at generation
    (`omni.anim.people` character spawn), never producing normal output — and the
    whole-workflow retry wrapper looped for ~24h without a single clean pass,
    blocking the round. (Confirms the retry-granularity risk noted in Stage 4: a
    theme that can't pass cleanly wedges the entire run.)
  - **Recovery:** killed the stuck run by PID, **disabled the `characters` theme**
    in the orchestrator (commented out; it was always optional per the plan), and
    relaunched as **5 themes × 2 rounds** with a fresh `--workspace-root full_run_ws2`
    and fresh `promoted_baseline_trajectory_v2` (clean cold start). Committed
    `d0308b6`. This run is healthy and in progress.
  - **TODO before re-enabling characters:** diagnose why `omni.anim.people`
    character spawn crashes Isaac at generation (Stage 7b feature). Until then it
    stays out of the search.
- **Stragglers noted (non-blocking).** Old-theme references remain in
  `README.md:146` (`camera-color` mention), `tensorleap_intgration_code/data_preprocess.py:1020–1021`
  (parses legacy output-folder names), and `tests/test_base_pool.py`
  (synthetic test data using `camera.camera_height_mean` — tests still pass
  because they don't hit the real schema). `sdg_config_mean_std.yaml` still
  present as a Stage 1 leftover.
- **Stage 6 — Search-space expansion + multi-seed evaluation + base_v4 seeds
  (2026-07-06, uncommitted at time of writing; NOT yet smoke-tested — Isaac is
  under repair, run the two checks below before the next real run).**
  - **Seed set switched `base_v1` → `base_v4`** across all six existing per-group
    project configs + the three new ones (21 seeds). Validated: schema resolves,
    every bound covered, and every base_v4 `materials.textures` value is within
    the `traj-scene` categorical choices (the seed-switch risk — clean).
  - **Three new themes in `config.py:SEARCH_SPACE_THEMES`:**
    - `traj-environment` (`environment.name`) — **promoted to FIRST** in the
      round (reverses §3.5's "do env last"): env is the biggest domain-gap lever
      and the promoted baseline it yields becomes the substrate every later theme
      conditions on, so it must lead, not trail. Safe because base_v4 already
      exercises all four warehouses in trajectory mode and their roam bounds
      differ by ≤1m (occupancy-planner tolerant) — no per-env geometry preset or
      Isaac-script change was needed. New `project_config_trajectory_environment.yaml`.
    - `traj-distractor-occurrence` (8 groups' `occurrence`; `0` removes a group,
      subsuming the old `.use`). New `project_config_trajectory_distractor_occurrence.yaml`.
    - `traj-distractor-diversity` (8 groups' `diversity`; script clamps to each
      group's asset-pool size, so out-of-range values are harmless). New
      `project_config_trajectory_distractor_diversity.yaml`. Staged AFTER
      occurrence (commented out in the orchestrator) — don't stack two fresh
      object themes in one round.
  - **`traj-scene` placement std deliberately NOT added.** With uniform scatter
    kept (shown better for the trajectory direction), `*.position_std` is a genuine
    no-op — `_scatter_position` ignores `position_mean/std` under
    `scatter: uniform` (`standalone_palletjack_trajectory_sdg.py:970-981`) — and
    every base_v4 seed has `rotation_std=[0,0,~104°]` (x/y dead for floor objects,
    yaw already near-saturated). So "reinstate scene-objects fully" collapses to
    `count_per_model`, which is already in `traj-scene`. `pallet_stacks.*` remains
    out — unimplemented in the trajectory script (future).
  - **Multi-seed per-trial evaluation.** Each candidate YAML is now generated once
    per seed in `IsaacConfig.eval_seeds` (each into `output_dir/seed_<k>/`) and all
    seeds' RGB frames are POOLED before embedding, so MMD-to-real reflects the
    config's DISTRIBUTION (many layouts/trajectories) instead of one realization.
    `num_frames_override` is now PER-SEED; the seed set is FIXED across trials
    (paired comparison) and folded into the run fingerprint. Orchestrator uses
    `sample_number: 15` × `eval_seeds: [1,2,3,4]` = 60 frames/trial (same budget as
    the old single-run 60, decorrelated across 4 layouts). Wiring: `IsaacConfig.eval_seeds`
    (default `[0]`, backward-compatible), `run_isaac_generation(..., seed=)` →
    `--seed`, the per-seed loop in `controller._materialize_and_execute_iteration`,
    and `eval_seeds` threaded through `run_theme_rounds._apply_common_overrides`.
  - **Replicator RNG now seeded from the episode seed (root-cause fix).** Object/
    light placement uses `rep.distribution.*` (via `rep_normal`/`_scatter_position`),
    which is governed by Replicator's global RNG — NOT `random.seed()`/
    `np.random.seed()`. Before this, `--seed` re-rolled only the trajectory path,
    leaving the scene layout identical across seeds, which would have silently
    defeated both multi-seed evaluation AND multi-seed dataset generation (e.g.
    base_v4's per-config seeds). Added a guarded `rep.set_global_seed(seed)` right
    after the `omni.replicator.core` import in `run_stage4`
    (`standalone_palletjack_trajectory_sdg.py`).
  - **Smoke checks owed before the next real run (Isaac was down): BOTH PASSED
    2026-07-06.** Ran 4 × 5-frame generations off `base_v4/exp10_reference_tight.yaml`
    via `/opt/IsaacSim/python.sh` (`--headless True --no_video`); all 4 succeeded
    on driver-attempt 1 (rc=0, 5 RGB frames each; no whole-workflow retry needed).
    1. **`rep.set_global_seed` decorrelation — CONFIRMED.** `full_warehouse` at
       seed 1 vs seed 2 produced visibly DIFFERENT scene layouts (distinct pallet/
       distractor scatter, forklift placement, lighting tint), not merely different
       camera paths — and the planned trajectories also diverged (19.78 m/33 wpts vs
       13.38 m/11 wpts). Object scatter world-positions are NOT dumped to disk and
       `_scatter_position` returns lazy `rep.distribution` nodes, so the check is
       necessarily visual (as the plan intended); montage saved in scratchpad.
    2. **Env-switch stall — CONFIRMED CLEAR.** `warehouse_multiple_shelves` and
       `warehouse_with_forklifts` both loaded + generated 5 frames with no
       `characters`-style wedge. Occupancy planner planned real paths (multishelf
       needed a 2nd *internal* planner attempt — normal `max_retries` behavior, not
       a workflow stall — len 14.37 m; forklifts 16.90 m attempt 1).
    - **CDN flake note:** both env runs logged one non-fatal
      "Maximum loading time reached while waiting for assets" WARNING from the
      Replicator orchestrator (the Stage-4 Omniverse-S3 signature) but did NOT
      exit 1 — all frames still rendered. So this signature is sometimes a warning,
      sometimes fatal; the per-run retry improvement (Stage 4 note) remains worth doing.
- **Stage 7 — Exploration-boundary optimization LANDED + smoke-verified
  (2026-07-07, uncommitted at time of writing).** The camera roam region is now a
  searchable axis via a constrained reparameterization (NOT the raw `bounds_xy`).
  - **Reparameterization.** New `trajectory.roam` block: `center_x_frac`,
    `center_y_frac` ∈ [-1,1] and `width_frac`, `height_frac` ∈ (0,1], interpreted
    as fractions of the *env envelope* — which is the config's pre-existing
    `trajectory.bounds_xy` (the per-env box the seed author tuned). `apply_roam_bounds()`
    (`standalone_palletjack_trajectory_sdg.py`) derives `bounds_xy` from these
    right after CLI overrides, before object scatter + occupancy planning read it
    and before `write_run_config` dumps the resolved config. Construction keeps
    the box strictly inside the envelope for any in-range frac (center offset
    scales by leftover slack → 0 as size→1), so it's env-relative (coupling #3)
    and cannot clip walls. At defaults (center 0, size 1.0) the box == envelope
    exactly → behavior unchanged when disabled/defaulted. Object scatter reads the
    same `bounds_xy` (coupling #2) so a smaller box also insets object placement.
    Emits a `stage5_roam_bounds_derived` event.
  - **Anti-wedge (coupling #1), two layers.** (a) `apply_roam_bounds` scales
    `occupancy.min_path_m` down toward the box's free-region diagonal. (b) THE REAL
    FIX: `_build_occupancy_waypoints` no longer raises when no attempt meets
    `min_path_m` — it tracks the longest valid, obstacle-clear path across attempts
    and falls back to it (`relaxed_below_min_path: true` in `planned_path.json`).
    Only raises if ZERO clear paths exist. Root cause found in smoke test: a
    centered 0.5×0.5 box in `full_warehouse` yields 20 clear paths of 1–10.5 m —
    all rejected as < 11.45 m — so the pre-fix code raised and the whole-workflow
    retry wrapper looped forever (the exact `characters`-style wedge).
  - **Short-path frame fill = ping-pong traversal + smooth in-place pivot** (both
    user ideas, combined). When a path is too short to space `num_frames` ≥
    `min_spacing_m` (default 0.3 m) apart, `_plan_traversal` reflects the route
    forward-and-back enough times to restore spacing (backward legs flip heading
    180°, facing the camera the opposite way = new information, not re-views), AND
    at each forward↔reverse turnaround the camera PIVOTS IN PLACE — a set of frames
    at the turnaround point whose yaw sweeps from the incoming heading to the
    reversed one — so the view rotates smoothly instead of snapping 180° between
    consecutive frames. Pivot cost is modelled as `pivot_arc_m` (default 1.5 m) of
    virtual arc so the frame budget divides smoothly between travel and rotation.
    Implemented via `seg_idx`+`yaw_rel` on the reflected waypoints (no change to
    the hot per-frame loop). Collision-free (same clear path). No-op for long paths
    / low frame counts. Config:
    `trajectory.short_path_fill.{enabled,min_spacing_m,max_passes,pivot_arc_m}`
    (script defaults, no seed change needed). Verified in `poses.jsonl`: yaw ramps
    (e.g. −32→+19→+71→+122° over 4 pivot frames) then the reverse leg departs
    continuously; montage `cmp_pivot.png` shows the view panning across the scene.
  - **New search axis.** `traj-exploration-bounds` theme (`config.py`) over the 4
    roam fracs; new `project_config_trajectory_exploration_bounds.yaml` (bounds:
    center ±0.6, size 0.5–1.0; raw `trajectory.bounds_xy` excluded — it's derived).
    `roam` block added to all 21 base_v4 seeds at full-roam defaults (schema
    intersects). Staged commented in `theme_rounds_trajectory.yaml` (add after
    `traj-environment` promotes a baseline env; don't stack a fresh axis mid-round).
  - **Offline validation.** All 10 per-group configs still load + validate (no
    regression); exploration config's filtered schema = exactly the 4 roam fracs
    (`enabled` + `bounds_xy` correctly dropped); `apply_roam_bounds` + `_plan_traversal`
    unit-tested (full=envelope, corner stays inside, tiny→min_path scaled, ping-pong
    restores spacing + flips heading, no zero-length segments).
  - **Isaac smoke — ALL 4 PASSED (driver-attempt 1, no wedge).** r1 full 0.5
    centered (path 10.46 m, relaxed, 5 frames); r2 full 0.5 CORNER offset (8.55 m,
    relaxed, within walls — 2 frames dark from facing a wall, valid but low-info);
    r3 multishelf 0.5 (9.62 m, relaxed, objects visible); r4 full 0.3 box (BELOW the
    0.5 search floor) + 24 frames → 4.14 m path, min_path auto-scaled to 4.81 m,
    ping-pong x-passes filled all 24 frames with forward→reverse heading flip
    (0°→−180°) confirmed in `poses.jsonl`. Objects visible + within walls in all;
    montage `cmp_stage7.png` in scratchpad.
  - **Follow-up (non-blocking):** extreme corner offsets (center ±0.6 with a small
    box) can push the camera flush against a wall → a few dark/low-info frames.
    Consider tightening `center_*_frac` bounds or adding a wall-proximity inset if
    those frames dilute MMD signal in the real run.

- **Stage 8 — full_warehouse camera-trajectory perimeter bounds VALIDATED
  (2026-07-07).** Iteratively dialed the ego-camera perimeter loop for
  `full_warehouse` by rendering a clockwise rectangle walk (waypoint_list mode,
  `yaw_rel=0` so the camera faces the direction of travel) and reviewing frames
  edge-by-edge. Per-iteration loop enforced: run sim → upload to S3 → analyze →
  wait for feedback.
  - **Final loop:** rectangle **x∈[-22.5, 3], y∈[-11, 29], FOV 120°**, clockwise,
    camera faces travel direction. Config `push8_run/perim_fw_push8.yaml`.
  - **Walls confirmed by render:** west wall ≈ x=-22.5; north wall ≈ y=29 (top edge
    runs the clean cross-aisle *past* the north shelves, tight to the wall); east
    set to x=3 (not the ~x=5 wall) so the east edge runs a rack aisle in its north
    half (y≈13–27) then opens to floor near the east wall below. East pushed to 10.5
    showed **exterior void** → do not exceed ~5 on east.
  - **Corrects the [-5,5,-11,13] "main hall" assumption** (see
    `base_v4_bounds_offset_bug`): the interior is ONE connected hall reaching
    x=-22.5 west — there is NO partition wall at x=-5; the occupancy-map "gray"
    west strip was just unscanned floor, not unreachable space.
  - **Caveat:** these are CAMERA (ghost waypoint) bounds — waypoint_list ignores
    occupancy — so they are NOT drop-in for OBJECT SCATTER (needs collision-free
    floor via occupancy). Keep camera-trajectory bounds separate from `ENV_BOUNDS`
    (scatter) in `_generate_base_v4.py`.
  - **Artifacts:** 9 iterations uploaded to
    `s3://nvidia-isaac-bucket/trajectory-tests/20260707_perim_full_warehouse_*`;
    progression montage `COMPARE_all_turns.png` in scratchpad.
  - **Written to repo (uncommitted):** (1) `perim_full_warehouse.yaml` = the exact
    camera loop x[-22.5,3] y[-11,29] FOV120. (2) `ENV_BOUNDS["full_warehouse"]` in
    `_generate_base_v4.py` = `[-22.5, 5.0, -11.0, 29.0]` (scatter envelope, east=5 at
    the wall). Before writing ENV_BOUNDS, an `occupancy_path` re-check over the box
    passed: planner found a clean 7-wp path reaching x=-18.6 west (not relaxed, met
    min_path), objects scattered in-bounds — so the west region is genuinely
    occupancy-reachable, not just camera-flyable. This large box should also relieve
    the min_path-12m pathfinding failure noted in the earlier main-hall bounds. Other
    3 envs (`warehouse`, `warehouse_multiple_shelves`, `warehouse_with_forklifts`) NOT
    yet re-validated this way.

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

> **Superseded in part by Stage 6 (see progress log).** `traj-distractor-groups`
> (now `traj-distractor-occurrence`) and `traj-distractor-diversity` have landed,
> and `traj-environment` was **promoted to FIRST**, not last: in the
> promoted-baseline flow the theme tuned first becomes the substrate everyone
> else conditions on, so the highest-leverage axis must lead. The "do env last
> because it invalidates prior tuning" rationale below only holds if env is a
> trailing add-on — it isn't anymore. The remaining entries (placement std,
> emissive, motion blur, color-aug, dataset noise) are still deferred as written.

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

### Stage 7 — Exploration-boundary optimization (LANDED 2026-07-07 — see progress log for the as-built summary; the design notes below are the original plan)

**Goal:** let Optuna optimize *where in the scene the camera is allowed to
roam* — i.e. treat the exploration area as a searchable knob instead of a
fixed per-config constant. Motivation: real capture is often confined to
sub-regions of a warehouse (specific aisles / zones), and constraining the
synthetic exploration boundary may close domain gap that a full-floor roam
can't. This is a NEW search axis (a candidate `traj-exploration-bounds`
theme), distinct from the object-placement axes.

**What defines the exploration area today.** The occupancy path planner roams
`trajectory.bounds_xy = [x_min, x_max, y_min, y_max]` (consumed at
`standalone_palletjack_trajectory_sdg.py:_build_occupancy_waypoints`, ~line
416), subject to `trajectory.occupancy.{boundary_margin_m, buffer_m,
min_path_m, max_retries, z_slice_m, cell_size_m}`. Bounds are currently fixed
per config (full ≈ `[-13,13,-13,15]`, plain ≈ `[-12,12,-12,14]`).

**Parameterization — do NOT search the 4 raw bounds directly.** Four
independent floats let Optuna propose invalid boxes (`x_min > x_max`) and make
the box size confound its position. Prefer a constrained reparameterization,
e.g. **center + extent**: `center_x, center_y` (offset within the warehouse
envelope) + `width, height` (or a single `roam_fraction` ∈ (0,1] that shrinks
the full envelope, plus an offset). The loop or the script then derives
`bounds_xy` from these. Recommended: add the derived-bounds computation in the
Isaac script so `bounds_xy` stays a deterministic function of the searched
params (mirrors how env geometry could be handled), keeping the loop
parameter-agnostic.

**Hard couplings to respect (each is a potential `characters`-style stall):**

1. **Occupancy feasibility.** Shrinking the box below what `min_path_m` (=12m)
   needs makes `_build_occupancy_waypoints` fail to plan a path → whole-workflow
   retry loop. Either bound the minimum area well above the min-path envelope,
   or scale `min_path_m` with the box size. A smoke test at the smallest
   proposed box is mandatory before a full run.
2. **Object scatter is coupled, not independent.** Uniform scatter places
   objects inside `trajectory.bounds_xy` (inset) — see `_scatter_position`
   (`~:970-981`). Shrinking the exploration box therefore also shrinks the
   object-placement region (objects stay camera-reachable — desirable — but this
   is a side effect to state explicitly, not an independent knob).
3. **Environment coupling.** The box must stay inside the *chosen* warehouse's
   physical extent. Since `traj-environment` is now searched (Stage 6), express
   the exploration bounds as **fractions of the env envelope**, not absolute
   metres, or the same absolute box will be valid in `full_warehouse` and
   out-of-walls in a smaller env.

**Implementation sketch:**
- Add `traj-exploration-bounds` to `SEARCH_SPACE_THEMES` (`config.py`) over the
  chosen reparameterized paths (e.g. `trajectory.roam.{center_x,center_y,
  width_frac,height_frac}` — new config keys the script consumes).
- Teach `standalone_palletjack_trajectory_sdg.py` to compute `bounds_xy` from
  those keys (clamped to the env envelope, with a feasibility floor vs
  `min_path_m`) before occupancy planning.
- Add these keys to every base_v4 seed (schema inference intersects across
  seeds) with sensible defaults that reproduce today's full-floor roam, and add
  a `project_config_trajectory_exploration_bounds.yaml`.
- Exclude the raw `trajectory.bounds_xy` indexed paths from search (they become
  derived, not tuned).

**Smoke checks owed:** smallest proposed box still plans a ≥`min_path_m` route;
box stays within walls for every searchable env; objects remain visible (not all
clipped outside the roam region).

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

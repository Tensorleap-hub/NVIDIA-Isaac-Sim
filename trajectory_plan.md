# Palletjack Trajectory SDG Refactor Plan

## Goal

Move `palletjack_sdg` from independent randomized still-frame generation to
episode-based trajectory generation:

1. Create or load an environment.
2. Place an ego camera, robot, or human-like agent in the environment.
3. Move the agent through the scene over time.
4. Export images, optional video, annotations, and trajectory metadata.
5. Keep YAML as the public control surface so `simulation_calibration_loop`
   can continue to materialize, run, and optimize simulation configs.

The safest implementation path is to build a new trajectory generator beside
the current random-frame generator, keep the same CLI contract, and migrate
features in stages.

## Current Integration Contract

The trajectory generator must preserve the interface used by
`simulation_calibration_loop`:

- Script path is configured in `simulation_calibration_loop/project_config*.yaml`
  under `isaac.script_path`.
- The loop launches Isaac from `simulation_calibration_loop/data.py` with:
  - `./python.sh <script_path>`
  - `--config <yaml_path>`
  - `--headless True|False`
  - `--data_dir <output_dir>`
  - optional `--num_frames <N>`
- The loop expects generated images to be discoverable under the run output
  directory. Today it first checks `Camera/rgb`, then falls back to the full
  output tree.
- Optimizable parameters are inferred from seed YAMLs. Any new path that should
  be optimized must exist in all seed YAMLs used by that loop.

Initial target script:

```text
palletjack_sdg/standalone_palletjack_trajectory_sdg.py
```

Existing script remains available during migration:

```text
palletjack_sdg/standalone_palletjack_sdg_mean_std.py
```

## IsaacSim Reference Map

Use these local IsaacSim references when resuming this work in another session.

### Best Overall Pattern: MobilityGen

MobilityGen is the closest reference for an environment plus moving agent plus
attached cameras plus frame/state recording.

- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen.examples/isaacsim/replicator/mobility_gen/examples/scenarios.py`
  - `RandomPathFollowingScenario`
  - Samples start/end points from an occupancy map.
  - Builds a path, computes lookahead target, and drives the robot.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen.examples/isaacsim/replicator/mobility_gen/examples/robots.py`
  - `WheeledMobilityGenRobot`
  - `PolicyMobilityGenRobot`
  - `CarterRobot`
  - `H1Robot`
  - Shows front camera paths and policy/wheeled control wrappers.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen/python/impl/robot.py`
  - Base robot interface.
  - Chase camera helper.
  - Shared path-following/random-action parameters.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen/python/impl/camera.py`
  - Camera/render-product wrapper.
  - RGB, segmentation, depth, normals annotators.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen/python/impl/writer.py`
  - Writes RGB, segmentation, depth, normals, state, config, stage, occupancy map.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen/python/impl/occupancy_map.py`
  - Occupancy map load/save and world/pixel conversion helpers.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.mobility_gen/python/impl/path_planner.py`
  - Path generation and path compression.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/replicator/mobility_gen/replay_directory.py`
  - Replays recorded scenarios and writes modalities.

### Replicator Capture Loop References

These show how to step simulation manually and capture frames without relying on
`rep.trigger.on_frame` random snapshots.

- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py`
  - Uses `World`.
  - Disables capture-on-play.
  - Steps the simulation and calls `rep.orchestrator.step(...)` explicitly.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.replicator.examples/custom_event_and_write.py`
  - Manual custom events and manual Replicator writes.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.replicator.examples/multi_camera.py`
  - Multiple camera prims, render products, and writer outputs.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.synthetic_recorder/isaacsim/replicator/synthetic_recorder/synthetic_recorder.py`
  - Programmatic recorder pattern.
  - Explicit render product and writer setup.

### Video Export References

- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.replicator.examples/cosmos_writer_warehouse.py`
  - Warehouse scene.
  - Carter robot navigation positions.
  - Front robot camera capture.
  - CosmosWriter-style video output.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.replicator.examples/python/tests/test_sdg_cosmos_writer.py`
  - Validates MP4 and PNG outputs.

### Robot And Human-Like Agent References

- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.robot.wheeled_robots.examples/jetbot_differential_move.py`
  - `World`, `WheeledRobot`, `DifferentialController`.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.robot.wheeled_robots.examples/kaya_holonomic_move.py`
  - Holonomic control.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py`
  - H1 policy robot with velocity commands.
- `/Users/orram/Tensorleap/IsaacSim/source/standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py`
  - Spot policy robot with velocity commands.
- `/Users/orram/Tensorleap/IsaacSim/source/tools/actor_sdg/sdg_scheduler.py`
  - Actor SDG scheduler.
  - Useful for human/character-oriented generation.
- `/Users/orram/Tensorleap/IsaacSim/source/tools/actor_sdg/default_config.yaml`
  - Actor SDG config shape for cameras, robots, characters, writer, and length.

### Environment And Navigation Map References

- `/Users/orram/Tensorleap/IsaacSim/source/tools/scene_blox/src/scene_blox/generate_scene.py`
  - Generates USD scenes from tile/constraint configs.
- `/Users/orram/Tensorleap/IsaacSim/source/tools/scene_blox/parameters/warehouse/tile_generation.yaml`
  - Warehouse tile generation settings.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.asset.gen.omap/python/tests/test_occupancy.py`
  - Occupancy map generation test.
- `/Users/orram/Tensorleap/IsaacSim/source/extensions/isaacsim.asset.gen.omap.ui/isaacsim/asset/gen/omap/ui/extension.py`
  - UI implementation for occupancy map generation bounds/cell size/collision APIs.

## Proposed YAML Shape

Keep old keys available for compatibility, but introduce explicit trajectory
sections. The calibration loop can optimize any scalar/list fields here once
they are present in all seed YAMLs and added to `SEARCH_SPACE_THEMES`.

```yaml
run:
  headless: true
  num_frames: 128
  data_dir: /isaac-sim/palletjack_sdg/palletjack_data/default

simulation:
  mode: trajectory       # random_frame | trajectory
  seed: 0
  physics_dt: 0.0166667
  capture_every_n_steps: 1

environment:
  name: full_warehouse
  usd_path: null         # optional override; otherwise use environment_urls

agent:
  type: camera_rig       # camera_rig | carter | h1 | human | ros2
  start_mode: fixed      # fixed | random_free_space | from_waypoint
  start_pose: [0.0, 0.0, 1.6, 0.0, 0.0, 0.0]
  speed_mps: 0.8
  turn_rate_dps: 90.0

trajectory:
  mode: waypoint_list    # waypoint_list | random_waypoints | occupancy_path | nav_agent
  waypoints:
    - [0.0, 0.0, 1.6, 0.0, 0.0, 0.0]
    - [3.0, 0.0, 1.6, 0.0, 0.0, 0.0]
  bounds_xy: [-6.0, 6.0, -6.0, 8.0]
  lookahead_m: 1.5
  smoothing: 0.2
  stop_on_collision: true

cameras:
  ego:
    enabled: true
    parent: agent
    resolution: [960, 544]
    height_m: 1.6
    fov_mean: 75.0
    fov_std: 0.0
  chase:
    enabled: false
    parent: agent
    resolution: [960, 544]

capture:
  rgb: true
  bounding_box_2d_tight: true
  semantic_segmentation: false
  depth: false
  video: false
  metadata: true
```

## Output Contract

Each run should write:

```text
<output_dir>/
  Camera/
    rgb/
      rgb_0000.png
      rgb_0001.png
  trajectory/
    poses.jsonl
    events.jsonl
  run_config.yaml
  run_manifest.json
  isaac.log
```

Later stages may add:

```text
<output_dir>/
  Camera/
    depth/
    semantic_segmentation/
    bounding_box_2d_tight/
  video/
    ego.mp4
  stage/
    scene.usd
  nav/
    occupancy_map.yaml
    occupancy_map.png
    planned_path.json
```

Minimum metadata per frame:

- frame index
- simulation time
- camera prim path
- camera world position
- camera world orientation
- agent prim path
- agent world position
- agent world orientation
- trajectory segment or waypoint index
- collision state, once collisions are enabled

## Progress Log

| Stage | Status | Notes |
|-------|--------|-------|
| 0 | ✅ done | Script skeleton, CLI, output tree, manifest, events log |
| 1 | ✅ done | Waypoint traversal, heading-relative yaw (roll=90 fix), RGB capture, poses.jsonl |
| 2 | ✅ done | Per-episode scene randomization (palletjacks, forklifts, pallets, distractors, lighting, materials); frame-0 consistency fix via warmup step before writer attach |
| 3 | ✅ done | Per-episode FOV sampling, chase camera, per-modality output folders, calibration loop discovery fix |
| 4 | ✅ done | CosmosWriter fixed: script nodes opt-in + timeline.play() + pause_timeline=False; 5 MP4s produced |
| 5 | ✅ done | Occupancy-map free-space path planning; boundary margin for undetected walls; BFS path + nav artifacts |
| 6 | ✅ done — **pivoted to OD-focused ghost camera** | After iterating through kinematic Carter, dynamic Carter, sweep tests, and rigid props, we landed on the conclusion that for an OD training-data pipeline the robot body is not part of the labels and only generates stuck trajectories. Default is now `agent.type: camera_rig` (ghost). Carter remains opt-in. Full camera mount config + per-frame sinusoidal jitter (pitch, roll) for handheld / uneven-terrain realism. |
| 6.1 | ✅ done | Camera realism: DOF (`fStop`+`focusDistance`), position jitter (lateral/vertical), `focal_length_mm` override, fisheye via OpenCV post-render. Motion blur is wired (time-sampled xform ops + shutter attrs + RTX motion-blur settings) but **kinematic camera-sweep blur does not engage** the RT pipeline (RTX integrates object velocities, not camera-xform time samples). Visible blur in this setup requires moving scene objects with physics velocity — see Stage 7.5. |
| 7 | ✅ MVP done | Humans as labeled OD targets. 5 characters/episode spawned at random xy in free corridor; class `person` semantic; BasicWriter bbox labels JSON includes person; 0–5 persons/frame across 60 frames (mean 1.3). 10 character variants. Animation deferred to **Stage 7b**: needs `omni.anim.people` BehaviorScript scaffolding + command files. |
| 7b | ✅ done | (A) Idle pose via `ApplyAnimationGraphAPICommand` from Biped_Setup — characters exit T-pose. (B) Basic movement: per-episode `commands.txt` (Idle / LookAround / GoTo) executed by `character_behavior.py` attached via `ApplyScriptingAPICommand`. Numerically verified — positions change frame-to-frame via `ag.get_character().get_world_transform()`. Config knob: `characters.animation.{enabled, movement.enabled, ...}`. Reference: `s3://nvidia-isaac-bucket/trajectory-tests/20260701_stage7b2_walking/`. |
| 7.5 | next | **Revisit Carter/robot agent specifically to unlock camera-motion blur.** Make the ghost camera optionally physics-driven (kinematically-targeted rigid body with `physics:velocity` set per-frame), or fall back to mounting the camera on a moving rigid prim that PhysX tracks. Goal: distinct camera-sweep blur like the official Isaac motion_blur_short.py example. |
| 11 | ✅ done (2026-07-15) | **In-process episode mode + random-frame capture.** `--seeds "s1 s2 …" --out_root X` renders ALL seeds in ONE Isaac session: scene randomization moved to a re-fireable `on_custom_event("randomize_scene")` trigger; per episode the scene re-rolls, camera intrinsics/jitter phases re-sample, occupancy map + path rebuild, and per-episode BasicWriters write the same `<config-stem>_seed<S>/` layout the one-process-per-seed wrappers produced (converters unchanged). Layout failures retry in-process with `seed+k*1000` (`--max_seed_retries`, dir keeps the label seed). Session-level artifacts go to `<out_root>/_session_<stem>/`. Constraints: `camera_rig` agent only, no `capture.video`, no chase. **`--capture_mode random`** skips path traversal — every frame gets an independent `UniformPoseSampler` freespace pose + uniform yaw (motion-blur time samples disabled); with `--num_frames 1` each image is a fresh env re-roll = the old disconnected random frames, but with all v4 scene fixes (occupancy z-band, wall-texture diversity, corrected bounds). Throughput ~7–11 s/image vs ~65–120 s/image per-process (~10×). Runner: `run_base_v4_128rand.sh`. GOTCHA: never edit this script while a per-process wrapper loop is live — each run re-reads it from disk (burned exp07 seeds 15–19/21 on 2026-07-15). |
| 8–10 | parked | Collision detection / Nav2-style nav / dynamic obstacles — deferred. |

### Stage 4 Validation (completed 2026-06-02)

- ✅ EXIT:0, 10 frames, no errors
- ✅ `Camera/rgb/` — 10 PNG ego frames
- ✅ `Camera_chase/` — 10 PNG chase frames
- ✅ `Camera/bounding_box_2d_tight/` — 10 .npy + labels/prim_paths JSON pairs
- ✅ `video/clip_0000/` — 5 MP4s (rgb, depth, edges, segmentation, shaded_seg) + 10 PNGs each
- ✅ `run_manifest.json` — `video_files` lists all 5 MP4 paths

### Stage 3 Validation (completed 2026-05-27)

- ✅ Ego-only run (chase disabled): clean output, no `Camera_chase/` directory
- ✅ Ego + chase run: `Camera/rgb/`, `Camera/bounding_box_2d_tight/`, `Camera_chase/` all populated
- ✅ Per-modality subfolders: each enabled modality gets its own subfolder under `Camera/` via separate BasicWriter instances
- ✅ Frame count parity: rgb == bounding_box_2d_tight (10/10)
- ✅ FOV variation: `fov_std: 5` produces different FOVs across seeds (seed=0 → 75.25°, seed=42 → 76.16°); recorded per-frame in poses.jsonl and in manifest `episode_camera`
- ✅ Chase yaw fix: chase camera uses `heading_yaw` only, not `heading_yaw + yaw_rel`; verified with yaw_rel=30 — ego looks 30° sideways, chase stays on heading
- ✅ Calibration loop `discover_generated_images(Camera/rgb)`: finds 10 ego frames, PASS
- ✅ Calibration loop post-Isaac fallback fixed in `controller.py:652`: changed `discover_generated_images(output_dir)` → `discover_generated_images(local_rgb_dir)` to prevent chase frames leaking into DINOv2 embeddings

### Pre-Stage-3 Validation (completed 2026-05-27)

- ✅ `run_manifest.json` image_count correct (10/10)
- ✅ Different scene layout per run (confirmed visually)
- ✅ Within-episode frame consistency (frames 1-N identical scene, confirmed after warmup-step fix)
- ✅ Manifest schema compatible with `discover_generated_images()` (rglob, no manifest parsing needed)
- ✅ Broken USD assets removed from `sdg_config_mean_std.yaml`:
  - Removed: `SM_PaletteB_01.usd`, `SM_CardBox_A_01/04/05.usd` (not available on Isaac 5.1 content server)
- ⚠️ Seed reproducibility: Replicator's CUDA-side RNG is not controlled by `random.seed()`; per-episode layout varies across runs with the same config seed. Acceptable for SDG diversity; not fixable without low-level Replicator API.

## Implementation Stages

Each stage should be independently runnable in Isaac Sim and should produce
images that can be visually inspected before moving to the next stage.

### Stage 0: Baseline And Debug Harness ✅

Purpose:

- Preserve the current generator.
- Add a new trajectory script entry point without changing behavior yet.
- Make it easy to compare current random-frame output with trajectory output.

Implementation:

- Create `standalone_palletjack_trajectory_sdg.py`.
- Support the same CLI args as the current script:
  - `--config`
  - `--headless`
  - `--data_dir`
  - `--num_frames`
- Load YAML and write `run_config.yaml`.
- Create the output directory structure.
- Add a small debug metadata writer.

Validation in simulator:

- Run with `num_frames: 3`.
- Confirm Isaac starts and exits cleanly.
- Confirm output directory exists.
- Confirm `run_config.yaml`, `run_manifest.json`, and `isaac.log` are written.
- Confirm no changes are required in `simulation_calibration_loop` when only
  `isaac.script_path` is changed.

Exit criteria:

- New script is launchable by the calibration loop.
- No image quality expectations yet.

### Stage 1: Static Environment Plus Fixed Ego Camera Path ✅

Purpose:

- Prove the new pipeline can load a warehouse, move a camera through time, and
  write ordered RGB frames.

Implementation:

- Load `environment.name` using existing `environment_urls`.
- Create one camera rig prim.
- Drive the camera along a deterministic waypoint list from YAML.
- Use explicit simulation/capture stepping instead of `rep.trigger.on_frame`.
- Write RGB frames under `Camera/rgb`.
- Write `trajectory/poses.jsonl`.

Validation in simulator:

- Run 10-30 frames in non-headless mode.
- Watch the camera move through the warehouse.
- Confirm images are sequential and visually different.
- Confirm no random object/camera jumps occur between frames.
- Confirm `discover_generated_images(output_dir / "Camera" / "rgb")` finds the frames.
- Confirm frame count equals `run.num_frames` or the CLI `--num_frames` override.

Exit criteria:

- The output is a coherent camera trajectory, not independent snapshots.
- The calibration loop can embed the generated RGB images.

### Stage 2: Reuse Current Static Scene Randomization Per Episode ✅

Purpose:

- Bring over current palletjack/forklift/pallet/distractor generation without
  reintroducing per-frame randomness.

Implementation:

- Reuse current YAML sections:
  - `palletjacks`
  - `forklifts`
  - `pallets`
  - `pallet_stacks`
  - `distractors`
  - `distractor_randomization`
  - `materials`
  - `lighting`
- Spawn objects once at episode setup.
- Apply material and lighting randomization once at episode setup.
- Keep image augmentation and dataset noise as post-write processing.
- Keep semantics compatible with current object-detection outputs.

Validation in simulator:

- Run the same trajectory with different seeds.
- Confirm object layout changes between runs.
- Confirm object layout does not jump within a run.
- Confirm palletjack/forklift/pallet labels still appear in annotation output.
- Compare a few frames against current generator output for rough scene fidelity.

Exit criteria:

- Current random-scene visual diversity exists at the episode level.
- Trajectory temporal consistency is preserved.

### Stage 3: Camera Controls, Multi-Camera, And Annotation Parity

Purpose:

- Make camera parameters and output modalities production-compatible.

Implementation:

- Map old camera fields onto trajectory cameras:
  - `camera.camera_height_mean/std`
  - `camera.camera_tilt_mean/std`
  - `camera.camera_yaw_mean/std`
  - `camera.camera_roll_mean/std`
  - `camera.fov_mean/std`
  - `camera.camera_type`
  - `camera.clipping_range`
- Add `cameras.ego`, `cameras.chase`, and optional named cameras.
- Support BasicWriter-style RGB, 2D boxes, semantic segmentation, instance
  segmentation, and depth.
- Ensure all camera outputs are grouped predictably under each camera name.

Validation in simulator:

- Run with ego only.
- Run with ego plus chase.
- Confirm all modalities have matching frame counts.
- Confirm 2D boxes line up visually with RGB frames.
- Confirm camera settings are reflected in image appearance.

Exit criteria:

- Output parity with current generator for RGB and object-detection annotations.
- New multi-camera output is deterministic and inspectable.

### Stage 4: Image Sequence To Video Export

Purpose:

- Add optional video export while keeping image export as the primary contract
  for calibration.

Implementation:

- Add `capture.video`.
- Start with image sequence as source of truth.
- Use an Isaac/Replicator writer when reliable, using the CosmosWriter warehouse
  example as reference.
- If video writing is disabled or unavailable, image export must still succeed.

Validation in simulator:

- Generate a short trajectory with `capture.video: true`.
- Confirm images still exist.
- Confirm MP4 exists under `video/`.
- Confirm MP4 frame count and ordering match the PNG sequence.
- Confirm calibration loop still uses the RGB images, not the MP4.

Exit criteria:

- Video export works as an optional product.
- Image output remains stable.

### Stage 4 Progress (2026-05-27)

**Done:**

- All stage strings, function name, default data_dir bumped from stage_3 → stage_4
- `capture.video: true` added to `sdg_config_trajectory.yaml`
- CosmosWriter wired into `run_stage4()`:
  - Created when `capture_cfg.video` is true
  - Initialized with `output_dir=video/`, `use_instance_id=True`
  - Attached to `rp_ego` alongside existing BasicWriters
  - `video/` directory created by `prepare_output_tree` equivalent
- `on_final_frame()` + `detach()` called after `rep.orchestrator.wait_until_complete()`
- `video_files` list (MP4 paths relative to output_dir) added to manifest `episode_camera` block and `stage4_complete` event
- Run confirmed clean exit (EXIT:0); `Camera/rgb/` and `Camera/bounding_box_2d_tight/` still produced correctly

**Fixed (2026-06-02):**

Root cause was three missing setup steps required by CosmosWriter:
1. `/app/omni.graph.scriptnode/opt_in = True` — CosmosWriter's Canny edge annotator uses OmniGraph script nodes; without opt-in the annotator chain silently refuses to attach and `write()` is never called
2. `timeline.play()` before the capture loop — CosmosWriter reads timeline FPS on first `write()` and expects the timeline live
3. `pause_timeline=False` in `rep.orchestrator.step()` — CosmosWriter internals require the timeline to keep running between steps

Fix: added all three in `run_stage4()`. Ghost camera still works correctly since it's driven by explicit USD translate/rotate ops, not physics.

### Stage 5 Validation (completed 2026-06-29)

- ✅ EXIT:0, 30 frames, no errors
- ✅ `Camera/rgb/` — 30 PNG ego frames
- ✅ `Camera_chase/` — 30 PNG chase frames
- ✅ `Camera/bounding_box_2d_tight/` — 30 .npy annotation files
- ✅ `nav/planned_path.json` — path found on attempt 1; 13.887m path, 3 waypoints; start (-2.6, 8.3) → end (-4.5, -4.8)
- ✅ `nav/map.png` + `nav/map.yaml` — occupancy map saved
- ✅ `video/clip_0000/` — 5 MP4s (rgb, depth, edges, segmentation, shaded_seg) × 30 frames each
- ✅ `trajectory/poses.jsonl` — 30 frames; frame-0 pos matches path start, frame-29 pos matches path end
- ✅ `trajectory/events.jsonl` — all stage5 events fired including `stage5_occupancy_path_sampled`
- ✅ `run_manifest.json` — image_count: 30, video_files lists all 5 MP4 paths
- ✅ `discover_generated_images(Camera/rgb)` — finds 30 frames, PASS

### Stage 6 Validation (completed 2026-06-29) — v1 kinematic Carter

- ✅ EXIT:0, 60 frames, no errors
- ✅ Nova Carter loaded as USD reference under `/World/Carter/body` (parent Xform owns the ops because the Carter USD root already has translate+orient+scale)
- ✅ Ego camera auto-discovered: `/World/Carter/body/chassis_link/sensors/front_hawk/left/camera_left` (FOV overridden to 75°)
- ✅ `Camera/rgb/` — 60 PNG ego frames from robot's front Hawk
- ✅ `Camera_chase/` — 60 PNG chase frames (robot body visible behind)
- ✅ `Camera/bounding_box_2d_tight/` — 60 .npy files
- ✅ `nav/planned_path.json` — 14.6m path, 3 waypoints; start (-2.6, 8.3) → end (-2.7, -6.3)
- ✅ `video/clip_0000/` — 5 MP4s × 60 frames
- ✅ `trajectory/poses.jsonl` — includes `agent_type=carter`, `agent_prim`, `agent_pos`, `agent_yaw_deg`

**Known limitation (v1):** Carter is kinematically posed each frame (SetTranslateOp+SetRotateXYZOp), so the body slides but wheels don't physically rotate. Frame-to-frame motion is continuous (~25 cm/frame). Stage 6.2 candidate work: drive via `DifferentialController.apply_wheel_actions` + pure-pursuit (MobilityGen pattern) so wheels actually spin.

### Stage 6 v1.1 fixes (2026-06-29)

Initial v1 run had the robot tipping over from colliding with cones/barrels/pallets (props below the z=1.6m path scan), because the Nova Carter USD's articulation is dynamic by default — PhysX accumulated tilt across frames once we teleported into a prop. Fix:

- Strip `UsdPhysics.ArticulationRootAPI` from the Carter subtree
- Set `kinematicEnabled=True` on every `RigidBodyAPI` under the robot (chassis_link, wheels, sensors)
- Spawn Carter at (-20, -20, 0) — off the occupancy map — so its footprint doesn't block path planning during the omap scan
- Switch ego from `front_hawk/left/camera_left` (offset stereo eye that hugs shelf walls in narrow aisles) to `front_owl/camera` (centered fisheye)

Behavior now matches spec:
- ✅ Planner avoids walls/shelves at z=1.6m scan
- ✅ Kinematic chassis pushes dynamic props (cones, barrels) without tipping
- ✅ Centered Owl camera gives clean ego views down warehouse aisles

### Stage 6 v2 → ghost-camera pivot (2026-06-30)

**Why the pivot:** After v1.1, we attempted v2 (physics-driven Carter via `DifferentialController` + pure-pursuit). Robot was upright and respected static walls, but two problems emerged:

1. **Carter got stuck constantly** — narrow aisles + heading drift = grinding into walls. Wheel friction integration also caused ~15% commanded speed efficiency, so most trajectories ran out of frames mid-path.
2. **For an OD training-data pipeline, the robot body is not in the labels.** Frames showing the Carter chassis are wasted pixels; the model never sees a Carter at inference. The "physical robot" abstraction adds complexity without OD value.

**Direction change:** Default agent is now `camera_rig` (ghost). Stages 1–5 already used this; we extend it with full mount config + per-frame jitter. Carter remains available (`agent.type: carter`) for users who do need AMR-style data with the body in view.

**What stage 6 became:** a configurable camera platform on top of the existing occupancy-path planner.

#### New camera config schema (all Optuna-searchable)

```yaml
cameras:
  ego:
    resolution: [960, 544]
    fov_mean: 75.0
    fov_std: 0.0
    projection: perspective       # perspective | orthographic
    height_m: 1.4                 # mount Z (ghost) / above chassis_link (carter)
    pitch_deg: 0.0                # static nose tilt (+ up)
    roll_deg: 0.0                 # static side tilt (+ right side down)
    pitch_jitter:                 # sinusoidal handheld / uneven-terrain feel
      amp_deg: 2.0
      hz: 1.5
    roll_jitter:
      amp_deg: 1.5
      hz: 1.2
    mount:                        # carter-only chassis-relative offsets
      forward_m: 0.30
      lateral_m: 0.00
```

#### Stage 6.1 outcome (completed 2026-06-30)

All five steps wired, four with strong visible effect:

1. ✅ **DOF (`fStop` + `focusDistance`)** — verified at f/1.4 (extreme bokeh) and f/5.6 (realistic warehouse-cam look). s3: `20260630_cam_stack_dof/`
2. ✅ **Position jitter** (`lateral_jitter_m`, `vertical_jitter_m`) — sinusoidal with own hz + per-episode phase; poses.jsonl shows oscillation, frames visibly shake at amp_m=0.15.
3. ✅ **Focal length override** (`focal_length_mm`) — bypasses FOV; verified 8mm gives 105.3° back-solved FOV with ultra-wide stretching.
4. ✅ **Fisheye via OpenCV post-render** — writes Camera/rgb_fisheye/ alongside perspective. OD labels remain valid for the perspective frames.
5. ⚠️ **Motion blur** — Wiring done (camera shutterOpen/shutterClose attrs in time codes, time-sampled translate/rotate ops at shutter open + close, `/omni/replicator/captureMotionBlur=True`, RTX motion-blur tuning). USD diagnostic confirms 61 time samples written at correct time codes; shutterClose set to 4.0 tc matching frame_dt × tcps. **But:** the RTX pipeline (both RaytracedLighting and PathTracing tested) integrates blur from object velocities — it does not appear to integrate camera-xform time samples for camera-sweep blur. Verified with deterministic A/B (same scene, same path, no props): shutter=0 vs shutter=1.0 produces only sub-1% pixel difference. Reference: official `motion_blur_short.py` shows dramatic blur on physics-driven objects (Cheez-It boxes) but its camera is static. **Resolution deferred to Stage 7.5.**

### Stage 7: Animated Human Characters As OD Targets

Pivot from the original "human-like agent driving the camera" framing — for an OD pipeline, the value is humans **in the scene as labeled detection targets**, not humans driving the trajectory.

Purpose:
- Add `person` to the set of detectable classes
- Realistic warehouse staffing (operator walking aisle, picker pushing cart) so trained OD models generalize to environments with humans

Implementation candidates:
- **Isaac `actor_sdg`** (`/opt/IsaacSim/source/tools/actor_sdg/`) — scheduler-driven character SDG with built-in animation paths
- **Omniverse Characters** — animated character prims with walk/idle motion clips
- **NVIDIA people assets** under `/Isaac/People/...` on the Nucleus asset server

Per-episode randomization:
- Number of characters (0–N)
- Character variants (multiple body/clothing assets)
- Path mode (random waypoints in free space, or scripted aisle walks)
- Animation (walk, idle, push-cart) — sampled per character
- Speed jitter

Validation:
- ✓ Characters appear in RGB ego frames at varied poses
- ✓ Each character carries class semantic `person`
- ✓ BasicWriter's `bounding_box_2d_tight` outputs include person bboxes
- ✓ At least one episode demonstrating multiple humans + palletjacks + forklifts coexisting

### Stage 7b Validation (completed 2026-07-01)

**Wiring done in `run_stage4()`:**
- Enable `omni.kit.scripting`, `omni.anim.timeline`, `omni.anim.graph.core`, `omni.anim.retarget.core`, `omni.anim.navigation.core`, `omni.anim.people` right after `SimulationApp` starts, before `open_stage()`.
- Set `/persistent/exts/omni.anim.people/character_prim_path = /World/Characters`, disable navmesh + dynamic-avoidance (straight-line GoTo is "very basic" enough for OD), enable `/rtx/raytracing/fractionalCutoutOpacity`.
- Load `Biped_Setup.usd` invisibly under `/World/Characters/Biped_Setup`.
- For each spawned character's `SkelRoot`: `ApplyAnimationGraphAPICommand(paths=[skel_root], animation_graph_path=<biped's AnimationGraph>)` + `ApplyScriptingAPICommand(paths=[skel_root])` + `omni:scripting:scripts = [path/to/character_behavior.py]`.
- Emit `characters/commands.txt` per episode with a random Idle / LookAround / GoTo mix (bounded to `spawn_bounds_xy`); point `/exts/omni.anim.people/command_settings/command_file_path` at it; set `number_of_loop = inf`.

**Three non-obvious gotchas (each one silently stalled the walk):**
1. **`/app/scripting/ignoreWarningDialog = True`** — `omni.kit.scripting` gates BehaviorScript execution behind a security popup. Headless can't dismiss it, so scripts silently never load. Without this flag every diagnostic looks fine (scripts registered, timeline playing) but nothing ticks.
2. **`ag.get_character(...).get_world_transform()`** is the ground truth position. The anim graph writes to fabric, not USD, so `xformOp:translate` and even `ComputeLocalToWorldTransform(Usd.TimeCode.Default())` return the spawn position forever. Only the anim.graph.core API sees the animated transform.
3. **~20 `simulation_app.update()` after attaching scripts** — the ScriptManager's dirty-USD processing is async. If `timeline.play()` fires before those ticks land, `on_play` runs on an empty script list.

**Results:**
- ✅ EXIT:0, 60 frames, 5 characters, all cycling Idle → GoTo → LookAround
- ✅ Position deltas confirm walking (e.g. person_01 (-0.8, 0.4) → (-1.7, 1.1) → (0.99, 0.76))
- ✅ Frame 5 shows a character mid-stride (legs apart)
- ✅ Artifacts: `s3://nvidia-isaac-bucket/trajectory-tests/20260701_stage7b2_walking/`

**Config surface** (`sdg_config_stage6.yaml`):
```yaml
characters:
  animation:
    enabled: true          # (A) exit T-pose, use Biped anim graph
    movement:
      enabled: true        # (B) walk via commands.txt
      commands_per_char: 3
      idle_prob: 0.4
      lookaround_prob: 0.2
      goto_prob: 0.4
      idle_duration_range: [3.0, 6.0]
      lookaround_duration_range: [5.0, 10.0]
      goto_max_dist_m: 2.5
```

### Stage 7.5: Physics-Driven Camera For Motion Blur

Purpose:
- Unlock the RTX motion-blur path that needs `physics:velocity` on a moving prim — kinematic teleport doesn't engage it (see Stage 6.1 outcome #5)

Approach options (try in order):
1. **Kinematic-target rigid body**: make the camera prim a kinematic rigid body, write `physics:velocity` each frame derived from path interpolation, set `kinematicEnabled=True` and `kinematicTarget` = next frame pose
2. **Carter-mounted ghost camera**: when `agent.type: carter`, the camera is already on the chassis_link (a physics body) — verify whether THAT produces blur and use it as the precedent
3. **Parent the camera prim to a rigid-body Xform that physics drives** along the planned path

Validation:
- Same A/B harness as Stage 6.1#5 (deterministic waypoints, no scene randomization)
- Pass criterion: ON vs OFF show clearly distinct blur on warehouse wall texture, comparable to Isaac's official example

### Stage 5: Occupancy Map And Random Free-Space Trajectories

Purpose:

- Move beyond hardcoded waypoints and generate valid paths through the scene.

Implementation:

- Add `trajectory.mode: occupancy_path`.
- Load or generate an occupancy map for the environment.
- Sample random start/end points in free space.
- Generate and compress a path using MobilityGen-style helpers.
- Save the planned path under `nav/planned_path.json`.
- Add debug rendering of the occupancy map and path if possible.

Validation in simulator:

- Run many short trajectories.
- Confirm starts and goals are inside free space.
- Confirm camera path avoids shelves and large static obstacles.
- Confirm planned path is saved and can be replayed.
- Confirm failed path sampling is visible in logs and does not silently produce
  invalid output.

Exit criteria:

- Random trajectories are valid enough to run unattended in batches.
- Every generated path is inspectable and replayable.

### Stage 6: Physical Robot Agent

Purpose:

- Replace the ghost camera rig with an actual robot body and attach cameras to
  robot frames.

Implementation:

- Add `agent.type: carter`.
- Load a Carter/Nova Carter USD, following MobilityGen robot patterns.
- Attach or use an existing front camera.
- Drive with differential or policy-style velocity commands.
- Keep ego camera capture identical from the calibration loop's perspective.
- Add optional chase camera for debugging.

Validation in simulator:

- Run in non-headless mode.
- Confirm robot body is visible from chase camera.
- Confirm ego view moves with robot front camera.
- Confirm robot follows path smoothly.
- Confirm robot does not teleport between frames.
- Confirm output image layout remains compatible with previous stages.

Exit criteria:

- Physical robot trajectories produce the same data products as camera-rig
  trajectories.

### Stage 7: Human-Like Or Animated Agent (superseded)

The original Stage 7 framed humans as the trajectory-driving AGENT (H1 humanoid, animated characters carrying the camera). After the Stage 6 ghost-camera pivot for OD, this framing no longer adds OD value — the agent is just transport for the camera, and ghost is simpler. The active Stage 7 (humans-as-SCENE-targets) is described above in the progress log.

### Stage 8: Collision Detection And Episode Events

Purpose:

- Make trajectory validity explicit and debuggable.

Implementation:

- Add collision state to `trajectory/events.jsonl`.
- Add `trajectory.stop_on_collision`.
- Add collision counters and final episode status to `run_manifest.json`.
- Mark frames where the agent is blocked, collided, or recovered.
- Support replay/debug mode for a failed run.

Validation in simulator:

- Intentionally run a path through an obstacle.
- Confirm collision is detected.
- Confirm behavior matches YAML:
  - stop
  - continue
  - resample
- Confirm metadata clearly reports the collision frame and object/prim if
  available.

Exit criteria:

- Invalid trajectories are observable and traceable.
- Batch runs do not silently produce bad paths.

### Stage 9: ROS2/Nav2-Style Navigation Agent

Purpose:

- Add a navigation-agent mode that can later use ROS2 or ROS2-like interfaces.

Implementation:

- Add `agent.type: ros2`.
- Add `trajectory.mode: nav_agent`.
- Define YAML for:
  - map source
  - localization source
  - goal list
  - replanning policy
  - max stuck time
  - recovery behavior
  - command topic or internal command adapter
- Start with an internal adapter that mimics ROS2 command inputs.
- Add real ROS2 bridge only after internal navigation semantics are stable.

Validation in simulator:

- Send one goal.
- Confirm agent moves toward goal.
- Confirm agent reports success/failure.
- Send multiple goals.
- Confirm each goal status is written to metadata.
- Test blocked path and recovery behavior.

Exit criteria:

- Navigation behavior is controlled by YAML and recorded in metadata.
- The implementation can later swap internal commands for real ROS2 bridge
  commands without changing the calibration loop contract.

### Stage 10: Dynamic Obstacles And Complex Scenes

Purpose:

- Support richer simulation settings for harder data generation.

Implementation:

- Add moving distractors or forklifts.
- Add dynamic human/robot agents.
- Add traffic rules or simple behavior policies.
- Add multi-agent settings:
  - ego agent
  - background agents
  - target objects
  - moving distractors
- Add environment variants from SceneBlox or custom USDs.
- Add configurable weather/lighting/material schedules if needed.

Validation in simulator:

- Confirm moving objects are temporally smooth.
- Confirm labels/boxes follow moving targets.
- Confirm ego camera does not overlap dynamic obstacles unless collision tests
  intentionally request it.
- Confirm long runs remain stable.
- Confirm image/video/metadata outputs stay synchronized.

Exit criteria:

- Complex scenarios can run in batch mode.
- Failures can be traced through metadata and replay artifacts.

## Calibration Loop Updates

Only make loop changes after the trajectory script works directly.

Expected updates:

- Add trajectory themes to `SEARCH_SPACE_THEMES`:
  - `agent.speed_mps`
  - `agent.turn_rate_dps`
  - `trajectory.lookahead_m`
  - `trajectory.smoothing`
  - `cameras.ego.height_m`
  - `cameras.ego.fov_mean`
  - `cameras.ego.fov_std`
  - `capture.video`
- Add new seed config directory for trajectory runs:
  - `palletjack_sdg/experiments/trajectory/base_v1`
- Add project config pointing at:
  - `palletjack_sdg/standalone_palletjack_trajectory_sdg.py`
- Keep `run.data_dir`, `run.num_frames`, and `run.headless` excluded from
  optimization unless explicitly needed.

Validation:

- Run one calibration-loop smoke test with `iteration_batch_size: 1`.
- Confirm YAML materialization includes trajectory sections.
- Confirm Isaac outputs images.
- Confirm embeddings are computed.
- Confirm state records the new generated YAML and output directory.

## Debugging Rules

Use small, visual checkpoints before adding complexity:

1. Non-headless first, then headless.
2. Three frames first, then 30, then 128+.
3. One camera first, then multiple cameras.
4. No objects first, then static objects, then dynamic objects.
5. Ghost camera first, then robot body, then ROS2/navigation agent.
6. Fixed path first, then random waypoints, then occupancy-map paths.
7. Images first, then annotations, then video.

Every stage should include at least one run that can be inspected by opening:

```text
<output_dir>/Camera/rgb/
<output_dir>/trajectory/poses.jsonl
<output_dir>/run_manifest.json
```

## Risks And Mitigations

### Risk: Current Script Is Too Monolithic

Mitigation:

- Do not rewrite the old script in place.
- Build the trajectory generator as a parallel script.
- Reuse small helpers from `palletjack_sdg/utils`.

### Risk: Calibration Loop Cannot See Images

Mitigation:

- Always write RGB images under `Camera/rgb`.
- Keep image filenames sorted and stable.
- Validate with `discover_generated_images(output_dir / "Camera" / "rgb")`.

### Risk: Per-Frame Randomization Breaks Temporal Consistency

Mitigation:

- Randomize scene layout once per episode.
- Randomize camera and agent parameters once per episode.
- Use motion for per-frame visual change.

### Risk: Navigation Adds Too Much Complexity Early

Mitigation:

- Start with ghost camera waypoints.
- Add physical robot only after image/metadata output is stable.
- Add occupancy maps before ROS2/Nav2-style control.

### Risk: Video Export Distracts From Calibration

Mitigation:

- Treat PNG image sequence as the primary data product.
- Treat MP4 as optional derived output.

## Recommended First Implementation Slice

Start with Stage 0 and Stage 1 only:

1. New script with compatible CLI.
2. Load warehouse.
3. Move an ego camera along two or three YAML waypoints.
4. Write `Camera/rgb` images.
5. Write `trajectory/poses.jsonl`.
6. Run once directly in Isaac.
7. Run once through `simulation_calibration_loop` with `iteration_batch_size: 1`.

Do not add robot physics, occupancy maps, ROS2, or video until this minimal
trajectory pipeline is producing the expected images.

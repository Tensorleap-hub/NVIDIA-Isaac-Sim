# base_v4 — generated image review

Reviewer comments on images generated from the `base_v4` trajectory yamls.

## Summary of recurring issues

1. **Camera clips through walls / shelves** — the dominant problem, seen across
   many seeds (exp01, 02, 04, 06, 07, 09, 13, 16, 17). Sometimes the camera even
   passes through a wall into a new area. This should not be allowed.
   → **[FIXED, verified]** Root cause was NOT missing collision (the warehouses
   are 100% collided). The occupancy scan was a zero-thickness slice at z=1.0m, so
   pallet racks (thin posts + gaps at that height) scanned as free floor and the
   planner routed straight through them. Now scans a vertical band. See Fixes #2.
2. **Objects placed in/behind walls** — object placement has no boundaries either.
   → **[OPEN]** Targets/distractors use `scatter: uniform` across `bounds_xy`,
   whose edges abut the walls, so some land in/behind them. Same family as the
   "edge of scene" dark frames (camera near a bound edge stares at the dim
   perimeter). Fix direction: inset scatter + camera path well inside the walls
   (raise `boundary_margin_m` / `scatter_inset_m`), or clip both to occupancy
   freespace. Not yet done.
3. **Lighting knob has no visible effect** — `intensity_mean` changes don't read;
   internal warehouse lights dominate (exp14 night, exp15 very-dark both look lit).
   → **[OPEN, diagnosed]** The lighting randomization only drives `RectLight`
   prims; the warehouse USD's own ceiling/dome lights dominate and are untouched,
   so night/very-dark stay lit. Fix direction: also scale/disable the env's
   built-in lights (or add exposure control). Not yet done.
4. **Distractors / clutter not appearing** — configured in yaml but missing or too
   few in renders (exp01 none, exp05 too few, exp20 "dense" not dense).
   → **[FIXED, verified]** Distractors were pinned to a central Gaussian while
   targets moved to uniform scatter + a roaming tight camera, so they were off-
   frame. Now uniform-scatter (co-located with targets) + raised `clutter_level`.
   See Fixes #1.
5. **Tight-set camera variance is invisible** — fov/height/pitch/low-angle look
   identical across exp01–10 (exp07 mid-fov, exp09 low-angle-closeup, exp10 == exp08);
   pitch may render as roll (exp12); "low" configs aren't low (exp19). FOV only
   reads at the extreme (exp18, 90°). Overhead height does read (exp11).
   → **[OPEN]** Needs: widen FOV deltas beyond the 53–60 band; push "low" configs
   to near-floor height; and verify the pitch→roll suspicion (exp12 -18° reading
   as roll) in how `pitch_deg` maps to the camera euler. Not yet done.
6. **Naming vs reality mismatches** — "tight", "steep downward", "night/very dark",
   "low", "dense clutter" don't match what the images show.
   → **[PARTIAL]** Resolves as #2/#3/#5 land; "dense clutter" (exp20) should now
   read after Fix #1 (clutter_level kept at its own high value + uniform scatter).

Debug/next-step ideas (see General notes): draw allowed trajectories from the env
nav/occupancy map and curate an acceptable-trajectory list; let Optuna control the
start point (favor near-shelf views); add more camera variation, objects, textures,
and noise.

## Fixes applied (SDG dev pass, 2026-07-06)

Code: `standalone_palletjack_trajectory_sdg.py`; configs regenerated via
`_generate_base_v4.py`. All 21 base_v4 yamls regenerated. Verified by smoke
renders of exp01 (distractors) and exp02 (wall-clipping); consolidated full
re-run pending.

1. **Distractors now co-located + denser** (issue #4). Distractor pose routes
   through `_scatter_position()`, honoring `scatter: uniform` (backward-compatible
   Gaussian fallback). Generator `FORCE_DISTRACTORS` sets `scatter: uniform` +
   `clutter_level: 3.0` on all non-density configs; `SCATTER_ONLY_DIST` gives
   dense/sparse uniform scatter while keeping their own counts (exp20=5.0,
   exp21=0.1). Verified: exp01 frames now show barrels/bottles/signs/buckets/
   crates across the floor.
2. **Occupancy scan = vertical band, not a slice** (issue #1). `set_transform`
   now scans `[scan_z_min_m, scan_z_max_m]` (defaults 0.1 → `z_slice_m`+1 ≈ camera
   height+1) instead of min_z=max_z=0. Racks/shelves/walls become solid obstacles.
   `buffer_m` raised 0.4 → 0.6 (generator `OCC_TUNE`) for edge clearance. Verified
   on exp02: occupied cells 24% → 33%; planned path routes around the racks; no
   through-wall clipping in renders; planner still finds long valid paths.
3. **Nav path-overlay QA artifact** (reviewer's requested debug aid). Each run now
   writes `nav/path_overlay.png`: the planned path (blue), start (green), end (red)
   over the occupancy grid with the `buffer_m` inflation shown as a red wash —
   so wall/shelf clipping is visually debuggable per run. `_save_path_overlay()`.

## Tight set (from base_v3)

- **exp01_tight_eye_level_bright**
  - No distractors appear in the images, yet distractors are configured in the yaml.
  - seed 123: camera seems to enter/clip through the wall — this should not be allowed.
  - seed 456: lighting seems to change midway through the trajectory.
  - seed 789: weird lights and scene.
- **exp02_low_pov_closeup**
  - seed 42, 123, 456: camera enters wall.
  - Distractors do seem to appear in this scene (unlike exp01).
  - seed 789: seems to go through a wall into a new place.
  - Idea: adding a nav-map / occupancy-map overlay could help debug the wall-clipping.
- **exp03_forklift_yard_tight** — looks good.
- **exp04_dim_shift_tight** — looks ok overall.
  - seed 456: camera seems to start off-scene at some point.
  - seed 789: just going through the shelves — no intelligible image.
- **exp05_multishelf_aisle_tight** — not enough distractors; otherwise ok.
- **exp06_bright_dense_mixed**
  - seed 42: walking through boxes, nothing visible.
  - seed 123: walking through a wall.
- **exp07_mid_fov_survey** — "mid-fov"? Camera looks the same as exp01–06; only env and objects differ, not the camera. (fov 60 vs 53–56 elsewhere is too subtle to read — the FOV variance in the tight set isn't visible.)
  - Same going-through-wall issue.
- **exp08_plain_warehouse_tight** — looks all right. What is "tight" supposed to mean here? (naming unclear)
- **exp09_low_angle_closeup** — camera again looks the same; no low angle, no close-up visible.
  - Same wall issue as previously mentioned.
- **exp10_reference_tight** — looks exactly like exp08.

## Variance imports (from base_v1)

- **exp11_overhead_survey** — high view! good. (camera height difference is finally visible here)
- **exp12_steep_downward** — "steep downward"? I see a horizontal tilt (roll), not a downward pitch, and maybe a slightly lower height. The -18° pitch is not reading as downward. (possible pitch/roll mix-up or pitch not applied)
- **exp13_narrow_telephoto** — all seeds behind a wall.
- **exp14_night_shift_dim** — "night shift"? External lighting seems to have no effect compared to the internal lights (the `intensity_mean` drop to 40k isn't reading — internal warehouse lights dominate).
- **exp15_very_dark** — "very dark" in name, but lights are on and the scene is very bright. The 15k `intensity_mean` has no visible effect — confirms the lighting knob doesn't control the dominant (internal) lights. Same root cause as exp14.
- **exp16_bright_daytime** — pretty generic; going into walls, but that's the general issue.
- **exp17_running_operator** — pretty generic; going into walls, but that's the general issue.
- **exp18_wide_low_fast** — does seem like a wide shot (the 90° FOV reads). Good — confirms FOV variance is visible at the extreme, just not within the tight 53–60 band.
- **exp19_patrol_robot_low** — "low"? not really. Push it to near-floor height (1.2m isn't reading as low).
- **exp20_dense_clutter** — not dense and no clutter (the clutter_level 2.4 / 8 characters aren't showing up).
- **exp21_sparse_minimal** — it is what it says (sparse/minimal reads correctly).

## General notes

- Note to self: do I actually see camera-height differences across configs? (verify the `height_m` variance is showing up in the renders)
- General (not only exp06): using the environment navigation/occupancy map, we could draw the allowed trajectories to debug/validate paths (recurring wall- and shelf-clipping across seeds).
- Haven't seen a proper walk *between* shelves yet — reinforces the idea of using the map to build a curated list of acceptable trajectories.
- Optuna optimization should be able to control the trajectory starting point (and other minimal but distinct options). Idea: images near shelves are likely better, so let Optuna explore there and decide they beat roaming in the open floor.
- A lot of objects are placed inside or behind walls — need to set clear boundaries for object placement (not just the camera path).
- Wishlist for next iteration: more camera variations; more objects and more variety in how objects are placed/arranged; more textures; add noise as well.

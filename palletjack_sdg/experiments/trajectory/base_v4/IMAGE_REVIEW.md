# base_v4 — generated image review

Reviewer comments on images generated from the `base_v4` trajectory yamls.

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
- **exp08_plain_warehouse_tight** —
- **exp09_low_angle_closeup** —
- **exp10_reference_tight** —

## Variance imports (from base_v1)

- **exp11_overhead_survey** —
- **exp12_steep_downward** —
- **exp13_narrow_telephoto** —
- **exp14_night_shift_dim** —
- **exp15_very_dark** —
- **exp16_bright_daytime** —
- **exp17_running_operator** —
- **exp18_wide_low_fast** —
- **exp19_patrol_robot_low** —
- **exp20_dense_clutter** —
- **exp21_sparse_minimal** —

## General notes

- Note to self: do I actually see camera-height differences across configs? (verify the `height_m` variance is showing up in the renders)
- General (not only exp06): using the environment navigation/occupancy map, we could draw the allowed trajectories to debug/validate paths (recurring wall- and shelf-clipping across seeds).
- Haven't seen a proper walk *between* shelves yet — reinforces the idea of using the map to build a curated list of acceptable trajectories.

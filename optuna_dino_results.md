# Optuna trajectory-SDG calibration — DINOv2 theme rounds (2026-07-09 → 07-11)

Results summary for the first full theme-rounds Optuna run over the trajectory
SDG, scored by MMD-to-real on stock DINOv2 embeddings. Plan of record and
design rationale: [optuna_search_trajectory.md](optuna_search_trajectory.md).

## Headline

**Best seed 0.4092 → final promoted 0.32604 — a ~20% MMD-to-real reduction**
over the hand-authored base_v4 seeds, from 600 scored trials
(5 themes × 3 rounds × 10 iterations × 4 candidates, each candidate = 4
seed-sims × 15 frames pooled before embedding).

## Setup

- **Objective:** `mmd_rbf` between pooled synthetic embeddings and a fixed
  858-image real reference set (loco_dataset + Roboflow valid), embedder
  `dinov2_vitb14_reg` (768-d, torch hub) — chosen over the RF-DETR backbone
  used in the prior (aborted) run.
- **Themes (per round):** camera (merged intrinsics+mount, 8 params), jitter
  (11), agent (3), scene (14 incl. palletjack color), distractor-occurrence
  (9). `environment.name` searched **jointly inside every theme** — the
  standalone environment theme was removed after round-1 evidence showed a
  4-way categorical searched alone wastes a full pass.
- **Search plumbing that mattered:** TPE `n_startup_trials: 10` (the
  `max(50, 3·params)` default left 10-iteration themes on pure random
  search), and pool priming with all 60 observations (22 base_v4 seed
  anchors backfilled + prior trials) at every theme start.
- **Promotion:** monotonic guard — a theme's best full YAML is promoted to
  the shared baseline only if it beats the standing objective; every later
  theme materializes candidates on top of it.

## Result trajectory

| Step | Theme (round) | MMD | What changed |
|---|---|---|---|
| seeds | exp16_bright_daytime best anchor | 0.4092 | — |
| 1 | camera (r1) | **0.3661** | FOV 105.5°±1.1 (ultrawide), height 2.48 m, pitch +2.4°, f/6.9 @ 5.9 m DoF |
| 2 | scene (r1) | **0.35747** | palletjack body tint, warm bright lighting (~169k), roughness 0.40, clutter 1.9 |
| 3 | occurrence (r3) | **0.32604** | cardboard-heavy distractors: CardBox 4 / TrafficSigns 3 / Bucket 3 / CratePlastic 2, **zero** barrels & pushcarts, clutter 3.0 |

All three promotions read as genuine domain alignment with the loco footage
(high wide-angle mounted cameras with real lens optics; box-and-cone floor
clutter). `full_warehouse` won every contested environment trial — partly
real signal, partly the known bounds confound (see caveats).

**Negative results (equally load-bearing):**
- Camera jitter never helped in 3 rounds (best 0.3651 vs incumbent) — the
  metric does not reward added shake.
- Agent speed/turn barely matters: three near-misses at 0.358–0.365 across
  rounds, never past the bar.
- Rounds 2 and most of 3 promoted nothing — the config converged early; the
  late occurrence win shows the *composition* axis was under-explored, not
  the continuous knobs.

## Deliverables

- **S3 (consolidated):** `s3://nvidia-isaac-bucket/trajectory-tests/20260712_optuna_dino_results/`
  — `best.yaml` + `best.json`, the three top-k exports
  (`best_top10{,_diverse,_diverse_latent}.yaml`), `manifest.json`, and
  `top_performers/rank01..10_*/` with each trial's `config.yaml` and full
  per-seed data (`Camera/rgb`, `Camera/bounding_box_2d_tight`).
- **S3 (per theme/round):** `s3://nvidia-isaac-bucket/trajectory-tests/20260708_optuna_theme_rounds_dino/<theme>/round_{01..03}/` — top-3 trials per study.
- **Local:** promoted dir `simulation_calibration_loop/promoted_baseline_trajectory_dino/`
  (best.yaml, pool with 60 scored observations), workspaces under
  `simulation_calibration_loop/rounds_ws_20260710_weekend/`, log
  `rounds_20260711... (rounds_weekend.log)`.
- Global top-10 (from `manifest.json`): ranks led by occurrence_r03
  iter006_run002 (0.32604) and scene_r01 iter009_run000 (0.35747); agent
  studies fill 5 of the next 8 slots — the agent axis hovers at the incumbent.

## Caveats & incidents

- **Top-k export scope:** `best_top10*.yaml` are per-THEME snapshots
  (overwritten by each study); the shipped ones are from occurrence_r03. The
  consolidated `top_performers/` folder is the *global* ranking instead.
- **Env/bounds confound:** `trajectory.bounds_xy` does not switch with
  `environment.name`; non-full_warehouse trials inherit full_warehouse
  bounds and get penalized partly for the wrong reason. Proper fix: couple
  bounds to env in the SDG script before trusting the env axis.
- **Deterministic layout failures:** the pose sampler can fail
  ("high <= 0") for a specific (config, seed) layout; this wedged camera_r02
  in an infinite outer-retry loop for hours. Fixed in `controller.py` by
  porting the shell runner's seed re-roll (seed + k·1000, `isaac-seed-retry`
  log marker).
- Transient Omniverse CDN fetch failures crashed single Isaac runs
  throughout; the retry wrapper recovered all of them.

## Next steps

1. Eyeball `top_performers/rank01..03` RGBs (S3) for visual sanity.
2. Build a training dump from `best.yaml` (optionally mixing the
   diverse top-k for coverage) and train RF-DETR against the
   `rfdetr_traj_v4b_base` baseline — does −20% MMD translate to +mAP?
3. If the env axis matters going forward, implement env-coupled bounds_xy.
4. Staged themes still unexplored: distractor-diversity, exploration-bounds,
   characters (crash-blocked).

# Cosmos v4: Base vs. Optuna-Optimized

## Goal

Compare Cosmos-augmented results between the trajectory **v4 base** config
(`palletjack_sdg/experiments/trajectory/base_v4` / `cosmos_v4`) and the
**Optuna-optimized** trajectory config (from the search described in
`optuna_search_trajectory.md`), to see whether Cosmos augmentation still
leaves a measurable difference between the two.

## How this fits into the bigger picture

This is **Pipeline A vs. Pipeline B** from `optimization-recipe.md`, extended
with a DINOv2/MMD check as a gate before paying for two fine-tunes:

```
base_v4 (hand-picked)      ──Isaac──▶ cosmos_v4 clips ──Cosmos──▶ augmented A
optuna winner (searched)   ──Isaac──▶ cosmos_*  clips ──Cosmos──▶ augmented B
                                                                     │
                                              DINOv2 embed + MMD(A, B)  ◄── this track's new step
                                                                     │
                                                          S3 upload, then
                                                     RT-DETR/RF-DETR fine-tune
                                                     A vs B, compare mAP
```

`optuna_search_trajectory.md` is the search itself (Isaac → DINOv2 → Optuna,
no Cosmos, scored against real LOCO frames). This track picks up **after** a
winner config exists: put both configs through the Cosmos-profile generator,
and instead of jumping straight to two fine-tunes, first check with MMD
whether Cosmos augmentation washes out the gap Optuna found.

## Plan → code map

### 1. Generate

- Base v4 → Cosmos-profile clips: already wired.
  `palletjack_sdg/run_cosmos_v4_train.sh` → `standalone_palletjack_trajectory_sdg.py`
  (`CosmosWriter`, Stage 4) → `palletjack_sdg/palletjack_data/trajectory/cosmos_v4_*/`.
  Only a curated 6-of-32 `base_v4` configs are materialized as `cosmos_v4/expNN...yaml`
  today (see `CURATED` in `_generate_cosmos_v4.py`).
- Optuna winner → Cosmos-profile clips: **no config exists yet.**
  `_generate_cosmos_v4.py` is the pattern to copy (it takes a base config and
  applies only the Cosmos-profile deltas — `num_frames`, `capture_dt`,
  `capture.video`, `short_path_fill` — verbatim style otherwise). Once a
  winner YAML exists, the same script (or a copy pointed at the winner) plus
  a `run_cosmos_optuna_train.sh` twin of `run_cosmos_v4_train.sh` produces the
  matching clip set.
- **Same prompt data**: the actual Cosmos text-prompt / stylize step is
  **external to this repo** — `optimization-recipe.md`'s "Cosmos post-process"
  is marked TBD there, and there's no prompt config anywhere in
  `standalone_palletjack_trajectory_sdg.py` or `experiments/trajectory/cosmos_v4/*`.
  This has to be pinned down (one fixed prompt/command invoked identically for
  both clip sets) before step 1 can be called done.

### 2. Compare (DINOv2 MMD)

- The MMD building block already exists and is generic:
  `DistributionMetrics.mmd(X, Y)` in `calibration_optuna/metrics.py` (RBF
  kernel, median-heuristic gamma — same math the Optuna loop uses to score
  against real data).
- Embedding two directories of images already exists too:
  `DINOv2Embedder` / `RFDETREmbedder` in `simulation_calibration_loop/data.py`
  (`embed_paths`). Per `optuna_search_trajectory.md` ("Embedder: use the
  trained RF-DETR backbone, not stock DINOv2"), prefer `RFDETREmbedder` for
  consistency with the search itself.
- **Missing piece:** there is no standalone script that just embeds two
  arbitrary directories and prints their MMD — today these two pieces are
  only wired together *inside* the Optuna loop (`controller.py`), scored
  against a fixed real-image reference pool, not against each other. Need a
  small script (e.g. `compare_cosmos_mmd.py` in this folder) that: embeds the
  base_v4-Cosmos set, embeds the optuna-Cosmos set, embeds the real LOCO
  reference set, and reports MMD(base, real), MMD(optuna, real), and
  MMD(base, optuna).

### 3. Publish (S3)

- Two existing, slightly different tools:
  - `simulation_calibration_loop/upload_top_runs_to_s3.py` — uploads the
    top-N *search trials* (rgb + embedding + yaml) from a workspace root.
  - `SimulationCalibrationController._export_best_runs_to_s3` /
    `_sync_directory_to_s3` in `controller.py` — the loop's own end-of-run S3 export.
  - Neither uploads a finished Cosmos clip set + comparison report — that's
    a new, small sync step (`aws s3 sync` of the two output dirs + the MMD
    numbers), not a new mechanism.

### 4. Train

- `optimization-recipe.md` Pipeline A/B point at `/home/ubuntu/RT-DETR/train.py`,
  which **does not exist on this machine**. The real, working trainer is
  `od_scripts/train_warehouse_real.py` (`RFDETRBase`/`RFDETRLarge` from
  `models/rf-detr`), documented end-to-end in `od_scripts/TRAINING_PROTOCOL.md`.
  Use that instead — it already expects Roboflow-COCO layout and has a
  documented path from Isaac's nested `Camera/` output to a training-ready
  dataset (`/tmp/convert_trajectory_synth.py` wrapping
  `od_scripts/prepare_synth_dataset.py`, see `TRAINING_PROTOCOL.md`).
- Two datasets to build (base-v4-Cosmos, optuna-Cosmos), each merged with the
  same real LOCO train/valid split so the eval set stays constant — mirrors
  how `warehouse3cls_traj_v1` was built.

## What's missing, in order

1. **An actual Optuna winner config.** Per `optuna_search_trajectory.md`, the
   search is mid-flight (Stage 8, 2026-07-07); no `state.json`/`workspace_*`
   exists on this machine yet, so there is no `best_config.yaml` to feed
   Cosmos. Either finish/resume the loop (`simulation_calibration_loop/run_main_loop.sh`)
   or pull the current best via `simulation_calibration_loop/collect_top_runs.py`
   against whatever workspace root the search has been running in.
2. **A Cosmos-profile config generator for the winner** (copy of
   `_generate_cosmos_v4.py`) + a run script (copy of `run_cosmos_v4_train.sh`).
3. **The actual Cosmos post-process command** (prompt + version), fixed and
   applied identically to both clip sets — currently undocumented anywhere in
   this repo.
4. **The MMD comparison script** described above (embed both sets + real
   reference, report pairwise MMD).
5. **An S3 sync step** for the two finished Cosmos datasets + the MMD report.
6. **Two `od_scripts/train_warehouse_real.py` runs** (base-v4-Cosmos vs.
   optuna-Cosmos, same real-data split, same hyperparams) + an mAP comparison,
   per the "Reporting" section of `optimization-recipe.md`.

## Status

Not started — this folder tracks the work as it progresses.

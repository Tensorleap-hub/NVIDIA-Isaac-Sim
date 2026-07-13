# Synthetic Data Generation Training Workflow

This repository contains two related warehouse synthetic-data workflows:

- Isaac Sim synthetic data generation for palletjack detection, used with Tensorleap recommendations
- an outer simulation calibration loop that uses DINOv2 embeddings and Optuna to search for better SDG parameter settings against real LOCO images

The current SDG runner is `palletjack_sdg/standalone_palletjack_sdg_mean_std.py`, and the base config it consumes is `palletjack_sdg/sdg_config_mean_std.yaml`.

## Overview

Use this repository in one of two ways:

- Tensorleap workflow: generate data, evaluate it in Tensorleap, export suggestions, convert the CSV into new YAMLs, and generate again
- Simulation calibration loop: optimize Isaac SDG parameters directly against real LOCO images with DINOv2 + Optuna

Detailed loop documentation lives in `simulation_calibration_loop/README.md`.

## Current Status

- The Tensorleap loop did not fully converge in my earlier runs. It looked like the suggestions kept expanding the distribution again and again. I may have moved to the EC2 loop too early and not verified every stage.
- I started testing targeted cases such as very high and very low camera setups and noise suggestions under `palletjack_sdg/experiments/experiment_mean_std/`.
- The standalone simulation calibration loop with DINOv2 embeddings does seem to converge: roughly from an initial distance around `0.6` to a current best around `0.35`, with more iterations still running.

## AWS / Infra

Saved best runs are in `s3://nvidia-isaac-bucket/`, and the loop runs are mainly under `s3://nvidia-isaac-bucket/optuna-ec2/`.

There is an instance ready to use called `nvidia`. Start it, connect, and enjoy.

Codex is installed there for my user, so have fun with it. Use it in `screen` so it will not disconnect when you do.

### Trajectory / Cosmos Test Data

Trajectory-SDG clips used for Cosmos-Transfer stylization live under:

- `s3://nvidia-isaac-bucket/trajectory-tests/20260708_cosmos_v4/`
- `s3://nvidia-isaac-bucket/trajectory-tests/20260712_cosmos_optuna/`

Download with:

```bash
aws s3 sync s3://nvidia-isaac-bucket/trajectory-tests/20260708_cosmos_v4/ ./20260708_cosmos_v4/
aws s3 sync s3://nvidia-isaac-bucket/trajectory-tests/20260712_cosmos_optuna/ ./20260712_cosmos_optuna/
```

Both prefixes are still being updated (as of 2026-07-12) — re-run the sync later today to pick up the latest clips before relying on them.

## Important Files

- `leap_integration.py`: Tensorleap integration entrypoint
- `simulation_calibration_loop/README.md`: detailed calibration-loop guide
- `simulation_calibration_loop/project_config.yaml`: default loop config
- `simulation_calibration_loop/theme_rounds.yaml`: ordered theme-round schedule
- `simulation_calibration_loop/run_main_loop_with_retry.sh`: retry wrapper for one loop config
- `simulation_calibration_loop/run_with_loop_venv.sh`: run loop scripts inside the dedicated loop venv
- `simulation_calibration_loop/run_theme_rounds.py`: themed multi-config loop runner
- `palletjack_sdg/standalone_palletjack_sdg_mean_std.py`: Isaac Sim SDG runner
- `palletjack_sdg/sdg_config_mean_std.yaml`: base mean/std SDG config
- `palletjack_sdg/experiments/generate_configs_mean_std.py`: generate YAMLs from Tensorleap CSV suggestions

## Tensorleap Workflow

The palletjack SDG config expresses randomized parameters as normal-distribution mean/std pairs.

Useful locations:

- `palletjack_sdg/experiments/experiment_mean_std/base_v2`: current base experiment family
- `palletjack_sdg/experiments/experiment_mean_std/`: generated experiments, suggestion folders, and ad hoc tests

Typical flow:

1. Generate data on EC2.
2. Upload the generated data to `s3://nvidia-isaac-bucket/` and download it locally if needed.
3. Evaluate the run in Tensorleap and export a suggestions CSV.
4. Convert the CSV into new Isaac YAMLs.
5. Push changes, pull them on EC2, and generate again.

Example generation command:

```bash
screen -dmS noise bash -lc 'cd /home/ubuntu/NVIDIA-Isaac-Sim/palletjack_sdg && bash run_experiments_mean_std.sh experiments/experiment_mean_std/base_v2 64'
```

Example YAML generation command:

```bash
python palletjack_sdg/experiments/generate_configs_mean_std.py \
  --csv palletjack_sdg/experiments/EXP-NAME/SUGGESTIONS_CSV.csv
```

## Simulation Calibration Loop

see [`simulation_calibration_loop/README.md`](...)
and
[`simulation_calibration_loop/ARCHITECTURE_REVIEW.md`](...)

The simulation calibration loop uses:

- DINOv2 as the feature extractor
- real LOCO subset-3 images as the reference distribution
- `calibration_optuna` as the suggestion engine

The main workflow is:

1. Start from one or more seed Isaac YAML configs.
2. Flatten selected SDG parameters into an Optuna search space.
3. Compute DINOv2 embeddings for the real LOCO reference set and cache them.
4. Materialize candidate parameter rows back into Isaac YAML files.
5. Run Isaac Sim to generate synthetic images with the palletjack SDG script.
6. Compute DINOv2 embeddings for the synthetic outputs.
7. Score synthetic-vs-real distance with `calibration_optuna`.
8. Ask Optuna for the next batch of SDG parameter suggestions.
9. Repeat for `N` iterations while tracking the best YAMLs.

The YAMLs in `simulation_calibration_loop/` define themed experiments. I split them into relatively orthogonal optimization stages.

Two things are supposed to happen between runs:

- the best base configuration is propagated forward, because not all parameters are optimized in each run
- curated initial configurations are carried forward into later Optuna runs through the base pool

Run all theme rounds:

```bash
screen -dmS sim-rounds bash -lc 'cd /home/ubuntu/NVIDIA-Isaac-Sim && python simulation_calibration_loop/run_theme_rounds.py \
  --round-config simulation_calibration_loop/theme_rounds.yaml \
  --workspace-root /home/ubuntu/NVIDIA-Isaac-Sim/simulation_calibration_loop/round_workspaces'
```
### Known Issues

- Embedding reuse through the historical pool is still not working the way I wanted. In practice, past embeddings are often rebuilt instead of being cleanly reused across iterations.

Run a single config:

```bash
bash simulation_calibration_loop/run_main_loop_with_retry.sh \
  --config simulation_calibration_loop/project_config.yaml
```

### Key Config Fields

The main loop config is `simulation_calibration_loop/project_config.yaml`.

Important fields:

- `project_name`: Optuna experiment name
- `workspace_dir`: output workspace for state, caches, materialized YAMLs, and Isaac outputs
- `seed_config_dir`: seed SDG YAML directory used as the initial search domain
- `real_dataset_root`: local LOCO dataset root
- `real_annotations_file`: LOCO annotations file
- `max_iterations`: number of optimization iterations
- `iteration_batch_size`: number of SDG configs evaluated per iteration
- `top_n_best_trials`: how many best trials to keep or export
- `isaac.script_path`: Isaac SDG runner, currently `../palletjack_sdg/standalone_palletjack_sdg_mean_std.py`
- `search_space`: parameter themes and include/exclude controls for optimization

Available search-space themes:

- `environment`
- `camera`
- `camera-color`
- `noise`
- `objects`
- `lighting`
- `materials`

### Workspace Output

The loop writes into `workspace_dir`, including:

- `state.json`: durable iteration ledger
- `main_loop_screen.log`: combined UI and Isaac log
- `cache/real/*.npy`: cached real-image embeddings
- `iteration_000/`, `iteration_001/`, ...: per-iteration outputs

Each iteration directory contains:

- `yamls/`: materialized SDG YAML files for that batch
- `outputs/`: Isaac output folders and logs
- `cache/`: cached synthetic embeddings

## Training Notes

The repository still includes the original local and cloud guides for training on generated data, including TAO-based workflows. Start from:

- `local/README.md`
- `cloud/README.md`

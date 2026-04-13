# Synthetic Data Generation Training Workflow

This repository contains two connected workflows for warehouse synthetic data:

- Isaac Sim synthetic data generation for palletjack detection - to be used with Tensorleap recommendations
- an outer simulation calibration loop that uses DINOv2 embeddings and Optuna to search for better SDG parameter settings against real LOCO images

The current SDG runner is [`palletjack_sdg/standalone_palletjack_sdg_mean_std.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/palletjack_sdg/standalone_palletjack_sdg_mean_std.py), and the base config it consumes is [`palletjack_sdg/sdg_config_mean_std.yaml`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/palletjack_sdg/sdg_config_mean_std.yaml).


## Guides
- Simulation calibration loop details: [simulation_calibration_loop/README.md](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/README.md)

## AWS 
Saved best runs are in  `s3://nvidia-isaac-bucket/` paricularly the runs from the loop are in
[`https://nvidia-isaac-bucket.s3.us-east-1.amazonaws.com/optuna-ec2/`]

There is a instance ready to be used call `nvidia` just, start, connect and enjoy!
Codex is installed with my user there so have fun with it ! Use it in screen so it won't disconnect when you do.

## Repository Workflow

The main calibration workflow is:

1. Start from one or more seed Isaac YAML configs.
2. Flatten selected SDG parameters into an Optuna search space.
3. Compute DINOv2 embeddings for the real LOCO reference set and cache them.
4. Materialize candidate parameter rows back into Isaac YAML files.
5. Run Isaac Sim to generate synthetic images with the palletjack SDG script.
6. Compute DINOv2 embeddings for the synthetic outputs.
7. Score synthetic vs. real distance with `calibration_optuna`.
8. Ask Optuna for the next batch of SDG parameter suggestions.
9. Repeat for `N` iterations while tracking the current best YAMLs.


## Important Files

- [`run_dinov2_optuna_loop.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/run_dinov2_optuna_loop.py): main Python entrypoint for the outer loop
- [`simulation_calibration_loop/project_config.yaml`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/project_config.yaml): default loop config
- [`simulation_calibration_loop/run_main_loop.sh`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/run_main_loop.sh): convenience wrapper for the loop
- [`simulation_calibration_loop/run_with_loop_venv.sh`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/run_with_loop_venv.sh): run any loop script inside the dedicated loop venv
- [`palletjack_sdg/standalone_palletjack_sdg_mean_std.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/palletjack_sdg/standalone_palletjack_sdg_mean_std.py): Isaac Sim SDG runner
- [`palletjack_sdg/sdg_config_mean_std.yaml`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/palletjack_sdg/sdg_config_mean_std.yaml): base mean/std SDG config
- [`palletjack_sdg/experiments/generate_configs_mean_std.py`](...) generate yaml from tensorleap csv 


## Tensorleap Flow with Palletjack SDG

The palletjack SDG config expresses randomized parameters as normal distribution mean/std pairs instead.

Under `palletjack_sdg/experiments` you can find the yamls used for generating data. 
`palletjack_sdg/experiments/experiment_mean_std/base_v2` stores the most recent base configuration to start the process

Flow with Tensorleap:
1. Create data in ec2 using
```bash
screen -dmS noise bash -lc 'bash run_experiments_mean_std.sh experiments/experiment_mean_std/basev2 64'
```
2. move the data to s3 bucket `s3://nvidia-isaac-bucket/` and download to local
3. Evaluate in Tensorleap and get csv - place csv in new folder under the experiments
4. Build yamls from the csv using 
```bash
python palletjack_sdg/experiments/generate_configs_mean_std.py 
       --csv palletjack_sdg/experiments/EXP-NAME/SUGGESTIONS_CSV.csv
```
5. push data --> pull in ec2
6. Create data and repeat


## Simulation Calibration Loop
See more elaborate [`simulation_calibration_loop/README.md`](...)
The loop uses:

- DINOv2 as the feature extractor
- real LOCO subset-3 images as the reference distribution
- `calibration_optuna` as the suggestion engine - updated a bit from the version in the engine

Yamls in the folder `simulation_calibration_loop` configure a base experiment.

### Main Run Commands

Run the loop workflow:
In this workflow the yamls are run one by one - starting from a configured base state, 
I divided them to thematic orthogonal optimizations. 
Two key things happen between runs:
1. Best Base configuration os propagated (as not all params are optimized in each run)
2. The initial configurations, used as the base for each optuna run, are curated and added after each iteration.
Todo: they are soppose the reuse the past embeddings in the pool, but this does not work and they are rebuilding them each iteration. This should be fixed. 

```bash
screen -S sim-rounds -dm bash -lc \
'cd /home/ubuntu/NVIDIA-Isaac-Sim && python simulation_calibration_loop/run_theme_rounds.py \
 --round-config simulation_calibration_loop/theme_rounds.yaml \
 --workspace-root /home/ubuntu/NVIDIA-Isaac-Sim/simulation_calibration_loop/round_workspaces'
```

### Run each config separately:
Each config can be run by itself:

```bash
bash simulation_calibration_loop/run_main_loop_with_retry.sh \
  --config simulation_calibration_loop/project_config.yaml
```

### Key Config Fields

The main loop config is [`simulation_calibration_loop/project_config.yaml`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/project_config.yaml).

Important fields:

- `project_name`: Optuna experiment name
- `workspace_dir`: output workspace for state, caches, materialized YAMLs, and Isaac outputs
- `seed_config_dir`: seed SDG YAML directory used as the initial search domain
- `real_dataset_root`: local LOCO dataset root
- `real_annotations_file`: LOCO annotations file
- `max_iterations`: number of optimization iterations
- `iteration_batch_size`: number of SDG configs evaluated per iteration
- `top_n_best_trials`: how many best trials to keep/export
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

The repository still includes the original local and cloud guides for training on generated data, including TAO-based workflows. For those steps, start from:

- [local/README.md](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/local/README.md)
- [cloud/README.md](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/cloud/README.md)

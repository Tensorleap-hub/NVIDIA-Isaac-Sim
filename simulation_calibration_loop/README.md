# Simulation Calibration Loop

This module implements an iterative calibration workflow for Isaac synthetic data generation using:

- DINOv2 as the feature extractor
- real LOCO subset-3 images as the reference distribution
- `calibration_optuna` as the suggestion engine

The workflow is:

1. Load a set of seed Isaac YAML configs.
2. Flatten the selected Isaac parameters into an Optuna-compatible search space.
3. Compute DINOv2 embeddings for the real dataset once and cache them.
4. Materialize the current parameter rows back into Isaac YAMLs.
5. Run Isaac to generate synthetic images.
6. Compute DINOv2 embeddings for the synthetic images.
7. Measure synthetic-vs-real distance with `calibration_optuna`.
8. Ask Optuna for the next batch of Isaac parameter suggestions.
9. Repeat for `N` iterations.


## Files

- `config.py`: typed config loader for the workflow
- `parameter_schema.py`: flattens Isaac YAMLs into Optuna rows and reconstructs them back
- `data.py`: DINOv2 embedding, real-image selection, Isaac subprocess execution, state store
- `controller.py`: main iterative workflow
- `ui.py`: terminal progress screen
- `project_config.yaml`: main workflow config
- `smoke_test_dinov2.py`: small offline DINOv2 + Optuna smoke test using existing images
- `test_isaac_small_loop.py`: small end-to-end Isaac loop test


## Main Workflow

Run from the repository root:

```bash
PYTHONPATH=. python -m simulation_calibration_loop.run_dinov2_optuna_loop
```

or:

```bash
PYTHONPATH=. python run_dinov2_optuna_loop.py --config simulation_calibration_loop/project_config.yaml
```

The main entrypoint loads `simulation_calibration_loop/project_config.yaml`, creates a workspace under `workspace_dir`, and persists progress in `state.json`.


## Config

The workflow is configured through `simulation_calibration_loop/project_config.yaml`.

Important fields:

- `project_name`: Optuna experiment name
- `workspace_dir`: output workspace for state, cache, YAMLs, outputs
- `seed_config_dir`: seed Isaac YAMLs used as the initial search domain
- `real_dataset_root`: local warehouse dataset root
- `real_annotations_file`: subset-3 annotations
- `max_iterations`: number of optimization iterations
- `iteration_batch_size`: number of Isaac YAMLs to evaluate per iteration

### DINO config

- `dino.model_name`: currently `dinov2_vitb14_reg`
- `dino.repo`: Torch Hub repo
- `dino.batch_size`
- `dino.image_size`
- `dino.resize_size`

### Isaac config

- `isaac.isaac_sim_path`: Isaac installation root
- `isaac.script_path`: SDG script to execute
- `isaac.headless`

### Search space

`search_space` controls which Isaac YAML parameters are exposed to Optuna.

If `include` is non-empty, only those exact paths are optimized.

Example:

```yaml
search_space:
  include:
    - camera.camera_height_mean
    - camera.fov_mean
    - lighting.intensity_mean
    - materials.textures
  exclude:
    - run.data_dir
    - run.num_frames
```

All non-selected Isaac fields remain fixed from the base template YAML used for materialization.


## Workspace Layout

The main workflow writes into `workspace_dir`:

- `state.json`: durable iteration ledger
- `cache/real/*.npy`: cached real DINO embeddings
- `iteration_000/`
- `iteration_001/`
- ...

Each iteration directory contains:

- `yamls/`: materialized Isaac YAMLs for that iteration
- `outputs/`: Isaac output folders and logs
- `cache/`: cached synthetic DINO embeddings


## Progress UI

The terminal UI shows:

- phase
- iteration progress
- current run id
- real cache hit/miss
- best trial so far
- best objective so far
- current iteration best / mean / median objective
- recent Isaac logs

The current objective headline is the first configured metric in `calibration_optuna`, which is currently `mmd_rbf`.


## DINO Smoke Test

This test does not launch Isaac. It uses existing real and synthetic images.

Run:

```bash
PYTHONPATH=. python -m simulation_calibration_loop.smoke_test_dinov2 --samples-per-domain 4 --device cpu
```

What it does:

- samples real subset-3 images
- samples synthetic images from an existing synthetic run directory
- embeds both with DINOv2
- prints latent-space distance summary stats
- runs a single Optuna suggestion pass using existing synthetic `run_config.yaml` files

Useful flags:

- `--synthetic-root`
- `--workflow-config`
- `--synthetic-experiments`
- `--synthetic-images-per-experiment`
- `--device`


## Small Isaac Loop Test

This test does launch Isaac, but keeps the run small.

Run:

```bash
PYTHONPATH=. python -m simulation_calibration_loop.test_isaac_small_loop --config simulation_calibration_loop/test_isaac_small_loop.yaml --device cpu
```

What it does:

- uses one initial Isaac YAML
- overrides Isaac to generate only `10` frames per iteration
- embeds the generated RGB images with DINOv2
- runs one Optuna update
- materializes the next suggestion into the next YAML
- repeats for a small number of iterations

This is the quickest real end-to-end validation path for:

- YAML materialization
- Isaac execution
- synthetic image collection
- DINO embedding
- Optuna suggestion generation


## Prerequisites

You need:

- a working Isaac Sim installation
- access to the local warehouse dataset root
- DINOv2 downloadable or already cached through `torch.hub`
- Python deps from `local_requirements.txt`

Notes:

- Isaac itself is not installed by Python requirements.
- The first DINOv2 run may download weights if they are not cached locally.


## Typical Usage Pattern

1. Edit `project_config.yaml`.
2. Narrow `search_space.include` to the Isaac parameters you actually want to optimize.
3. Run the smoke test if you want to validate DINOv2 and Optuna on existing data first.
4. Run the small Isaac loop if you want a cheap end-to-end validation.
5. Run the main workflow for the full iterative calibration loop.


## Implementation Notes

- Seed YAMLs are loaded with `extends` resolved.
- The search domain is inferred from the seed YAML family.
- Resume is implemented by replaying completed iterations from `state.json`.
- The workflow currently treats the synthetic domain as a single Optuna group named `simulation_1`.

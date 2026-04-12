# Simulation Calibration Loop Review

## Findings

### 1. High: Existing workspace outputs can be silently reused for the wrong trial parameters

The main loop uses a stable run id format, `iterXXX_runYYY`, for each iteration batch. In
[`_materialize_and_execute_iteration()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L249),
the output directory is derived only from the iteration index and run index:

- `outputs/iter000_run000`
- `outputs/iter000_run001`

Later in that same method, the workflow checks whether RGB images already exist under
[`output_dir / "Camera" / "rgb"`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L282).
If images are found, Isaac is skipped and those images are reused. That means a rerun in the
same workspace can accidentally evaluate stale synthetic outputs against newly materialized YAML
parameters if:

- the search space changed
- the seed set changed
- the base template changed
- the stored suggestions changed
- the workflow was partially rerun in an old workspace

The DINO embedding cache does not fully protect against this because the manifest includes the
stable YAML path, not a hash of the YAML contents.

### 2. High: Resume correctness depends on replay re-creating the same hidden Optuna ask state

Resume currently rebuilds the in-memory Optuna study by replaying completed iterations through
[`_replay_completed_iterations()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L216).
That method rehydrates old artifacts and then calls
[`_run_optimizer_iteration()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L329)
for each completed iteration.

The problem is that `_run_optimizer_iteration()` does not only register old evaluations. It also
asks for the next Optuna batch through `runner.evaluate_iteration(...)`. Those replay-generated
suggestions are discarded, and the workflow instead resumes from the suggestions stored in
`state.json`.

This works only if replay reproduces the exact same ask order and trial numbering as the original
run. Any change in sampler behavior, Optuna version, config, or replay path can desynchronize the
live study from the persisted state.

### 3. Medium: The shared workflow log still contains raw Isaac spam and is truncated on startup

The stdout flood was fixed for non-interactive runs, but the log file itself is still noisy.
[`WorkflowUI.append_log()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/ui.py#L90)
still writes every Isaac line into the shared UI log file through
[`_write_line()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/ui.py#L184).

As a result, `workspace/main_loop_screen.log` still contains all Isaac logs mixed with workflow
status lines. On top of that,
[`WorkflowUI.__init__()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/ui.py#L48)
truncates the shared log file on every startup, so the top-level workflow history is lost after a
resume.

### 4. Medium: Seed-config discovery is brittle to base-folder reorganization

[`load_yaml_configs()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/parameter_schema.py#L192)
only loads YAMLs from the top level of `seed_config_dir` via `glob("*.yaml")`.

That means:

- moving the base YAMLs into subfolders will cause them to be ignored
- changing the relative folder structure can break `extends` resolution in
  [`_load_yaml_with_extends()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/parameter_schema.py#L223)

The current workflow therefore assumes a flat seed directory and stable relative inheritance paths.


## Flow

### Bash entrypoint

The main entrypoint is
[`run_dinov2_optuna_loop.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/run_dinov2_optuna_loop.py).
It does three things:

1. parse `--config`
2. load the workflow config with
   [`load_workflow_config()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/config.py)
3. create
   [`SimulationCalibrationController`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py)
   and call `run()`

So all real workflow logic starts inside the controller.


### Controller startup

In
[`SimulationCalibrationController.__init__()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L45),
the controller performs the static setup:

1. Load all seed YAMLs from `seed_config_dir`.
2. Resolve YAML inheritance.
3. Infer the full flattening schema from the seed family.
4. Filter that schema through `search_space.include` and `search_space.exclude`.
5. Build the seed rows for iteration 0.
6. Infer parameter bounds and types for Optuna from the seed metadata.
7. Build the `ExperimentRunner`.
8. Build the `DINOv2Embedder`.

At that point the controller knows:

- which Isaac fields are optimizable
- how to flatten and reconstruct configs
- what Optuna bounds to use
- how to run Isaac and embed images


### Real reference setup

At the start of
[`run()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L111),
the controller:

1. starts the UI
2. computes or loads cached real embeddings via
   [`_prepare_real_embeddings()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L179)
3. sets those real embeddings into the Optuna runner
4. loads `state.json`
5. replays completed iterations into the in-memory study

The real embeddings are the fixed target distribution for the whole loop.


### How one iteration works

Each iteration has three phases: generate, optimize, persist.

#### 1. Generate

Generation happens in
[`_materialize_and_execute_iteration()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L249).

For each current row:

1. create a run id like `iter003_run002`
2. materialize a nested Isaac YAML from the flat Optuna-style parameter row
3. force `run.data_dir` to the iteration-specific output directory
4. write the YAML under `iteration_xxx/yamls/`
5. look for existing RGB files under `outputs/<run_id>/Camera/rgb`
6. if needed, copy RGBs from a configured synthetic base directory
7. if still needed, launch Isaac
8. collect generated RGBs from `Camera/rgb`
9. embed those RGB images with DINOv2
10. create a `RunArtifact`

The `RunArtifact` stores:

- run id
- YAML path
- output dir
- Isaac log path
- embedding cache path
- image count
- flattened parameters
- optional Optuna trial number

#### 2. Optimize

Optimization happens in
[`_run_optimizer_iteration()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L329).

That method:

1. loads the synthetic embeddings from the run artifacts
2. builds the `current_distributions` structure expected by `calibration_optuna`
3. passes trial numbers through so the optimizer knows whether each row is:
   - a seed/external trial, or
   - a previously asked Optuna trial
4. calls `runner.evaluate_iteration(...)`
5. receives:
   - next suggestions
   - metrics for the current batch
6. computes `iteration_best`, `iteration_mean`, and `iteration_median`

The objective currently shown by the workflow is the first configured optimization metric,
which is `mmd_rbf`.

#### 3. Persist

Back in
[`run()`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py#L128),
the controller:

1. writes objective values back into the run artifacts
2. queries the current best trials from Optuna
3. updates the UI
4. appends a completed iteration record to `state.json`
5. optionally exports the current top runs to S3
6. uses the returned suggestions as the input rows for the next iteration


### Seed trials vs Optuna-issued trials

This is an important correctness point in the current design.

Seed rows are treated as externally evaluated distributions:

- they start with `optuna_trial_number = None`
- they are imported into the study as completed trials

Suggestions returned by Optuna are different:

- they are stored with an explicit `optuna_trial_number`
- when they come back from Isaac evaluation, they are completed with `tell(...)`

That distinction is what makes the iterative ask/tell loop valid after the initial seed batch.


## Files

### `config.py`

[`config.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/config.py)
is the typed config loader.

It is responsible for:

- loading `project_config.yaml`
- resolving relative paths
- parsing DINO, Isaac, and search-space settings
- expanding higher-level search-space themes like `camera` and `lighting`

This file is the main user-facing configuration surface.


### `parameter_schema.py`

[`parameter_schema.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/parameter_schema.py)
is the YAML-to-Optuna translation layer.

It is responsible for:

- resolving YAML `extends`
- inferring optimizable parameter paths
- flattening nested YAMLs into Optuna rows
- writing flat parameter rows back into nested configs
- filtering the schema through the configured search space

This file defines what one trial actually means in terms of Isaac parameters.


### `data.py`

[`data.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/data.py)
contains the side-effecting runtime helpers.

It is responsible for:

- selecting real LOCO images from subset annotations
- loading and running DINOv2
- caching embeddings on disk
- finding generated RGB images
- launching Isaac
- streaming Isaac logs into per-run log files
- loading and saving `state.json`


### `controller.py`

[`controller.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/controller.py)
is the workflow brain.

It is responsible for:

- startup configuration
- real embedding preparation
- iteration materialization and generation
- Optuna evaluation
- UI updates
- resume/replay
- checkpoint persistence
- optional S3 export

If there is one file to review first for pipeline correctness, it is this one.


### `ui.py`

[`ui.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/ui.py)
is the reporting layer.

It provides:

- an interactive dashboard for TTY sessions
- compact status lines for non-interactive runs
- recent-log tracking
- summary metrics such as best trial, best objective, and iteration best/mean/median


### `README.md`

[`README.md`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/README.md)
is the operational guide.

It explains:

- how to run the main loop
- how to run the smoke test
- how to run the small Isaac loop test
- how the config file is structured


### `smoke_test_dinov2.py`

[`smoke_test_dinov2.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/smoke_test_dinov2.py)
is a cheap offline validation script.

It does not launch Isaac. It uses:

- real subset-3 images
- existing synthetic RGB outputs
- DINOv2 embeddings
- one Optuna suggestion pass

This is the fastest way to validate DINO and the optimizer wiring.


### `test_isaac_small_loop.py`

[`test_isaac_small_loop.py`](/Users/orram/Tensorleap/synthetic_data_generation_training_workflow/simulation_calibration_loop/test_isaac_small_loop.py)
is the smallest true end-to-end loop.

It does:

- one YAML per iteration
- a small number of frames
- real Isaac execution
- DINO embedding
- one Optuna update at a time

This is the quickest full-stack validation path.

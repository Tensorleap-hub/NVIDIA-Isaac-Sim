# How to Run a Calibration Optimization

A hands-on walkthrough for taking a clean checkout to a running calibration loop. This complements the conceptual overview in [README.md](README.md).

The optimization closes the loop:
**seed Isaac YAMLs → Isaac renders → DINOv2 (or RF-DETR) embeddings → MMD vs. real reference → Optuna suggests better Isaac params → repeat.**

---

## 1. One-time setup

### 1.1. Outer-loop Python env

The calibration loop runs in its own venv, not in Isaac Sim's bundled Python. Create it once:

```bash
bash simulation_calibration_loop/setup_loop_venv.sh
```

This builds `./.sim_loop_venv` from Isaac Sim's Python 3.11 and installs `local_requirements.txt`.

### 1.2. Datasets

You need:

- a working Isaac Sim install (path goes into `isaac.isaac_sim_path` below)
- the LOCO warehouse dataset accessible locally (`real_dataset_root` + `real_annotations_file`)
- DINOv2 weights — auto-downloaded by `torch.hub` on first run

### 1.3. Seed Isaac YAMLs

Put one or more starting-point Isaac YAML configs into a `seed_config_dir`. The optimizer:

- uses these seeds as the *initial* observations registered with Optuna (warm-start trials);
- uses their structure to discover which parameter paths exist (the schema);
- does **not** read bounds from them — bounds are declared in the project YAML (see §2.3).

A reasonable seed family is 1–N YAMLs that span the variations you ultimately want Optuna to explore (e.g. different camera presets).

---

## 2. The project YAML

The project YAML (e.g. `project_config.yaml`) is the single config file you point the loop at. Below are the sections you'll touch.

### 2.1. Workspace + data

```yaml
project_name: my_run
workspace_dir: ./workspace_my_run
seed_config_dir: ../palletjack_sdg/experiments/my_seeds
real_dataset_root: ../loco_dataset
real_annotations_file: ../loco_dataset/labels/loco-sub3-v1-train.json
max_iterations: 50
iteration_batch_size: 8
top_n_best_trials: 10
```

- `project_name` is also the Optuna study name.
- `workspace_dir` holds state, caches, per-iteration outputs, and resumes from there.

### 2.2. Embedder + Isaac

```yaml
embedder_backend: dinov2          # or "rfdetr"
dino:
  model_name: dinov2_vitb14_reg
  repo: facebookresearch/dinov2
  batch_size: 32
  image_size: 224
  resize_size: 256
isaac:
  isaac_sim_path: /opt/IsaacSim
  script_path: ../palletjack_sdg/standalone_palletjack_sdg_mean_std.py
  headless: true
```

For the RF-DETR backbone, see `RFDETREmbedderConfig` in `config.py`.

### 2.3. Search space — **this is the key section**

The search space has two halves:

| Field    | What it controls                                  |
|----------|---------------------------------------------------|
| `themes` / `include` / `exclude` | **Which** parameter paths Optuna sees |
| `bounds` | **What range** Optuna is allowed to try for each |

The final set of optimized paths is `expanded(themes) + include − exclude`. Every surviving path **must** have a matching entry in `bounds` — the controller fails fast at startup if any are missing, and prints the exact list to add.

Types are inferred automatically from the seed YAMLs (`int` / `float` / `bool` / `str`):

- numeric paths: `bounds[path] = [min, max]`
- non-numeric paths (bool, string, serialized JSON): `bounds[path] = [<allowed values>]`

Example:

```yaml
search_space:
  themes:
    - camera
    - lighting
  include:
    - materials.textures
  exclude:
    - camera.camera_roll_std
  bounds:
    # numeric: [min, max]
    camera.camera_height_mean: [1.0, 3.0]
    camera.camera_height_std:  [0.0, 0.5]
    camera.camera_tilt_mean:   [-15.0, 15.0]
    camera.camera_tilt_std:    [0.0, 10.0]
    camera.fov_mean:           [40.0, 90.0]
    camera.fov_std:            [0.0, 20.0]
    lighting.intensity_mean:   [500.0, 5000.0]
    lighting.intensity_std:    [0.0, 1000.0]
    # categorical: list of allowed values
    lighting.visibility_choices: ["clear", "overcast", "stormy"]
    materials.textures:          ["wood", "metal", "concrete"]
```

#### Discovering which paths to declare

Don't guess from memory — let the controller tell you. Run with `bounds: {}` once:

```yaml
search_space:
  themes: [camera, lighting]
  include: []
  exclude: []
  bounds: {}
```

The controller raises:

```
ValueError: Missing search_space.bounds entries for the following parameter paths:
  - camera.camera_height_mean
  - camera.camera_height_std
  - camera.camera_tilt_mean
  ...
Add them to the project YAML under `search_space.bounds`.
```

Copy that list into `bounds` and fill in ranges.

#### Min/max parameter pairs

When a path pair ends in `_min` / `_max` (e.g. `camera.camera_tilt_min`, `camera.camera_tilt_max`), the optimizer internally re-parameterizes them as `min + delta = max` so it can guarantee `max ≥ min` during sampling. Just declare bounds for each path independently — the rewrite happens transparently.

### 2.4. Optional: themed-rounds, base pool, S3 export, promoted baseline

See README.md §`Config`, §`Base pool`, §`Theme Rounds`.

---

## 3. Smoke-test before the real loop

Two cheaper tests are available before the full loop.

### 3.1. Smoke test (no Isaac)

Validates DINOv2 + Optuna against existing real/synthetic images:

```bash
bash simulation_calibration_loop/run_with_loop_venv.sh \
  -m simulation_calibration_loop.smoke_test_dinov2 \
  --samples-per-domain 4 \
  --device cpu
```

### 3.2. Small Isaac loop

Launches Isaac but with `num_frames_override=10` and a tiny iteration count. Quickest real end-to-end check:

```bash
bash simulation_calibration_loop/run_small_loop.sh --device cpu
```

This drives `test_isaac_small_loop.py` against `test_isaac_small_loop.yaml`. The small loop infers its own bounds from the `bounds_seed_dir` rather than the project YAML — it's a self-contained smoke test, not the production path.

---

## 4. Run the full loop

```bash
bash simulation_calibration_loop/run_main_loop.sh
```

or, with a custom project YAML:

```bash
bash simulation_calibration_loop/run_with_loop_venv.sh \
  run_dinov2_optuna_loop.py \
  --config simulation_calibration_loop/project_config.yaml
```

Auto-restart on crash, with a 60 s backoff:

```bash
bash simulation_calibration_loop/run_main_loop_with_retry.sh \
  --config simulation_calibration_loop/project_config.yaml
```

The loop is resumable — if `workspace_dir/state.json` already has completed iterations, the controller rebuilds the in-memory Optuna study from disk and continues from the last suggested batch.

---

## 5. What to watch while it runs

The terminal UI shows:

- current phase (`real-cache`, `generate`, `optimize`, `complete`)
- iteration index + completed/total runs in this batch
- current Isaac run id
- the best trial id and best objective seen so far
- the current iteration's best / mean / median objective
- recent Isaac stdout lines

The headline objective is the first metric in `calibration_optuna`'s config — by default `mmd_rbf` (RBF-kernel MMD between real and synthetic DINOv2 embeddings; lower = closer).

The full log is at `<workspace_dir>/main_loop_screen.log`.

---

## 6. Workspace layout (where outputs land)

```
<workspace_dir>/
├── state.json                   # durable iteration ledger (resume reads from here)
├── base_pool.json               # optional, when base_pool.enabled is true
├── main_loop_screen.log
├── cache/real/<key>.npy         # cached real DINO embeddings
├── iteration_000/
│   ├── yamls/iter000_run000.yaml
│   ├── outputs/iter000_run000__<hash>/
│   │   ├── Camera/rgb/*.png
│   │   ├── isaac.log
│   │   └── run_manifest.json
│   └── cache/iter000_run000__<hash>_<embedder>.npy
├── iteration_001/...
└── optuna/<project_name>/       # Optuna SQLite study (when persisted)
```

If `promoted_baseline_dir` is set, the workflow also writes `best.yaml` + `best.json` there.

If `s3_best_runs_prefix` is set, the workflow syncs the current top trials there after each iteration.

---

## 7. Common errors

### `Missing search_space.bounds entries for the following parameter paths`

You declared paths via `themes` / `include` but left `bounds` empty (or missing entries). Copy the list from the error into `bounds` and add ranges. See §2.3.

### `Numeric bound for '<path>' must be [min, max]; got ...`

A bounds entry for a numeric path is malformed — must be a 2-element list.

### `Categorical bound for '<path>' must be a non-empty list; got ...`

A bounds entry for a non-numeric path needs at least one allowed value.

### `WARNING: seed value <x> for '<path>' falls outside declared bounds [...]; clamping.`

A seed YAML carries a value outside the declared range. The optimizer clamps to keep the trial valid, but you should fix one of the two — either widen the bound or update the seed. This is *expected* behavior (no longer a silent rewrite) but flag-worthy.

### `Seed config N is missing required path '<path>'`

A seed YAML doesn't contain the path the schema inferred from the others. Make the seed family structurally consistent, or move the divergent seed out of `seed_config_dir`.

---

## 8. Tweaking the optimizer itself

These live in `calibration_optuna/config.py`'s `DEFAULT_CONFIG` (overridable through the project YAML):

| Field                          | Default                | Meaning                                  |
|--------------------------------|------------------------|------------------------------------------|
| `optimization_metrics`         | `['mmd_rbf']`          | List of objectives (multi-objective if >1) |
| `logit_bounds`                 | `(-5.0, 5.0)`          | Range for shape logits (mass per simulation group) |
| `mmd_max_samples`              | `1000`                 | Max samples used per MMD computation     |
| `optimizer.multivariate`       | `True`                 | TPE multivariate sampler                 |
| `optimizer.constant_liar`      | `True`                 | TPE constant-liar for batched ask        |
| `optimizer.n_startup_trials`   | auto (≥ 50, scales w/ params) | Random trials before TPE kicks in |

The project YAML can override the nested `optimizer.*` knobs through `tpe_sampler:` (see `WorkflowConfig` in `config.py`).

---

## 9. Glossary

- **Param path**: dotted YAML path like `camera.fov_mean`. Lists use `[i]` suffix (`palletjacks.position_std[0]`).
- **Trial**: one Isaac run with one parameter row, evaluated to one objective value.
- **Iteration / batch**: `iteration_batch_size` trials run together, then registered with Optuna; Optuna then suggests the next batch.
- **Warm-start / seed trial**: a row not produced by `ask()` — typically the initial seed YAMLs or pool-replay observations. Registered via `add_trial(...)` instead of `tell(...)`.
- **Promoted baseline**: the best-so-far YAML, written to `promoted_baseline_dir/best.yaml`. Multiple themed runs in sequence share this so each new theme starts from the previous theme's best.
- **Base pool**: optional persistent set of scored base YAMLs that future iterations sample from instead of always overlaying onto a single rolling best.

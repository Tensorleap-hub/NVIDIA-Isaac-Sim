# Optimization Recipe

End-to-end recipe from a trajectory SDG config to a fine-tuned RT-DETR model,
covering both the vanilla path (single generation) and the Optuna-optimized
path (loop-search then final generation). Also documents the sanity runs used
to prove the new pipeline reproduces our previously-published pre/post-optim
numbers.

## Pipelines

### Pipeline A — Baseline (no optim, with Cosmos)

Purpose: produce a "reference" fine-tune from a hand-picked base config.

```
base_config.yaml
      │
      ▼
Isaac (standalone_palletjack_trajectory_sdg.py, dense-in-time)
      │  Camera/rgb/*.png + Camera/bounding_box_2d_tight/*
      │  video/clip_0000/rgb.mp4
      ▼
Cosmos (in-run CosmosWriter → mp4)
      │
      ▼
Cosmos post-process (upscale / stylize / augment — external step)
      │  augmented dataset
      ▼
RT-DETR fine-tune (RT-DETR/train.py)
      │
      ▼
Validation metrics
```

### Pipeline B — Optuna-optimized

Purpose: search the config space for a better generation config, then produce
the fine-tune dataset from that winner.

```
base_config.yaml
      │
      ▼
Isaac (sparse-in-time, no Cosmos)  ◄──────────────────────┐
      │  Camera/rgb/*.png                                  │
      ▼                                                    │
DINOv2 / RF-DETR embedding + score                         │
      │                                                    │
      ▼                                                    │
Optuna suggests next config ──────────────────────────────┘
      │  (repeat N rounds; simulation_calibration_loop/controller.py)
      ▼
best_config.yaml
      │
      ▼
Isaac (dense-in-time, WITH Cosmos)  ← same script, different YAML profile
      │
      ▼
Cosmos post-process
      │
      ▼
RT-DETR fine-tune
      │
      ▼
Validation metrics  →  compare against Pipeline A
```

### Pipeline C — Sanity: reproduce prior pre/post-optim results

Purpose: prove the new trajectory pipeline reproduces our previously-published
random-frame results when Cosmos is disabled. Two runs, same fine-tune recipe:

```
prev_baseline_config.yaml (pre-optuna) ── Isaac (no Cosmos) ── RT-DETR fine-tune ── metrics
prev_optuna_winner.yaml   (post-optuna)── Isaac (no Cosmos) ── RT-DETR fine-tune ── metrics
```

Pass criterion: within ~1 mAP of the previously-published numbers for the same
configs on the same eval set.

## Concrete commands

Placeholders in `<ANGLE_BRACKETS>` — fill per run.

### A. Baseline generate → Cosmos → fine-tune

```bash
# 1. Generate (dense-in-time, Cosmos on)
./palletjack_sdg/run_trajectory_stage6.sh \
    --config palletjack_sdg/experiments/trajectory/cosmos_v1/exp01_operator_walk_steady.yaml \
    --num_frames 128

# 2. Cosmos post-process (external, exact command TBD in Cosmos repo)
python <cosmos_pipeline>/augment.py \
    --input palletjack_sdg/palletjack_data/.../video/clip_0000/rgb.mp4 \
    --output <augmented_dataset_dir>

# 3. Fine-tune
python /home/ubuntu/RT-DETR/train.py \
    --data <augmented_dataset_dir> \
    --weights <pretrained_rt_detr.pt> \
    --project runs/baseline_A --name exp01
```

### B. Optuna-optimized generate → Cosmos → fine-tune

```bash
# 1. Optuna loop: N rounds of (suggest → Isaac → embed → score)
./simulation_calibration_loop/run_main_loop.sh \
    --project-config simulation_calibration_loop/project_config_trajectory.yaml \
    --base-config palletjack_sdg/experiments/trajectory/training_v1/exp01_operator_walk_steady.yaml \
    --n-trials <N>

# 2. Take the best config (loop writes it out) and generate the full dataset
./palletjack_sdg/run_trajectory_stage6.sh \
    --config <loop_output_dir>/best_config.yaml \
    --num_frames 128

# 3. Cosmos post-process (same as A)
# 4. RT-DETR fine-tune (same as A) into runs/optuna_B
```

### C. Sanity — reproduce prior results without Cosmos

```bash
# pre-optuna
./palletjack_sdg/run_trajectory_stage6.sh \
    --config <prev_baseline_config>.yaml
python /home/ubuntu/RT-DETR/train.py --data <isaac_out> --project runs/sanity_pre

# post-optuna
./palletjack_sdg/run_trajectory_stage6.sh \
    --config <prev_optuna_winner>.yaml
python /home/ubuntu/RT-DETR/train.py --data <isaac_out> --project runs/sanity_post
```

## Reporting

For each pipeline (A, B, C-pre, C-post), report:
- Generation config used (path + git SHA)
- Number of frames, number of episodes/inits
- Fine-tune weights checkpoint + hyperparams
- mAP@[0.5], mAP@[0.5:0.95], per-class AP for {palletjack, forklift, pallet, person}
- Runtime: gen wall-clock, fine-tune wall-clock

The apples-to-apples comparison is:
- **A vs B**: does Optuna beat a hand-picked config? (measured in mAP delta on the same eval set)
- **C-pre vs C-post**: does the trajectory pipeline reproduce the random-frame pipeline's mAP within tolerance? (regression gate)

## Implementation details

### Two temporal densities

Frame-to-frame in a trajectory is highly redundant: at `capture_dt=0.4s` and
1 m/s camera speed, consecutive frames overlap ~90% of pixel content and the
DINOv2 / RF-DETR embeddings across them are near-duplicates. This wastes
both the Optuna scoring budget and the labelled-training budget.

Two YAML profiles under `agent:` control this:

- **Training profile** (Optuna + fine-tune consumption, no Cosmos):
  ```yaml
  run:
    num_frames: 24
  agent:
    capture_dt: 2.5        # ~2.5 m between samples at 1 m/s → spatially independent
  capture:
    video: false           # skip MP4 encoder; PNGs are the product
  ```
  24 spatially-separated frames per episode. Combined with N Optuna trials
  → N × 24 diverse samples.

- **Cosmos profile** (video output for downstream augmentation):
  ```yaml
  run:
    num_frames: 128
  agent:
    capture_dt: 0.0333     # 30 fps
  capture:
    video: true
  ```
  Smooth playback at 30 fps for Cosmos's temporal augmentation stage.

The 5 example configs under `experiments/trajectory/base_v1/` currently use
Cosmos-profile timings. When splitting, keep the style (mount height, jitter,
lens, characters) identical between the two profiles — only the timing +
`capture.video` differ.

### Frame diversity vs episode diversity

Within one episode:
- Scene randomization (palletjacks/forklifts/pallets/distractors/lighting/materials)
  fires **once**, at spawn. It does not change per frame.
- Character positions do change (Stage 7b BehaviorScripts drive `GoTo` etc.).
- Camera position changes along the planned path.

So `num_frames` inside one episode only diversifies camera pose + character pose.
Object layout and lighting variation only come from re-running Isaac with a
different seed. **For training data quality, prefer more short episodes over
one long episode** — the Optuna loop naturally provides this by re-running
Isaac for every trial.

### Optuna loop status quo

- Entry point: `simulation_calibration_loop/run_main_loop.sh`
- Controller: `simulation_calibration_loop/controller.py`
- Search space: declared in the config, resolved by `parameter_schema.py`
- Scorer: DINOv2 / RF-DETR embedding distance to a reference set (see
  `simulation_calibration_loop/base_pool.py`)
- Trial output: each trial writes an Isaac run dir; the loop persists the
  best config as a materialised YAML for downstream use in Pipeline B.

### Cosmos post-process

Currently external to this repo. Consumes `video/clip_0000/rgb.mp4` (from
CosmosWriter, wired in Stage 4 of the trajectory pipeline) and emits an
augmented dataset. Exact command + version TBD.

### RT-DETR fine-tune

- Entry point: `/home/ubuntu/RT-DETR/train.py`
- Consumes the Isaac RGB + bounding-box outputs (or the Cosmos-augmented
  variant) in whatever format the RT-DETR trainer expects — data-loader
  wiring TBD if not already in place.

## Open items

- [ ] Materialise `experiments/trajectory/training_v1/*.yaml` (sparse-time
      variants of the 5 base_v1 examples).
- [ ] Materialise `experiments/trajectory/cosmos_v1/*.yaml` (dense-time
      variants — essentially rename of base_v1 with `capture.video: true`).
- [ ] Add `--config` passthrough in `run_trajectory_stage6.sh` (currently
      hard-coded in the wrapper).
- [ ] Confirm the Optuna loop's project-config format accepts the new
      trajectory search-space keys (`characters.count`, `agent.capture_dt`,
      camera jitter, DOF, etc.).
- [ ] Document Cosmos post-process command + version once nailed down.
- [ ] Confirm RT-DETR trainer accepts Isaac's BasicWriter output format;
      write an adapter if needed.

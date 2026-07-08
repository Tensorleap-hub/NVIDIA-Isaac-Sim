# RF-DETR Warehouse-3cls Training Protocol

**Task.** Detect `pallet_truck`, `forklift`, `pallet` in warehouse imagery. Real
data comes from LOCO; synthetic data comes from Isaac Sim trajectory SDG
(`palletjack_sdg/standalone_palletjack_trajectory_sdg.py`).

## Model & environment
- Model: `rfdetr.RFDETRBase` (~32 M params), `num_classes=3`, `class_names=["pallet_truck", "forklift", "pallet"]`.
- Pretrain weights: **RF-DETR COCO** (default — do not pass `--pretrain-weights` unless you know the checkpoint was already fine-tuned on the same 3 classes).
- Trainer: `od_scripts/train_warehouse_real.py` → `RFDETRBase.train()` (PyTorch Lightning under the hood).
- GPU: single NVIDIA L40S (46 GB). Peak GPU memory in fine-tune configs below is ~12 GB with `batch_size=4, grad_accum_steps=4`.
- Python venv: `/home/ubuntu/NVIDIA-Isaac-Sim/.venv` (Python 3.12).

## Datasets

RF-DETR expects Roboflow-style COCO layout:
```
<dataset-dir>/
    train/
        _annotations.coco.json
        <image files>
    valid/
        _annotations.coco.json
        <image files>
```
Category IDs are 1-indexed in insertion order:
```
1: pallet_truck    2: forklift    3: pallet
```

| Dir | Train | Valid | Notes |
|---|---:|---:|---|
| `~/warehouse3cls_real` | 4110 | 858 | LOCO only. Built once via `od_scripts/prepare_loco_dataset.py` (see docstring for LOCO subset splits). |
| `~/warehouse3cls_mixed` | 9035 | 915 | LOCO real + older random-frame Isaac synth. Valid ≠ real valid (mixed). |
| `~/warehouse3cls_traj_v1` | 6016 | 858 | LOCO real train **+ 1906 new trajectory-SDG frames**. **Valid = LOCO real valid only** (used as the shared evaluation set below). Built by symlinking `warehouse3cls_real/{train,valid}` into the new dir, copying its annotations, then merging synth frames via `/tmp/convert_trajectory_synth.py` (thin wrapper around `od_scripts/prepare_synth_dataset.py`). |

### Real (LOCO) → COCO
```bash
python od_scripts/prepare_loco_dataset.py \
    --train-ann loco-sub1-v1-val.json \
    --train-ann loco-sub2-v1-train.json \
    --train-ann loco-sub4-v1-val.json \
    --train-ann loco-sub5-v1-train.json \
    --train-imgs /home/ubuntu/loco_dataset \
    --val-ann   loco-sub3-v1-train.json \
    --val-imgs  /home/ubuntu/loco_dataset/subset-3 \
    --output-dir /home/ubuntu/warehouse3cls_real
```
Filters LOCO's original label set down to the 3 warehouse classes and remaps
category IDs. Images are symlinked, not copied.

### Random-frame synth → COCO
```bash
python od_scripts/prepare_synth_dataset.py \
    --input-dirs /path/to/iter_run_dirs/* \
    --output-dir /home/ubuntu/warehouse3cls_synth \
    --val-fraction 0.1
```
Assumes flat layout `<run_dir>/rgb_XXXX.png` + `bounding_box_2d_tight_XXXX.{npy,json}` from Isaac Sim's `BasicWriter`.

### Trajectory synth → COCO (nested `Camera/` layout)
Trajectory SDG writes:
```
<run_dir>/Camera/rgb/rgb_XXXX.png
<run_dir>/Camera/bounding_box_2d_tight/bounding_box_2d_tight_XXXX.npy
<run_dir>/Camera/bounding_box_2d_tight/bounding_box_2d_tight_labels_XXXX.json
<run_dir>/Camera/bounding_box_2d_tight/bounding_box_2d_tight_prim_paths_XXXX.json
```
`prepare_synth_dataset.py` uses flat globs and would (a) miss the files and
(b) collide filenames across seeds if you passed `Camera/` as input. Use
`/tmp/convert_trajectory_synth.py`, which:
- Recurses into `Camera/rgb/` and `Camera/bounding_box_2d_tight/`.
- Uses the seed-dir name (e.g. `exp01_operator_walk_steady_seed42`) as `run_prefix` so filenames stay unique.
- Puts **all** frames into `train/` (val set is real-only for a clean evaluation).
- Reuses `prepare_synth_dataset.frames_to_coco()` for the actual COCO merge (root-mesh deduplication via `bounding_box_2d_tight_prim_paths_*.json`, `palletjack → pallet_truck` class remap, symlinked images).

### Deduplication note (matters for forklift/pallet counts)
Isaac Sim's `bounding_box_2d_tight` writer emits one row per semantic prim,
which includes both the object root prim and its child meshes (`/Ref/S_ForkliftBody`,
`/Ref/SM_PaletteA_01`, etc.). Naively keeping every row double-counts
forklifts and pallets. Naively keeping only `/Ref/…` rows drops palletjacks,
which have no child mesh. `prepare_synth_dataset.frames_to_coco()` keeps only
rows whose prim path does **not** contain `/Ref/` — the object root entry —
which is the one instance per object. Do not "simplify" this logic.

## Training hyperparameters

The two established configurations in this repo are:

### Config A — short fine-tune (matches `rfdetr_real_base_tuned`)
```bash
python od_scripts/train_warehouse_real.py \
    --dataset-dir <path> \
    --output-dir  <path>/output/rfdetr_traj_tuned \
    --epochs 35 \
    --lr 5e-5 --lr-encoder 1e-5 \
    --lr-drop 30 --warmup-epochs 1.0 \
    --batch-size 4 --grad-accum-steps 4 --num-workers 4
```
Historical baseline `rfdetr_real_base_tuned` used the same knobs but
`--epochs 40`. We now default to **35 epochs** — validation mAP plateaued
before epoch 40 in the historical run, so the extra epochs are wasted GPU
time.

### Config B — long "from-scratch-ish" (matches `rfdetr_base`)
```bash
python od_scripts/train_warehouse_real.py \
    --dataset-dir <path> \
    --output-dir  <path>/output/rfdetr_traj_base \
    --epochs 35 \
    --lr 1e-4 --lr-encoder 1.5e-4 \
    --lr-drop 100 --warmup-epochs 0.0 \
    --batch-size 4 --grad-accum-steps 4 --num-workers 4
```
Historical `rfdetr_base` used `--epochs 100` on `warehouse3cls_mixed`. When
comparing a new dataset against it, trim to **35 epochs** for the new run
first; only run the full 100 if 35 is competitive and you want to see the
ceiling.

Notes about defaults that come from `RFDETRBase.train()` and are **not**
exposed as CLI flags in `train_warehouse_real.py`:
- `ema_decay=0.993`, `use_ema=True`, `ema_update_interval=1` — the best EMA
  checkpoint (`checkpoint_best_ema.pth`) is usually the deployable one.
- `multi_scale=True`, `expanded_scales=True`, `square_resize_div_64=True`.
- `ia_bce_loss=True`, `group_detr=13`, `num_select=300`.
- `weight_decay=1e-4`, `clip_max_norm=0.1`.
- `lr_scheduler="step"` (drops 10× at `lr-drop`).
- `dataset_file="roboflow"` — that's what makes RF-DETR expect the
  `train/`, `valid/` COCO layout.

## Evaluation

Always evaluate on **real valid only** (`warehouse3cls_real/valid` = 858 imgs).
This is the shared eval set across configs. Comparisons that mix real+synth
into the valid split (as `rfdetr_base`'s own metrics.csv does) are not
apples-to-apples with real-only fine-tunes.

To eval an existing checkpoint on real valid without further training, just
run one training epoch pointed at a dataset whose `valid/` **is** the real
valid — the initial validation pass happens before any optimizer step and
gives a clean read of the checkpoint's zero-shot performance. That's how we
recovered `rfdetr_base`'s real-valid numbers (see the comparison table below).

## Current results (as of 2026-07-02)

All numbers below are on **`warehouse3cls_real/valid` (858 images)** unless
otherwise noted. `mAP_50` and `mAP_50_95` are on EMA weights (best EMA
epoch), which is the deployable checkpoint (`checkpoint_best_ema.pth`).
Per-class AP columns in `metrics.csv` (`val/AP/{class}`) are AP@50:95, not
AP@50.

| Model | Pretrain | Train data | Epochs | best ep | mAP@50 | mAP@50:95 | AP@50:95 truck / fork / pallet |
|---|---|---|---:|---:|---:|---:|---|
| `rfdetr_real_base_tuned` | RF-DETR COCO | real only (4110) | 40 | 39 | 0.391 | 0.179 | 0.033 / 0.328 / 0.177 |
| `rfdetr_base` (zero-shot on real valid) † | RF-DETR COCO | real + old random-frame synth (9035) | 100 | 100 | **0.602** | **0.380** | 0.036 / **0.751** / 0.355 |
| `rfdetr_base` (own metrics — mixed valid) †† | RF-DETR COCO | real + old random-frame synth (9035) | 100 | 46 | 0.560 | 0.302 | 0.234 / 0.477 / 0.180 |
| `rfdetr_traj_tuned` (Config A) | RF-DETR COCO | real + new trajectory synth (6016) | 35 | 28 | 0.363 | 0.155 | 0.030 / 0.282 / 0.165 |
| **`rfdetr_traj_base` (Config B)** | RF-DETR COCO | real + new trajectory synth (6016) | 35 | **14** | **0.443** | **0.206** | 0.054 / 0.350 / 0.173 |
| `rfdetr_traj_v4_base` (Config B) | RF-DETR COCO | real + traj synth v4 07-05 dump (9244) | 35 | 26 | 0.468 | 0.208 | 0.060 / 0.333 / 0.174 |
| **`rfdetr_traj_v4b_base` (Config B)** | RF-DETR COCO | real + traj synth v4b 07-07 dump — wall-fix+corrected-bounds (7847) | 35 | 27 | **0.464** | **0.214** | 0.052 / 0.369 / 0.188 |

† Re-evaluated: `rfdetr_base`'s checkpoint has never been evaluated on the
LOCO-real valid split during training. To get an apples-to-apples number we
loaded `rfdetr_base/checkpoint_best_regular.pth` as `--pretrain-weights`
and read the pre-training-step val pass on `warehouse3cls_real/valid`.
Numbers are directly comparable with the other rows.

†† `rfdetr_base`'s own `metrics.csv` numbers are on `warehouse3cls_mixed/valid`,
which is a mix of LOCO real + old random-frame synth — a different, easier
distribution than real-only. Do not compare across `†` and `††`; they are
different eval sets.

### Read of the results
- **New trajectory synth helps vs. real-only** (0.443 mAP@50 vs. 0.391 — +5.2 pp at mAP@50, +2.7 pp at mAP@50:95). Confirms the new SDG pipeline is providing usable signal.
- **Old random-frame synth still wins the shootout** on real valid (0.560 vs. 0.443 mAP@50 for the closest apples-to-apples). The clean comparison:
  - `warehouse3cls_mixed/valid` is 858 real jpgs + 56 synth pngs — effectively the real valid split, not a truly mixed eval.
  - Historical `rfdetr_base` saturated on EMA mAP@50 by epoch ~10–15 (0.53 at epoch 9, drift to 0.56 by epoch 46, noise for the rest). Our 35-epoch budget is well past its saturation point.
  - Training data composition (verified by symlink mtimes vs. tfevents start times):
    - `rfdetr_base` = real (4110) + **themed random-frame synth only** (33 configs from `ec2-loop/base_v2/exp01–exp33.yaml`, 2868 frames after filtering). Total 6978 images. → **0.560 mAP@50**.
    - `rfdetr_synth_opt0` = real + themed + Optuna-loop trial frames (40 more `iterNNN_runYYY__<hash>` configs, +2056 frames). Total 9034 images. → **0.625 mAP@50** (+6.5 pp from adding the Optuna trials).
    - `rfdetr_traj_base` (this work) = real + **trajectory synth** (40 episodes = 20 configs × 2 seeds, 1906 correlated frames). Total 6016 images. → **0.443 mAP@50**.
  - **`rfdetr_base` (0.560) vs. `rfdetr_traj_base` (0.443) is the fair random-frame-vs-trajectory head-to-head** at matched dataset scale (6978 vs. 6016 train). Random-frame themed synth wins by 11.7 pp. Compute-normalized picture is worse for trajectory: at matched optimizer steps (~7900), `rfdetr_base` = 0.522 vs. `rfdetr_traj_base` = 0.401 — 12 pp gap.
  - The gap is **synth configuration diversity + per-frame independence**, not frame count. Themed random-frame = 33 distinct configs × ~87 independently randomized frames each. Trajectory = 20 configs × 2 seeds × ~48 *consecutive* frames from a moving camera → each episode's frames are highly correlated, so effective diversity is ~40 configs' worth of "unique looks", not 1906.
- **Config B (lr 1e-4) beats Config A (lr 5e-5)** on this data (+8.0 pp mAP@50). Same pattern as the historical `rfdetr_base` beating `rfdetr_real_base_tuned`. Default to Config B for new datasets.
- **Config B saturates fast**: peak at epoch 14, no gain over the remaining 21 epochs. **15 epochs would be sufficient** for Config B on ~6 k images. Config A was still slowly rising through epoch 28 and mostly flat after the LR drop at 30.
- **Pallet_truck AP is uniformly weak** (0.03 – 0.05 on real valid). Adding synth of the class did *not* move the needle much. Worth investigating whether the synth palletjack renders match the LOCO pallet_truck class distribution (viewpoints, occlusions, lighting).

### Recommended next runs
1. Config B + 15 epochs on `warehouse3cls_traj_v1` (sanity — should match run (b)'s peak with 60% less compute).
2. Config B + 35 epochs on a **larger** trajectory synth dump (~5 k frames, matching old synth volume) to isolate synth quality vs. synth volume.
3. Diagnostic: eval `rfdetr_traj_base` and `rfdetr_base` on a pallet_truck-only slice of real valid to see which classes each model is trading off.

## Reproducibility log — what was actually run for each row
- **`rfdetr_real_base_tuned`**: `train_warehouse_real.py` default hyperparams (Config A but with 40 epochs), no `--pretrain-weights`, `--dataset-dir /home/ubuntu/warehouse3cls_real`, 40 epochs. Output at `/home/ubuntu/warehouse3cls_mixed/output/rfdetr_real_base_tuned/`.
- **`rfdetr_base`**: Config B with `--epochs 100`, no `--pretrain-weights`, `--dataset-dir /home/ubuntu/warehouse3cls_mixed`. Output at `/home/ubuntu/warehouse3cls_mixed/output/rfdetr_base/`.
- **`rfdetr_traj_tuned` (this run)**: Config A above (35 epochs), no `--pretrain-weights`, `--dataset-dir /home/ubuntu/warehouse3cls_traj_v1`. Output at `/home/ubuntu/warehouse3cls_traj_v1/output/rfdetr_traj_tuned/`.
- **`rfdetr_traj_base` (this run)**: Config B above (35 epochs), no `--pretrain-weights`, `--dataset-dir /home/ubuntu/warehouse3cls_traj_v1`. Output at `/home/ubuntu/warehouse3cls_traj_v1/output/rfdetr_traj_base/`.

## Gotchas seen while setting this up
1. **`train_warehouse_real.py` has no `--pretrain-weights` default** — omitting the flag falls back to RF-DETR's COCO pretrain (the right choice for a fresh fine-tune). Passing a prior 3-class checkpoint (e.g. `rfdetr_base/checkpoint_best_regular.pth`) works but is only sensible if you know the checkpoint's classes match; you'll also see `load_pretrain_weights: args.num_queries absent; inferred ckpt_num_queries=300 from tensor rows 3900 ÷ ckpt_group_detr=13` warnings.
2. **Trajectory SDG output is nested** (`Camera/rgb/`, `Camera/bounding_box_2d_tight/`) — don't point `prepare_synth_dataset.py` at it directly. Use `/tmp/convert_trajectory_synth.py`.
3. **Filename uniqueness across runs** — `frames_to_coco()`'s output filenames are `f"{run_prefix}_{rgb_path.name}"`. If two source dirs produce the same `run_prefix` (e.g. all `Camera/` for trajectory), all frames after the first collide on the symlink and the annotation JSON ends up with orphans. Use the seed-directory name as prefix.
4. **RF-DETR resumes silently from `<output-dir>/last.ckpt`** if you re-launch into a non-empty output dir. If you kill a run and restart with different hyperparams, `rm -rf <output-dir>` first or Lightning will pick up the last checkpoint.
5. **Best checkpoint file to deploy**: `checkpoint_best_ema.pth`. `checkpoint_best_regular.pth` is the non-EMA weights (usually a bit weaker). `checkpoint_best_total.pth` is a Lightning-side artifact and not intended for inference.

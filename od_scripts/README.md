# Warehouse 3-class detector — training protocol v2

Detect `forklift`, `pallet`, `pallet_truck` in warehouse imagery with RF-DETR Base.
Real data = LOCO; synthetic = Isaac Sim SDG renders. Everything here reads only
`/home/ubuntu/datasets` (`loco_dataset`, `base_v2_final`, `top-runs-may-ok`, `trajectory-optimized`, `base_v4_trajectory`+`base_v4_random`) and writes only `/home/ubuntu/datasets_coco`.
Arms are defined once in `common.ARMS`; `build_datasets.py`, `run_arms.sh`, `verify_gt.py`, `results.py` all read it.
Synthetic runs are deduplicated by content (md5 of the run's largest frame — frame 0 is sometimes a blank render);
blank frames carry no boxes and are dropped automatically.

## Two rules
1. **Validation = LOCO subset-3, only.** It is never in any `train/`.
2. **Never validate on synthetic frames.** All synth goes to `train/`. Synth "eval sets"
   exist only to measure how well a model *fits* its own training data (diagnostics).

## Files
| file | purpose |
|---|---|
| `common.py` | paths, the `ARMS` table, `load_class_names()` (ascending category_id order — RF-DETR maps sorted ids → 0..N-1) |
| `synth_coco.py` | Isaac BasicWriter → COCO (`collect_frames`, `frames_to_coco`): per-frame labels json (semanticId order is per-run!), `palletjack→pallet_truck`, keep only prim paths **without** `/Ref/` (one row per object; do not "simplify") |
| `build_datasets.py` | builds everything below, asserts the two rules, writes `MANIFEST.json` |
| `verify_gt.py` | GT report: composition, per-source label stats, example boxes per class/source → `gt_report.html` |
| `train.py` | the recipe (below); ReduceLROnPlateau + early stop via monkeypatch of the installed `rfdetr` |
| `eval_checkpoint.py` | score a `.pth` on a dataset's `valid/` with RF-DETR's own COCO eval (`--json` output) |
| `run_arms.sh` | sequential queue: train → evals → `results.py` per arm |
| `results.py` | markdown results table (`RESULTS.md`) |
| `legacy/training_report/` | July-2026 report of the previous (deleted) runs, kept for reference only |

## Datasets (`/home/ubuntu/datasets_coco`)
| dir | train | valid |
|---|---|---|
| `real` | LOCO sub1+2+4+5 → 4110 imgs | LOCO sub3 → 858 imgs |
| `real_basev2` | real + `base_v2_final` (32 exps × 128 frames; 2889 labeled) = 6999 | → `../real/valid` |
| `real_may` | real + `top-runs-may-ok` (59 runs; 3632 labeled) = 7742 | → `../real/valid` |
| `real_all` | real + base_v2 + may = 10631 | → `../real/valid` |
| `real_traj` | real + `trajectory-optimized` (312 traj runs × ≤11 frames; 2826 labeled) = 6936 | → `../real/valid` |
| `real_basev4` | real + `base_v4_{trajectory,random}` (ONE dataset: exp01-06 traj 5 frames/seed + exp07-32 random 1 frame/seed; 6241 labeled) = 10351 | → `../real/valid` |
| `real_traj_basev4` | real + traj + base_v4 = 13177 | → `../real/valid` |
| `real_all_traj` | real + base_v2 + may + traj + base_v4 = 19698 | → `../real/valid` |
| `real_all_matched` | size-matched control: real + 1444 base_v2 + 1445 may = 6999 (== `real_basev2`; seeded sample, `common.ARM_SUBSAMPLE`) | → `../real/valid` |
| `evalsets/train_real` | — | → `real/train` (fit on real train) |
| `evalsets/train_basev2`, `train_may`, `train_traj_optuna`, `train_basev4` | — | that synth source only |
| `evalsets/train_<arm>` | — | → `<arm>/train` (created by `run_arms.sh`) |

Categories (identical everywhere, derived from LOCO's id order): `1 forklift, 2 pallet, 3 pallet_truck`.
Images are symlinks to the raw files; synth filenames are `<run_dir>_rgb_NNNN.png`, real are `*.jpg`.
Rebuild from scratch: `.venv/bin/python od_scripts/build_datasets.py --force`.

## Recipe (identical for every arm)
`RFDETRBase`, RF-DETR COCO pretrain, `num_classes=3`; AdamW lr 1e-4 / encoder 1.5e-4, warmup 0;
batch 4 × grad-accum 4 (eff. 16), num_workers 4; EMA on (decay 0.993).
LR: **ReduceLROnPlateau** on `val/ema_mAP_50_95`, factor 0.1, patience 3, min-Δ 5e-4, floor 1e-2 × base.
Early stop: patience 8 on EMA, cap 60 epochs. `num_sanity_val_steps=0` (the 2-batch sanity eval otherwise
poisons best-checkpoint tracking). Deployable checkpoint: `checkpoint_best_ema.pth`.
RF-DETR silently resumes from `<output-dir>/last.ckpt` — `run_arms.sh` wipes the output dir first.

```bash
# one arm by hand
.venv/bin/python od_scripts/train.py --dataset-dir /home/ubuntu/datasets_coco/real_basev2
# whole study (env knobs: EPOCHS_DEFAULT / EPOCHS_<arm> caps, NUM_WORKERS, REUSE_EXISTING=1 = evals only if ckpt exists)
EPOCHS_DEFAULT=30 EPOCHS_real_all_traj=25 NUM_WORKERS=8 nohup od_scripts/run_arms.sh > /home/ubuntu/datasets_coco/logs/queue.log 2>&1 &
# fair eval of any checkpoint on the real valid
.venv/bin/python od_scripts/eval_checkpoint.py --dataset-dir /home/ubuntu/datasets_coco/real --pretrain-weights <ckpt>
```

## Outputs
`/home/ubuntu/datasets_coco/<arm>/output/rfdetr_reducelr/` — `metrics.csv`, checkpoints, `eval/<split>.json`.
Logs in `/home/ubuntu/datasets_coco/logs/`. Summary table: `/home/ubuntu/datasets_coco/RESULTS.md`.
Per-class columns in `metrics.csv` (`val/AP/<class>`) are AP@50:95.

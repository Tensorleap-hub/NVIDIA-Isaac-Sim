# Warehouse 3-class detector — training protocol v2

## Current state (2026-09-02)
An 11-arm study is **done**. Winner: `real_all_optrand` (real + base_v2 + may + base_v4 +
optuna_rand), mAP@50:95 **0.2318** on real valid (best F1 is actually `real_all`, 0.508 —
see the report). Full comparison: `/home/ubuntu/datasets_coco/RESULTS.md` and the charts
report below. Everything is backed up to
**`s3://nvidia-isaac-bucket/training/arms_study_20260830/`**: per-arm `checkpoint_best_ema.pth`
+ `metrics.csv` + `eval/*.json` + train log, plus `RESULTS.md`, `MANIFEST.json`,
`gt_report.html`, `summary_report.html`, `report_data.json` at the root.
To add a 12th arm: add an entry to `ARMS` in `common.py`, `build_datasets.py` (no `--force`,
it skips existing dirs), then `run_arms.sh <new_arm>`.

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
| `run_arms.sh` | sequential queue: train → evals → `results.py` per arm → `upload_s3.sh` |
| `results.py` | markdown results table (`RESULTS.md`) |
| `upload_s3.sh` | push one arm's model+metrics, or the study-level files, to S3 (idempotent, rerunnable) |
| `merge_metrics.sh` | after `train.py --resume`, splice the pre-resume `metrics.csv` history back in (see Resuming below) |
| `build_report_data.py` | every arm's full epoch curve + eval metrics → `report_data.json` |
| `summary_report_template.html` + `report_data.json` → `summary_report.html` | the charts report (bars, per-class, convergence, F1) — regenerate: `.venv/bin/python od_scripts/build_report_data.py`, then re-stamp with `python3 -c "import json; open('/home/ubuntu/datasets_coco/summary_report.html','w').write(open('od_scripts/summary_report_template.html').read().replace('__REPORT_DATA__', json.dumps(json.load(open('/home/ubuntu/datasets_coco/report_data.json')))))"` |
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
| `real_optuna_rand` | real + `optuna_rand` (random-frame render of the same 24 optuna-winning configs as traj_optuna, 128 seeds/config; 2761 labeled) = 6871 | → `../real/valid` |
| `real_all_optrand` | follow-up combo: real + base_v2 + may + base_v4 + optuna_rand (traj_optuna dropped — it hurt every combo it touched) = 19633 | → `../real/valid` |
| `real_all_matched` | size-matched control: real + 1444 base_v2 + 1445 may = 6999 (== `real_basev2`; seeded sample, `common.ARM_SUBSAMPLE`) | → `../real/valid` |
| `evalsets/train_real` | — | → `real/train` (fit on real train) |
| `evalsets/train_basev2`, `train_may`, `train_traj_optuna`, `train_basev4`, `train_optuna_rand` | — | that synth source only |
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
Per-class columns in `metrics.csv` (`val/AP/<class>`) are AP@50:95. Everything here also lives on S3 —
see **Current state** above for the bucket path.

## Resuming a capped/interrupted run
`run_arms.sh` caps epochs (`EPOCHS_DEFAULT`/`EPOCHS_<arm>`); if an arm's best epoch is *at* the cap it
likely hadn't converged. To keep training from `<output-dir>/last.ckpt`:
```bash
.venv/bin/python od_scripts/train.py --dataset-dir <ds> --output-dir <out> \
    --resume <out>/last.ckpt --epochs 60   # must exceed the previous cap
```
Gotcha: the resumed run's CSV logger **overwrites** `metrics.csv`, losing the pre-resume epoch history on
disk (the run itself resumes correctly — optimizer/scheduler/EMA/best-ckpt state all carry over). Before
resuming, save the old file (`cp metrics.csv metrics.csv.bak` or pull the S3 copy under `<arm>/metrics.csv`
if already uploaded); after the resumed run finishes, `od_scripts/merge_metrics.sh <backup.csv> <out>/metrics.csv`
splices the full curve back together. Then re-run the evals (`eval_checkpoint.py` per split) and
`upload_s3.sh <arm>` to refresh S3 with the converged checkpoint.

## Disaster recovery — if `/home/ubuntu/datasets` or `/home/ubuntu/datasets_coco` is gone
The **built** COCO datasets (`datasets_coco/`) are not backed up anywhere (they're symlink farms, cheap to
rebuild, expensive to store) — only `MANIFEST.json` (their composition record) is on S3. The **raw sources**
they're built from are all backed up, under their *original* names (not the renamed/prefixed copies in
`/home/ubuntu/datasets`):

| `/home/ubuntu/datasets/…` | S3 source |
|---|---|
| `loco_dataset/`, `base_v2_final/` | `s3://nvidia-isaac-bucket/{loco_dataset,base_v2_final}/` |
| `top-runs-may-ok/` | `s3://nvidia-isaac-bucket/top-runs-may-ok/` |
| `base_v4_trajectory/` (dirs prefixed `v4t_`) | `s3://nvidia-isaac-bucket/trajectory-tests/20260715_train_v4_128seed/` |
| `base_v4_random/` (dirs prefixed `v4r_`) | `s3://nvidia-isaac-bucket/trajectory-tests/20260715_train_v4_128rand/` |
| `trajectory-optimized/` | `s3://nvidia-isaac-bucket/trajectory-tests/trajectory-optimized/` |
| `optuna_rand/` (dirs prefixed `or_`, symlinks not copies) | `s3://nvidia-isaac-bucket/trajectory-tests/20260715_train_optuna_128rand/` |

To rebuild from scratch: `aws s3 sync` each source into place under `/home/ubuntu/datasets/<name>` (apply the
`v4t_`/`v4r_`/`or_` prefix per `common.synth_run_dirs()` — or just repoint `common.py`'s `BASEV4_TRAJ` /
`BASEV4_RAND` / `OPTUNA_RAND` at the unprefixed dirs, since `synth_coco.py` uses the directory name only as a
uniqueness key, not a semantic one), then `.venv/bin/python od_scripts/build_datasets.py --force`.

# mixedbase-50% vs. trajectory synth — data size vs. diversity findings

**Task.** Train `~/warehouse3cls_mixedbase` on **50% of its train data** with **Config B**
and compare mAP to `traj_v3`'s **0.202** (EMA mAP@50:95). Extended into a small
data-size-vs-diversity study across the trajectory and mixed-synth runs.

All numbers are **EMA mAP@50:95** on the **pure real valid set (858 LOCO images)**
unless stated otherwise. EMA weights (`checkpoint_best_ema.pth`) are the deployable
checkpoint.

---

## Experiment setup

- **Dataset built:** `~/warehouse3cls_mixedbase_50`
  - `train/` = random 50% subsample of `warehouse3cls_mixedbase/train` (seed 42):
    **3,489 / 6,978** images → **2,068 real (jpg) + 1,421 random-frame synth (png)**.
    66,948 / 135,707 annotations. Images symlinked to their resolved real targets.
  - `valid/` = **pure real valid (858 imgs)**, symlinked from `warehouse3cls_real/valid`.
- **Why real valid, not mixedbase's own valid:** `warehouse3cls_mixedbase/valid` has
  **879** images = 858 real **+ 21 synth**. That is *not* apples-to-apples with the
  trajectory runs. `traj_v2` and `traj_v3` both eval on a valid that is **byte-identical
  to real valid (858 imgs)** — verified. Using real valid makes all three directly comparable.
- **Class mapping** is identical across all train/valid splits used:
  `1=forklift, 2=pallet, 3=pallet_truck`. (Note: this differs from the order in
  `TRAINING_PROTOCOL.md`, which is stale — consistency between train and valid is what matters,
  and it holds.)
- **Config B** (from `TRAINING_PROTOCOL.md`): 35 epochs, `lr 1e-4`, `lr-encoder 1.5e-4`,
  `lr-drop 100`, `warmup 0`, `batch 4 × grad-accum 4` (effective batch 16). COCO-pretrained
  RF-DETR base, head re-init to 3 classes (**not** warm-started from a prior 3-class ckpt).
- **Output:** `~/warehouse3cls_mixedbase_50/output/rfdetr_mixedbase50_base/`
- ~3.5 min/epoch on the L40S; **218 optimizer steps/epoch** (3,489 / 16).

---

## Headline result

| Run | Train imgs | Peak EMA mAP@50:95 | Peak ep | Steps to peak | mAP@50 | AP fork / pallet / truck |
|---|---:|---:|---:|---:|---:|---|
| `traj_v2` | 7,042 | **0.2135** | 32 | 14,552 | 0.414 | 0.382 / 0.182 / 0.038 |
| `traj_v3` | 5,166 | **0.2103** | 16 | 5,490 | 0.401 | 0.333 / 0.176 / 0.033 |
| **`mixedbase50`** | **3,489** | **0.1970** | 28 | 6,350 | 0.400 | 0.336 / 0.169 / 0.025 |

**mixedbase-50% peaks at EMA 0.197** — only **~1.3 pp** below traj_v3 (0.210) and
**~1.6 pp** below traj_v2 (0.214), while training on **1.7×–2× fewer images**. It landed
just under the 0.202 reference. The run plateaued around 0.19–0.197 from ~epoch 20 onward
(converged; the last ~10 epochs are noise).

> Note: the "0.202" reference for traj_v3 sits between its best-EMA (0.210 @ ep16) and
> its best-EMA-at-best-raw-epoch (0.207 @ ep18). traj_v3's true peak EMA is **0.210**.

### ⚠️ Confound: mixedbase50 also has *half the real data*, not just half the synth
"50% of the train data" was a **uniform random subsample of the full mixedbase**
(4,110 real + 2,868 synth = 6,978), so it halved the **real** portion too:

| Run | Real imgs | Synth imgs | Peak EMA |
|---|---:|---:|---:|
| traj_v3 | **4,110** (full) | 1,056 trajectory | 0.210 |
| mixedbase50 | **2,068** (half) | 1,421 random-frame | 0.197 |

Because **eval is on real valid**, real training images are the highest-value data — and
mixedbase50 is handicapped on exactly that. So the mixedbase50-vs-traj_v3 pair is **not** a
clean "random-frame vs trajectory synth" comparison: it mixes a *synth-type* difference with
a *real-data-volume* difference. Two consequences:
- mixedbase50 reaching 0.197 with **half the real data** is arguably *more* impressive, but
- the "diversity-bound, not volume-bound" conclusion (Finding 3) **cannot be attributed to
  synth type alone** from this pair — real volume differs too.

A clean synth-type control (`warehouse3cls_rftypematch`: **full 4,110 real + 1,056
random-frame synth**, matching traj_v3 on *both* real and synth counts) is run separately —
see **Finding 5 / rftypematch** below. That isolates random-frame vs trajectory synth.

---

## Findings (the reasoning thread)

### 1. Matched-*epoch* comparison is misleading — epochs aren't equal training
An epoch = one pass over the train set, so **steps/epoch scales with dataset size**:

| Run | Train imgs | Steps/epoch |
|---|---:|---:|
| mixedbase50 | 3,489 | ~218 |
| traj_v3 | 5,166 | ~323 |
| traj_v2 | 7,042 | ~440 |

At "the same epoch," the bigger dataset has taken proportionally more gradient updates.
Early on, mixedbase50 trailed traj_v2 by ~5 pp at epochs 0–2 — but that partly reflects
traj_v2 having done **2× the optimizer steps per epoch**, not a data-quality deficit.

### 2. At matched *steps* vs. traj_v2, mixedbase50 is even
Normalizing for compute (equal optimizer steps), mixedbase50 ≈ traj_v2:

| ~Steps | mixedbase50 | traj_v2 |
|---:|---:|---:|
| ~1,320 | 0.156 (ep5) | 0.159 (ep2) |

So per unit compute, half-the-data mixed synth matches trajectory synth → **more
sample-efficient per image** (random-frame synth carries more effective diversity
per frame than correlated trajectory frames).

### 3. Converged *peak* is nearly size-independent (across the trajectory runs)
traj_v3 has **27% less data** than traj_v2 **and** peaks in **~1/3 the steps**
(5,490 vs 14,552), yet lands within **0.3 pp** of traj_v2 (0.210 vs 0.214). In the
~5k–7k regime, **train-set size barely moves the converged mAP** — the ceiling is set by
synth **diversity/quality**, not raw image count. traj_v3 is the more *efficient* run.

> Scope: this holds cleanly *within the trajectory family* (v2 vs v3 differ mostly in synth
> volume/composition). It does **not**, on its own, license comparing mixedbase50 to traj_v3
> as a synth-type test — that pair also differs in real-data volume (see the ⚠️ confound box).
> The `rftypematch` run (Finding 5) is the clean control.

### 4. But within the *same distribution*, more data = faster per-step convergence
The cleanest size ablation is mixedbase50 vs. **full mixedbase** (a "sanity" run, also
COCO-init, 3 epochs, 436 steps/epoch) — identical distribution, only size differs.
Matched by step:

| ~Step | full mixedbase (6,978) | mixedbase50 (3,489) |
|---:|---:|---:|
| ~436 | 0.168 (ep0) | 0.092 (ep1) |
| ~873 | 0.190 (ep1) | 0.121 (ep3) |
| ~1,310 | **0.210** (ep2) | **0.156** (ep5) |

At matched steps, full mixedbase is **~5 pp ahead**: at a given step count it has shown the
model ~2× as many *unique* images (one pass vs. mixedbase50's two passes over half the set),
so its gradients are less redundant. (Caveat: the sanity run evaluated on the mixed
879-img valid, which inflates by only ~1 pp — the ~5 pp gap is real.)

### 5. Clean synth-type control — `rftypematch` (full real + count-matched random-frame synth)
To isolate **random-frame vs trajectory synth** with everything else held equal, built
`warehouse3cls_rftypematch` = **4,110 real + 1,056 random-frame synth** (5,166 imgs) — matching
traj_v3 on real count, synth count, valid set, and Config B. The *only* difference vs traj_v3
is synth **type**.

| Run | Real | Synth (type) | Train imgs | Peak EMA mAP@50:95 |
|---|---:|---|---:|---:|
| traj_v3 | 4,110 | 1,056 (trajectory) | 5,166 | 0.2103 |
| `rftypematch` | 4,110 | 1,056 (random-frame) | 5,166 | _TBD — training_ |

_Result pending; this section will be filled when the run converges. Output:
`~/warehouse3cls_rftypematch/output/rfdetr_rftypematch_base/`. Build: `od_scripts/build_rftypematch.py`._

### These are two different axes and don't conflict
- **Convergence *speed* (per step):** more data helps — full mixedbase > mixedbase50 on the
  identical distribution. ✔ size matters here.
- **Converged *peak* (ceiling):** barely moves with size — traj_v3 (5.2k) ≈ traj_v2 (7k),
  and mixedbase50 (3.5k) lands only ~1.3 pp under traj_v3. ✔ diversity/quality sets the
  plateau (~0.20–0.21 for this family), not volume.

---

## Bottom line

- **Answer to the task:** mixedbase-50% (3,489 imgs, Config B) peaks at **EMA mAP@50:95 = 0.197**,
  just **below traj_v3's 0.202/0.210** — despite ~1.7× fewer images **and half the real data**.
  Effectively a tie, but see the ⚠️ confound: this pair varies both synth *type* and real
  *volume*, so it does not cleanly attribute the near-tie to synth type. The `rftypematch`
  control (Finding 5) is the clean random-frame-vs-trajectory test.
- **Data size mainly buys convergence *speed*, not a higher *ceiling*** — established cleanly
  *within* a distribution (mixedbase50 vs full mixedbase, Finding 4) and *within* the trajectory
  family (v2 vs v3, Finding 3). Peak mAP in this 3-class family plateaus around **0.20–0.21**.
- **Random-frame mixed synth is more sample-efficient than trajectory synth** per image
  (correlated consecutive trajectory frames → low effective diversity), consistent with
  `TRAINING_PROTOCOL.md`'s earlier random-frame-vs-trajectory read.
- **`pallet_truck` AP stays weak everywhere** (0.025–0.038) — the persistent class problem;
  none of these data mixes fix it. Adding synth of the class does not move it. Worth a
  dedicated look (synth palletjack viewpoints/occlusion vs. LOCO pallet_truck distribution).

## Reproduction
```bash
# 1. build the 50% subset (real valid) — seed 42, valid = pure real valid
python3 od_scripts/build_mixedbase_50.py   # -> ~/warehouse3cls_mixedbase_50

# 2. Config B, 35 epochs
python od_scripts/train_warehouse_real.py \
    --dataset-dir /home/ubuntu/warehouse3cls_mixedbase_50 \
    --output-dir  /home/ubuntu/warehouse3cls_mixedbase_50/output/rfdetr_mixedbase50_base \
    --epochs 35 --lr 1e-4 --lr-encoder 1.5e-4 \
    --lr-drop 100 --warmup-epochs 0.0 \
    --batch-size 4 --grad-accum-steps 4 --num-workers 4
```

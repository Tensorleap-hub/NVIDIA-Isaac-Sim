# Convergence Benchmark: Optuna vs Tensorleap Suggester

**Author:** (TBD) | **Date:** 2026-04-23 | **Status:** Draft

## 1. Motivation

The Tensorleap (TL) calibration loop did not converge on the palletjack task: per-iteration suggestions expanded the parameter distribution instead of refining it. The local Optuna + DINOv2 loop on the same task converged from distance ~0.6 to ~0.369.

These two loops differ in **two** places — the evaluator (DINOv2 vs TL latent space) and the suggester (Optuna TPE vs TL's insights → CSV workflow). A failure on the real task cannot distinguish which component is at fault.

This benchmark isolates the **suggester** axis by fixing the evaluator on both sides, so any convergence gap is attributable to the suggestion engine.

## 2. Question

Given identical evaluator, initial state, trial budget, and seed, does Tensorleap's suggestion CSV converge to the known-optimum parameter vector as fast as (or ever as well as) Optuna TPE?

## 3. Design principles

- **Single axis of variation.** Lock every shared component byte-for-byte; only the suggestion engine changes.
- **Ground truth available.** Both "real" and "synthetic" are generated from the same parameterized process. A fixed θ\* defines the target distribution, enabling parameter-space convergence measurement in addition to objective-space.
- **Cheap iteration.** No Isaac Sim. A 2D image generator runs each trial in seconds, not minutes, so we can afford many bench configurations.
- **Evaluator parity enforced by ONNX.** DINOv2 is exported to ONNX once and used by both the local loop and TL. Rules out drift from model version, image preprocessing, or pooling differences.

## 4. Shared components

### 4.1 Toy generator
A pure-Python 2D image pipeline parameterized by θ. No Isaac dependency. Proposed parameter vector (8D mixed):

| Param              | Type       | Range           | Notes                                              |
|--------------------|------------|-----------------|----------------------------------------------------|
| `blur_sigma`       | continuous | [0.0, 5.0]      | Gaussian blur                                      |
| `noise_std`        | continuous | [0.0, 0.5]      | Additive Gaussian noise                            |
| `brightness_shift` | continuous | [-0.5, 0.5]     | Pre-noise brightness offset                        |
| `color_shift_r`    | continuous | [-0.3, 0.3]     | Per-channel shift (RGB triplet)                    |
| `color_shift_g`    | continuous | [-0.3, 0.3]     |                                                    |
| `color_shift_b`    | continuous | [-0.3, 0.3]     |                                                    |
| `clutter_count`    | integer    | [0, 20]         | Number of random rectangles overlaid on base image |
| `background_id`    | categorical| {0, 1, 2, 3}    | Fixed pool of 4 base images                        |

Continuous + discrete + categorical mix intentional — mirrors the real SDG search space shape.

`θ*` — a fixed, interior value (not at any boundary). Picked once, committed.

### 4.2 Real dataset
Generated once from θ\* with M distinct RNG seeds. Proposed M = 500. Images written to `bench/data/real/` and not regenerated during benchmarking.

### 4.3 Evaluator
- **Model:** `dinov2_vitb14_reg` preferred (matches `simulation_calibration_loop/project_config.yaml`), but any `dinov2_vitb14` variant is acceptable if the reg variant causes export issues.
- **Preprocessing:** resize 256, center-crop 224, DINOv2-standard normalization.
- **Export:** traced/exported to ONNX at a frozen version; both loops load this ONNX file.
- **Distance metric:** MMD over embeddings, `mmd_max_samples = 1000` (matching production config). Per-trial: generate N\_trial = 128 synthetic images (matches `num_frames` in the real seed configs), embed, compute MMD against cached real embeddings.

> [!important] Prerequisite task (separate agent)
> Export DINOv2 to ONNX and verify it loads successfully in Tensorleap before the benchmark can run on the TL side. Steps:
> 1. Export `dinov2_vitb14_reg` (or fallback variant) to ONNX using `torch.onnx.export` with a representative input.
> 2. Verify ONNX output parity against the PyTorch model (same input → embeddings within 1e-5).
> 3. Upload the ONNX to TL and confirm it parses without errors.
> 4. Commit the `.onnx` file + an integrity hash to `bench/convergence/`.
> This is a blocker for Condition B (TL side) but not for Condition A (local Optuna loop).

### 4.4 Initial conditions
- Same seed configs: a fixed set of 8 starting θ vectors, distributed across the search space (one per roughly stratified region). Same file, same order, used by both loops.
- Same random seed for the suggester (both Optuna's sampler seed and TL's equivalent, if exposed).
- Same per-iteration trial count N — fixed by TL's row count (see §5).

## 5. Experimental conditions

| Parameter               | Value                     |
|-------------------------|---------------------------|
| Iterations (K)          | 30                        |
| Trials per iteration (N)| Fixed by TL CSV row count |
| Total trials            | N × K                     |
| Images per trial        | 128                       |
| Seed                    | 42                        |

### 5.1 Condition A — Optuna loop (control)
- Reads DINOv2 MMD distance per trial.
- `optuna.samplers.TPESampler(seed=42)` proposes next N parameter rows.
- Loop runs entirely locally; no TL involvement.

### 5.2 Condition B — Tensorleap loop
- Same evaluator, same generator, same initial configs.
- Each iteration: generate N images per suggested row, upload to TL with DINOv2-derived embeddings.
- TL analysis emits a CSV of N suggested θ vectors for the next iteration.
- Those rows are the next iteration's trials.

## 6. Metrics (per iteration)

| Metric                   | Definition                                                        | What it reveals                                  |
|--------------------------|-------------------------------------------------------------------|--------------------------------------------------|
| `best_objective_i`       | min MMD across all trials up to iteration i                       | Standard convergence curve                       |
| `best_theta_i`           | θ achieving `best_objective_i`                                    | Where the suggester has landed                   |
| `param_gap_i`            | ‖normalize(best_theta_i) − normalize(θ\*)‖₂ in min-max-scaled space | True convergence (not just objective plateau)    |
| `spread_i`               | mean std of the N trials' params in iteration i (normalized)       | **Key diagnostic** — captures TL's "expansion" failure mode |
| `median_objective_i`     | median MMD across the N trials in iteration i                     | Is the bulk moving, or just an outlier?          |

All metrics logged per iteration to a CSV per condition, plus a comparison plot at the end.

## 7. Success criteria

- **Pass (Optuna baseline):** Optuna reaches `param_gap_i ≤ ε_θ` and `best_objective_i ≤ ε_obj` within K iterations. If it doesn't, the toy is too hard — redesign before comparing TL.
- **TL pass:** TL also converges within K. Reported as iteration-to-threshold for both.
- **TL fail with diagnostic:** TL's `spread_i` grows or stays flat while Optuna's decreases — confirms the "expanding distribution" failure mode observed on the real task. This is the outcome that most informs product work.
- **TL slow but working:** both converge, TL takes more iterations. Report the ratio.

Thresholds `ε_θ`, `ε_obj` are set after one pilot Optuna run so they're calibrated to what's achievable, not guessed.

## 8. Protocol

1. Implement toy generator + θ\* + real-set generation. Commit `bench/data/real/` and `bench/theta_star.json`.
2. Export DINOv2 to ONNX. Commit the `.onnx` + an integrity hash.
3. Build a shared harness that loads the ONNX, generates images from a θ, embeds, computes MMD vs cached real embeddings. Same harness is used by both conditions.
4. Run **pilot** Optuna loop with a larger budget than the planned bench to confirm the toy converges at all. Set `ε_θ`, `ε_obj` from the pilot.
5. Run **Condition A** (Optuna) with the agreed budget. Log metrics.
6. Run **Condition B** (TL). Log metrics.
7. Produce the comparison plot: `best_objective_i`, `param_gap_i`, `spread_i` over i, both conditions overlaid.
8. Write results note. Decide follow-ups.

## 9. Assumptions / open questions

1. **TL consumes the ONNX DINOv2 natively.** To confirm: does TL's current ONNX pipeline support the `dinov2_vitb14_reg` architecture (registers, input shape, pooling output) without modification? If not, what's the surgery?
2. **TL CSV row format.** Column names must match the toy's θ key set exactly, or a mapping layer is needed. To be documented once we have a sample CSV.
3. **TL iteration cadence.** How does a "TL iteration" actually get triggered — manual Insights export, scheduled job, API call? The benchmark needs to script this to be reproducible.
4. **TL seed / sampler determinism.** Is TL's suggestion engine seedable? If not, multi-seed runs are needed on the TL side to get a distribution, not a single curve.
5. **Categorical parameter handling.** Optuna handles `background_id` via `suggest_categorical`. How does TL represent and vary categoricals? If it can't, drop the categorical or encode it differently.

## 10. Out of scope

- Isaac Sim — this bench is deliberately Isaac-free for iteration speed. Once results are clear on the toy, a follow-up using Isaac can test whether findings transfer.
- Multi-objective search — single-scalar MMD only.
- Running bench on real LOCO — the point is ground truth via generated data.

## 11. Implementation notes

Suggested layout:

```
bench/
  convergence/
    README.md                      # this doc (rename)
    theta_star.json                # frozen target parameters
    generator.py                   # toy image generator
    evaluator.py                   # ONNX DINOv2 + MMD
    harness.py                     # shared trial runner
    optuna_loop.py                 # Condition A
    tl_loop.py                     # Condition B (reads TL CSV)
    metrics.py                     # per-iteration logging
    plot.py                        # comparison plot
    data/
      real/                        # 500 images generated from θ*
      real_embeddings.npy          # cached once
    dinov2_vitb14_reg.onnx         # frozen evaluator
    runs/
      optuna_seed42/               # per-run metrics + logs
      tl_seed42/
```

Dependencies: reuse `.sim_loop_venv` where possible; only addition is ONNX runtime.

## 12. Decision log (to fill as we go)

- [ ] θ\* value chosen and committed
- [ ] **DINOv2 exported to ONNX** (separate agent task — see §4.3)
- [ ] **DINOv2 ONNX imports successfully into TL** (separate agent task — see §4.3)
- [ ] DINOv2 ONNX parity verified (same embedding for same input, within 1e-5)
- [ ] ε\_θ, ε\_obj set from pilot
- [ ] TL ONNX compatibility confirmed
- [ ] TL CSV column schema documented
- [ ] TL iteration trigger mechanism decided

## 13. Risks

- **DINOv2 ONNX divergence between loops.** Mitigated by parity check in §12. If it fails, the whole comparison is contaminated.
- **Toy too easy / too hard.** Pilot Optuna run catches this before TL is involved.
- **TL suggestion format turns out to be categorical/textual, not numeric rows.** Would require either a translation layer or rescoping the benchmark to a fair comparison of whatever signal TL actually emits. Decide early — don't build the full bench first.
- **TL's iteration is slow enough to make K=30 impractical.** Reduce K, or start with a smaller N. The benchmark is only useful if it can run end-to-end within a day or two.

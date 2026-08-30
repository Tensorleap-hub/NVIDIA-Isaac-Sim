# Option 3-lite: refresh the calibration_optuna fork to the engine loop, run it on the TL latent space

> **Status (2026-08-30): superseded as the primary path by
> [`option2-engine-synthetic-job.md`](option2-engine-synthetic-job.md).**
> Phase 1 (vendoring) is DONE (commit b05760f) and is kept as the fallback loop: if the TL-LS
> search in the engine job doesn't converge, run the vendored `engine_loop.CalibrationLoop` with
> the DINOv2 embedder. Phases 2–5 below are on hold: replicating the engine's default LS headless
> turned out to require capturing an ES-driven layer pick + the engine's converted TF graph
> (an expensive export), which the engine job provides natively.

Goal: run the engine's exact `CalibrationLoop` semantics (auto n_samples sizing, noise-floor
stopping, MAD outlier gate, dimensionality-scaled budget) headlessly on the EC2 Isaac machine,
scoring MMD in the Tensorleap latent space instead of DINOv2 — as the de-risking step before
building the full `@tensorleap_simulation` engine job (Option 2).

## What runs where

| Where | What |
|---|---|
| Mac (engine + code-loader checkouts, platform) | Phase 1 vendoring (copy engine files into this repo, commit); Phase 2 export script (needs engine env + platform storage) |
| EC2 `nvidia` box | Everything else: `git pull` this repo, receive `tl_ls_basis.npz` + ONNX, run the loop against local Isaac |

Neither the engine repo nor a code-loader checkout is needed on EC2. `code-loader` (pip) is only
needed if pushing the integration from there.

## Phase 1 — Vendor the engine loop into `calibration_optuna/`

Source: `engine/src_tensorleap/trainer/ds_curation/calibration_optuna/`.
Our `calibration_optuna/` is an older fork of this package; refresh it:

1. Copy `convergence.py` and `metrics.py` verbatim (both are pure numpy/scipy — verified).
2. Copy `loop.py` as `calibration_optuna/engine_loop.py` and de-platform it:
   - Replace `from src_tensorleap.contract...` imports with a new local `calibration_optuna/contracts.py`
     defining the two small dataclasses: `SingleSimulationData(sim_name, params, n_samples, seed)`
     and `SimulationInstance(name, sim_config)`.
   - Drop `LatentSpaceType` / `OPTUNA_CALIBRATION_LS_TYPE` (only used as a dict key in the engine
     dispatcher; our dispatcher returns arrays directly).
   - Replace `leaplogger` with stdlib `logging` (keep the `extra={...}` payloads — they're the
     progress telemetry).
3. Keep our existing `optimizer.py`, `config.py`, `experiment_runner.py` (the engine loop only
   needs `OptunaOptimizer.ask(n)`, `.tell(dist_id, score)`, `.mark_failed(dist_id)`,
   `.n_startup_trials` — all present in the fork). Diff the fork's `optimizer.py` against the
   engine's during the copy and note (not blindly adopt) any divergence.
4. Do NOT modify `controller.py` in this phase; the engine loop lands alongside, not instead.

## Phase 2 — Pin the latent space and export the basis (Mac / platform side)

1. Add `@tensorleap_custom_latent_space` to the integration (same tensor the loop's
   `RFDETREmbedder.layer_index` targets today), push, run one evaluate. This makes the LS layer
   explicit and identical between platform and headless runs.
2. New script `scripts/export_tl_ls_basis.py` (runs in the engine env): load the version's
   `latent_space_after_pca` blob, dump to `tl_ls_basis.npz`:
   `mean_vec`, `std_vec`, `pca_components`, plus `real_embeddings` — the target-cluster
   embeddings selected by the same filter the platform job would use, already PCA-projected.
3. Copy the `.npz` (few MB) to EC2. Re-export whenever the model or the evaluate changes —
   record the source version uid inside the npz for traceability.

## Phase 3 — `TLLatentSpaceEmbedder` (EC2 side)

New backend in `simulation_calibration_loop/data.py`, matching the existing
`embed_paths(paths, batch_size, cache_path, manifest)` interface:

- ONNX `InferenceSession` on the RF-DETR model, run to the custom-LS output tensor,
  using the same image preprocessing as the integration's input encoder.
- Project: `((x - mean_vec) / std_vec) @ pca_components.T` (this is the entirety of the
  engine's `apply_normalized_pca`).
- Reuse the existing embedding cache/manifest mechanics; `model_name` in the manifest =
  `tl_ls_<version_uid>` so cached embeddings invalidate when the basis changes.
- Config: `embedder_backend: tl_ls` + `tl_ls: {npz_path, onnx_path, ls_output_name}`.

## Phase 4 — `IsaacSampleDispatcher` implementing the engine seam

New `simulation_calibration_loop/engine_dispatcher.py` exposing the engine loop's interface:

- `generate(sim_data_by_dist) -> {dist_id: [sample_ids]}`: for each `SingleSimulationData`,
  materialize the YAML (`materialize_config` on the base template + params), then
  `run_isaac_generation(..., num_frames_override=n_samples, seed=seed)` with the existing
  seed-retry (`seed + k*1000` on no-freespace layouts). Sample ids = frame paths.
- `collect_ls(sample_ids_by_dist) -> {dist_id: np.ndarray}`: embed each trial's frames via
  `TLLatentSpaceEmbedder`, return projected arrays. Partial generation: return what rendered;
  the engine loop's keep-threshold gating handles the rest.
- `prune_caches(keep_ids)`: evict embedding caches outside the running top-K (optional, no-op ok).
- Multi-seed pooling (controller's `eval_seeds`) lives INSIDE `generate` per trial if wanted —
  the engine loop is seed-agnostic.
- Search-space mapping: build `SimulationInstance(name="simulation_1", sim_config=...)` from the
  project config's `search_space.bounds` (numeric `[min,max]` -> `{type, bounds:{min,max}}`,
  lists -> categorical `values`) — the same shapes `@tensorleap_simulation` takes, so the
  Option-2 migration is a rename.

## Phase 5 — Runner + A/B

1. `simulation_calibration_loop/run_engine_loop.py`: load project config + npz, build dispatcher,
   run `CalibrationLoop(simulations, dispatcher, real_embeddings)`, write
   `best_trials.csv`-style output + a state json compatible with the existing plots.
   Note: the loop's auto-sizing fires ONE extra "ping" Isaac run (center-of-bounds params) at start.
2. Tests in `tests/test_engine_loop_port.py`: fake dispatcher with synthetic Gaussians —
   assert convergence stop-reason, outlier suppression, and that a partial-generation trial is
   gated (mirror the engine's own unit tests).
3. The experiment that decides Option 2: same theme round, same bounds, TL-LS backend vs the
   converging DINOv2 baseline. Compare best-MMD trajectory + whether suggestions expand the
   distribution again (the failure mode of the earlier TL loop).

## Deferred (phase B, only if the A/B converges)

- Wrap the engine loop under `controller.py` so base pool / themes / promoted-baseline chain
  around it, or go straight to Option 2 (`@tensorleap_simulation` + engine job) where those
  become manual rounds.
- Fork-sync policy: re-diff against engine `calibration_optuna/` per engine release we care about.

# Option 2: engine synthetic job on the EC2 Isaac machine

Decision (2026-08-30): go Option-2-first. The engine's synthetic job gives the exact TL latent
space natively (model inference + ES-picked default LS + evaluate-fit PCA, all inside the job),
which removes the LS-extractor export that made the headless path expensive. The vendored engine
loop from `option3-lite-fork-refresh.md` Phase 1 (done, commit b05760f) is kept as the fallback:
if the TL-LS search doesn't converge, revert to DINOv2 embeddings on the updated loop code — a
config change, not new work.

Deployment answer: full TL install co-located with Isaac on the EC2 `nvidia` box. The remote
variant (ship YAMLs / download samples per trial) is ruled out — per-trial round-trips and egress
make the calibration loop transfer-bound.

## Preconditions to verify first

- Integration uses STRING sample ids (hard requirement — `workersynthetic.py` rejects int-id projects).
- EC2 instance has headroom to run k3d + platform beside Isaac (RAM/disk; both want the GPU —
  evaluate and Isaac generation won't overlap within the job, but check memory).
- `leap check` runs each simulation once with N=1, seed=0 — the stub-first phasing below keeps
  Isaac out of the check until the plumbing exists.

## Phase 1 — Platform on the box

1. k3d TL install on EC2, cluster created with a shared volume mount
   (`--volume /shared/isaac:/shared/isaac`) so generation pods and the host see one directory.
2. Warehouse/LOCO data present (already on the box for the standalone loop); wire
   `project_config.yaml` paths.
3. `leap push` the existing integration; run evaluate on this install — its PCA is the basis the
   synthetic job will use. No custom LS, no graph surgery: the job uses the default LS.

## Phase 2 — Stub simulation, end-to-end validation

1. Add a trivial `@tensorleap_simulation` (fast synthetic images with one controllable statistic,
   e.g. brightness drawn from a param) returning a `PreprocessResponse` the existing input
   encoder can read. Keyword contract: sim_params keys + `N` + `seed`, deterministic per seed.
2. Push, evaluate, trigger the synthetic job from the UI with a small target filter.
3. Success criteria: job completes; top-K samples persisted as `additional`; `synthetic_results`
   cluster filter visible; report bundle + `best_trials.csv` in storage; best params move toward
   the target statistic. This proves push → pods → encoders → LS → PCA → MMD → persistence
   without Isaac in the loop.

## Phase 3 — Isaac behind the sim function

1. Host-side runner: a small watcher (systemd unit or screen loop) polling
   `/shared/isaac/requests/`, executing the existing `run_isaac_generation` per request file,
   writing frames + a done-marker to `/shared/isaac/outputs/<request_id>/`.
2. `@tensorleap_simulation("isaac_trajectory", sim_params=...)`: materialize the YAML from the
   base template + params (`materialize_config`), drop a request file, block on the done-marker,
   return a `PreprocessResponse` over the rendered RGB paths.
3. Seed-retry (`seed + k*1000` on no-freespace layouts) lives in the host runner, mirroring
   `run_base_v4_train.sh` — the engine only sees produced-vs-requested and fails below an 80%
   generation ratio.
4. Timeout/failure contract: runner writes an error-marker on Isaac failure; the sim function
   raises so the trial is marked failed rather than hanging the queue.

## Phase 4 — Real search space

1. Trim to the top ~8–10 params by fANOVA importance (`more_points_param_importance.csv`,
   `state.json` param_importances) — NOT ~20: engine budget defaults scale with dimensionality
   (~11 startup trials per dim, batch ≤ 8) and are not overridable from the job request; at d≈20
   that is ~700 Isaac sessions worst-case, at d≈8–10 roughly half with a real chance of early
   stale-stop.
2. Map bounds from `search_space.bounds` into `sim_params` (same shapes; the Option-3 dispatcher
   mapping notes apply verbatim).
3. Fixed params (previous best / promoted baseline) fold into the base YAML inside the sim
   function.
4. Run against the real target filter. Read results: UI cluster filter (sim vs real),
   report bundle (MMD curve, noise-floor ratio, importances, stop reason), `best_trials.csv`.

## Phase 5 — Decide

- Converges (best-MMD trend down, suggestions not re-expanding the distribution): adopt as the
  production loop; theme rounds = per-round `sim_params` edit + re-push + new job (manual, or a
  script around the leap CLI). Consider an engine PR to expose budget knobs in the job request
  if runs are frequent.
- Doesn't converge: fallback — the vendored `calibration_optuna.engine_loop.CalibrationLoop` on
  EC2 with the DINOv2 embedder (Option 3 wiring minus the TL-LS embedder), keeping the engine's
  loop semantics either way.

## Open questions

- Trial latency: Isaac app startup dominates; if per-trial cost is too high, use the SDG script's
  `--seeds` episode mode inside the runner to render a trial's frames in one session.
- Whether the auto-sizing ping trial's center-of-bounds params always produce a navigable layout
  (runner's seed-retry should absorb it).

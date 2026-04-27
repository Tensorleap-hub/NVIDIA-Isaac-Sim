from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration_optuna.optimizer import OptunaOptimizer
from calibration_optuna.metrics import compute_all_metrics, DistributionMetrics

from .config import (
    THETA_STAR_PATH, N_ITERATIONS, N_TRIALS_PER_ITER, N_IMAGES_PER_TRIAL, SEED, RUNS_DIR,
    THETA_KEYS, THETA_BOUNDS, MMD_MAX_SAMPLES, seed_thetas,
)
from .evaluator import Embedder, mmd_rbf
from .generate_tl_seed import _SEED_THETAS, _SEED_N_IMAGES
from .harness import run_trial
from .metrics import MetricsLogger, IterationRecord

_GROUP = "simulation_1"

_PARAM_BOUNDS = {_GROUP: {
    "blur_sigma":       [0.0, 5.0],
    "noise_std":        [0.0, 0.5],
    "brightness_shift": [-0.5, 0.5],
    "color_shift_r":    [-0.3, 0.3],
    "color_shift_g":    [-0.3, 0.3],
    "color_shift_b":    [-0.3, 0.3],
    "clutter_count":    [0, 20],
    "background_id":    [0, 1, 2, 3],
}}

_PARAM_TYPE = {_GROUP: {
    "blur_sigma":       "float",
    "noise_std":        "float",
    "brightness_shift": "float",
    "color_shift_r":    "float",
    "color_shift_g":    "float",
    "color_shift_b":    "float",
    "clutter_count":    "int",
    "background_id":    "categorical",
}}



def _theta_to_params(theta: dict) -> dict:
    params = {f"shape_logit_{_GROUP}": 0.0}
    for k, v in theta.items():
        params[f"{_GROUP}__{k}"] = v
    return params


def _params_to_theta(params: dict) -> dict:
    prefix = f"{_GROUP}__"
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def run_optuna_loop(
    real_embeddings: np.ndarray,
    run_dir: Path,
    n_iterations: int = N_ITERATIONS,
    n_trials_per_iter: int = N_TRIALS_PER_ITER,
    n_images: int = N_IMAGES_PER_TRIAL,
    seed: int = SEED,
) -> list[IterationRecord]:
    run_dir = Path(run_dir)
    theta_star = json.loads(THETA_STAR_PATH.read_text())
    logger = MetricsLogger(run_dir / "metrics.csv", theta_star)
    embedder = Embedder()

    config = {
        "experiment_name": f"bench_seed{seed}",
        "random_seed": seed,
        "iteration_batch_size": n_trials_per_iter,
        "optimization_metrics": ["mmd_rbf"],
        "mmd_max_samples": MMD_MAX_SAMPLES,
        "optimizer": {
            "n_startup_trials": 10,
            "multivariate": True,
        },
    }

    optimizer = OptunaOptimizer(
        experiment_dir=run_dir,
        config=config,
        param_bounds=_PARAM_BOUNDS,
        param_type=_PARAM_TYPE,
    )

    rng_real = np.random.RandomState(42)
    if len(real_embeddings) > MMD_MAX_SAMPLES:
        real_idx = rng_real.choice(len(real_embeddings), MMD_MAX_SAMPLES, replace=False)
        real_sub = real_embeddings[real_idx]
    else:
        real_sub = real_embeddings
    rbf_gamma = DistributionMetrics._compute_gamma_median_heuristic(real_sub, real_sub)

    trial_seed = seed
    current_distributions = [
        (f"seed_{i}", _theta_to_params(t))
        for i, t in enumerate(_SEED_THETAS)
    ]

    for iteration in range(n_iterations):
        iter_n_images = _SEED_N_IMAGES if iteration == 0 else n_images
        trial_results = []
        metrics_list = []
        all_syn_embs = []
        for _dist_id, params in current_distributions:
            theta = _params_to_theta(params)
            _dist, syn_embs = run_trial(theta, iter_n_images, real_embeddings, embedder, seed=trial_seed)
            trial_seed += 1
            full_metrics = compute_all_metrics(syn_embs, real_sub, rbf_gamma=rbf_gamma)
            trial_results.append((theta, full_metrics["mmd_rbf"]))
            metrics_list.append(full_metrics)
            all_syn_embs.append(syn_embs)

        pooled = np.concatenate(all_syn_embs, axis=0)
        all_samples_obj = mmd_rbf(pooled, real_embeddings)
        record = logger.log(iteration, trial_results, all_samples_obj)
        print(
            f"[optuna] iter={iteration:02d}  best={record.best_objective:.4f}"
            f"  gap={record.param_gap:.4f}  spread={record.spread:.4f}"
        )

        current_distributions = optimizer.suggest_next_distributions(
            current_distributions=current_distributions,
            metrics_list=metrics_list,
            config=config,
            trial_numbers=None,
        )

    return logger.load()


if __name__ == "__main__":
    embs = np.load(str(RUNS_DIR.parent / "real_embeddings.npy"))
    run_dir = RUNS_DIR / f"optuna_seed{SEED}"
    run_optuna_loop(real_embeddings=embs, run_dir=run_dir)

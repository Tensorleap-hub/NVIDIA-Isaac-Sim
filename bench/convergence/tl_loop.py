from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    THETA_STAR_PATH, N_IMAGES_PER_TRIAL, N_TRIALS_PER_ITER, SEED, RUNS_DIR,
)
from .evaluator import Embedder
from .harness import run_trial
from .metrics import MetricsLogger, IterationRecord

_THETA_COLUMNS = [
    "blur_sigma", "noise_std", "brightness_shift",
    "color_shift_r", "color_shift_g", "color_shift_b",
    "clutter_count", "background_id",
]


def run_tl_loop(
    real_embeddings: np.ndarray,
    csv_path: Path,
    run_dir: Path,
    n_images: int = N_IMAGES_PER_TRIAL,
    n_trials_per_iter: int = N_TRIALS_PER_ITER,
    seed: int = SEED,
) -> list[IterationRecord]:
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    theta_star = json.loads(THETA_STAR_PATH.read_text())
    logger = MetricsLogger(run_dir / "metrics.csv", theta_star)
    embedder = Embedder()

    df = pd.read_csv(csv_path)
    missing = [c for c in _THETA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TL CSV missing columns: {missing}")

    trial_seed = seed
    chunks = [df.iloc[i : i + n_trials_per_iter] for i in range(0, len(df), n_trials_per_iter)]
    for iteration, chunk in enumerate(chunks):
        trial_results = []
        for _, row in chunk.iterrows():
            theta = {k: row[k] for k in _THETA_COLUMNS}
            dist, _ = run_trial(theta, n_images, real_embeddings, embedder, seed=trial_seed)
            trial_seed += 1
            trial_results.append((theta, dist))
        record = logger.log(iteration, trial_results)
        print(
            f"[tl]     iter={iteration:02d}  best={record.best_objective:.4f}"
            f"  gap={record.param_gap:.4f}  spread={record.spread:.4f}"
        )

    return logger.load()

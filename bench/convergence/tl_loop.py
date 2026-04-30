from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    THETA_STAR_PATH, N_IMAGES_PER_TRIAL, SEED, RUNS_DIR,
)
from .evaluator import Embedder
from .generate_tl_seed import _SEED_THETAS, _SEED_N_IMAGES
from .evaluator import mmd_rbf
from .harness import run_trial
from .metrics import MetricsLogger, IterationRecord

_PREFIX = "metadata.theta_"


def _parse_next_trials_csv(csv_path: Path) -> list[dict]:
    from .config import THETA_KEYS
    df = pd.read_csv(csv_path)
    thetas = []
    for _, row in df.iterrows():
        theta = {k: float(row[f"{_PREFIX}{k}"]) for k in THETA_KEYS}
        theta["clutter_count"] = int(round(theta["clutter_count"]))
        thetas.append(theta)
    return thetas


def run_tl_loop(
    real_embeddings: np.ndarray,
    iter_csvs: list[Path],
    run_dir: Path,
    seed: int = SEED,
) -> list[IterationRecord]:
    run_dir = Path(run_dir)
    theta_star = json.loads(THETA_STAR_PATH.read_text())
    logger = MetricsLogger(run_dir / "metrics.csv", theta_star)
    embedder = Embedder()

    iterations = [("seed", _SEED_THETAS, _SEED_N_IMAGES)] + [
        (str(p.name), _parse_next_trials_csv(p), N_IMAGES_PER_TRIAL)
        for p in iter_csvs
    ]

    trial_seed = seed
    for i, (_label, thetas, n_imgs) in enumerate(iterations):
        trial_results = []
        all_syn_embs = []
        for theta in thetas:
            _dist, syn_embs = run_trial(theta, n_imgs, real_embeddings, embedder, seed=trial_seed)
            trial_seed += 1
            trial_results.append((theta, _dist))
            all_syn_embs.append(syn_embs)
        pooled = np.concatenate(all_syn_embs, axis=0)
        all_samples_obj = mmd_rbf(pooled, real_embeddings)
        record = logger.log(i, trial_results, all_samples_obj)
        print(
            f"[tl]     iter={i:02d}  best={record.best_objective:.4f}"
            f"  gap={record.param_gap:.4f}  spread={record.spread:.4f}"
        )

    return logger.load()


if __name__ == "__main__":
    from .config import REAL_EMBEDDINGS_PATH
    real_embs = np.load(str(REAL_EMBEDDINGS_PATH))
    trials_dir = Path(__file__).parent.parent / "trials_from_tl"
    iter_csvs = sorted(trials_dir.glob("next-trials-*.csv"))
    run_dir = RUNS_DIR / f"tl_seed{SEED}"
    run_tl_loop(real_embeddings=real_embs, iter_csvs=iter_csvs, run_dir=run_dir)

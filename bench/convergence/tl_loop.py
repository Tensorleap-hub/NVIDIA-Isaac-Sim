from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    THETA_STAR_PATH, N_IMAGES_PER_TRIAL, SEED, RUNS_DIR, MMD_MAX_SAMPLES,
)
from .evaluator import Embedder, mmd_rbf
from .generate_tl_seed import _SEED_THETAS, _SEED_N_IMAGES
from .generator import generate_images
from .metrics import MetricsLogger, IterationRecord

_PREFIX = "metadata.theta_"


_INT_KEYS = {"clutter_count", "background_id"}


def _parse_next_trials_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    theta_cols = [c for c in df.columns if c.startswith(_PREFIX)]
    thetas = []
    for _, row in df.iterrows():
        theta = {}
        for col in theta_cols:
            key = col[len(_PREFIX):]
            val = float(row[col])
            theta[key] = int(round(val)) if key in _INT_KEYS else val
        thetas.append(theta)
    return thetas


def _save_iter_data(run_dir: Path, iter_idx: int, theta_images: list[tuple[dict, list]]) -> None:
    iter_dir = run_dir / f"iter_{iter_idx:02d}"
    images_dir = iter_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dist_idx, (theta, images) in enumerate(theta_images):
        for img_idx, img in enumerate(images):
            fname = f"dist_{dist_idx:03d}_{img_idx:04d}.png"
            img.save(images_dir / fname)
            rows.append({"image_path": str(images_dir / fname), **theta})
    pd.DataFrame(rows).to_csv(iter_dir / "metadata.csv", index=False)


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
        is_seed = _label == "seed"
        theta_images = []

        for theta in thetas:
            images = generate_images(theta, n_imgs, seed=trial_seed)
            syn_embs = embedder.embed(images)
            dist = mmd_rbf(syn_embs, real_embeddings, max_samples=MMD_MAX_SAMPLES)
            trial_seed += 1
            trial_results.append((theta, dist))
            all_syn_embs.append(syn_embs)
            if not is_seed:
                theta_images.append((theta, images))

        if not is_seed:
            _save_iter_data(run_dir, i, theta_images)

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

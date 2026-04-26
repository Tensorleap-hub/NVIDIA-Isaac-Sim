from __future__ import annotations

import numpy as np
from .generator import generate_images
from .evaluator import mmd_rbf
from .config import MMD_MAX_SAMPLES


def run_trial(
    theta: dict,
    n_images: int,
    real_embeddings: np.ndarray,
    embedder,
    seed: int = 0,
    mmd_max_samples: int = MMD_MAX_SAMPLES,
) -> tuple[float, np.ndarray]:
    images = generate_images(theta, n=n_images, seed=seed)
    syn_embeddings = embedder.embed(images)
    distance = mmd_rbf(syn_embeddings, real_embeddings, max_samples=mmd_max_samples)
    return distance, syn_embeddings

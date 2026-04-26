"""
Generate the initial TL synth dataset — identical to iteration 0 of the local benchmark.

Produces:
  <DATA_ROOT>/tl_seed/images/seed_<D>_<I>.png   — raw images
  <DATA_ROOT>/tl_seed/metadata.csv               — one row per image, theta columns
  <DATA_ROOT>/tl_seed/embeddings.npy             — DINOv2 embeddings (for validation)

The metadata CSV is in the format expected by
calibration_optuna.data_utils.prepare_client_data_for_optimizer:
  - one row per image (no distribution_id column needed — detected by grouping identical rows)
  - columns are the raw theta param names
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DATA_ROOT, N_TRIALS_PER_ITER, N_IMAGES_PER_TRIAL, SEED, THETA_KEYS, seed_thetas,
)
from .evaluator import Embedder
from .generator import generate_images

TL_SEED_DIR = DATA_ROOT / "tl_seed"
IMAGES_DIR = TL_SEED_DIR / "images"
METADATA_PATH = TL_SEED_DIR / "metadata.csv"
EMBEDDINGS_PATH = TL_SEED_DIR / "embeddings.npy"


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    thetas = seed_thetas(N_TRIALS_PER_ITER, SEED)
    print(f"Generating {N_TRIALS_PER_ITER} distributions × {N_IMAGES_PER_TRIAL} images …")

    all_images = []
    metadata_rows = []

    for dist_idx, theta in enumerate(thetas):
        images = generate_images(theta, n=N_IMAGES_PER_TRIAL, seed=dist_idx)
        for img_idx, img in enumerate(images):
            fname = f"seed_{dist_idx:03d}_{img_idx:04d}.png"
            img.save(IMAGES_DIR / fname)
            row = {"image_path": str(IMAGES_DIR / fname), **theta}
            metadata_rows.append(row)
        all_images.extend(images)
        if (dist_idx + 1) % 5 == 0:
            print(f"  {dist_idx + 1}/{N_TRIALS_PER_ITER} distributions done")

    metadata_df = pd.DataFrame(metadata_rows, columns=["image_path"] + THETA_KEYS)
    metadata_df.to_csv(METADATA_PATH, index=False)
    print(f"Metadata saved → {METADATA_PATH}  ({len(metadata_df)} rows)")

    print("Embedding with DINOv2 …")
    embedder = Embedder()
    embeddings = embedder.embed(all_images)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings saved → {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

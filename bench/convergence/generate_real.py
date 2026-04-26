from __future__ import annotations

import json
import numpy as np

from .config import (
    REAL_DIR,
    REAL_EMBEDDINGS_PATH,
    RUNS_DIR,
    THETA_STAR_PATH,
    N_REAL_IMAGES,
)
from .evaluator import Embedder
from .generator import generate_images


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    theta_star = json.loads(THETA_STAR_PATH.read_text())
    print(f"Generating {N_REAL_IMAGES} images from θ* …")
    images = generate_images(theta_star, n=N_REAL_IMAGES, seed=0)

    for i, img in enumerate(images):
        img.save(REAL_DIR / f"{i:04d}.png")
    print(f"Saved PNGs → {REAL_DIR}")

    print("Embedding with DINOv2 …")
    embedder = Embedder()
    embeddings = embedder.embed(images)
    np.save(REAL_EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved → {REAL_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()

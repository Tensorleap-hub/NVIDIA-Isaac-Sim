from pathlib import Path

DATA_ROOT = Path.home() / "tensorleap" / "data" / "synth-data-benchmark"
REAL_DIR = DATA_ROOT / "real"
REAL_EMBEDDINGS_PATH = DATA_ROOT / "real_embeddings.npy"
RUNS_DIR = DATA_ROOT / "runs"
THETA_STAR_PATH = Path(__file__).parent / "theta_star.json"
N_REAL_IMAGES = 500

THETA_KEYS = [
    "blur_sigma",
    "clutter_count",
]
THETA_BOUNDS = {
    "blur_sigma":    (0.0, 5.0),
    "clutter_count": (0.0, 20.0),
}

def seed_thetas(n: int, seed: int) -> list:
    import numpy as np
    rng = np.random.RandomState(seed)
    thetas = []
    for _ in range(n):
        theta = {
            "blur_sigma": float(rng.uniform(0.0, 5.0)),
            "clutter_count": int(rng.randint(0, 21)),
        }
        thetas.append(theta)
    return thetas


IMAGE_SIZE = 256
N_IMAGES_PER_TRIAL = 16
N_ITERATIONS = 10
N_TRIALS_PER_ITER = 6
SEED = 42
MMD_MAX_SAMPLES = 1000

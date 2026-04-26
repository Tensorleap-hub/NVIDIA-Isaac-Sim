from pathlib import Path

DATA_ROOT = Path.home() / "tensorleap" / "data" / "synth-data-benchmark"
REAL_DIR = DATA_ROOT / "real"
REAL_EMBEDDINGS_PATH = DATA_ROOT / "real_embeddings.npy"
RUNS_DIR = DATA_ROOT / "runs"
THETA_STAR_PATH = Path(__file__).parent / "theta_star.json"
N_REAL_IMAGES = 500

THETA_KEYS = [
    "blur_sigma", "noise_std", "brightness_shift",
    "color_shift_r", "color_shift_g", "color_shift_b",
    "clutter_count", "background_id",
]
THETA_BOUNDS = {
    "blur_sigma":        (0.0,  5.0),
    "noise_std":         (0.0,  0.5),
    "brightness_shift":  (-0.5, 0.5),
    "color_shift_r":     (-0.3, 0.3),
    "color_shift_g":     (-0.3, 0.3),
    "color_shift_b":     (-0.3, 0.3),
    "clutter_count":     (0.0,  20.0),
    "background_id":     (0.0,  3.0),
}

IMAGE_SIZE = 256
N_IMAGES_PER_TRIAL = 128
N_ITERATIONS = 30
N_TRIALS_PER_ITER = 8
SEED = 42
MMD_MAX_SAMPLES = 1000

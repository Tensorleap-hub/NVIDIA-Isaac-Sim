import numpy as np
from unittest.mock import MagicMock
from convergence.harness import run_trial

_THETA = {
    "blur_sigma": 1.5, "noise_std": 0.1, "brightness_shift": 0.05,
    "color_shift_r": 0.0, "color_shift_g": 0.0, "color_shift_b": 0.0,
    "clutter_count": 3, "background_id": 0,
}


def test_returns_float_and_array():
    real = np.random.randn(50, 768).astype(np.float32)
    embedder = MagicMock()
    embedder.embed.return_value = np.random.randn(8, 768).astype(np.float32)

    dist, embs = run_trial(_THETA, n_images=8, real_embeddings=real, embedder=embedder, seed=0)

    assert isinstance(dist, float)
    assert dist >= 0.0
    assert embs.shape == (8, 768)
    embedder.embed.assert_called_once()


def test_embedder_receives_correct_image_count():
    real = np.random.randn(50, 768).astype(np.float32)
    embedder = MagicMock()
    embedder.embed.return_value = np.random.randn(16, 768).astype(np.float32)

    run_trial(_THETA, n_images=16, real_embeddings=real, embedder=embedder, seed=0)

    called_images = embedder.embed.call_args[0][0]
    assert len(called_images) == 16


def test_mmd_zero_when_embeddings_match_real():
    rng = np.random.RandomState(42)
    real = rng.randn(50, 32).astype(np.float32)
    syn = real[:8].copy()
    embedder = MagicMock()
    embedder.embed.return_value = syn

    dist, _ = run_trial(_THETA, n_images=8, real_embeddings=real, embedder=embedder, seed=0)
    assert dist < 0.5

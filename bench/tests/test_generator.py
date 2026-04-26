import numpy as np
import pytest
from PIL import Image
from convergence.generator import generate_images

_BASE = {
    "blur_sigma": 0.0, "noise_std": 0.0, "brightness_shift": 0.0,
    "color_shift_r": 0.0, "color_shift_g": 0.0, "color_shift_b": 0.0,
    "clutter_count": 0, "background_id": 0,
}


def test_returns_correct_count():
    imgs = generate_images(_BASE, n=4, seed=0)
    assert len(imgs) == 4


def test_returns_pil_images():
    imgs = generate_images(_BASE, n=2, seed=0)
    assert all(isinstance(img, Image.Image) for img in imgs)


def test_output_size():
    imgs = generate_images(_BASE, n=1, seed=0)
    assert imgs[0].size == (256, 256)


def test_deterministic():
    theta = {**_BASE, "noise_std": 0.1, "clutter_count": 3}
    a = generate_images(theta, n=2, seed=42)
    b = generate_images(theta, n=2, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(np.array(x), np.array(y))


def test_different_seeds_differ():
    theta = {**_BASE, "noise_std": 0.3}
    i1 = generate_images(theta, n=1, seed=1)[0]
    i2 = generate_images(theta, n=1, seed=2)[0]
    assert not np.array_equal(np.array(i1), np.array(i2))


@pytest.mark.parametrize("bg_id", [0, 1, 2, 3])
def test_all_background_ids(bg_id):
    theta = {**_BASE, "background_id": bg_id}
    imgs = generate_images(theta, n=1, seed=0)
    assert len(imgs) == 1


def test_blur_reduces_high_frequency():
    theta_noisy = {**_BASE, "background_id": 3}
    no_blur = np.array(generate_images({**theta_noisy, "blur_sigma": 0.0}, n=1, seed=0)[0]).astype(float)
    blurred = np.array(generate_images({**theta_noisy, "blur_sigma": 3.0}, n=1, seed=0)[0]).astype(float)
    assert blurred.std() < no_blur.std()


def test_clutter_changes_image():
    no_clutter = np.array(generate_images({**_BASE, "clutter_count": 0}, n=1, seed=5)[0])
    with_clutter = np.array(generate_images({**_BASE, "clutter_count": 15}, n=1, seed=5)[0])
    assert not np.array_equal(no_clutter, with_clutter)

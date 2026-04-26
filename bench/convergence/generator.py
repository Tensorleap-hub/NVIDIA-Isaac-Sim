import numpy as np
from PIL import Image, ImageFilter
from .config import IMAGE_SIZE


def _make_background(bg_id: int) -> np.ndarray:
    h = w = IMAGE_SIZE
    rng = np.random.RandomState(int(bg_id) * 12345)
    if bg_id == 0:
        arr = np.zeros((h, w, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(0.2, 0.8, w)
        arr[:, :, 1] = np.linspace(0.5, 0.3, w)
        arr[:, :, 2] = np.linspace(0.8, 0.2, w)
    elif bg_id == 1:
        block = 32
        grid = (np.indices((h, w)).sum(axis=0) // block) % 2
        v = (0.3 + grid * 0.4).astype(np.float32)
        arr = np.stack([v, v, v], axis=-1)
    elif bg_id == 2:
        arr = np.zeros((h, w, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(0.1, 0.7, h)[:, None]
        arr[:, :, 1] = np.linspace(0.6, 0.2, h)[:, None]
        arr[:, :, 2] = np.linspace(0.3, 0.9, h)[:, None]
    else:
        arr = rng.uniform(0.2, 0.8, (h, w, 3)).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def generate_images(theta: dict, n: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    bg = _make_background(int(theta["background_id"]))
    images = []
    for _ in range(n):
        img = bg.copy()

        img += float(theta["brightness_shift"])
        img[:, :, 0] += float(theta["color_shift_r"])
        img[:, :, 1] += float(theta["color_shift_g"])
        img[:, :, 2] += float(theta["color_shift_b"])
        img = np.clip(img, 0.0, 1.0)

        n_rects = int(theta["clutter_count"])
        sz = IMAGE_SIZE
        for _ in range(n_rects):
            x1 = rng.randint(0, sz - 1)
            y1 = rng.randint(0, sz - 1)
            x2 = rng.randint(x1 + 1, min(x1 + sz // 4 + 1, sz))
            y2 = rng.randint(y1 + 1, min(y1 + sz // 4 + 1, sz))
            img[y1:y2, x1:x2] = rng.rand(3).astype(np.float32)

        noise_std = float(theta["noise_std"])
        if noise_std > 0:
            img += rng.normal(0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        blur_sigma = float(theta["blur_sigma"])
        pil = Image.fromarray((img * 255).astype(np.uint8))
        if blur_sigma > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        images.append(pil)
    return images

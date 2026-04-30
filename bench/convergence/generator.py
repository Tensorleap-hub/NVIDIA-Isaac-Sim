import numpy as np
from PIL import Image, ImageFilter
from .config import IMAGE_SIZE

_BG_BLOCK = 32
_BG: np.ndarray | None = None


def _get_background() -> np.ndarray:
    global _BG
    if _BG is None:
        h = w = IMAGE_SIZE
        grid = (np.indices((h, w)).sum(axis=0) // _BG_BLOCK) % 2
        v = (0.3 + grid * 0.4).astype(np.float32)
        _BG = np.stack([v, v, v], axis=-1)
    return _BG


def generate_images(theta: dict, n: int, seed: int) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    bg = _get_background()
    sz = IMAGE_SIZE
    blur_sigma = float(theta["blur_sigma"])
    n_rects = int(theta["clutter_count"])

    images = []
    for _ in range(n):
        img = bg.copy()

        for _ in range(n_rects):
            x1 = rng.randint(0, sz - 1)
            y1 = rng.randint(0, sz - 1)
            x2 = rng.randint(x1 + 1, min(x1 + sz // 4 + 1, sz))
            y2 = rng.randint(y1 + 1, min(y1 + sz // 4 + 1, sz))
            img[y1:y2, x1:x2] = rng.rand(3).astype(np.float32)

        pil = Image.fromarray((img * 255).astype(np.uint8))
        if blur_sigma > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        images.append(pil)
    return images

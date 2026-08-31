import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_latent_space

from .config import CONFIG


def _ls_config() -> dict:
    return CONFIG["dino_latent_space"]


def _cache_path_for_image(image_path: str) -> Path:
    cfg = _ls_config()
    key = hashlib.sha1(os.path.abspath(image_path).encode()).hexdigest()
    return Path(cfg["cache_dir"]) / f"{key}_{cfg['model_name']}.npy"


def _load_embedding(image_path: str) -> np.ndarray:
    cache_path = _cache_path_for_image(image_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing DINOv2 embedding for {image_path} (expected {cache_path}). "
            "Run: poetry run python -m tensorleap_intgration_code.latent_space"
        )
    return np.load(str(cache_path)).astype(np.float32)


@tensorleap_custom_latent_space()
def dino_latent_space(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return _load_embedding(preprocess.data[idx]["path"])


def extract_latent_space(subsets: List[PreprocessResponse]) -> None:
    import torch
    from PIL import Image
    from torchvision import transforms

    cfg = _ls_config()
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[Path, str] = {}
    for subset in subsets:
        for sample_id in subset.sample_ids:
            image_path = os.path.abspath(subset.data[sample_id]["path"])
            cache_path = _cache_path_for_image(image_path)
            if not cache_path.exists():
                pending[cache_path] = image_path
    if not pending:
        return

    device_name = cfg.get("device") or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(device_name)
    model = torch.hub.load(cfg["repo"], cfg["model_name"])
    model.eval()
    model.to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(cfg["resize_size"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    items = list(pending.items())
    batch_size = int(cfg["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            images = []
            for _, image_path in batch:
                with Image.open(image_path) as image:
                    images.append(transform(image.convert("RGB")))
            features = model(torch.stack(images, dim=0).to(device)).detach().cpu().numpy().astype(np.float32)
            for (cache_path, _), feature in zip(batch, features):
                np.save(str(cache_path), feature)


if __name__ == "__main__":
    from .data_preprocess import preprocess_func_leap

    extract_latent_space(preprocess_func_leap())

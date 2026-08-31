import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_latent_space

from .config import CONFIG


def _resolve_device(cfg: dict) -> str:
    if cfg.get("device"):
        return cfg["device"]
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _dino_latent_space_impl(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
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

    device = torch.device(_resolve_device(cfg))
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


def _rfdetr_neck_ls_config() -> dict:
    return CONFIG["rfdetr_neck_latent_space"]


def _rfdetr_neck_cache_path_for_image(image_path: str) -> Path:
    cfg = _rfdetr_neck_ls_config()
    key = hashlib.sha1(os.path.abspath(image_path).encode()).hexdigest()
    return Path(cfg["cache_dir"]) / f"{key}_rfdetr_neck.npy"


def _load_rfdetr_neck_embedding(image_path: str) -> np.ndarray:
    cache_path = _rfdetr_neck_cache_path_for_image(image_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing RF-DETR neck embedding for {image_path} (expected {cache_path}). "
            "Run: poetry run python -m tensorleap_intgration_code.latent_space"
        )
    return np.load(str(cache_path)).astype(np.float32)


def _rfdetr_neck_latent_space_impl(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return _load_rfdetr_neck_embedding(preprocess.data[idx]["path"])


def extract_rfdetr_neck_latent_space(subsets: List[PreprocessResponse]) -> None:
    import sys

    from .config import abs_path_from_root

    repo_root = abs_path_from_root("")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from simulation_calibration_loop.data import RFDETRNeckEmbedder

    cfg = _rfdetr_neck_ls_config()
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[Path, str] = {}
    for subset in subsets:
        for sample_id in subset.sample_ids:
            image_path = os.path.abspath(subset.data[sample_id]["path"])
            cache_path = _rfdetr_neck_cache_path_for_image(image_path)
            if not cache_path.exists():
                pending[cache_path] = image_path
    if not pending:
        return

    embedder = RFDETRNeckEmbedder(
        checkpoint_path=abs_path_from_root(cfg["checkpoint_path"]),
        device=_resolve_device(cfg),
        image_size=int(cfg["image_size"]),
        resize_size=int(cfg["resize_size"]),
        out_channels=int(cfg.get("out_channels", 256)),
    )

    items = list(pending.items())
    batch_size = int(cfg["batch_size"])
    scratch_path = cache_dir / "_rfdetr_neck_batch_scratch.npy"
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        image_paths = [Path(image_path) for _, image_path in batch]
        features = embedder.embed_paths(
            image_paths,
            batch_size=batch_size,
            cache_path=scratch_path,
            manifest={"paths": [str(p) for p in image_paths]},
        )
        for (cache_path, _), feature in zip(batch, features):
            np.save(str(cache_path), feature)
    scratch_path.unlink(missing_ok=True)
    scratch_path.with_suffix(".manifest.json").unlink(missing_ok=True)


_LATENT_SPACE_IMPLS = {
    "dino": _dino_latent_space_impl,
    "rfdetr_neck": _rfdetr_neck_latent_space_impl,
}

_ACTIVE_LATENT_SPACE_BACKEND = CONFIG.get("active_latent_space", "dino")
if _ACTIVE_LATENT_SPACE_BACKEND not in _LATENT_SPACE_IMPLS:
    raise ValueError(
        f"Unknown active_latent_space '{_ACTIVE_LATENT_SPACE_BACKEND}', "
        f"expected one of {list(_LATENT_SPACE_IMPLS)}"
    )

active_latent_space = tensorleap_custom_latent_space()(_LATENT_SPACE_IMPLS[_ACTIVE_LATENT_SPACE_BACKEND])


if __name__ == "__main__":
    from .data_preprocess import preprocess_func_leap

    subsets = preprocess_func_leap()
    if _ACTIVE_LATENT_SPACE_BACKEND == "rfdetr_neck":
        extract_rfdetr_neck_latent_space(subsets)
    else:
        extract_latent_space(subsets)

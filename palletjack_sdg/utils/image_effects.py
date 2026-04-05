import io
import os
import random

import numpy as np
from PIL import Image


def sample_normal(mean, std, lower=None, upper=None, integer=False):
    """Sample a scalar from N(mean, std), with optional clamping."""
    if mean is None:
        return None

    std = 0.0 if std is None else float(std)
    value = float(mean) if std <= 0 else random.gauss(float(mean), std)
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    if integer:
        return int(round(value))
    return value


def get_dataset_noise_cfg(cam_cfg):
    """Resolve per-image dataset-noise config from explicit or legacy fields."""
    ds_cfg = (cam_cfg.get("dataset_noise") or {}).copy()
    if ds_cfg:
        ds_cfg.setdefault("enabled", False)
        ds_cfg.setdefault("mode", "gaussian")
        ds_cfg.setdefault("sigma_mean", 0.0)
        ds_cfg.setdefault("sigma_std", 0.0)
        ds_cfg.setdefault("jpeg_quality_mean", 95)
        ds_cfg.setdefault("jpeg_quality_std", 2.0)
        ds_cfg.setdefault("shot_scale_mean", 100.0)
        ds_cfg.setdefault("shot_scale_std", 0.0)
        ds_cfg.setdefault("seed", -1)
        if (
            ds_cfg.get("enabled")
            or float(ds_cfg.get("sigma_mean", 0.0)) > 0.0
            or float(ds_cfg.get("sigma_std", 0.0)) > 0.0
        ):
            return ds_cfg

    legacy_mean = cam_cfg.get("noise_std_mean", 0.0)
    legacy_std = cam_cfg.get("noise_std_std", 0.0)
    return {
        "enabled": (legacy_mean is not None and float(legacy_mean) > 0.0)
        or (legacy_std is not None and float(legacy_std) > 0.0),
        "mode": "gaussian",
        "sigma_mean": 0.0 if legacy_mean is None else float(legacy_mean),
        "sigma_std": 0.0 if legacy_std is None else float(legacy_std),
        "jpeg_quality_mean": 95,
        "jpeg_quality_std": 2.0,
        "shot_scale_mean": 100.0,
        "shot_scale_std": 0.0,
        "seed": -1,
    }


def resolve_image_augmentation_cfg(image_augmentation_cfg, cam_cfg):
    """Resolve post-write image augmentation config, including legacy camera color."""
    aug_cfg = (image_augmentation_cfg or {}).copy()
    aug_cfg.setdefault("enabled", False)
    aug_cfg.setdefault("brightness_gain_mean", 1.0)
    aug_cfg.setdefault("brightness_gain_std", 0.0)
    aug_cfg.setdefault("contrast_gain_mean", 1.0)
    aug_cfg.setdefault("contrast_gain_std", 0.0)
    aug_cfg.setdefault("gamma_mean", 1.0)
    aug_cfg.setdefault("gamma_std", 0.0)

    if "color_gain_mean" not in aug_cfg or "color_gain_std" not in aug_cfg:
        color_mean = cam_cfg.get("color_mean")
        color_std = cam_cfg.get("color_std")
        if color_mean is not None and color_std is not None:
            aug_cfg.setdefault("color_gain_mean", list(color_mean))
            aug_cfg.setdefault("color_gain_std", list(color_std))
    aug_cfg.setdefault("color_gain_mean", [1.0, 1.0, 1.0])
    aug_cfg.setdefault("color_gain_std", [0.0, 0.0, 0.0])

    neutral = (
        aug_cfg["brightness_gain_mean"] == 1.0
        and aug_cfg["brightness_gain_std"] == 0.0
        and aug_cfg["contrast_gain_mean"] == 1.0
        and aug_cfg["contrast_gain_std"] == 0.0
        and aug_cfg["gamma_mean"] == 1.0
        and aug_cfg["gamma_std"] == 0.0
        and list(aug_cfg["color_gain_mean"]) == [1.0, 1.0, 1.0]
        and list(aug_cfg["color_gain_std"]) == [0.0, 0.0, 0.0]
    )
    if not neutral:
        aug_cfg["enabled"] = True
    return aug_cfg


def sample_image_augmentation_params(aug_cfg):
    return {
        "brightness_gain": sample_normal(
            aug_cfg["brightness_gain_mean"],
            aug_cfg["brightness_gain_std"],
            lower=0.0,
        ),
        "contrast_gain": sample_normal(
            aug_cfg["contrast_gain_mean"],
            aug_cfg["contrast_gain_std"],
            lower=0.0,
        ),
        "gamma": sample_normal(
            aug_cfg["gamma_mean"],
            aug_cfg["gamma_std"],
            lower=1e-6,
        ),
        "color_gain": tuple(
            max(0.0, sample_normal(channel_mean, channel_std, lower=0.0))
            for channel_mean, channel_std in zip(
                aug_cfg["color_gain_mean"], aug_cfg["color_gain_std"]
            )
        ),
    }


def apply_image_augmentation(image_data, aug_params):
    data = image_data.astype(np.float32)
    color_gain = np.asarray(aug_params["color_gain"], dtype=np.float32).reshape(1, 1, 3)
    data *= color_gain
    data *= float(aug_params["brightness_gain"])
    data = (data - 127.5) * float(aug_params["contrast_gain"]) + 127.5
    gamma = max(1e-6, float(aug_params["gamma"]))
    data = 255.0 * np.power(np.clip(data, 0.0, 255.0) / 255.0, gamma)
    return np.clip(data, 0.0, 255.0)


def apply_jpeg_artifacts(image_data, quality):
    quality = int(max(1, min(100, round(quality))))
    with io.BytesIO() as buffer:
        Image.fromarray(image_data.astype(np.uint8), mode="RGB").save(
            buffer, format="JPEG", quality=quality
        )
        buffer.seek(0)
        with Image.open(buffer) as img:
            return np.asarray(img.convert("RGB"), dtype=np.float32)


def apply_shot_noise(image_data, shot_scale, rng):
    shot_scale = max(1e-6, float(shot_scale))
    normalized = np.clip(image_data, 0.0, 255.0) / 255.0
    photons = np.clip(normalized * shot_scale, 0.0, None)
    noisy = rng.poisson(photons).astype(np.float32) / shot_scale
    return np.clip(noisy * 255.0, 0.0, 255.0)


def find_rgb_image_paths(output_dir):
    candidate_dirs = [
        os.path.join(output_dir, "Camera", "rgb"),
        os.path.join(output_dir, "image_2"),
        os.path.join(output_dir, "images"),
    ]
    extensions = (".png", ".jpg", ".jpeg")

    for candidate_dir in candidate_dirs:
        if os.path.isdir(candidate_dir):
            image_paths = sorted(
                os.path.join(candidate_dir, name)
                for name in os.listdir(candidate_dir)
                if name.lower().endswith(extensions)
            )
            if image_paths:
                return candidate_dir, image_paths

    image_paths = []
    for root, _, files in os.walk(output_dir):
        for name in files:
            if name.lower().endswith(extensions):
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return (os.path.dirname(image_paths[0]), image_paths) if image_paths else (None, [])


def apply_post_write_effects_to_saved_rgb(output_dir, noise_cfg, aug_cfg):
    """Apply image augmentation, per-image noise, and optional JPEG artifacts."""
    if not noise_cfg.get("enabled", False) and not aug_cfg.get("enabled", False):
        print("Image augmentation and dataset noise disabled — skipping RGB augmentation")
        return

    image_dir, image_paths = find_rgb_image_paths(output_dir)
    if not image_paths:
        print(f"No RGB image directory found under {output_dir}; skipping RGB augmentation")
        return

    sigma_mean = max(0.0, float(noise_cfg.get("sigma_mean", 0.0)))
    sigma_std = max(0.0, float(noise_cfg.get("sigma_std", 0.0)))
    jpeg_quality_mean = float(noise_cfg.get("jpeg_quality_mean", 95))
    jpeg_quality_std = max(0.0, float(noise_cfg.get("jpeg_quality_std", 2.0)))
    shot_scale_mean = max(1e-6, float(noise_cfg.get("shot_scale_mean", 100.0)))
    shot_scale_std = max(0.0, float(noise_cfg.get("shot_scale_std", 0.0)))
    mode = str(noise_cfg.get("mode", "gaussian"))
    seed = int(noise_cfg.get("seed", -1))
    rng = np.random.default_rng(None if seed < 0 else seed)

    print(
        f"Applying post-write effects to {len(image_paths)} image(s) in {image_dir}: "
        f"aug_enabled={aug_cfg.get('enabled', False)}, noise_mode={mode}"
    )

    for image_path in image_paths:
        with Image.open(image_path) as img:
            data = np.asarray(img.convert("RGB"), dtype=np.float32)
        if aug_cfg.get("enabled", False):
            data = apply_image_augmentation(data, sample_image_augmentation_params(aug_cfg))
        if mode in ("gaussian", "gaussian_jpeg") and noise_cfg.get("enabled", False):
            sigma = sigma_mean if sigma_std <= 0 else max(0.0, float(rng.normal(sigma_mean, sigma_std)))
            if sigma > 0:
                data = np.clip(data + rng.normal(0.0, sigma, size=data.shape), 0.0, 255.0)
        if mode in ("shot", "shot_jpeg") and noise_cfg.get("enabled", False):
            shot_scale = shot_scale_mean if shot_scale_std <= 0 else max(1e-6, float(rng.normal(shot_scale_mean, shot_scale_std)))
            data = apply_shot_noise(data, shot_scale, rng)
        if mode in ("jpeg", "gaussian_jpeg") and noise_cfg.get("enabled", False):
            quality = jpeg_quality_mean if jpeg_quality_std <= 0 else float(rng.normal(jpeg_quality_mean, jpeg_quality_std))
            data = apply_jpeg_artifacts(np.clip(data, 0.0, 255.0), quality)
        if mode == "shot_jpeg" and noise_cfg.get("enabled", False):
            quality = jpeg_quality_mean if jpeg_quality_std <= 0 else float(rng.normal(jpeg_quality_mean, jpeg_quality_std))
            data = apply_jpeg_artifacts(np.clip(data, 0.0, 255.0), quality)
        Image.fromarray(np.clip(data, 0.0, 255.0).astype(np.uint8), mode="RGB").save(image_path)

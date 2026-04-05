import math

import omni.replicator.core as rep


def _maybe_tuple(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def rep_normal(mean, std):
    """Create a replicator normal distribution for scalar or vector values."""
    mean = _maybe_tuple(mean)
    std = _maybe_tuple(std)
    if isinstance(mean, tuple) and not isinstance(std, tuple):
        std = tuple(float(std or 0.0) for _ in mean)
    if std is None:
        std = 0.0 if not isinstance(mean, tuple) else tuple(0.0 for _ in mean)
    return rep.distribution.normal(mean, std)


def fov_to_focal_length(horizontal_aperture, fov_degrees):
    return horizontal_aperture / (2 * math.tan(math.radians(fov_degrees) / 2))


def normalize_projection_type(camera_type):
    if camera_type == "fisheyeEquidistant":
        return "fisheyePolynomial"
    if camera_type == "fisheyePolynomial":
        return "fisheyePolynomial"
    return camera_type or "pinhole"


def is_fisheye_projection(camera_type):
    return normalize_projection_type(camera_type) != "pinhole"


def get_fisheye_max_fov_mean_std(cam_cfg):
    if cam_cfg.get("fisheye_max_fov") is not None:
        return float(cam_cfg["fisheye_max_fov"])
    if cam_cfg.get("fov_mean") is not None:
        fov_std = float(cam_cfg.get("fov_std", 0.0) or 0.0)
        return float(cam_cfg["fov_mean"]) + 2.0 * fov_std
    return 200.0

import math

import numpy as np
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


def euler_yaw_first_xyz(tilt_deg, yaw_deg, roll_deg=0.0):
    """Return intrinsic XYZ Euler angles (degrees) for R = Rz(yaw) * Rx(tilt).

    Applying yaw around world Z first, then tilt around the camera's local X,
    ensures that roll_deg=0 always produces a level horizon regardless of yaw.
    The optional roll_deg is added to the Y component (small-angle approximation
    around the camera forward axis — exact when tilt and yaw are both small).
    """
    t = math.radians(tilt_deg)
    y = math.radians(yaw_deg)
    sin_b = math.sin(y) * math.sin(t)
    b = math.asin(max(-1.0, min(1.0, sin_b)))
    cos_b = math.cos(b)
    if abs(cos_b) > 1e-7:
        a = math.atan2(math.cos(y) * math.sin(t), math.cos(t))
        c = math.atan2(math.sin(y) * math.cos(t), math.cos(y))
    else:
        # Gimbal lock (tilt≈90° and yaw≈90°): keep tilt, zero c
        a = t
        c = 0.0
    return (math.degrees(a), math.degrees(b) + roll_deg, math.degrees(c))


def sample_camera_rotation_choices(tilt_mean, tilt_std, yaw_mean, yaw_std, roll_mean, roll_std, n):
    """Pre-compute n rotation tuples with correct yaw-first ordering for rep.distribution.choice."""
    tilts = np.random.normal(tilt_mean, tilt_std, n) if tilt_std > 0 else np.full(n, tilt_mean)
    yaws = np.random.normal(yaw_mean, yaw_std, n) if yaw_std > 0 else np.full(n, yaw_mean)
    rolls = np.random.normal(roll_mean, roll_std, n) if roll_std > 0 else np.full(n, roll_mean)
    return [euler_yaw_first_xyz(float(t), float(y), float(r)) for t, y, r in zip(tilts, yaws, rolls)]


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

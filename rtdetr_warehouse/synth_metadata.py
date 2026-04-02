import os

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

# Fixed number of texture slots — matches the base sdg_config.yaml texture pool
_NUM_TEXTURES = 25

# Distractor group names — must match sdg_config.yaml distractors.groups keys
_DISTRACTOR_GROUPS = [
    "CardBox", "BarelPlastic", "BottlePlastic", "CratePlastic",
    "TrafficSigns", "Bucket", "RackPile", "PushCart",
]

_NAN = float("nan")

def _count_distractor_instances(rc: dict) -> float:
    """Total distractor instances = sum over groups of diversity × occurrence × clutter_level."""
    dist = rc.get("distractors", {})
    clutter = dist.get("clutter_level", 1.0)
    total = 0
    for g in dist.get("groups", {}).values():
        diversity  = min(g.get("diversity", len(g.get("assets", []))), len(g.get("assets", [])))
        occurrence = g.get("occurrence", 1)
        total += diversity * max(1, round(occurrence * clutter))
    return float(total) if total > 0 else _NAN


_SENTINEL = {
    "synth_run_number":                  _NAN,
    "synth_experiment":                  "",
    "synth_render_width":                _NAN,
    "synth_render_height":               _NAN,
    "synth_env_name":                    "",
    "synth_camera_type":                 "",
    "synth_camera_height_min":           _NAN,
    "synth_camera_height_max":           _NAN,
    "synth_camera_tilt_min":             _NAN,
    "synth_camera_tilt_max":             _NAN,
    "synth_camera_yaw_min":              _NAN,
    "synth_camera_yaw_max":              _NAN,
    "synth_camera_roll_min":             _NAN,
    "synth_camera_roll_max":             _NAN,
    "synth_fov_min":                     _NAN,
    "synth_fov_max":                     _NAN,
    "synth_noise_std_min":               _NAN,
    "synth_noise_std_max":               _NAN,
    "synth_motion_blur_min":             _NAN,
    "synth_motion_blur_max":             _NAN,
    "synth_jpeg_quality_min":            _NAN,
    "synth_jpeg_quality_max":            _NAN,
    "synth_distractors":                 "",
    "synth_clutter_level":               _NAN,
    "synth_palletjack_count_per_model":  _NAN,
    "synth_palletjack_rotation_max_z":   _NAN,
    "synth_palletjack_color_randomized": _NAN,
    "synth_lighting_intensity_mean":     _NAN,
    "synth_lighting_intensity_std":      _NAN,
    "synth_materials_roughness_min":     _NAN,
    "synth_materials_roughness_max":     _NAN,
    **{f"synth_texture_{i + 1}": "" for i in range(_NUM_TEXTURES)},
    "synth_num_distractor_instances":    _NAN,
    **{f"synth_dist_{g}_diversity":  _NAN for g in _DISTRACTOR_GROUPS},
    **{f"synth_dist_{g}_occurrence": _NAN for g in _DISTRACTOR_GROUPS},
    **{f"synth_dist_{g}_instances":  _NAN for g in _DISTRACTOR_GROUPS},
    "synth_num_objects":                 _NAN,
}


@tensorleap_metadata("synth_metadata")
def synth_metadata(idx: str, preprocess: PreprocessResponse) -> dict:
    record = preprocess.data[idx]
    rc = record.get("run_config") if isinstance(record, dict) else None

    if not rc:
        return _SENTINEL.copy()

    cam = rc.get("camera", {})
    pj = rc.get("palletjacks", {})
    light = rc.get("lighting", {})
    mat = rc.get("materials", {})
    render = rc.get("render", {})
    dist = rc.get("distractors", {})
    textures = mat.get("textures", [])

    tilt_min = cam.get("camera_tilt_min")
    tilt_max = cam.get("camera_tilt_max")

    return {
        "synth_run_number":                  int(record.get("run_number", 0)),
        "synth_experiment":                  str(record.get("experiment", "")),
        "synth_render_width":                int(render.get("width", 0)),
        "synth_render_height":               int(render.get("height", 0)),
        "synth_env_name":                    str(rc.get("environment", {}).get("name", "")),
        "synth_camera_type":                 str(cam.get("camera_type", "pinhole")),
        "synth_camera_height_min":           float(cam.get("camera_height_min", _NAN)),
        "synth_camera_height_max":           float(cam.get("camera_height_max", _NAN)),
        "synth_camera_tilt_min":             float(tilt_min) if tilt_min is not None else _NAN,
        "synth_camera_tilt_max":             float(tilt_max) if tilt_max is not None else _NAN,
        "synth_camera_yaw_min":              float(cam.get("camera_yaw_min", _NAN)),
        "synth_camera_yaw_max":              float(cam.get("camera_yaw_max", _NAN)),
        "synth_camera_roll_min":             float(cam.get("camera_roll_min", _NAN)),
        "synth_camera_roll_max":             float(cam.get("camera_roll_max", _NAN)),
        "synth_fov_min":                     float(cam.get("fov_min", _NAN)),
        "synth_fov_max":                     float(cam.get("fov_max", _NAN)),
        "synth_noise_std_min":               float(cam.get("noise_std_min", _NAN)),
        "synth_noise_std_max":               float(cam.get("noise_std_max", _NAN)),
        "synth_motion_blur_min":             float(cam.get("motion_blur_strength_min", _NAN)),
        "synth_motion_blur_max":             float(cam.get("motion_blur_strength_max", _NAN)),
        "synth_jpeg_quality_min":            float(cam.get("jpeg_quality_min", _NAN)),
        "synth_jpeg_quality_max":            float(cam.get("jpeg_quality_max", _NAN)),
        "synth_distractors":                 str(rc.get("run", {}).get("distractors", "")),
        "synth_clutter_level":               float(dist.get("clutter_level", _NAN)),
        "synth_palletjack_count_per_model":  int(pj.get("count_per_model", 0)),
        "synth_palletjack_rotation_max_z":   float((pj.get("rotation_max") or [_NAN, _NAN, _NAN])[2]),
        "synth_palletjack_color_randomized": float(any(v > 0 for v in (pj.get("color_max") or [0, 0, 0]))),
        "synth_lighting_intensity_mean":     float(light.get("intensity_mean", _NAN)),
        "synth_lighting_intensity_std":      float(light.get("intensity_std", _NAN)),
        "synth_materials_roughness_min":     float(mat.get("roughness_min", _NAN)),
        "synth_materials_roughness_max":     float(mat.get("roughness_max", _NAN)),
        **{
            f"synth_texture_{i + 1}": os.path.basename(textures[i]) if i < len(textures) else ""
            for i in range(_NUM_TEXTURES)
        },
        "synth_num_distractor_instances":    _count_distractor_instances(rc),
        **{
            f"synth_dist_{g}_diversity": float(
                dist.get("groups", {}).get(g, {}).get("diversity", _NAN)
            )
            for g in _DISTRACTOR_GROUPS
        },
        **{
            f"synth_dist_{g}_occurrence": float(
                dist.get("groups", {}).get(g, {}).get("occurrence", _NAN)
            )
            for g in _DISTRACTOR_GROUPS
        },
        **{
            f"synth_dist_{g}_instances": float(
                min(
                    dist.get("groups", {}).get(g, {}).get("diversity",
                        len(dist.get("groups", {}).get(g, {}).get("assets", []))),
                    len(dist.get("groups", {}).get(g, {}).get("assets", []))
                ) * max(1, round(
                    dist.get("groups", {}).get(g, {}).get("occurrence", 1)
                    * dist.get("clutter_level", 1.0)
                ))
            ) if dist.get("groups", {}).get(g) else _NAN
            for g in _DISTRACTOR_GROUPS
        },
        "synth_num_objects":                 len(record.get("anns", [])),
    }

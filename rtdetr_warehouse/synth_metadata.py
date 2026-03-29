import os

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

# Fixed number of texture slots — matches the base sdg_config.yaml texture pool
_NUM_TEXTURES = 25

_NAN = float("nan")

_SENTINEL = {
    "synth_run_number":                  _NAN,
    "synth_experiment":                  "",
    "synth_distractors":                 "",
    "synth_render_width":                _NAN,
    "synth_render_height":               _NAN,
    "synth_env_url":                     "",
    "synth_camera_height_min":           _NAN,
    "synth_camera_height_max":           _NAN,
    "synth_fov_min":                     _NAN,
    "synth_fov_max":                     _NAN,
    "synth_noise_std_min":               _NAN,
    "synth_noise_std_max":               _NAN,
    "synth_motion_blur_min":             _NAN,
    "synth_motion_blur_max":             _NAN,
    "synth_jpeg_quality_min":            _NAN,
    "synth_jpeg_quality_max":            _NAN,
    "synth_palletjack_count_per_model":  _NAN,
    "synth_palletjack_rotation_max_z":   _NAN,
    "synth_palletjack_color_randomized": _NAN,
    "synth_lighting_intensity_mean":     _NAN,
    "synth_lighting_intensity_std":      _NAN,
    "synth_materials_roughness_min":     _NAN,
    "synth_materials_roughness_max":     _NAN,
    **{f"synth_texture_{i + 1}": "" for i in range(_NUM_TEXTURES)},
    "synth_num_warehouse_distractors":   _NAN,
    "synth_num_additional_distractors":  _NAN,
    "synth_num_objects":                 _NAN,
}


@tensorleap_metadata("synth_metadata")
def synth_metadata(idx: int, preprocess: PreprocessResponse) -> dict:
    record = preprocess.data[idx]
    rc = record.get("run_config") if isinstance(record, dict) else None

    if not rc:
        return _SENTINEL.copy()

    cam = rc.get("camera", {})
    pj = rc.get("palletjacks", {})
    light = rc.get("lighting", {})
    mat = rc.get("materials", {})
    run = rc.get("run", {})
    render = rc.get("render", {})
    textures = mat.get("textures", [])

    return {
        "synth_run_number":                  int(record.get("run_number", 0)),
        "synth_experiment":                  str(record.get("experiment", "")),
        "synth_distractors":                 str(run.get("distractors", "")),
        "synth_render_width":                int(render.get("width", 0)),
        "synth_render_height":               int(render.get("height", 0)),
        "synth_env_url":                     str(rc.get("environment", {}).get("env_url", "")),
        "synth_camera_height_min":           float(cam.get("camera_height_min", _NAN)),
        "synth_camera_height_max":           float(cam.get("camera_height_max", _NAN)),
        "synth_fov_min":                     float(cam.get("fov_min", _NAN)),
        "synth_fov_max":                     float(cam.get("fov_max", _NAN)),
        "synth_noise_std_min":               float(cam.get("noise_std_min", _NAN)),
        "synth_noise_std_max":               float(cam.get("noise_std_max", _NAN)),
        "synth_motion_blur_min":             float(cam.get("motion_blur_strength_min", _NAN)),
        "synth_motion_blur_max":             float(cam.get("motion_blur_strength_max", _NAN)),
        "synth_jpeg_quality_min":            int(cam.get("jpeg_quality_min", 0)),
        "synth_jpeg_quality_max":            int(cam.get("jpeg_quality_max", 0)),
        "synth_palletjack_count_per_model":  int(pj.get("count_per_model", 0)),
        "synth_palletjack_rotation_max_z":   float((pj.get("rotation_max") or [_NAN, _NAN, _NAN])[2]),
        "synth_palletjack_color_randomized": any(v > 0 for v in (pj.get("color_max") or [0, 0, 0])),
        "synth_lighting_intensity_mean":     float(light.get("intensity_mean", _NAN)),
        "synth_lighting_intensity_std":      float(light.get("intensity_std", _NAN)),
        "synth_materials_roughness_min":     float(mat.get("roughness_min", _NAN)),
        "synth_materials_roughness_max":     float(mat.get("roughness_max", _NAN)),
        **{
            f"synth_texture_{i + 1}": os.path.basename(textures[i]) if i < len(textures) else ""
            for i in range(_NUM_TEXTURES)
        },
        "synth_num_warehouse_distractors":   len(rc.get("distractors_warehouse", {}).get("assets", [])),
        "synth_num_additional_distractors":  len(rc.get("distractors_additional", {}).get("assets", [])),
        "synth_num_objects":                 len(record.get("anns", [])),
    }

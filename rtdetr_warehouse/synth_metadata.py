from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata


def _rc(idx: int, preprocess: PreprocessResponse):
    record = preprocess.data[idx]
    return record.get("run_config") if isinstance(record, dict) else None


@tensorleap_metadata("synth_metadata")
def synth_metadata(idx: int, preprocess: PreprocessResponse) -> dict:
    rc = _rc(idx, preprocess)
    record = preprocess.data[idx]

    if not rc:
        return {}

    cam = rc.get("camera", {})
    pj = rc.get("palletjacks", {})
    light = rc.get("lighting", {})
    mat = rc.get("materials", {})
    run = rc.get("run", {})
    render = rc.get("render", {})

    return {
        "synth_num_frames":                  int(run.get("num_frames", 0)),
        "synth_distractors":                 str(run.get("distractors", "")),
        "synth_render_width":                int(render.get("width", 0)),
        "synth_render_height":               int(render.get("height", 0)),
        "synth_env_url":                     str(rc.get("environment", {}).get("env_url", "")),
        "synth_camera_height_min":           float(cam.get("camera_height_min", 0.0)),
        "synth_camera_height_max":           float(cam.get("camera_height_max", 0.0)),
        "synth_fov_min":                     float(cam.get("fov_min", 0.0)),
        "synth_fov_max":                     float(cam.get("fov_max", 0.0)),
        "synth_noise_std_min":               float(cam.get("noise_std_min", 0.0)),
        "synth_noise_std_max":               float(cam.get("noise_std_max", 0.0)),
        "synth_motion_blur_min":             float(cam.get("motion_blur_strength_min", 0.0)),
        "synth_motion_blur_max":             float(cam.get("motion_blur_strength_max", 0.0)),
        "synth_jpeg_quality_min":            int(cam.get("jpeg_quality_min", 0)),
        "synth_jpeg_quality_max":            int(cam.get("jpeg_quality_max", 0)),
        "synth_palletjack_count_per_model":  int(pj.get("count_per_model", 0)),
        "synth_palletjack_rotation_max_z":   float((pj.get("rotation_max") or [0, 0, 0])[2]),
        "synth_palletjack_color_randomized": any(v > 0 for v in (pj.get("color_max") or [0, 0, 0])),
        "synth_lighting_intensity_mean":     float(light.get("intensity_mean", 0.0)),
        "synth_lighting_intensity_std":      float(light.get("intensity_std", 0.0)),
        "synth_materials_roughness_min":     float(mat.get("roughness_min", 0.0)),
        "synth_materials_roughness_max":     float(mat.get("roughness_max", 0.0)),
        "synth_num_warehouse_distractors":   len(rc.get("distractors_warehouse", {}).get("assets", [])),
        "synth_num_additional_distractors":  len(rc.get("distractors_additional", {}).get("assets", [])),
        "synth_num_objects":                 len(record.get("anns", [])),
    }

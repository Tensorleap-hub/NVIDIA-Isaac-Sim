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


def _float_or_nan(value):
    return float(value) if value is not None else _NAN


def _vector_item(values, idx):
    if isinstance(values, list) and idx < len(values):
        return _float_or_nan(values[idx])
    return _NAN


def _basename_or_empty(paths, idx):
    if idx < len(paths):
        return os.path.basename(paths[idx])
    return ""


def _bool_to_float(value):
    return float(bool(value))


def _get_record_and_config(idx: str, preprocess: PreprocessResponse):
    record = preprocess.data[idx]
    rc = record.get("run_config") if isinstance(record, dict) else None
    return record, rc

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


def _distractor_group_metadata(dist: dict) -> dict:
    return {
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
                    dist.get("groups", {}).get(g, {}).get(
                        "diversity",
                        len(dist.get("groups", {}).get(g, {}).get("assets", [])),
                    ),
                    len(dist.get("groups", {}).get(g, {}).get("assets", [])),
                ) * max(
                    1,
                    round(
                        dist.get("groups", {}).get(g, {}).get("occurrence", 1)
                        * dist.get("clutter_level", 1.0)
                    ),
                )
            ) if dist.get("groups", {}).get(g) else _NAN
            for g in _DISTRACTOR_GROUPS
        },
    }


_SENTINEL = {
    "synth_source":                      "",
    "synth_optuna_bucket":               "",
    "synth_optuna_theme":                "",
    "synth_optuna_trial_number":         _NAN,
    "synth_optuna_rank":                 _NAN,
    "synth_optuna_objective_value":      _NAN,
    "synth_iteration":                   _NAN,
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


_MEAN_STD_SENTINEL = {
    "synth_source":                             "",
    "synth_optuna_bucket":                      "",
    "synth_optuna_theme":                       "",
    "synth_optuna_trial_number":                _NAN,
    "synth_optuna_rank":                        _NAN,
    "synth_optuna_objective_value":             _NAN,
    "synth_iteration":                          _NAN,
    "synth_run_number":                           _NAN,
    "synth_experiment":                           "",
    "synth_render_width":                         _NAN,
    "synth_render_height":                        _NAN,
    "synth_env_name":                             "",
    "synth_camera_type":                          "",
    "synth_distractors":                          "",
    "synth_clutter_level":                        _NAN,
    "synth_camera_position_mean_x":               _NAN,
    "synth_camera_position_mean_y":               _NAN,
    "synth_camera_position_std_x":                _NAN,
    "synth_camera_position_std_y":                _NAN,
    "synth_camera_height_mean":                   _NAN,
    "synth_camera_height_std":                    _NAN,
    "synth_camera_tilt_mean":                     _NAN,
    "synth_camera_tilt_std":                      _NAN,
    "synth_camera_yaw_mean":                      _NAN,
    "synth_camera_yaw_std":                       _NAN,
    "synth_camera_roll_mean":                     _NAN,
    "synth_camera_roll_std":                      _NAN,
    "synth_focal_length_mean":                    _NAN,
    "synth_focal_length_std":                     _NAN,
    "synth_fov_mean":                             _NAN,
    "synth_fov_std":                              _NAN,
    "synth_camera_color_mean_r":                  _NAN,
    "synth_camera_color_mean_g":                  _NAN,
    "synth_camera_color_mean_b":                  _NAN,
    "synth_camera_color_std_r":                   _NAN,
    "synth_camera_color_std_g":                   _NAN,
    "synth_camera_color_std_b":                   _NAN,
    "synth_noise_std_mean":                       _NAN,
    "synth_noise_std_std":                        _NAN,
    "synth_motion_blur_mean":                     _NAN,
    "synth_motion_blur_std":                      _NAN,
    "synth_jpeg_quality_mean":                    _NAN,
    "synth_jpeg_quality_std":                     _NAN,
    "synth_dataset_noise_enabled":                _NAN,
    "synth_dataset_noise_mode":                   "",
    "synth_dataset_noise_sigma_mean":             _NAN,
    "synth_dataset_noise_sigma_std":              _NAN,
    "synth_dataset_noise_jpeg_quality_mean":      _NAN,
    "synth_dataset_noise_jpeg_quality_std":       _NAN,
    "synth_dataset_noise_shot_scale_mean":        _NAN,
    "synth_dataset_noise_shot_scale_std":         _NAN,
    "synth_dataset_noise_seed":                   _NAN,
    "synth_image_augmentation_enabled":           _NAN,
    "synth_image_brightness_gain_mean":           _NAN,
    "synth_image_brightness_gain_std":            _NAN,
    "synth_image_contrast_gain_mean":             _NAN,
    "synth_image_contrast_gain_std":              _NAN,
    "synth_image_gamma_mean":                     _NAN,
    "synth_image_gamma_std":                      _NAN,
    "synth_image_color_gain_mean_r":              _NAN,
    "synth_image_color_gain_mean_g":              _NAN,
    "synth_image_color_gain_mean_b":              _NAN,
    "synth_image_color_gain_std_r":               _NAN,
    "synth_image_color_gain_std_g":               _NAN,
    "synth_image_color_gain_std_b":               _NAN,
    "synth_palletjack_count_per_model":           _NAN,
    "synth_palletjack_rotation_mean_z":           _NAN,
    "synth_palletjack_rotation_std_z":            _NAN,
    "synth_palletjack_color_randomized":          _NAN,
    "synth_palletjack_color_mean_r":              _NAN,
    "synth_palletjack_color_mean_g":              _NAN,
    "synth_palletjack_color_mean_b":              _NAN,
    "synth_palletjack_color_std_r":               _NAN,
    "synth_palletjack_color_std_g":               _NAN,
    "synth_palletjack_color_std_b":               _NAN,
    "synth_distractor_position_mean_x":           _NAN,
    "synth_distractor_position_mean_y":           _NAN,
    "synth_distractor_position_mean_z":           _NAN,
    "synth_distractor_position_std_x":            _NAN,
    "synth_distractor_position_std_y":            _NAN,
    "synth_distractor_position_std_z":            _NAN,
    "synth_distractor_rotation_mean_z":           _NAN,
    "synth_distractor_rotation_std_z":            _NAN,
    "synth_distractor_scale_mean":                _NAN,
    "synth_distractor_scale_std":                 _NAN,
    "synth_lighting_color_mean_r":                _NAN,
    "synth_lighting_color_mean_g":                _NAN,
    "synth_lighting_color_mean_b":                _NAN,
    "synth_lighting_color_std_r":                 _NAN,
    "synth_lighting_color_std_g":                 _NAN,
    "synth_lighting_color_std_b":                 _NAN,
    "synth_lighting_intensity_mean":              _NAN,
    "synth_lighting_intensity_std":               _NAN,
    "synth_materials_roughness_mean":             _NAN,
    "synth_materials_roughness_std":              _NAN,
    "synth_materials_emissive_intensity_mean":    _NAN,
    "synth_materials_emissive_intensity_std":     _NAN,
    **{f"synth_texture_{i + 1}": "" for i in range(_NUM_TEXTURES)},
    "synth_num_distractor_instances":             _NAN,
    **{f"synth_dist_{g}_diversity":  _NAN for g in _DISTRACTOR_GROUPS},
    **{f"synth_dist_{g}_occurrence": _NAN for g in _DISTRACTOR_GROUPS},
    **{f"synth_dist_{g}_instances":  _NAN for g in _DISTRACTOR_GROUPS},
    "synth_num_objects":                          _NAN,
}


@tensorleap_metadata("synth_metadata")
def synth_metadata(idx: str, preprocess: PreprocessResponse) -> dict:
    record, rc = _get_record_and_config(idx, preprocess)

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
        "synth_source":                      str(record.get("subset", "")),
        "synth_optuna_bucket":               str(record.get("optuna_bucket", "")),
        "synth_optuna_theme":                str(record.get("optuna_theme", "")),
        "synth_optuna_trial_number":         _float_or_nan(record.get("trial_number")),
        "synth_optuna_rank":                 _float_or_nan(record.get("optuna_rank")),
        "synth_optuna_objective_value":      _float_or_nan(record.get("optuna_objective_value")),
        "synth_iteration":                   _float_or_nan(record.get("iteration")),
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
            f"synth_texture_{i + 1}": _basename_or_empty(textures, i)
            for i in range(_NUM_TEXTURES)
        },
        "synth_num_distractor_instances":    _count_distractor_instances(rc),
        **_distractor_group_metadata(dist),
        "synth_num_objects":                 len(record.get("anns", [])),
    }


@tensorleap_metadata("synth_metadata_mean_std")
def synth_metadata_mean_std(idx: str, preprocess: PreprocessResponse) -> dict:
    record, rc = _get_record_and_config(idx, preprocess)

    if not rc:
        return _MEAN_STD_SENTINEL.copy()

    cam = rc.get("camera", {})
    pj = rc.get("palletjacks", {})
    light = rc.get("lighting", {})
    mat = rc.get("materials", {})
    render = rc.get("render", {})
    dist = rc.get("distractors", {})
    dist_rand = rc.get("distractor_randomization", {})
    dataset_noise = cam.get("dataset_noise", {})
    image_aug = rc.get("image_augmentation", {})
    textures = mat.get("textures", [])

    camera_color_mean = cam.get("color_mean")
    camera_color_std = cam.get("color_std")
    lighting_color_mean = light.get("color_mean")
    lighting_color_std = light.get("color_std")
    image_color_gain_mean = image_aug.get("color_gain_mean")
    image_color_gain_std = image_aug.get("color_gain_std")
    palletjack_color_mean = pj.get("color_mean")
    palletjack_color_std = pj.get("color_std")

    return {
        "synth_source":                             str(record.get("subset", "")),
        "synth_optuna_bucket":                      str(record.get("optuna_bucket", "")),
        "synth_optuna_theme":                       str(record.get("optuna_theme", "")),
        "synth_optuna_trial_number":                _float_or_nan(record.get("trial_number")),
        "synth_optuna_rank":                        _float_or_nan(record.get("optuna_rank")),
        "synth_optuna_objective_value":             _float_or_nan(record.get("optuna_objective_value")),
        "synth_iteration":                          _float_or_nan(record.get("iteration")),
        "synth_run_number":                        int(record.get("run_number", 0)),
        "synth_experiment":                        str(record.get("experiment", "")),
        "synth_render_width":                      int(render.get("width", 0)),
        "synth_render_height":                     int(render.get("height", 0)),
        "synth_env_name":                          str(rc.get("environment", {}).get("name", "")),
        "synth_camera_type":                       str(cam.get("camera_type", "pinhole")),
        "synth_distractors":                       str(rc.get("run", {}).get("distractors", "")),
        "synth_clutter_level":                     float(dist.get("clutter_level", _NAN)),
        "synth_camera_position_mean_x":            _vector_item(cam.get("position_mean"), 0),
        "synth_camera_position_mean_y":            _vector_item(cam.get("position_mean"), 1),
        "synth_camera_position_std_x":             _vector_item(cam.get("position_std"), 0),
        "synth_camera_position_std_y":             _vector_item(cam.get("position_std"), 1),
        "synth_camera_height_mean":                _float_or_nan(cam.get("camera_height_mean")),
        "synth_camera_height_std":                 _float_or_nan(cam.get("camera_height_std")),
        "synth_camera_tilt_mean":                  _float_or_nan(cam.get("camera_tilt_mean")),
        "synth_camera_tilt_std":                   _float_or_nan(cam.get("camera_tilt_std")),
        "synth_camera_yaw_mean":                   _float_or_nan(cam.get("camera_yaw_mean")),
        "synth_camera_yaw_std":                    _float_or_nan(cam.get("camera_yaw_std")),
        "synth_camera_roll_mean":                  _float_or_nan(cam.get("camera_roll_mean")),
        "synth_camera_roll_std":                   _float_or_nan(cam.get("camera_roll_std")),
        "synth_focal_length_mean":                 _float_or_nan(cam.get("focal_length_mean")),
        "synth_focal_length_std":                  _float_or_nan(cam.get("focal_length_std")),
        "synth_fov_mean":                          _float_or_nan(cam.get("fov_mean")),
        "synth_fov_std":                           _float_or_nan(cam.get("fov_std")),
        "synth_camera_color_mean_r":               _vector_item(camera_color_mean, 0),
        "synth_camera_color_mean_g":               _vector_item(camera_color_mean, 1),
        "synth_camera_color_mean_b":               _vector_item(camera_color_mean, 2),
        "synth_camera_color_std_r":                _vector_item(camera_color_std, 0),
        "synth_camera_color_std_g":                _vector_item(camera_color_std, 1),
        "synth_camera_color_std_b":                _vector_item(camera_color_std, 2),
        "synth_noise_std_mean":                    _float_or_nan(cam.get("noise_std_mean")),
        "synth_noise_std_std":                     _float_or_nan(cam.get("noise_std_std")),
        "synth_motion_blur_mean":                  _float_or_nan(cam.get("motion_blur_strength_mean")),
        "synth_motion_blur_std":                   _float_or_nan(cam.get("motion_blur_strength_std")),
        "synth_jpeg_quality_mean":                 _float_or_nan(cam.get("jpeg_quality_mean")),
        "synth_jpeg_quality_std":                  _float_or_nan(cam.get("jpeg_quality_std")),
        "synth_dataset_noise_enabled":             _bool_to_float(dataset_noise.get("enabled", False)),
        "synth_dataset_noise_mode":                str(dataset_noise.get("mode", "")),
        "synth_dataset_noise_sigma_mean":          _float_or_nan(dataset_noise.get("sigma_mean")),
        "synth_dataset_noise_sigma_std":           _float_or_nan(dataset_noise.get("sigma_std")),
        "synth_dataset_noise_jpeg_quality_mean":   _float_or_nan(dataset_noise.get("jpeg_quality_mean")),
        "synth_dataset_noise_jpeg_quality_std":    _float_or_nan(dataset_noise.get("jpeg_quality_std")),
        "synth_dataset_noise_shot_scale_mean":     _float_or_nan(dataset_noise.get("shot_scale_mean")),
        "synth_dataset_noise_shot_scale_std":      _float_or_nan(dataset_noise.get("shot_scale_std")),
        "synth_dataset_noise_seed":                _float_or_nan(dataset_noise.get("seed")),
        "synth_image_augmentation_enabled":        _bool_to_float(image_aug.get("enabled", False)),
        "synth_image_brightness_gain_mean":        _float_or_nan(image_aug.get("brightness_gain_mean")),
        "synth_image_brightness_gain_std":         _float_or_nan(image_aug.get("brightness_gain_std")),
        "synth_image_contrast_gain_mean":          _float_or_nan(image_aug.get("contrast_gain_mean")),
        "synth_image_contrast_gain_std":           _float_or_nan(image_aug.get("contrast_gain_std")),
        "synth_image_gamma_mean":                  _float_or_nan(image_aug.get("gamma_mean")),
        "synth_image_gamma_std":                   _float_or_nan(image_aug.get("gamma_std")),
        "synth_image_color_gain_mean_r":           _vector_item(image_color_gain_mean, 0),
        "synth_image_color_gain_mean_g":           _vector_item(image_color_gain_mean, 1),
        "synth_image_color_gain_mean_b":           _vector_item(image_color_gain_mean, 2),
        "synth_image_color_gain_std_r":            _vector_item(image_color_gain_std, 0),
        "synth_image_color_gain_std_g":            _vector_item(image_color_gain_std, 1),
        "synth_image_color_gain_std_b":            _vector_item(image_color_gain_std, 2),
        "synth_palletjack_count_per_model":        int(pj.get("count_per_model", 0)),
        "synth_palletjack_rotation_mean_z":        _vector_item(pj.get("rotation_mean"), 2),
        "synth_palletjack_rotation_std_z":         _vector_item(pj.get("rotation_std"), 2),
        "synth_palletjack_color_randomized":       _bool_to_float(
            any(v > 0 for v in (palletjack_color_std or [0, 0, 0]))
        ),
        "synth_palletjack_color_mean_r":           _vector_item(palletjack_color_mean, 0),
        "synth_palletjack_color_mean_g":           _vector_item(palletjack_color_mean, 1),
        "synth_palletjack_color_mean_b":           _vector_item(palletjack_color_mean, 2),
        "synth_palletjack_color_std_r":            _vector_item(palletjack_color_std, 0),
        "synth_palletjack_color_std_g":            _vector_item(palletjack_color_std, 1),
        "synth_palletjack_color_std_b":            _vector_item(palletjack_color_std, 2),
        "synth_distractor_position_mean_x":        _vector_item(dist_rand.get("position_mean"), 0),
        "synth_distractor_position_mean_y":        _vector_item(dist_rand.get("position_mean"), 1),
        "synth_distractor_position_mean_z":        _vector_item(dist_rand.get("position_mean"), 2),
        "synth_distractor_position_std_x":         _vector_item(dist_rand.get("position_std"), 0),
        "synth_distractor_position_std_y":         _vector_item(dist_rand.get("position_std"), 1),
        "synth_distractor_position_std_z":         _vector_item(dist_rand.get("position_std"), 2),
        "synth_distractor_rotation_mean_z":        _vector_item(dist_rand.get("rotation_mean"), 2),
        "synth_distractor_rotation_std_z":         _vector_item(dist_rand.get("rotation_std"), 2),
        "synth_distractor_scale_mean":             _float_or_nan(dist_rand.get("scale_mean")),
        "synth_distractor_scale_std":              _float_or_nan(dist_rand.get("scale_std")),
        "synth_lighting_color_mean_r":             _vector_item(lighting_color_mean, 0),
        "synth_lighting_color_mean_g":             _vector_item(lighting_color_mean, 1),
        "synth_lighting_color_mean_b":             _vector_item(lighting_color_mean, 2),
        "synth_lighting_color_std_r":              _vector_item(lighting_color_std, 0),
        "synth_lighting_color_std_g":              _vector_item(lighting_color_std, 1),
        "synth_lighting_color_std_b":              _vector_item(lighting_color_std, 2),
        "synth_lighting_intensity_mean":           _float_or_nan(light.get("intensity_mean")),
        "synth_lighting_intensity_std":            _float_or_nan(light.get("intensity_std")),
        "synth_materials_roughness_mean":          _float_or_nan(mat.get("roughness_mean")),
        "synth_materials_roughness_std":           _float_or_nan(mat.get("roughness_std")),
        "synth_materials_emissive_intensity_mean": _float_or_nan(mat.get("emissive_intensity_mean")),
        "synth_materials_emissive_intensity_std":  _float_or_nan(mat.get("emissive_intensity_std")),
        **{
            f"synth_texture_{i + 1}": _basename_or_empty(textures, i)
            for i in range(_NUM_TEXTURES)
        },
        "synth_num_distractor_instances":          _count_distractor_instances(rc),
        **_distractor_group_metadata(dist),
        "synth_num_objects":                       len(record.get("anns", [])),
    }

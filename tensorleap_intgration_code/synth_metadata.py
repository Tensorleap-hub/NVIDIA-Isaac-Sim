import csv
import os

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata
from tensorleap_intgration_code.config import CONFIG, abs_path_from_root

# Fixed number of texture slots — matches the base sdg_config.yaml texture pool
_NUM_TEXTURES = 25

# Distractor group names — must match sdg_config.yaml distractors.groups keys
_DISTRACTOR_GROUPS = [
    "CardBox", "BarelPlastic", "BottlePlastic", "CratePlastic",
    "TrafficSigns", "Bucket", "RackPile", "PushCart",
]

_NAN = float("nan")

_DINO_PERF_CSV = CONFIG.get(
    "dino_distances_csv",
    "simulation_calibration_loop/population_view/dino-population_performance.csv",
)

_DINO_METRIC_COLS = {
    "synth_dino_mmd_rbf":               "mmd_rbf_to_real",
    "synth_dino_centroid_l2":           "centroid_l2_to_real",
    "synth_dino_centroid_cosine":       "centroid_cosine_to_real",
    "synth_dino_pca_centroid_l2":       "pca_centroid_l2_to_real",
    "synth_dino_syn_to_real_nn_mean":   "syn_to_real_nn_mean",
    "synth_dino_syn_to_real_nn_median": "syn_to_real_nn_median",
    "synth_dino_real_to_syn_nn_mean":   "real_to_syn_nn_mean",
}

_OPT_RUN_SPLITS = (5, 10, 20)
_OPT_RUN_GROUP_KEYS = [f"synth_dino_{n}_opt_run_id" for n in _OPT_RUN_SPLITS]
_OPT_RUN_ORDER = CONFIG.get("dino_opt_run_order", [])

_DINO_NAN_ROW = {**{k: _NAN for k in _DINO_METRIC_COLS}, **{k: _NAN for k in _OPT_RUN_GROUP_KEYS}}



def _load_dino_lookups() -> tuple:
    csv_path = abs_path_from_root(_DINO_PERF_CSV)
    optuna_lookup = {}
    base_lookup = {}
    if not os.path.isfile(csv_path):
        return optuna_lookup, base_lookup

    base_rows = []
    optuna_pending = []  # (rank, lookup_key, metrics)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            metrics = {}
            for meta_key, csv_col in _DINO_METRIC_COLS.items():
                val = row.get(csv_col, "")
                try:
                    metrics[meta_key] = float(val)
                except (ValueError, TypeError):
                    metrics[meta_key] = _NAN

            category = row.get("category", "")
            trial_id = row.get("trial_id", "")
            run_id = row.get("run_id", "")

            if category and trial_id and run_id:
                cache_path = row.get("cache_path", "")
                label = row.get("label", "")
                optuna_pending.append(((category, trial_id, run_id), metrics))
            else:
                exp_name = os.path.basename(row.get("cache_path", ""))
                if exp_name:
                    for k in _OPT_RUN_GROUP_KEYS:
                        metrics[k] = 0
                    base_rows.append((exp_name, metrics))

    # When order list is set: listed runs get group IDs, unlisted get "".
    # When order list is empty: fall back to assigning groups to all rows by rank.
    if _OPT_RUN_ORDER:
        listed   = [(r, k, m) for r, k, m in optuna_pending if r >= 0]
        unlisted = [(r, k, m) for r, k, m in optuna_pending if r < 0]
    else:
        listed   = optuna_pending
        unlisted = []

    listed.sort(key=lambda x: x[0])
    total = len(listed)
    for i, (key, metrics) in enumerate(listed):
        for n in _OPT_RUN_SPLITS:
            group = i * n // total + 1
            metrics[f"synth_dino_{n}_opt_run_id"] = group
        optuna_lookup[key] = metrics

    for _, key, metrics in unlisted:
        for k in _OPT_RUN_GROUP_KEYS:
            metrics[k] = _NAN
        optuna_lookup[key] = metrics

    for exp_name, metrics in base_rows:
        base_lookup[exp_name] = metrics

    return optuna_lookup, base_lookup


_DINO_OPTUNA_LOOKUP, _DINO_BASE_LOOKUP = _load_dino_lookups()


# ---------------------------------------------------------------------------
# Collection CSVs — live ratio metadata
# ---------------------------------------------------------------------------

_COLLECTIONS_DIR = "tensorleap_intgration_code/collections"


def _load_collections() -> dict:
    """Returns {csv_stem: (ratio, frozenset_of_ids)}"""
    dir_path = abs_path_from_root(_COLLECTIONS_DIR)
    result = {}
    if not os.path.isdir(dir_path):
        return result
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".csv"):
            continue
        stem = fname[:-4]
        ids = set()
        with open(os.path.join(dir_path, fname), newline="") as f:
            for row in csv.DictReader(f):
                val = row.get("Index", "").strip()
                if val:
                    ids.add(val)
        if ids:
            result[stem] = (1.0 / len(ids), frozenset(ids))
    return result


_COLLECTIONS = _load_collections()
_COLLECTION_KEYS = [f"{stem}_ratio" for stem in _COLLECTIONS]


def _get_collection_meta(idx: str) -> dict:
    idx_str = str(idx)
    return {
        f"{stem}_ratio": ratio if idx_str in ids else _NAN
        for stem, (ratio, ids) in _COLLECTIONS.items()
    }


def _get_dino_metrics(record: dict) -> dict:
    experiment = record.get("experiment", "")
    if not experiment:
        return _DINO_NAN_ROW
    theme = record.get("optuna_theme", "")
    if not theme:
        # base_synth: experiment is the exp dir name, keyed directly
        return _DINO_BASE_LOOKUP.get(experiment, _DINO_NAN_ROW)
    # Tensorleap-Optimized: optuna_repetition is set by _load_flat_run_records
    # ("trial_132"), but empty by _load_optuna_records which sets trial_number.
    rep = record.get("optuna_repetition", "")
    if not rep:
        trial_num = record.get("trial_number")
        if trial_num is None:
            return _DINO_NAN_ROW
        rep = f"trial_{trial_num}"
    run_id = experiment.split("__")[0]
    return _DINO_OPTUNA_LOOKUP.get((theme, rep, run_id), _DINO_NAN_ROW)


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


_MEAN_STD_SENTINEL = {
    "synth_source":                             "",
    "synth_optuna_bucket":                      "",
    "synth_optuna_theme":                       "",
    "synth_optuna_repetition":                  "",
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
    **{k: _NAN for k in _DINO_METRIC_COLS},
    **{k: _NAN for k in _OPT_RUN_GROUP_KEYS},
    **{k: _NAN for k in _COLLECTION_KEYS},
}

@tensorleap_metadata("synth_metadata")
def synth_metadata(idx: str, preprocess: PreprocessResponse) -> dict:
    record, rc = _get_record_and_config(idx, preprocess)
    collection_meta = _get_collection_meta(idx)

    if not rc:
        return {**_MEAN_STD_SENTINEL, **collection_meta}

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
        "synth_optuna_repetition":                  str(record.get("optuna_repetition", "")),
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
        **_get_dino_metrics(record),
        **collection_meta,
    }

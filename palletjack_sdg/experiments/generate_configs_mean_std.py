"""Generate one mean/std SDG YAML config per row in a next-trials CSV.

Usage:
    python generate_configs_mean_std.py --csv <path-to-csv> --out-dir <output-directory>

Defaults:
    --csv     next to the script (looks for any *.csv nearby)
    --out-dir directory of the CSV file
"""

import argparse
import csv
import math
import os

import yaml


TEXTURE_PREFIX = "/Isaac/Materials/Textures/Patterns/"
METADATA_PREFIX = "metadata.synth_metadata_mean_std_"
TEXTURE_KEYS = [f"{METADATA_PREFIX}synth_texture_{i}" for i in range(1, 26)]
DISTRACTOR_GROUPS = [
    "CardBox",
    "BarelPlastic",
    "BottlePlastic",
    "CratePlastic",
    "TrafficSigns",
    "Bucket",
    "RackPile",
    "PushCart",
]


def metadata_key(suffix):
    return f"{METADATA_PREFIX}{suffix}"


def raw_value(row, suffix):
    return (row.get(metadata_key(suffix)) or "").strip()


def opt_str(row, suffix):
    value = raw_value(row, suffix)
    return value or None


def opt_float(row, suffix, ndigits=4):
    value = raw_value(row, suffix)
    if not value:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return round(parsed, ndigits)


def opt_int(row, suffix):
    value = opt_float(row, suffix, ndigits=8)
    if value is None:
        return None
    return int(round(value))


def opt_bool(row, suffix):
    value = raw_value(row, suffix)
    if not value:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return bool(parsed)


def complete_vector(*values):
    if any(value is None for value in values):
        return None
    return list(values)


def collect_textures(row):
    seen = set()
    textures = []
    for key in TEXTURE_KEYS:
        value = (row.get(key) or "").strip()
        if value and value not in seen:
            seen.add(value)
            textures.append(TEXTURE_PREFIX + value)
    return textures


def set_if_present(target, key, value):
    if value is None:
        return False
    target[key] = value
    return True


def build_config(row, extends_path):
    cfg = {
        "extends": extends_path,
        "run": {
            "num_frames": int(float(row["n_samples"])),
            "data_dir": "/placeholder",
        },
    }
    has_override = False

    distractors_mode = opt_str(row, "synth_distractors")
    if distractors_mode is not None:
        cfg["run"]["distractors"] = distractors_mode
        has_override = True

    render = {}
    has_override |= set_if_present(render, "width", opt_int(row, "synth_render_width"))
    has_override |= set_if_present(render, "height", opt_int(row, "synth_render_height"))
    if render:
        cfg["render"] = render

    env_name = opt_str(row, "synth_env_name")
    if env_name is not None:
        cfg["environment"] = {"name": env_name}
        has_override = True

    camera = {}
    has_override |= set_if_present(camera, "camera_type", opt_str(row, "synth_camera_type"))
    has_override |= set_if_present(
        camera,
        "position_mean",
        complete_vector(
            opt_float(row, "synth_camera_position_mean_x"),
            opt_float(row, "synth_camera_position_mean_y"),
        ),
    )
    has_override |= set_if_present(
        camera,
        "position_std",
        complete_vector(
            opt_float(row, "synth_camera_position_std_x"),
            opt_float(row, "synth_camera_position_std_y"),
        ),
    )
    has_override |= set_if_present(camera, "camera_height_mean", opt_float(row, "synth_camera_height_mean"))
    has_override |= set_if_present(camera, "camera_height_std", opt_float(row, "synth_camera_height_std"))
    has_override |= set_if_present(camera, "camera_tilt_mean", opt_float(row, "synth_camera_tilt_mean"))
    has_override |= set_if_present(camera, "camera_tilt_std", opt_float(row, "synth_camera_tilt_std"))
    has_override |= set_if_present(camera, "camera_yaw_mean", opt_float(row, "synth_camera_yaw_mean"))
    has_override |= set_if_present(camera, "camera_yaw_std", opt_float(row, "synth_camera_yaw_std"))
    has_override |= set_if_present(camera, "camera_roll_mean", opt_float(row, "synth_camera_roll_mean"))
    has_override |= set_if_present(camera, "camera_roll_std", opt_float(row, "synth_camera_roll_std"))
    has_override |= set_if_present(camera, "focal_length_mean", opt_float(row, "synth_focal_length_mean"))
    has_override |= set_if_present(camera, "focal_length_std", opt_float(row, "synth_focal_length_std"))
    has_override |= set_if_present(camera, "fov_mean", opt_float(row, "synth_fov_mean"))
    has_override |= set_if_present(camera, "fov_std", opt_float(row, "synth_fov_std"))
    has_override |= set_if_present(
        camera,
        "color_mean",
        complete_vector(
            opt_float(row, "synth_camera_color_mean_r"),
            opt_float(row, "synth_camera_color_mean_g"),
            opt_float(row, "synth_camera_color_mean_b"),
        ),
    )
    has_override |= set_if_present(
        camera,
        "color_std",
        complete_vector(
            opt_float(row, "synth_camera_color_std_r"),
            opt_float(row, "synth_camera_color_std_g"),
            opt_float(row, "synth_camera_color_std_b"),
        ),
    )
    has_override |= set_if_present(camera, "noise_std_mean", opt_float(row, "synth_noise_std_mean"))
    has_override |= set_if_present(camera, "noise_std_std", opt_float(row, "synth_noise_std_std"))
    has_override |= set_if_present(
        camera,
        "motion_blur_strength_mean",
        opt_float(row, "synth_motion_blur_mean"),
    )
    has_override |= set_if_present(
        camera,
        "motion_blur_strength_std",
        opt_float(row, "synth_motion_blur_std"),
    )
    has_override |= set_if_present(camera, "jpeg_quality_mean", opt_float(row, "synth_jpeg_quality_mean"))
    has_override |= set_if_present(camera, "jpeg_quality_std", opt_float(row, "synth_jpeg_quality_std"))

    dataset_noise = {}
    has_override |= set_if_present(
        dataset_noise,
        "enabled",
        opt_bool(row, "synth_dataset_noise_enabled"),
    )
    has_override |= set_if_present(dataset_noise, "mode", opt_str(row, "synth_dataset_noise_mode"))
    has_override |= set_if_present(
        dataset_noise,
        "sigma_mean",
        opt_float(row, "synth_dataset_noise_sigma_mean"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "sigma_std",
        opt_float(row, "synth_dataset_noise_sigma_std"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "jpeg_quality_mean",
        opt_float(row, "synth_dataset_noise_jpeg_quality_mean"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "jpeg_quality_std",
        opt_float(row, "synth_dataset_noise_jpeg_quality_std"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "shot_scale_mean",
        opt_float(row, "synth_dataset_noise_shot_scale_mean"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "shot_scale_std",
        opt_float(row, "synth_dataset_noise_shot_scale_std"),
    )
    has_override |= set_if_present(
        dataset_noise,
        "seed",
        opt_int(row, "synth_dataset_noise_seed"),
    )
    if dataset_noise:
        camera["dataset_noise"] = dataset_noise

    if camera:
        cfg["camera"] = camera

    image_augmentation = {}
    has_override |= set_if_present(
        image_augmentation,
        "enabled",
        opt_bool(row, "synth_image_augmentation_enabled"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "brightness_gain_mean",
        opt_float(row, "synth_image_brightness_gain_mean"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "brightness_gain_std",
        opt_float(row, "synth_image_brightness_gain_std"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "contrast_gain_mean",
        opt_float(row, "synth_image_contrast_gain_mean"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "contrast_gain_std",
        opt_float(row, "synth_image_contrast_gain_std"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "gamma_mean",
        opt_float(row, "synth_image_gamma_mean"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "gamma_std",
        opt_float(row, "synth_image_gamma_std"),
    )
    has_override |= set_if_present(
        image_augmentation,
        "color_gain_mean",
        complete_vector(
            opt_float(row, "synth_image_color_gain_mean_r"),
            opt_float(row, "synth_image_color_gain_mean_g"),
            opt_float(row, "synth_image_color_gain_mean_b"),
        ),
    )
    has_override |= set_if_present(
        image_augmentation,
        "color_gain_std",
        complete_vector(
            opt_float(row, "synth_image_color_gain_std_r"),
            opt_float(row, "synth_image_color_gain_std_g"),
            opt_float(row, "synth_image_color_gain_std_b"),
        ),
    )
    if image_augmentation:
        cfg["image_augmentation"] = image_augmentation

    palletjacks = {}
    has_override |= set_if_present(
        palletjacks,
        "count_per_model",
        opt_int(row, "synth_palletjack_count_per_model"),
    )
    has_override |= set_if_present(
        palletjacks,
        "rotation_mean",
        complete_vector(
            0.0,
            0.0,
            opt_float(row, "synth_palletjack_rotation_mean_z"),
        ),
    )
    has_override |= set_if_present(
        palletjacks,
        "rotation_std",
        complete_vector(
            0.0,
            0.0,
            opt_float(row, "synth_palletjack_rotation_std_z"),
        ),
    )
    has_override |= set_if_present(
        palletjacks,
        "color_mean",
        complete_vector(
            opt_float(row, "synth_palletjack_color_mean_r"),
            opt_float(row, "synth_palletjack_color_mean_g"),
            opt_float(row, "synth_palletjack_color_mean_b"),
        ),
    )
    has_override |= set_if_present(
        palletjacks,
        "color_std",
        complete_vector(
            opt_float(row, "synth_palletjack_color_std_r"),
            opt_float(row, "synth_palletjack_color_std_g"),
            opt_float(row, "synth_palletjack_color_std_b"),
        ),
    )
    if palletjacks:
        cfg["palletjacks"] = palletjacks

    distractor_randomization = {}
    has_override |= set_if_present(
        distractor_randomization,
        "position_mean",
        complete_vector(
            opt_float(row, "synth_distractor_position_mean_x"),
            opt_float(row, "synth_distractor_position_mean_y"),
            opt_float(row, "synth_distractor_position_mean_z"),
        ),
    )
    has_override |= set_if_present(
        distractor_randomization,
        "position_std",
        complete_vector(
            opt_float(row, "synth_distractor_position_std_x"),
            opt_float(row, "synth_distractor_position_std_y"),
            opt_float(row, "synth_distractor_position_std_z"),
        ),
    )
    has_override |= set_if_present(
        distractor_randomization,
        "rotation_mean",
        complete_vector(
            0.0,
            0.0,
            opt_float(row, "synth_distractor_rotation_mean_z"),
        ),
    )
    has_override |= set_if_present(
        distractor_randomization,
        "rotation_std",
        complete_vector(
            0.0,
            0.0,
            opt_float(row, "synth_distractor_rotation_std_z"),
        ),
    )
    has_override |= set_if_present(
        distractor_randomization,
        "scale_mean",
        opt_float(row, "synth_distractor_scale_mean"),
    )
    has_override |= set_if_present(
        distractor_randomization,
        "scale_std",
        opt_float(row, "synth_distractor_scale_std"),
    )
    if distractor_randomization:
        cfg["distractor_randomization"] = distractor_randomization

    distractors = {}
    has_override |= set_if_present(
        distractors,
        "clutter_level",
        opt_float(row, "synth_clutter_level"),
    )

    groups = {}
    for group_name in DISTRACTOR_GROUPS:
        group_cfg = {}
        has_group_value = False
        has_group_value |= set_if_present(
            group_cfg,
            "diversity",
            opt_int(row, f"synth_dist_{group_name}_diversity"),
        )
        has_group_value |= set_if_present(
            group_cfg,
            "occurrence",
            opt_int(row, f"synth_dist_{group_name}_occurrence"),
        )
        if has_group_value:
            groups[group_name] = group_cfg
            has_override = True
    if groups:
        distractors["groups"] = groups
    if distractors:
        cfg["distractors"] = distractors

    lighting = {}
    has_override |= set_if_present(
        lighting,
        "color_mean",
        complete_vector(
            opt_float(row, "synth_lighting_color_mean_r"),
            opt_float(row, "synth_lighting_color_mean_g"),
            opt_float(row, "synth_lighting_color_mean_b"),
        ),
    )
    has_override |= set_if_present(
        lighting,
        "color_std",
        complete_vector(
            opt_float(row, "synth_lighting_color_std_r"),
            opt_float(row, "synth_lighting_color_std_g"),
            opt_float(row, "synth_lighting_color_std_b"),
        ),
    )
    has_override |= set_if_present(
        lighting,
        "intensity_mean",
        opt_float(row, "synth_lighting_intensity_mean"),
    )
    has_override |= set_if_present(
        lighting,
        "intensity_std",
        opt_float(row, "synth_lighting_intensity_std"),
    )
    if lighting:
        cfg["lighting"] = lighting

    materials = {}
    has_override |= set_if_present(
        materials,
        "roughness_mean",
        opt_float(row, "synth_materials_roughness_mean"),
    )
    has_override |= set_if_present(
        materials,
        "roughness_std",
        opt_float(row, "synth_materials_roughness_std"),
    )
    has_override |= set_if_present(
        materials,
        "emissive_intensity_mean",
        opt_float(row, "synth_materials_emissive_intensity_mean"),
    )
    has_override |= set_if_present(
        materials,
        "emissive_intensity_std",
        opt_float(row, "synth_materials_emissive_intensity_std"),
    )
    textures = collect_textures(row)
    if textures:
        materials["textures"] = textures
        has_override = True
    if materials:
        cfg["materials"] = materials

    if not has_override:
        distribution_id = (row.get("distribution_id") or "<unknown>").strip() or "<unknown>"
        raise ValueError(
            f"Row {distribution_id} does not contain any usable "
            f"`{METADATA_PREFIX}synth_*` parameters."
        )

    return cfg


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate mean/std SDG YAML configs from a next-trials CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the next-trials CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Directory where YAML configs will be written. "
            "Created if it doesn't exist. "
            "Defaults to a directory named after the CSV file."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        candidates = [name for name in os.listdir(script_dir) if name.endswith(".csv")]
        if len(candidates) == 1:
            csv_path = os.path.join(script_dir, candidates[0])
        else:
            parser.error("--csv is required when there is not exactly one CSV next to this script.")
    csv_path = os.path.abspath(csv_path)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    sdg_config_abs = os.path.abspath(
        os.path.join(script_dir, "..", "sdg_config_mean_std.yaml")
    )
    extends_path = os.path.relpath(sdg_config_abs, out_dir)

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        distribution_id = (row.get("distribution_id") or "").strip()
        cfg = build_config(row, extends_path)
        out_path = os.path.join(out_dir, f"{distribution_id}.yaml")
        with open(out_path, "w") as handle:
            yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  wrote {distribution_id}.yaml")

    print(f"\nDone - {len(rows)} configs in {out_dir}")


if __name__ == "__main__":
    main()

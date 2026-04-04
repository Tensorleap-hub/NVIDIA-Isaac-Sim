# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
#  SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import io
import math
import random
import argparse
import yaml
import datetime
import numpy as np
from PIL import Image

# ── Argument parsing ──────────────────────────────────────────────────────────
# All run-level args default to None so we can detect explicit overrides.
parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "sdg_config.yaml"),
                    help="Path to YAML config file (default: sdg_config.yaml next to this script)")
parser.add_argument("--headless", type=lambda x: x.lower() == "true", default=None,
                    help="Override: launch headless")
parser.add_argument("--height", type=int, default=None, help="Override: image height")
parser.add_argument("--width", type=int, default=None, help="Override: image width")
parser.add_argument("--num_frames", type=int, default=None, help="Override: number of frames")
parser.add_argument("--environment", type=str, default=None,
                    help="Override: environment name (warehouse | full_warehouse | warehouse_multiple_shelves | warehouse_with_forklifts)")
parser.add_argument("--clutter_level", type=float, default=None,
                    help="Override: distractor clutter level multiplier (0 = no distractors)")
parser.add_argument("--data_dir", type=str, default=None, help="Override: output directory")

args, unknown_args = parser.parse_known_args()

# ── Load YAML config (with optional `extends` inheritance) ────────────────────
def _deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _load_cfg(config_path):
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    if "extends" not in raw_cfg:
        return raw_cfg

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(config_path)),
        raw_cfg.pop("extends"),
    )
    base_cfg = _load_cfg(base_path)
    return _deep_merge(base_cfg, raw_cfg)

CFG = _load_cfg(args.config)

# ── Merge CLI overrides (CLI wins over YAML) ──────────────────────────────────
if args.headless      is not None: CFG["run"]["headless"]                    = args.headless
if args.height        is not None: CFG["render"]["height"]                   = args.height
if args.width         is not None: CFG["render"]["width"]                    = args.width
if args.num_frames    is not None: CFG["run"]["num_frames"]                  = args.num_frames
if args.environment   is not None: CFG["environment"]["name"]                = args.environment
if args.clutter_level is not None: CFG["distractors"]["clutter_level"]       = args.clutter_level
if args.data_dir      is not None: CFG["run"]["data_dir"]                    = args.data_dir

# Backward compat: old experiment YAMLs set run.distractors = "None" / None / "additional"
# "None" / null  → no distractors; "warehouse" / "additional" → use default groups
_distractor_setting = CFG.get("run", {}).get("distractors")
if _distractor_setting in (None, "None", "none"):
    CFG["distractors"]["clutter_level"] = 0

# ── Launch simulation ─────────────────────────────────────────────────────────
from omni.isaac.kit import SimulationApp

LAUNCH_CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": CFG["run"]["headless"],
    "width":    CFG["render"]["width"],
    "height":   CFG["render"]["height"],
    "num_frames": CFG["run"]["num_frames"],
}

simulation_app = SimulationApp(launch_config=LAUNCH_CONFIG)

# ── Post-launch imports (must come after SimulationApp) ───────────────────────
import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep
from omni.isaac.core.utils.semantics import get_semantics

rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)


# ── Helpers ───────────────────────────────────────────────────────────────────
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path


def update_semantics(stage, keep_semantics=[]):
    """Remove semantics from the stage except for keep_semantics classes."""
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(property.GetPath())
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        continue
                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()
                    for semantic_class in keep_semantics:
                        if data_attr.Get() == semantic_class:
                            continue
                        else:
                            prim.RemoveProperty(type_attr.GetName())
                            prim.RemoveProperty(data_attr.GetName())
                            prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)


def full_textures_list():
    return [prefix_with_isaac_asset_server(p) for p in CFG["materials"]["textures"]]


def get_dataset_noise_cfg(cam_cfg):
    """Resolve per-image dataset-noise config from explicit or legacy fields."""
    ds_cfg = (cam_cfg.get("dataset_noise") or {}).copy()
    if ds_cfg:
        ds_cfg.setdefault("enabled", False)
        ds_cfg.setdefault("mode", "gaussian")
        ds_cfg.setdefault("sigma_min", 0.0)
        ds_cfg.setdefault("sigma_max", 0.0)
        ds_cfg.setdefault("jpeg_quality_min", 95)
        ds_cfg.setdefault("jpeg_quality_max", 100)
        ds_cfg.setdefault("seed", -1)
        if (
            ds_cfg.get("enabled")
            or float(ds_cfg.get("sigma_min", 0.0)) > 0.0
            or float(ds_cfg.get("sigma_max", 0.0)) > 0.0
        ):
            return ds_cfg

    legacy_min = cam_cfg.get("noise_std_min", 0.0)
    legacy_max = cam_cfg.get("noise_std_max", 0.0)
    return {
        "enabled": (legacy_min is not None and float(legacy_min) > 0.0)
        or (legacy_max is not None and float(legacy_max) > 0.0),
        "mode": "gaussian",
        "sigma_min": 0.0 if legacy_min is None else float(legacy_min),
        "sigma_max": 0.0 if legacy_max is None else float(legacy_max),
        "jpeg_quality_min": 95,
        "jpeg_quality_max": 100,
        "seed": -1,
    }


def resolve_image_augmentation_cfg(cam_cfg):
    """Resolve post-write image augmentation config, including legacy camera color."""
    aug_cfg = (CFG.get("image_augmentation") or {}).copy()
    aug_cfg.setdefault("enabled", False)
    aug_cfg.setdefault("brightness_gain_min", 1.0)
    aug_cfg.setdefault("brightness_gain_max", 1.0)
    aug_cfg.setdefault("contrast_gain_min", 1.0)
    aug_cfg.setdefault("contrast_gain_max", 1.0)
    aug_cfg.setdefault("gamma_min", 1.0)
    aug_cfg.setdefault("gamma_max", 1.0)

    if "color_gain_min" not in aug_cfg or "color_gain_max" not in aug_cfg:
        color_min = cam_cfg.get("color_min")
        color_max = cam_cfg.get("color_max")
        if color_min is not None and color_max is not None:
            aug_cfg.setdefault("color_gain_min", list(color_min))
            aug_cfg.setdefault("color_gain_max", list(color_max))
    aug_cfg.setdefault("color_gain_min", [1.0, 1.0, 1.0])
    aug_cfg.setdefault("color_gain_max", [1.0, 1.0, 1.0])

    neutral = (
        aug_cfg["brightness_gain_min"] == 1.0
        and aug_cfg["brightness_gain_max"] == 1.0
        and aug_cfg["contrast_gain_min"] == 1.0
        and aug_cfg["contrast_gain_max"] == 1.0
        and aug_cfg["gamma_min"] == 1.0
        and aug_cfg["gamma_max"] == 1.0
        and list(aug_cfg["color_gain_min"]) == [1.0, 1.0, 1.0]
        and list(aug_cfg["color_gain_max"]) == [1.0, 1.0, 1.0]
    )
    if not neutral:
        aug_cfg["enabled"] = True
    return aug_cfg


def sample_uniform_scalar(lo, hi):
    return random.uniform(float(lo), float(hi))


def sample_uniform_vector(lo, hi):
    return tuple(sample_uniform_scalar(a, b) for a, b in zip(lo, hi))


def sample_image_augmentation_params(aug_cfg):
    return {
        "brightness_gain": sample_uniform_scalar(
            aug_cfg["brightness_gain_min"], aug_cfg["brightness_gain_max"]
        ),
        "contrast_gain": sample_uniform_scalar(
            aug_cfg["contrast_gain_min"], aug_cfg["contrast_gain_max"]
        ),
        "gamma": sample_uniform_scalar(aug_cfg["gamma_min"], aug_cfg["gamma_max"]),
        "color_gain": sample_uniform_vector(
            aug_cfg["color_gain_min"], aug_cfg["color_gain_max"]
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

    sigma_min = max(0.0, float(noise_cfg.get("sigma_min", 0.0)))
    sigma_max = max(sigma_min, float(noise_cfg.get("sigma_max", sigma_min)))
    jpeg_quality_min = int(noise_cfg.get("jpeg_quality_min", 95))
    jpeg_quality_max = int(noise_cfg.get("jpeg_quality_max", 100))
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
            sigma = float(rng.uniform(sigma_min, sigma_max))
            if sigma > 0:
                data = np.clip(data + rng.normal(0.0, sigma, size=data.shape), 0.0, 255.0)
        if mode in ("jpeg", "gaussian_jpeg") and noise_cfg.get("enabled", False):
            quality = int(rng.integers(min(jpeg_quality_min, jpeg_quality_max), max(jpeg_quality_min, jpeg_quality_max) + 1))
            data = apply_jpeg_artifacts(np.clip(data, 0.0, 255.0), quality)
        Image.fromarray(np.clip(data, 0.0, 255.0).astype(np.uint8), mode="RGB").save(image_path)


def sample_camera_color_uniform(cam_cfg):
    color_min = cam_cfg.get("color_min")
    color_max = cam_cfg.get("color_max")
    if color_min is None or color_max is None:
        return None

    sampled = tuple(
        max(0.0, random.uniform(float(lo), float(hi)))
        for lo, hi in zip(color_min, color_max)
    )
    return sampled


def apply_camera_color_gain(camera_item, color_gain):
    if color_gain is None:
        return

    rep.modify.attribute(
        "omni:sensor:core:colorCorrectionWhiteBalance",
        color_gain,
        attribute_type="float3",
        input_prims=camera_item,
    )
    print(f"Camera color gain: {color_gain}")


def normalize_projection_type(camera_type):
    if camera_type == "fisheyeEquidistant":
        return "fisheyePolynomial"
    if camera_type == "fisheyePolynomial":
        return "fisheyePolynomial"
    return camera_type or "pinhole"


def is_fisheye_projection(camera_type):
    return normalize_projection_type(camera_type) != "pinhole"


def get_fisheye_max_fov(cam_cfg):
    if cam_cfg.get("fisheye_max_fov") is not None:
        return float(cam_cfg["fisheye_max_fov"])
    if cam_cfg.get("fov_max") is not None:
        return float(cam_cfg["fov_max"])
    if cam_cfg.get("fov_min") is not None:
        return float(cam_cfg["fov_min"])
    return 200.0


def add_palletjacks():
    pj_cfg = CFG["palletjacks"]
    rep_obj_list = [
        rep.create.from_usd(asset, semantics=[("class", "palletjack")], count=pj_cfg["count_per_model"])
        for asset in pj_cfg["assets"]
    ]
    return rep.create.group(rep_obj_list)


def add_distractors():
    """Spawn distractor groups according to diversity, occurrence, and clutter_level.

    For each group:
      - randomly pick `diversity` variants from the asset pool
      - spawn `max(1, round(occurrence × clutter_level))` instances of each variant
    Returns a rep group, or None if clutter_level is 0.
    """
    dist_cfg = CFG["distractors"]
    clutter = dist_cfg.get("clutter_level", 1.0)
    if clutter <= 0:
        print("clutter_level=0 — no distractors added")
        return None

    all_prims = []
    for group_name, group_cfg in dist_cfg.get("groups", {}).items():
        pool = group_cfg.get("assets", [])
        if not pool:
            continue
        diversity  = min(group_cfg.get("diversity", len(pool)), len(pool))
        count      = max(1, round(group_cfg.get("occurrence", 1) * clutter))
        selected   = random.sample(pool, diversity)
        for asset in selected:
            all_prims.append(
                rep.create.from_usd(prefix_with_isaac_asset_server(asset), count=count)
            )
        print(f"  {group_name}: {diversity} variant(s) × {count} instance(s)")

    if not all_prims:
        return None
    return rep.create.group(all_prims)


def run_orchestrator():
    rep.orchestrator.run()
    while not rep.orchestrator.get_is_started():
        simulation_app.update()
    while rep.orchestrator.get_is_started():
        simulation_app.update()
    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def write_run_config(output_dir):
    """Dump the resolved config + run metadata to output_dir/run_config.yaml."""
    import subprocess
    meta = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config_file": os.path.abspath(args.config),
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        meta["git_commit"] = git_hash
    except Exception:
        meta["git_commit"] = "unavailable"

    run_record = {"meta": meta, **CFG}
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "run_config.yaml")
    with open(out_path, "w") as f:
        yaml.dump(run_record, f, default_flow_style=False, sort_keys=False)
    print(f"Run config written to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env_name = CFG["environment"]["name"]
    env_rel  = CFG["environment_urls"][env_name]
    env_url  = prefix_with_isaac_asset_server(env_rel)
    print(f"Loading Stage {env_name} → {env_url}")
    open_stage(env_url)
    stage = get_current_stage()

    for i in range(100):
        if i % 10 == 0:
            print(f"App update {i}..")
        simulation_app.update()

    textures = full_textures_list()
    rep_palletjack_group = add_palletjacks()

    rep_distractor_group = add_distractors()

    update_semantics(stage=stage, keep_semantics=["palletjack"])

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_cfg = CFG["camera"]

    # --- Per-run post-processing values (sampled once) ---
    dataset_noise_cfg = get_dataset_noise_cfg(cam_cfg)
    motion_blur = random.uniform(
        cam_cfg["motion_blur_strength_min"], cam_cfg["motion_blur_strength_max"]
    )
    jpeg_quality = random.randint(cam_cfg["jpeg_quality_min"], cam_cfg["jpeg_quality_max"])

    settings = carb.settings.get_settings()
    if motion_blur > 0:
        settings.set("/rtx/post/motionblur/enabled", True)
        settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", float(motion_blur))
    # --- Focal length range (pinhole only) ---
    aperture = cam_cfg.get("horizontal_aperture") or 20.955
    projection_type = normalize_projection_type(cam_cfg.get("camera_type", "pinhole"))
    fl_min = cam_cfg.get("focal_length_min")
    fl_max = cam_cfg.get("focal_length_max")
    if not is_fisheye_projection(projection_type) and fl_min is None and cam_cfg.get("fov_min") is not None:
        # FOV (degrees) → focal length: fl = aperture / (2 * tan(fov/2))
        fl_min = aperture / (2 * math.tan(math.radians(cam_cfg["fov_max"]) / 2))
        fl_max = aperture / (2 * math.tan(math.radians(cam_cfg["fov_min"]) / 2))

    # --- Camera creation ---
    cam_kwargs = {
        "clipping_range": tuple(cam_cfg["clipping_range"]),
        "projection_type": projection_type,
    }
    if cam_cfg.get("focus_distance") is not None:
        cam_kwargs["focus_distance"] = cam_cfg["focus_distance"]
    if cam_cfg.get("f_stop") is not None:
        cam_kwargs["f_stop"] = cam_cfg["f_stop"]
    cam_kwargs["horizontal_aperture"] = aperture
    if is_fisheye_projection(projection_type):
        cam_kwargs["fisheye_max_fov"] = get_fisheye_max_fov(cam_cfg)
    cam = rep.create.camera(**cam_kwargs)

    # ── Replicator pipeline ───────────────────────────────────────────────────
    pj_cfg  = CFG["palletjacks"]
    dr_cfg  = CFG["distractor_randomization"]
    lt_cfg  = CFG["lighting"]
    mat_cfg = CFG["materials"]

    with rep.trigger.on_frame(num_frames=CFG["run"]["num_frames"]):

        with cam:
            pos_min = tuple(cam_cfg["position_min"]) + (cam_cfg["camera_height_min"],)
            pos_max = tuple(cam_cfg["position_max"]) + (cam_cfg["camera_height_max"],)
            pose_kwargs = {"position": rep.distribution.uniform(pos_min, pos_max)}
            if cam_cfg.get("camera_tilt_min") is not None:
                yaw_min  = cam_cfg.get("camera_yaw_min",  0.0)
                yaw_max  = cam_cfg.get("camera_yaw_max",  360.0)
                roll_min = cam_cfg.get("camera_roll_min", 0.0)
                roll_max = cam_cfg.get("camera_roll_max", 0.0)
                pose_kwargs["rotation"] = rep.distribution.uniform(
                    (cam_cfg["camera_tilt_min"], roll_min, yaw_min),
                    (cam_cfg["camera_tilt_max"], roll_max, yaw_max),
                )
            else:
                pose_kwargs["look_at"] = tuple(cam_cfg["look_at"])
            rep.modify.pose(**pose_kwargs)

            if not is_fisheye_projection(projection_type) and fl_min is not None:
                rep.modify.attribute("focalLength", rep.distribution.uniform(fl_min, fl_max))

        with rep.get.prims(path_pattern="SteerAxles"):
            rep.randomizer.color(
                colors=rep.distribution.uniform(
                    tuple(pj_cfg["color_min"]),
                    tuple(pj_cfg["color_max"]),
                )
            )

        with rep_palletjack_group:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    tuple(pj_cfg["position_min"]),
                    tuple(pj_cfg["position_max"]),
                ),
                rotation=rep.distribution.uniform(
                    tuple(pj_cfg["rotation_min"]),
                    tuple(pj_cfg["rotation_max"]),
                ),
                scale=rep.distribution.uniform(
                    tuple(pj_cfg["scale_min"]),
                    tuple(pj_cfg["scale_max"]),
                ),
            )

        if rep_distractor_group is not None:
            with rep_distractor_group:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        tuple(dr_cfg["position_min"]),
                        tuple(dr_cfg["position_max"]),
                    ),
                    rotation=rep.distribution.uniform(
                        tuple(dr_cfg["rotation_min"]),
                        tuple(dr_cfg["rotation_max"]),
                    ),
                    scale=rep.distribution.uniform(
                        dr_cfg["scale_min"],
                        dr_cfg["scale_max"],
                    ),
                )

        with rep.get.prims(path_pattern="RectLight"):
            rep.modify.attribute(
                "color",
                rep.distribution.uniform(tuple(lt_cfg["color_min"]), tuple(lt_cfg["color_max"])),
            )
            rep.modify.attribute(
                "intensity",
                rep.distribution.normal(lt_cfg["intensity_mean"], lt_cfg["intensity_std"]),
            )
            rep.modify.visibility(
                rep.distribution.choice(lt_cfg["visibility_choices"])
            )

        random_mat_floor = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(textures),
            roughness=rep.distribution.uniform(mat_cfg["roughness_min"], mat_cfg["roughness_max"]),
            metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep.distribution.uniform(
                mat_cfg["emissive_intensity_min"], mat_cfg["emissive_intensity_max"]
            ),
        )
        with rep.get.prims(path_pattern="SM_Floor"):
            rep.randomizer.materials(random_mat_floor)

        random_mat_wall = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(textures),
            roughness=rep.distribution.uniform(mat_cfg["roughness_min"], mat_cfg["roughness_max"]),
            metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep.distribution.uniform(
                mat_cfg["emissive_intensity_min"], mat_cfg["emissive_intensity_max"]
            ),
        )
        with rep.get.prims(path_pattern="SM_Wall"):
            rep.randomizer.materials(random_mat_wall)

    # ── Writer ────────────────────────────────────────────────────────────────
    output_directory = CFG["run"]["data_dir"]
    print("Outputting data to", output_directory)

    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=output_directory, omit_semantic_type=True)

    RESOLUTION = (CFG["render"]["width"], CFG["render"]["height"])
    render_product = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    run_orchestrator()
    simulation_app.update()
    apply_post_write_effects_to_saved_rgb(
        output_directory,
        dataset_noise_cfg,
        resolve_image_augmentation_cfg(cam_cfg),
    )

    # ── Dump resolved config alongside the data ───────────────────────────────
    write_run_config(output_directory)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

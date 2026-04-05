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
import math
import random
import argparse
import sys
import yaml
import datetime

# ── Argument parsing ──────────────────────────────────────────────────────────
# All run-level args default to None so we can detect explicit overrides.
parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "sdg_config_mean_std.yaml"),
                    help="Path to YAML config file (default: sdg_config_mean_std.yaml next to this script)")
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from palletjack_sdg.utils.camera import (
    fov_to_focal_length,
    get_fisheye_max_fov_mean_std,
    is_fisheye_projection,
    normalize_projection_type,
    rep_normal,
)
from palletjack_sdg.utils.image_effects import (
    apply_post_write_effects_to_saved_rgb,
    get_dataset_noise_cfg,
    resolve_image_augmentation_cfg,
    sample_normal,
)

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


def resolve_output_cfg():
    """Resolve writer/output modalities with defaults matching the old behavior."""
    output_cfg = (CFG.get("output") or {}).copy()
    output_cfg.setdefault("writer", "auto")
    output_cfg.setdefault("rgb", True)
    output_cfg.setdefault("od", True)
    output_cfg.setdefault("depth", False)
    output_cfg.setdefault("depth_annotator", "distance_to_camera")
    output_cfg.setdefault("semantic_segmentation", False)
    output_cfg.setdefault("instance_segmentation", False)
    return output_cfg


def initialize_writer(output_directory, output_cfg):
    """Create the requested writer and return (writer, rgb_enabled, writer_name)."""
    writer_mode = str(output_cfg.get("writer", "auto")).lower()
    wants_extra_modalities = any(
        (
            output_cfg.get("depth", False),
            output_cfg.get("semantic_segmentation", False),
            output_cfg.get("instance_segmentation", False),
        )
    )
    wants_basic_writer = (
        writer_mode == "basic"
        or wants_extra_modalities
        or not output_cfg.get("rgb", True)
        or not output_cfg.get("od", True)
    )

    if writer_mode not in {"auto", "kitti", "basic"}:
        raise ValueError(
            f"Unsupported output.writer={output_cfg['writer']!r}. Expected auto, kitti, or basic."
        )

    if writer_mode == "kitti" and (
        wants_extra_modalities
        or not output_cfg.get("rgb", True)
        or not output_cfg.get("od", True)
    ):
        raise ValueError(
            "KittiWriter only supports the default RGB+OD export here. "
            "Use output.writer=basic for custom modality combinations."
        )

    if writer_mode == "kitti" or (writer_mode == "auto" and not wants_basic_writer):
        writer = rep.WriterRegistry.get("KittiWriter")
        writer.initialize(output_dir=output_directory, omit_semantic_type=True)
        return writer, True, "KittiWriter"

    depth_annotator = str(output_cfg.get("depth_annotator", "distance_to_camera"))
    if depth_annotator not in {"distance_to_camera", "distance_to_image_plane"}:
        raise ValueError(
            "output.depth_annotator must be distance_to_camera or distance_to_image_plane"
        )

    writer = rep.WriterRegistry.get("BasicWriter")
    writer_kwargs = {
        "output_dir": output_directory,
        "rgb": bool(output_cfg.get("rgb", True)),
        "bounding_box_2d_tight": bool(output_cfg.get("od", True)),
        "semantic_segmentation": bool(output_cfg.get("semantic_segmentation", False)),
        "instance_segmentation": bool(output_cfg.get("instance_segmentation", False)),
    }
    if output_cfg.get("depth", False):
        writer_kwargs[depth_annotator] = True
    writer.initialize(**writer_kwargs)
    return writer, bool(output_cfg.get("rgb", True)), "BasicWriter"


def sample_camera_color_normal(cam_cfg):
    color_mean = cam_cfg.get("color_mean")
    color_std = cam_cfg.get("color_std")
    if color_mean is None or color_std is None:
        return None

    sampled = tuple(
        max(0.0, sample_normal(channel_mean, channel_std, lower=0.0))
        for channel_mean, channel_std in zip(color_mean, color_std)
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
    motion_blur = sample_normal(
        cam_cfg["motion_blur_strength_mean"],
        cam_cfg.get("motion_blur_strength_std", 0.0),
        lower=0.0,
    )
    jpeg_quality = sample_normal(
        cam_cfg["jpeg_quality_mean"],
        cam_cfg.get("jpeg_quality_std", 0.0),
        lower=0.0,
        upper=100.0,
        integer=True,
    )

    settings = carb.settings.get_settings()
    if motion_blur > 0:
        settings.set("/rtx/post/motionblur/enabled", True)
        settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", float(motion_blur))
    # --- Focal length distribution (pinhole only) ---
    aperture = cam_cfg.get("horizontal_aperture") or 20.955
    projection_type = normalize_projection_type(cam_cfg.get("camera_type", "pinhole"))
    fl_mean = cam_cfg.get("focal_length_mean")
    fl_std = cam_cfg.get("focal_length_std", 0.0)
    if not is_fisheye_projection(projection_type) and fl_mean is None and cam_cfg.get("fov_mean") is not None:
        fov_mean = cam_cfg["fov_mean"]
        fov_std = cam_cfg.get("fov_std", 0.0)
        fl_mean = fov_to_focal_length(aperture, fov_mean)
        if fov_std and fov_std > 0:
            fl_lo = fov_to_focal_length(aperture, max(1e-3, fov_mean + fov_std))
            fl_hi = fov_to_focal_length(aperture, max(1e-3, fov_mean - fov_std))
            fl_std = abs(fl_hi - fl_lo) / 2
        else:
            fl_std = 0.0

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
        cam_kwargs["fisheye_max_fov"] = get_fisheye_max_fov_mean_std(cam_cfg)
    cam = rep.create.camera(**cam_kwargs)

    # ── Replicator pipeline ───────────────────────────────────────────────────
    pj_cfg  = CFG["palletjacks"]
    dr_cfg  = CFG["distractor_randomization"]
    lt_cfg  = CFG["lighting"]
    mat_cfg = CFG["materials"]

    with rep.trigger.on_frame(num_frames=CFG["run"]["num_frames"]):

        with cam:
            pos_mean = tuple(cam_cfg["position_mean"]) + (cam_cfg["camera_height_mean"],)
            pos_std = tuple(cam_cfg.get("position_std", (0.0, 0.0))) + (
                cam_cfg.get("camera_height_std", 0.0),
            )
            pose_kwargs = {"position": rep_normal(pos_mean, pos_std)}
            if cam_cfg.get("camera_tilt_mean") is not None:
                yaw_mean = cam_cfg.get("camera_yaw_mean", 180.0)
                yaw_std = cam_cfg.get("camera_yaw_std", 360.0 / math.sqrt(12))
                roll_mean = cam_cfg.get("camera_roll_mean", 0.0)
                roll_std = cam_cfg.get("camera_roll_std", 0.0)
                pose_kwargs["rotation"] = rep_normal(
                    (cam_cfg["camera_tilt_mean"], roll_mean, yaw_mean),
                    (
                        cam_cfg.get("camera_tilt_std", 0.0),
                        roll_std,
                        yaw_std,
                    ),
                )
            else:
                pose_kwargs["look_at"] = tuple(cam_cfg["look_at"])
            rep.modify.pose(**pose_kwargs)

            if not is_fisheye_projection(projection_type) and fl_mean is not None:
                rep.modify.attribute("focalLength", rep_normal(fl_mean, fl_std))

        with rep.get.prims(path_pattern="SteerAxles"):
            rep.randomizer.color(
                colors=rep_normal(
                    tuple(pj_cfg["color_mean"]),
                    tuple(pj_cfg["color_std"]),
                )
            )

        with rep_palletjack_group:
            rep.modify.pose(
                position=rep_normal(
                    tuple(pj_cfg["position_mean"]),
                    tuple(pj_cfg["position_std"]),
                ),
                rotation=rep_normal(
                    tuple(pj_cfg["rotation_mean"]),
                    tuple(pj_cfg["rotation_std"]),
                ),
                scale=rep_normal(
                    tuple(pj_cfg["scale_mean"]),
                    tuple(pj_cfg["scale_std"]),
                ),
            )

        if rep_distractor_group is not None:
            with rep_distractor_group:
                rep.modify.pose(
                    position=rep_normal(
                        tuple(dr_cfg["position_mean"]),
                        tuple(dr_cfg["position_std"]),
                    ),
                    rotation=rep_normal(
                        tuple(dr_cfg["rotation_mean"]),
                        tuple(dr_cfg["rotation_std"]),
                    ),
                    scale=rep_normal(
                        dr_cfg["scale_mean"],
                        dr_cfg["scale_std"],
                    ),
                )

        with rep.get.prims(path_pattern="RectLight"):
            rep.modify.attribute(
                "color",
                rep_normal(tuple(lt_cfg["color_mean"]), tuple(lt_cfg["color_std"])),
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
            roughness=rep_normal(
                mat_cfg["roughness_mean"],
                mat_cfg["roughness_std"],
            ),
            metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep_normal(
                mat_cfg["emissive_intensity_mean"],
                mat_cfg["emissive_intensity_std"],
            ),
        )
        with rep.get.prims(path_pattern="SM_Floor"):
            rep.randomizer.materials(random_mat_floor)

        random_mat_wall = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(textures),
            roughness=rep_normal(
                mat_cfg["roughness_mean"],
                mat_cfg["roughness_std"],
            ),
            metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
            emissive_texture=rep.distribution.choice(textures),
            emissive_intensity=rep_normal(
                mat_cfg["emissive_intensity_mean"],
                mat_cfg["emissive_intensity_std"],
            ),
        )
        with rep.get.prims(path_pattern="SM_Wall"):
            rep.randomizer.materials(random_mat_wall)

    # ── Writer ────────────────────────────────────────────────────────────────
    output_directory = CFG["run"]["data_dir"]
    print("Outputting data to", output_directory)

    output_cfg = resolve_output_cfg()
    writer, rgb_output_enabled, writer_name = initialize_writer(output_directory, output_cfg)
    print("Using writer", writer_name, "with output config", output_cfg)

    RESOLUTION = (CFG["render"]["width"], CFG["render"]["height"])
    render_product = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    run_orchestrator()
    simulation_app.update()
    if rgb_output_enabled:
        apply_post_write_effects_to_saved_rgb(
            output_directory,
            dataset_noise_cfg,
            resolve_image_augmentation_cfg(CFG.get("image_augmentation"), cam_cfg),
        )
    else:
        print("RGB output disabled; skipping post-write RGB augmentation")

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

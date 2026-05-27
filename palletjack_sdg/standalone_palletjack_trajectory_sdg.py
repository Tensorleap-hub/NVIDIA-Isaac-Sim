"""Stage-2 trajectory SDG entry point.

Loads a warehouse environment once, randomizes scene objects (palletjacks,
forklifts, pallets, distractors, lighting, materials) once per episode, then
moves a ghost ego camera along a deterministic waypoint list and writes ordered
RGB frames plus per-frame pose metadata.

Scene randomization uses a single rep.trigger.on_frame(num_frames=1) block
fired during orchestrator.preview() so objects are placed before capture and
remain fixed for the entire trajectory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "sdg_config_trajectory.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Palletjack trajectory dataset generator")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--headless",
        type=lambda v: v.lower() == "true",
        default=None,
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_cfg(config_path: Path) -> dict[str, Any]:
    raw_cfg = yaml.safe_load(config_path.read_text())

    distractors_cfg = raw_cfg.get("distractors")
    if (
        isinstance(distractors_cfg, dict)
        and "groups" in distractors_cfg
        and distractors_cfg["groups"] is None
    ):
        distractors_cfg.pop("groups")

    if "extends" not in raw_cfg:
        return raw_cfg

    base_path = (config_path.parent / raw_cfg.pop("extends")).resolve()
    base_cfg = load_cfg(base_path)
    return deep_merge(base_cfg, raw_cfg)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("run", {})
    cfg.setdefault("render", {})
    cfg.setdefault("environment", {})

    if args.headless is not None:
        cfg["run"]["headless"] = args.headless
    if args.height is not None:
        cfg["render"]["height"] = args.height
    if args.width is not None:
        cfg["render"]["width"] = args.width
    if args.num_frames is not None:
        cfg["run"]["num_frames"] = args.num_frames
    if args.environment is not None:
        cfg["environment"]["name"] = args.environment
    if args.data_dir is not None:
        cfg["run"]["data_dir"] = args.data_dir
    if args.seed is not None:
        cfg.setdefault("simulation", {})
        cfg["simulation"]["seed"] = args.seed


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    data_dir = cfg.get("run", {}).get("data_dir")
    if data_dir is None:
        return SCRIPT_DIR / "palletjack_data" / "trajectory_stage2"
    return Path(data_dir).resolve()


def prepare_output_tree(output_dir: Path) -> dict[str, Path]:
    paths = {
        "output": output_dir,
        "rgb": output_dir / "Camera" / "rgb",
        "trajectory": output_dir / "trajectory",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unavailable"


def write_run_config(output_dir: Path, cfg: dict[str, Any], config_path: Path) -> Path:
    record = {
        "meta": {
            "timestamp": utc_timestamp(),
            "generator": "standalone_palletjack_trajectory_sdg.py",
            "generator_stage": "stage_2",
            "config_file": str(config_path.resolve()),
            "git_commit": git_commit(),
        },
        **cfg,
    }
    path = output_dir / "run_config.yaml"
    path.write_text(yaml.dump(record, default_flow_style=False, sort_keys=False))
    return path


def append_event(events_path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {"timestamp": utc_timestamp(), "event": event, **payload}
    with events_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_manifest(
    output_dir: Path,
    cfg: dict[str, Any],
    config_path: Path,
    environment_url: str | None,
    stage_loaded: bool,
    image_count: int = 0,
    pose_count: int = 0,
) -> Path:
    manifest = {
        "generator": "standalone_palletjack_trajectory_sdg.py",
        "generator_stage": "stage_2",
        "timestamp": utc_timestamp(),
        "config_file": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "headless": bool(cfg.get("run", {}).get("headless", True)),
        "num_frames": int(cfg.get("run", {}).get("num_frames", 0)),
        "environment": cfg.get("environment", {}).get("name"),
        "environment_url": environment_url,
        "stage_loaded": stage_loaded,
        "image_count": image_count,
        "trajectory_pose_count": pose_count,
        "events_path": "trajectory/events.jsonl",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def prefix_with_isaac_asset_server(relative_path: str) -> str:
    if (
        relative_path.startswith("http://")
        or relative_path.startswith("https://")
        or relative_path.startswith("omniverse://")
    ):
        return relative_path

    from omni.isaac.core.utils.nucleus import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path


def resolve_environment_url(cfg: dict[str, Any]) -> str:
    env_cfg = cfg.get("environment", {})
    if env_cfg.get("usd_path"):
        return prefix_with_isaac_asset_server(str(env_cfg["usd_path"]))

    env_name = env_cfg["name"]
    env_rel = cfg["environment_urls"][env_name]
    return prefix_with_isaac_asset_server(str(env_rel))


def launch_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "renderer": "RayTracedLighting",
        "headless": bool(cfg.get("run", {}).get("headless", True)),
        "width": int(cfg.get("render", {}).get("width", 960)),
        "height": int(cfg.get("render", {}).get("height", 544)),
    }


def _interpolate_waypoints(
    waypoints: list[list[float]], num_frames: int
) -> list[tuple[float, ...]]:
    """Return num_frames poses (x,y,z,roll,pitch,yaw,seg_idx) interpolated along arc length."""
    if num_frames == 1:
        wp = waypoints[0]
        return [(wp[0], wp[1], wp[2], wp[3], wp[4], wp[5], 0)]

    n_segs = len(waypoints) - 1
    seg_lengths = []
    for i in range(n_segs):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        dz = waypoints[i + 1][2] - waypoints[i][2]
        seg_lengths.append(math.sqrt(dx * dx + dy * dy + dz * dz))

    total_length = sum(seg_lengths)
    if total_length < 1e-9:
        wp = waypoints[0]
        return [(wp[0], wp[1], wp[2], wp[3], wp[4], wp[5], 0)] * num_frames

    cumulative = [0.0]
    for sl in seg_lengths:
        cumulative.append(cumulative[-1] + sl)

    result = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        arc = t * total_length

        seg = n_segs - 1
        for j in range(n_segs):
            if arc <= cumulative[j + 1]:
                seg = j
                break

        seg_len = seg_lengths[seg]
        local_t = 0.0 if seg_len < 1e-9 else (arc - cumulative[seg]) / seg_len
        local_t = max(0.0, min(1.0, local_t))

        wp0 = waypoints[seg]
        wp1 = waypoints[seg + 1]
        interp = [wp0[k] + local_t * (wp1[k] - wp0[k]) for k in range(6)]
        result.append(tuple(interp) + (seg,))

    return result


def run_stage2(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)
    apply_cli_overrides(cfg, args)

    seed = int(cfg.get("simulation", {}).get("seed", 0))
    random.seed(seed)

    output_dir = resolve_output_dir(cfg)
    paths = prepare_output_tree(output_dir)
    events_path = paths["trajectory"] / "events.jsonl"
    events_path.write_text("")

    write_run_config(output_dir, cfg, config_path)
    append_event(
        events_path,
        "stage2_output_tree_created",
        {"output_dir": str(output_dir.resolve()), "seed": seed},
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(launch_config=launch_config(cfg))

    import carb.settings
    import omni.replicator.core as rep
    import omni.usd
    from omni.isaac.core.utils.stage import open_stage
    from pxr import Gf, Semantics, UsdGeom
    from palletjack_sdg.utils.camera import rep_normal

    # ── Environment ───────────────────────────────────────────────────────────
    environment_url = resolve_environment_url(cfg)
    print(f"Loading environment: {environment_url}")
    open_stage(environment_url)
    simulation_app.update()

    append_event(
        events_path,
        "stage2_environment_loaded",
        {"environment": cfg["environment"]["name"], "environment_url": environment_url},
    )

    stage = omni.usd.get_context().get_stage()

    # ── Scene objects — spawn once ────────────────────────────────────────────
    def _add_palletjacks():
        pj_cfg = cfg.get("palletjacks", {})
        assets = pj_cfg.get("assets", [])
        count = pj_cfg.get("count_per_model", 0)
        if not assets or count == 0:
            return None
        groups = [
            rep.create.from_usd(asset, semantics=[("class", "palletjack")], count=count)
            for asset in assets
        ]
        return rep.create.group(groups)

    def _add_forklifts():
        fl_cfg = cfg.get("forklifts", {})
        assets = fl_cfg.get("assets", [])
        count = fl_cfg.get("count_per_model", 0)
        if not assets or count == 0:
            return None
        groups = [
            rep.create.from_usd(
                prefix_with_isaac_asset_server(asset),
                semantics=[("class", "forklift")],
                count=count,
            )
            for asset in assets
        ]
        return rep.create.group(groups)

    def _add_pallets():
        pa_cfg = cfg.get("pallets", {})
        assets = pa_cfg.get("assets", [])
        count = pa_cfg.get("count_per_model", 0)
        if not assets or count == 0:
            return None
        groups = [
            rep.create.from_usd(
                prefix_with_isaac_asset_server(asset),
                semantics=[("class", "pallet")],
                count=count,
            )
            for asset in assets
        ]
        return rep.create.group(groups)

    def _add_distractors():
        dist_cfg = cfg.get("distractors", {})
        clutter = dist_cfg.get("clutter_level", 1.0)
        if clutter <= 0:
            print("clutter_level=0 — no distractors")
            return None
        groups = dist_cfg.get("groups") or {}
        all_prims = []
        for group_name, group_cfg in groups.items():
            if not group_cfg:
                continue
            pool = group_cfg.get("assets", [])
            if not pool:
                continue
            diversity = min(group_cfg.get("diversity", len(pool)), len(pool))
            count = round(group_cfg.get("occurrence", 1) * clutter)
            if count == 0:
                continue
            selected = random.sample(pool, diversity)
            for asset in selected:
                all_prims.append(
                    rep.create.from_usd(prefix_with_isaac_asset_server(asset), count=count)
                )
            print(f"  distractor {group_name}: {diversity} variant(s) × {count}")
        return rep.create.group(all_prims) if all_prims else None

    def _update_semantics(keep=("palletjack", "forklift", "pallet")):
        for prim in stage.Traverse():
            if not prim.HasAPI(Semantics.SemanticsAPI):
                continue
            seen = set()
            for prop in prim.GetProperties():
                if not Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                    continue
                inst = prop.SplitName()[1]
                if inst in seen:
                    continue
                seen.add(inst)
                sem = Semantics.SemanticsAPI.Get(prim, inst)
                if sem.GetSemanticDataAttr().Get() not in keep:
                    prim.RemoveProperty(sem.GetSemanticTypeAttr().GetName())
                    prim.RemoveProperty(sem.GetSemanticDataAttr().GetName())
                    prim.RemoveAPI(Semantics.SemanticsAPI, inst)

    print("Spawning scene objects...")
    pj_group = _add_palletjacks()
    fl_group = _add_forklifts()
    pa_group = _add_pallets()
    dist_group = _add_distractors()
    _update_semantics()

    append_event(events_path, "stage2_scene_spawned", {
        "palletjacks": pj_group is not None,
        "forklifts": fl_group is not None,
        "pallets": pa_group is not None,
        "distractors": dist_group is not None,
    })

    # ── One-shot scene randomization via on_frame(num_frames=1) ──────────────
    # This fires once during orchestrator.preview() and is exhausted before
    # the trajectory capture loop begins, so the scene stays fixed per episode.
    pj_cfg = cfg.get("palletjacks", {})
    fl_cfg = cfg.get("forklifts", {})
    pa_cfg = cfg.get("pallets", {})
    dr_cfg = cfg.get("distractor_randomization", {})
    lt_cfg = cfg.get("lighting", {})
    mat_cfg = cfg.get("materials", {})
    textures = [prefix_with_isaac_asset_server(p) for p in mat_cfg.get("textures", [])]

    with rep.trigger.on_frame(num_frames=1):
        if pj_group is not None:
            with pj_group:
                rep.modify.pose(
                    position=rep_normal(tuple(pj_cfg["position_mean"]), tuple(pj_cfg["position_std"])),
                    rotation=rep_normal(tuple(pj_cfg["rotation_mean"]), tuple(pj_cfg["rotation_std"])),
                    scale=rep_normal(tuple(pj_cfg["scale_mean"]), tuple(pj_cfg["scale_std"])),
                )
        if fl_group is not None:
            with fl_group:
                rep.modify.pose(
                    position=rep_normal(tuple(fl_cfg["position_mean"]), tuple(fl_cfg["position_std"])),
                    rotation=rep_normal(tuple(fl_cfg["rotation_mean"]), tuple(fl_cfg["rotation_std"])),
                    scale=rep_normal(tuple(fl_cfg["scale_mean"]), tuple(fl_cfg["scale_std"])),
                )
        if pa_group is not None:
            with pa_group:
                rep.modify.pose(
                    position=rep_normal(tuple(pa_cfg["position_mean"]), tuple(pa_cfg["position_std"])),
                    rotation=rep_normal(tuple(pa_cfg["rotation_mean"]), tuple(pa_cfg["rotation_std"])),
                    scale=rep_normal(tuple(pa_cfg["scale_mean"]), tuple(pa_cfg["scale_std"])),
                )
        if dist_group is not None:
            with dist_group:
                rep.modify.pose(
                    position=rep_normal(tuple(dr_cfg["position_mean"]), tuple(dr_cfg["position_std"])),
                    rotation=rep_normal(tuple(dr_cfg["rotation_mean"]), tuple(dr_cfg["rotation_std"])),
                    scale=rep_normal(dr_cfg["scale_mean"], dr_cfg["scale_std"]),
                )
        if lt_cfg:
            with rep.get.prims(path_pattern="RectLight"):
                rep.modify.attribute(
                    "color",
                    rep_normal(tuple(lt_cfg["color_mean"]), tuple(lt_cfg["color_std"])),
                )
                rep.modify.attribute(
                    "intensity",
                    rep.distribution.normal(lt_cfg["intensity_mean"], lt_cfg["intensity_std"]),
                )
                rep.modify.visibility(rep.distribution.choice(lt_cfg["visibility_choices"]))
        if textures and mat_cfg:
            floor_mat = rep.create.material_omnipbr(
                diffuse_texture=rep.distribution.choice(textures),
                roughness=rep_normal(mat_cfg["roughness_mean"], mat_cfg["roughness_std"]),
                metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
                emissive_texture=rep.distribution.choice(textures),
                emissive_intensity=rep_normal(
                    mat_cfg["emissive_intensity_mean"], mat_cfg["emissive_intensity_std"]
                ),
            )
            with rep.get.prims(path_pattern="SM_Floor"):
                rep.randomizer.materials(floor_mat)
            wall_mat = rep.create.material_omnipbr(
                diffuse_texture=rep.distribution.choice(textures),
                roughness=rep_normal(mat_cfg["roughness_mean"], mat_cfg["roughness_std"]),
                metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
                emissive_texture=rep.distribution.choice(textures),
                emissive_intensity=rep_normal(
                    mat_cfg["emissive_intensity_mean"], mat_cfg["emissive_intensity_std"]
                ),
            )
            with rep.get.prims(path_pattern="SM_Wall"):
                rep.randomizer.materials(wall_mat)

    # ── Camera ────────────────────────────────────────────────────────────────
    cam_cfg = cfg.get("cameras", {}).get("ego", {})
    render_cfg = cfg.get("render", {})
    resolution_cfg = cam_cfg.get("resolution", [render_cfg.get("width", 960), render_cfg.get("height", 544)])
    width, height = int(resolution_cfg[0]), int(resolution_cfg[1])

    fov_deg = float(cam_cfg.get("fov_mean", cfg.get("camera", {}).get("fov_mean", 75.0)))
    horizontal_aperture = float(cfg.get("camera", {}).get("horizontal_aperture", 20.955))
    focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    clipping = cfg.get("camera", {}).get("clipping_range", [0.1, 1000000.0])

    # ── Trajectory ────────────────────────────────────────────────────────────
    traj_cfg = cfg.get("trajectory", {})
    waypoints_raw = traj_cfg.get("waypoints", [])
    if len(waypoints_raw) < 2:
        raise ValueError("trajectory.waypoints requires at least 2 entries")
    waypoints = [list(map(float, wp)) for wp in waypoints_raw]
    num_frames = int(cfg.get("run", {}).get("num_frames", 30))
    poses = _interpolate_waypoints(waypoints, num_frames)

    camera_prim_path = "/World/EgoCamera"
    camera_prim = stage.DefinePrim(camera_prim_path, "Camera")

    cam_schema = UsdGeom.Camera(camera_prim)
    cam_schema.GetHorizontalApertureAttr().Set(horizontal_aperture)
    cam_schema.GetFocalLengthAttr().Set(focal_length)
    cam_schema.GetClippingRangeAttr().Set(Gf.Vec2f(float(clipping[0]), float(clipping[1])))
    cam_schema.GetProjectionAttr().Set("perspective")

    xformable = UsdGeom.Xformable(camera_prim)
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()

    x0, y0, z0, r0, p0, yaw0_rel, _ = poses[0]
    seg0_dx = waypoints[1][0] - waypoints[0][0]
    seg0_dy = waypoints[1][1] - waypoints[0][1]
    yaw0 = math.degrees(math.atan2(-seg0_dx, seg0_dy)) + yaw0_rel
    translate_op.Set((x0, y0, z0))
    rotate_op.Set((r0, p0, yaw0))
    simulation_app.update()

    # ── Writer — images go to Camera/rgb/ ────────────────────────────────────
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

    rp = rep.create.render_product(camera_prim_path, (width, height))

    # on_frame(num_frames=1) fires on the first step(), not during preview().
    # Do one silent warmup step (no writer attached) to fire and exhaust the
    # trigger so the scene is fully placed before any frames are captured.
    rep.orchestrator.preview()
    simulation_app.update()

    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
    simulation_app.update()
    simulation_app.update()  # let materials/textures stream to GPU

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(output_dir / "Camera" / "rgb"), rgb=True)
    writer.attach(rp)

    append_event(events_path, "stage2_capture_start", {"num_frames": num_frames})

    poses_path = paths["trajectory"] / "poses.jsonl"
    poses_path.write_text("")

    physics_dt = float(cfg.get("simulation", {}).get("physics_dt", 1.0 / 60.0))

    for frame_index in range(num_frames):
        x, y, z, roll, pitch, yaw_rel, seg_idx = poses[frame_index]

        seg = int(seg_idx)
        dx = waypoints[seg + 1][0] - waypoints[seg][0]
        dy = waypoints[seg + 1][1] - waypoints[seg][1]
        heading_yaw = math.degrees(math.atan2(-dx, dy))
        yaw = heading_yaw + yaw_rel

        translate_op.Set((x, y, z))
        rotate_op.Set((roll, pitch, yaw))

        simulation_app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)

        pose_record = {
            "camera_prim": camera_prim_path,
            "camera_pos": [x, y, z],
            "camera_rot_euler_deg": [roll, pitch, yaw],
            "heading_yaw_deg": round(heading_yaw, 4),
            "relative_yaw_deg": round(yaw_rel, 4),
            "frame": frame_index,
            "sim_time": round(frame_index * physics_dt, 6),
            "waypoint_segment": seg,
        }
        with poses_path.open("a") as f:
            f.write(json.dumps(pose_record) + "\n")

        if frame_index % 10 == 0:
            print(f"Frame {frame_index + 1}/{num_frames}")

    rep.orchestrator.wait_until_complete()

    image_count = len(list(paths["rgb"].glob("*.png")))

    write_manifest(
        output_dir=output_dir,
        cfg=cfg,
        config_path=config_path,
        environment_url=environment_url,
        stage_loaded=True,
        image_count=image_count,
        pose_count=num_frames,
    )
    append_event(
        events_path,
        "stage2_complete",
        {"image_count": image_count, "num_frames": num_frames},
    )
    simulation_app.close()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring unknown arguments: {' '.join(unknown_args)}")
    run_stage2(args)


if __name__ == "__main__":
    main()

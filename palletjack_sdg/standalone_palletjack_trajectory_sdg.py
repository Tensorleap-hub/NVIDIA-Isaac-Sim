"""Stage-5 trajectory SDG entry point.

Loads a warehouse environment once, randomizes scene objects (palletjacks,
forklifts, pallets, distractors, lighting, materials) once per episode, then
moves a ghost ego camera along a waypoint path and writes ordered frames plus
per-frame pose metadata.

Stage-4: CosmosWriter video export (capture.video: true)
Stage-5: Occupancy-map path planning (trajectory.mode: occupancy_path)
  Generates a PhysX occupancy map of the settled scene, samples a random
  free-space start and BFS-plans a path to a random reachable end point.
  Saves nav/map.png, nav/map.yaml, nav/planned_path.json each run.
  Falls back to trajectory.waypoints when mode: waypoint_list.
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
        return SCRIPT_DIR / "palletjack_data" / "trajectory_stage5"
    return Path(data_dir).resolve()


def prepare_output_tree(output_dir: Path, chase_enabled: bool = False) -> dict[str, Path]:
    paths = {
        "output": output_dir,
        "ego": output_dir / "Camera",
        "trajectory": output_dir / "trajectory",
    }
    if chase_enabled:
        paths["chase"] = output_dir / "Camera_chase"
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
            "generator_stage": "stage_5",
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
    episode_camera: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "generator": "standalone_palletjack_trajectory_sdg.py",
        "generator_stage": "stage_5",
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
    if episode_camera:
        manifest["episode_camera"] = episode_camera
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


def _build_occupancy_waypoints(
    cfg: dict[str, Any],
    simulation_app,
    nav_dir: Path,
    camera_z: float,
    seed: int,
) -> list[list[float]]:
    """Generate random free-space waypoints via a PhysX occupancy map.

    Called after the warmup step so that placed scene objects are baked into
    the collision geometry before the map is generated.  Returns waypoints in
    the same format as trajectory.waypoints: [x, y, z, roll, pitch, yaw_rel].
    """
    import numpy as np
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Sdf, UsdGeom, UsdPhysics

    enable_extension("isaacsim.asset.gen.omap")
    enable_extension("isaacsim.replicator.mobility_gen")
    simulation_app.update()

    from isaacsim.asset.gen.omap.bindings import _omap
    from isaacsim.replicator.mobility_gen.impl.occupancy_map import OccupancyMap, OccupancyMapDataValue
    from isaacsim.replicator.mobility_gen.impl.path_planner import compress_path, generate_paths
    from isaacsim.replicator.mobility_gen.impl.pose_samplers import UniformPoseSampler

    # Ensure a PhysicsScene exists — without one the omap sees nothing.
    # The warehouse USD often embeds its own scene; this only creates one when absent.
    import carb
    stage = omni.usd.get_context().get_stage()
    existing_scenes = [p for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"]
    if not existing_scenes:
        from pxr import PhysxSchema
        physics_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        physx_scene_api.GetEnableGPUDynamicsAttr().Set(False)
        physx_scene_api.GetBroadphaseTypeAttr().Set("MBP")
        print("  OMap: created PhysicsScene (CPU/MBP)")
    else:
        print(f"  OMap: using existing PhysicsScene at {existing_scenes[0].GetPath()}")

    # Keep PhysX debug visualization off — loading omni.physx can enable it.
    carb.settings.get_settings().set("/physics/visualizationMode", 0)

    for _ in range(2):
        simulation_app.update()

    traj_cfg = cfg.get("trajectory", {})
    occ_cfg = traj_cfg.get("occupancy", {})
    bounds_xy = traj_cfg.get("bounds_xy", [-6.0, 6.0, -6.0, 8.0])
    x_min, x_max, y_min, y_max = [float(v) for v in bounds_xy]

    cell_size = float(occ_cfg.get("cell_size_m", 0.1))
    z_slice = float(occ_cfg.get("z_slice_m", 1.0))
    buffer_m = float(occ_cfg.get("buffer_m", 0.5))
    min_path_m = float(occ_cfg.get("min_path_m", 3.0))
    max_retries = int(occ_cfg.get("max_retries", 20))

    np.random.seed(seed)

    print(f"Occupancy map: bounds=({x_min},{y_min})→({x_max},{y_max}) z={z_slice}m cell={cell_size}m")
    om_iface = _omap.acquire_omap_interface()
    om_iface.set_cell_size(cell_size)
    om_iface.set_transform(
        (0.0, 0.0, z_slice),
        (x_min, y_min, 0.0),
        (x_max, y_max, 0.0),
    )
    om_iface.update()
    simulation_app.update()
    om_iface.generate()
    simulation_app.update()

    dims = om_iface.get_dimensions()   # [width_px, height_px]
    min_b = om_iface.get_min_bound()   # [x_min, y_min, z_min] world coords
    raw_buf = list(om_iface.get_buffer())

    # The omap extension loads isaacsim.util.debug_draw and draws the occupancy
    # grid as 3D lines that appear in camera renders.  Clear them immediately so
    # they don't contaminate the captured frames.
    try:
        from isaacsim.util.debug_draw import _debug_draw
        dd = _debug_draw.acquire_debug_draw_interface()
        dd.clear_lines()
        dd.clear_points()
        print("  OMap: debug draw cleared")
    except Exception as _e:
        print(f"  OMap: debug draw clear skipped ({_e})")

    # dims[0]=width(X cols), dims[1]=height(Y rows); buffer is row-major
    width_px = int(dims[0])
    height_px = int(dims[1])
    print(f"  Grid: {width_px}×{height_px} px, origin=({min_b[0]:.2f},{min_b[1]:.2f})")

    buf_arr = np.array(raw_buf, dtype=np.float32)
    data = buf_arr.reshape(height_px, width_px)

    # _omap: 0.0→freespace, 1.0→occupied, else→unknown
    occ_data = np.full((height_px, width_px), OccupancyMapDataValue.UNKNOWN, dtype=np.uint8)
    occ_data[data == 0.0] = OccupancyMapDataValue.FREESPACE
    occ_data[data == 1.0] = OccupancyMapDataValue.OCCUPIED

    # Mark the outer boundary cells as occupied to create a virtual safety margin
    # at the scan edges.  Warehouse walls often lack physics collision so the omap
    # sees them as freespace — without this, paths can run straight into geometry.
    boundary_margin_m = float(occ_cfg.get("boundary_margin_m", 2.0))
    boundary_px = max(1, int(boundary_margin_m / cell_size))
    occ_data[:boundary_px, :] = OccupancyMapDataValue.OCCUPIED   # south
    occ_data[-boundary_px:, :] = OccupancyMapDataValue.OCCUPIED  # north
    occ_data[:, :boundary_px] = OccupancyMapDataValue.OCCUPIED   # west
    occ_data[:, -boundary_px:] = OccupancyMapDataValue.OCCUPIED  # east
    print(f"  Boundary margin: {boundary_margin_m}m ({boundary_px}px) marked occupied")

    free_px = int(np.sum(occ_data == OccupancyMapDataValue.FREESPACE))
    print(f"  Freespace: {free_px}/{width_px * height_px} px ({100*free_px/(width_px*height_px):.1f}%)")

    origin = (float(min_b[0]), float(min_b[1]), 0.0)
    omap = OccupancyMap(data=occ_data, resolution=cell_size, origin=origin)
    omap_buffered = omap.buffered_meters(buffer_m)

    nav_dir.mkdir(parents=True, exist_ok=True)
    omap.save_ros(str(nav_dir))

    sampler = UniformPoseSampler()
    for attempt in range(max_retries):
        start_pose = sampler.sample(omap_buffered)
        start_px = omap.world_to_pixel_numpy(np.array([[start_pose.x, start_pose.y]]))
        # generate_paths expects (row, col) = (y_px, x_px)
        start_ij = (int(start_px[0, 1]), int(start_px[0, 0]))

        freespace = omap_buffered.freespace_mask()
        result = generate_paths(start_ij, freespace)

        valid = result.get_valid_end_points()
        if len(valid[0]) < 2:
            print(f"  Attempt {attempt+1}: no reachable ends, retry")
            continue

        end_ij = result.sample_random_end_point()
        path_ij = result.unroll_path(end_ij)          # (N,2) [row, col]
        path_ij, _ = compress_path(path_ij)
        path_xy_px = path_ij[:, ::-1]                 # → [col, row] = x_px, y_px
        path_world = omap.pixel_to_world_numpy(path_xy_px)  # (N,2) world [x,y]

        diffs = np.diff(path_world, axis=0)
        path_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
        if path_length < min_path_m:
            print(f"  Attempt {attempt+1}: path {path_length:.1f}m < {min_path_m}m, retry")
            continue

        waypoints = [[float(p[0]), float(p[1]), camera_z, 90.0, 0.0, 0.0] for p in path_world]
        print(f"  Path found: {len(waypoints)} pts, {path_length:.1f}m (attempt {attempt+1})")

        path_record = {
            "attempt": attempt + 1,
            "path_length_m": round(path_length, 3),
            "num_waypoints": len(waypoints),
            "start_world_xy": [round(float(start_pose.x), 4), round(float(start_pose.y), 4)],
            "end_world_xy": [round(float(path_world[-1, 0]), 4), round(float(path_world[-1, 1]), 4)],
            "waypoints_world_xy": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in path_world],
        }
        (nav_dir / "planned_path.json").write_text(json.dumps(path_record, indent=2))
        # Release the omap interface so its C++ render callbacks are unregistered,
        # preventing debug lines from being redrawn in captured frames.
        _omap.release_omap_interface(om_iface)
        return waypoints

    raise RuntimeError(f"No valid occupancy path found after {max_retries} attempts")


def run_stage4(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)
    apply_cli_overrides(cfg, args)

    seed = int(cfg.get("simulation", {}).get("seed", 0))
    random.seed(seed)

    cameras_cfg = cfg.get("cameras", {})
    capture_cfg = cfg.get("capture", {})
    chase_cfg = cameras_cfg.get("chase", {})
    chase_enabled = bool(chase_cfg.get("enabled", False))

    output_dir = resolve_output_dir(cfg)
    paths = prepare_output_tree(output_dir, chase_enabled=chase_enabled)
    events_path = paths["trajectory"] / "events.jsonl"
    events_path.write_text("")

    write_run_config(output_dir, cfg, config_path)
    append_event(
        events_path,
        "stage5_output_tree_created",
        {"output_dir": str(output_dir.resolve()), "seed": seed},
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(launch_config=launch_config(cfg))

    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from omni.isaac.core.utils.stage import open_stage
    from pxr import Gf, Semantics, UsdGeom
    from palletjack_sdg.utils.camera import rep_normal

    # CosmosWriter uses OmniGraph script nodes (Canny edge annotator) — without
    # this, the annotator chain silently fails to attach and write() is never called.
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
    # DLSS quality mode recommended by all CosmosWriter reference examples.
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # ── Environment ───────────────────────────────────────────────────────────
    environment_url = resolve_environment_url(cfg)
    print(f"Loading environment: {environment_url}")
    open_stage(environment_url)
    simulation_app.update()

    append_event(
        events_path,
        "stage5_environment_loaded",
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

    append_event(events_path, "stage5_scene_spawned", {
        "palletjacks": pj_group is not None,
        "forklifts": fl_group is not None,
        "pallets": pa_group is not None,
        "distractors": dist_group is not None,
    })

    # ── One-shot scene randomization via on_frame(num_frames=1) ──────────────
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

    # ── Camera parameters — sampled once per episode ──────────────────────────
    ego_cam_cfg = cameras_cfg.get("ego", {})
    legacy_cam = cfg.get("camera", {})
    render_cfg = cfg.get("render", {})

    resolution_cfg = ego_cam_cfg.get(
        "resolution", [render_cfg.get("width", 960), render_cfg.get("height", 544)]
    )
    width, height = int(resolution_cfg[0]), int(resolution_cfg[1])

    fov_mean = float(ego_cam_cfg.get("fov_mean", legacy_cam.get("fov_mean", 75.0)))
    fov_std = float(ego_cam_cfg.get("fov_std", legacy_cam.get("fov_std", 0.0)))
    fov_deg = random.gauss(fov_mean, fov_std) if fov_std > 0 else fov_mean
    fov_deg = max(10.0, min(170.0, fov_deg))

    horizontal_aperture = float(legacy_cam.get("horizontal_aperture", 20.955))
    focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    clipping = legacy_cam.get("clipping_range", [0.1, 1000000.0])

    print(f"Episode camera: fov={fov_deg:.1f}° focal_length={focal_length:.3f}mm")
    append_event(events_path, "stage5_camera_sampled", {
        "fov_deg": round(fov_deg, 3),
        "focal_length_mm": round(focal_length, 4),
        "resolution": [width, height],
    })

    # ── Trajectory ────────────────────────────────────────────────────────────
    traj_cfg = cfg.get("trajectory", {})
    traj_mode = traj_cfg.get("mode", "waypoint_list")
    num_frames = int(cfg.get("run", {}).get("num_frames", 30))
    camera_z = float(cameras_cfg.get("ego", {}).get("height_m", 1.6))

    if traj_mode == "waypoint_list":
        waypoints_raw = traj_cfg.get("waypoints", [])
        if len(waypoints_raw) < 2:
            raise ValueError("trajectory.waypoints requires at least 2 entries")
        waypoints = [list(map(float, wp)) for wp in waypoints_raw]
        poses = _interpolate_waypoints(waypoints, num_frames)
        x0, y0, z0, r0, p0, yaw0_rel, _ = poses[0]
        seg0_dx = waypoints[1][0] - waypoints[0][0]
        seg0_dy = waypoints[1][1] - waypoints[0][1]
        init_pos = (x0, y0, z0)
        init_rot = (r0, p0, math.degrees(math.atan2(-seg0_dx, seg0_dy)) + yaw0_rel)
    else:
        # occupancy_path: placeholder position for warmup; real path computed
        # after timeline.play() when the settled scene is visible to the omap.
        waypoints = None
        poses = None
        init_pos = (0.0, 0.0, camera_z)
        init_rot = (90.0, 0.0, 0.0)

    # ── Ego camera prim ───────────────────────────────────────────────────────
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

    translate_op.Set(init_pos)
    rotate_op.Set(init_rot)

    # ── Chase camera prim (optional) ──────────────────────────────────────────
    chase_camera_prim_path = None
    chase_translate_op = None
    chase_rotate_op = None
    chase_dist_m = float(chase_cfg.get("distance_m", 3.0))
    chase_height_m = float(chase_cfg.get("height_m", 2.0))

    if chase_enabled:
        chase_camera_prim_path = "/World/ChaseCamera"
        chase_prim = stage.DefinePrim(chase_camera_prim_path, "Camera")
        chase_schema = UsdGeom.Camera(chase_prim)
        chase_schema.GetHorizontalApertureAttr().Set(horizontal_aperture)
        chase_schema.GetFocalLengthAttr().Set(focal_length)
        chase_schema.GetClippingRangeAttr().Set(Gf.Vec2f(float(clipping[0]), float(clipping[1])))
        chase_schema.GetProjectionAttr().Set("perspective")
        chase_xformable = UsdGeom.Xformable(chase_prim)
        chase_translate_op = chase_xformable.AddTranslateOp()
        chase_rotate_op = chase_xformable.AddRotateXYZOp()
        # tilt_down_deg: geometric angle to look from chase position toward ego
        # roll=90 points forward; adding tilt_down_deg tilts nose down toward ego
        tilt_down_deg = -math.degrees(math.atan2(chase_height_m, chase_dist_m))
        print(f"Chase camera: dist={chase_dist_m}m above={chase_height_m}m tilt={tilt_down_deg:.1f}°")

    simulation_app.update()

    # ── Writer setup ──────────────────────────────────────────────────────────
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

    rp_ego = rep.create.render_product(camera_prim_path, (width, height))

    # Warmup: fires the on_frame(num_frames=1) trigger before writer is attached
    # so the scene settles (objects placed, textures loaded) before capture.
    rep.orchestrator.preview()
    simulation_app.update()
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
    simulation_app.update()
    simulation_app.update()

    ego_root = paths["ego"]
    if bool(capture_cfg.get("rgb", True)):
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=str(ego_root / "rgb"), rgb=True)
        w.attach(rp_ego)
    if bool(capture_cfg.get("bounding_box_2d_tight", False)):
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=str(ego_root / "bounding_box_2d_tight"), bounding_box_2d_tight=True)
        w.attach(rp_ego)
    if bool(capture_cfg.get("semantic_segmentation", False)):
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=str(ego_root / "semantic_segmentation"), semantic_segmentation=True)
        w.attach(rp_ego)
    if bool(capture_cfg.get("depth", False)):
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=str(ego_root / "depth"), distance_to_camera=True)
        w.attach(rp_ego)

    # ── CosmosWriter (optional video) ─────────────────────────────────────────
    # Runs alongside BasicWriter on the same render product. One trajectory =
    # one clip. Produces clip_0000/{rgb,depth,segmentation,shaded_seg,edges}.mp4
    # under video/ after on_final_frame() is called.
    cosmos_writer = None
    if bool(capture_cfg.get("video", False)):
        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
        cosmos_writer.initialize(output_dir=str(video_dir), use_instance_id=True)
        cosmos_writer.attach(rp_ego)
        print(f"CosmosWriter attached: video → {video_dir}")

    rp_chase = None
    if chase_enabled:
        chase_res = chase_cfg.get("resolution", [width, height])
        rp_chase = rep.create.render_product(
            chase_camera_prim_path, (int(chase_res[0]), int(chase_res[1]))
        )
        chase_writer = rep.WriterRegistry.get("BasicWriter")
        chase_writer.initialize(output_dir=str(paths["chase"]), rgb=True)
        chase_writer.attach(rp_chase)

    append_event(events_path, "stage5_capture_start", {
        "num_frames": num_frames,
        "modalities": {
            "rgb": bool(capture_cfg.get("rgb", True)),
            "bounding_box_2d_tight": bool(capture_cfg.get("bounding_box_2d_tight", False)),
            "semantic_segmentation": bool(capture_cfg.get("semantic_segmentation", False)),
            "depth": bool(capture_cfg.get("depth", False)),
            "video": bool(capture_cfg.get("video", False)),
        },
        "chase_enabled": chase_enabled,
    })

    poses_path = paths["trajectory"] / "poses.jsonl"
    poses_path.write_text("")

    physics_dt = float(cfg.get("simulation", {}).get("physics_dt", 1.0 / 60.0))

    # CosmosWriter requires the timeline to be playing and pause_timeline=False.
    # The ghost camera is driven by explicit USD ops, so physics running is harmless.
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    # ── Occupancy path (computed here so the settled scene is in collision) ──
    if traj_mode == "occupancy_path":
        nav_dir = output_dir / "nav"
        waypoints = _build_occupancy_waypoints(cfg, simulation_app, nav_dir, camera_z, seed)
        poses = _interpolate_waypoints(waypoints, num_frames)
        append_event(events_path, "stage5_occupancy_path_sampled", {
            "num_waypoints": len(waypoints),
            "nav_dir": str(nav_dir),
        })
        # Belt-and-suspenders: clear any debug draw that may have been redrawn since
        # _build_occupancy_waypoints() cleared it.  Must happen before first frame.
        try:
            from isaacsim.util.debug_draw import _debug_draw
            _debug_draw.acquire_debug_draw_interface().clear_lines()
            _debug_draw.acquire_debug_draw_interface().clear_points()
        except Exception:
            pass

    for frame_index in range(num_frames):
        x, y, z, roll, pitch, yaw_rel, seg_idx = poses[frame_index]

        seg = int(seg_idx)
        dx = waypoints[seg + 1][0] - waypoints[seg][0]
        dy = waypoints[seg + 1][1] - waypoints[seg][1]
        heading_yaw = math.degrees(math.atan2(-dx, dy))
        yaw = heading_yaw + yaw_rel

        translate_op.Set((x, y, z))
        rotate_op.Set((roll, pitch, yaw))

        if chase_enabled and chase_translate_op is not None:
            seg_len = math.sqrt(dx * dx + dy * dy)
            ux, uy = (dx / seg_len, dy / seg_len) if seg_len > 1e-9 else (0.0, 1.0)
            cx = x - ux * chase_dist_m
            cy = y - uy * chase_dist_m
            cz = z + chase_height_m
            chase_translate_op.Set((cx, cy, cz))
            # roll=90+tilt_down makes the camera look at the ego from behind and above
            # yaw tracks ego heading only — not ego's lateral look offset (yaw_rel)
            chase_rotate_op.Set((90.0 + tilt_down_deg, 0.0, heading_yaw))

        simulation_app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)

        pose_record = {
            "camera_prim": camera_prim_path,
            "camera_pos": [x, y, z],
            "camera_rot_euler_deg": [roll, pitch, yaw],
            "heading_yaw_deg": round(heading_yaw, 4),
            "relative_yaw_deg": round(yaw_rel, 4),
            "fov_deg": round(fov_deg, 3),
            "frame": frame_index,
            "sim_time": round(frame_index * physics_dt, 6),
            "waypoint_segment": seg,
        }
        with poses_path.open("a") as f:
            f.write(json.dumps(pose_record) + "\n")

        if frame_index % 10 == 0:
            print(f"Frame {frame_index + 1}/{num_frames}")

    rep.orchestrator.wait_until_complete()

    # Finalise CosmosWriter: stitches per-frame PNGs into MP4s for each modality.
    video_files: list[str] = []
    if cosmos_writer is not None:
        cosmos_writer.on_final_frame()
        cosmos_writer.detach()
        video_files = sorted(str(p.relative_to(output_dir)) for p in (output_dir / "video").rglob("*.mp4"))
        print(f"Video files written: {video_files}")

    image_count = len(list((paths["ego"] / "rgb").glob("*.png")))

    write_manifest(
        output_dir=output_dir,
        cfg=cfg,
        config_path=config_path,
        environment_url=environment_url,
        stage_loaded=True,
        image_count=image_count,
        pose_count=num_frames,
        episode_camera={
            "fov_deg": round(fov_deg, 3),
            "focal_length_mm": round(focal_length, 4),
            "resolution": [width, height],
            "chase_enabled": chase_enabled,
            "video_files": video_files,
        },
    )
    append_event(
        events_path,
        "stage5_complete",
        {"image_count": image_count, "num_frames": num_frames, "video_files": video_files},
    )
    simulation_app.close()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring unknown arguments: {' '.join(unknown_args)}")
    run_stage4(args)


if __name__ == "__main__":
    main()

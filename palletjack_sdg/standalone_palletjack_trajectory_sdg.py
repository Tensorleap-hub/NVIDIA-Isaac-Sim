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
    parser.add_argument("--seeds", type=str, default=None,
                        help="Space-separated seed list rendered in ONE Isaac session "
                             "(episode mode). Requires --out_root; each episode writes "
                             "<out_root>/<config-stem>_seed<S>/ like the per-process wrapper.")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Output root for --seeds episode mode")
    parser.add_argument("--max_seed_retries", type=int, default=4,
                        help="Episode mode: retries per seed (seed+k*1000) on layout failure")
    parser.add_argument("--capture_mode", type=str, default=None,
                        choices=["trajectory", "random"],
                        help="random = every frame an independent freespace pose "
                             "(disconnected snapshots, no path traversal)")
    parser.add_argument("--capture_dt", type=float, default=None,
                        help="Override agent.capture_dt (sim seconds per frame)")
    parser.add_argument("--no_video", action="store_true",
                        help="Force capture.video=false for training-profile sweeps")
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
    if args.capture_dt is not None:
        cfg.setdefault("agent", {})
        cfg["agent"]["capture_dt"] = args.capture_dt
    if args.capture_mode is not None:
        cfg["run"]["capture_mode"] = args.capture_mode
    if args.no_video:
        cfg.setdefault("capture", {})
        cfg["capture"]["video"] = False


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


def _generate_character_commands(
    characters: list[dict],
    spawn_bounds_xy: list[float],
    movement_cfg: dict[str, Any],
    rng: Any,
) -> list[str]:
    """Build lines for omni.anim.people's command file.

    Each character emits a random sequence of Idle / LookAround / GoTo commands
    inside spawn_bounds_xy. Without navmesh, GoTo walks straight; keep targets
    modest so characters don't clip walls.
    """
    commands_per_char = int(movement_cfg.get("commands_per_char", 3))
    idle_prob = float(movement_cfg.get("idle_prob", 0.4))
    lookaround_prob = float(movement_cfg.get("lookaround_prob", 0.2))
    goto_prob = float(movement_cfg.get("goto_prob", 0.4))
    idle_lo, idle_hi = movement_cfg.get("idle_duration_range", [3.0, 8.0])
    la_lo, la_hi = movement_cfg.get("lookaround_duration_range", [5.0, 10.0])
    goto_max_dist = float(movement_cfg.get("goto_max_dist_m", 3.0))
    xmin, xmax, ymin, ymax = spawn_bounds_xy
    total = idle_prob + lookaround_prob + goto_prob
    if total <= 0:
        return []
    weights = [idle_prob / total, lookaround_prob / total, goto_prob / total]
    kinds = ["Idle", "LookAround", "GoTo"]
    lines: list[str] = []
    for entry in characters:
        name = entry["prim"].rsplit("/", 1)[-1]
        cx, cy = entry["pos"][0], entry["pos"][1]
        # Start each sequence with a short idle so the anim graph settles.
        lines.append(f"{name} Idle {round(rng.uniform(idle_lo, idle_hi), 1)}")
        for _ in range(commands_per_char):
            r = rng.random()
            if r < weights[0]:
                lines.append(f"{name} Idle {round(rng.uniform(idle_lo, idle_hi), 1)}")
            elif r < weights[0] + weights[1]:
                lines.append(f"{name} LookAround {round(rng.uniform(la_lo, la_hi), 1)}")
            else:
                # Pick a target within goto_max_dist of current position, clamped
                # to spawn bounds. yaw = heading toward target.
                theta = rng.uniform(0.0, 2.0 * math.pi)
                dist = rng.uniform(0.5, goto_max_dist)
                tx = max(xmin, min(xmax, cx + dist * math.cos(theta)))
                ty = max(ymin, min(ymax, cy + dist * math.sin(theta)))
                yaw_deg = round(math.degrees(math.atan2(ty - cy, tx - cx)), 1)
                lines.append(
                    f"{name} GoTo {round(tx, 3)} {round(ty, 3)} 0.0 {yaw_deg}"
                )
                cx, cy = tx, ty  # walk continues from new position
    return lines


def _compressed_path_is_clear(path_ij: "np.ndarray", freespace: "np.ndarray") -> bool:
    """Return True if every straight-line segment in the compressed path stays in freespace.

    compress_path() is purely geometric — it never checks that shortcuts stay within
    the buffered freespace mask.  Walk each segment via Bresenham's line and reject
    paths that clip any occupied cell.
    """
    import numpy as _np
    rows, cols = freespace.shape
    for i in range(len(path_ij) - 1):
        r0, c0 = int(path_ij[i, 0]), int(path_ij[i, 1])
        r1, c1 = int(path_ij[i + 1, 0]), int(path_ij[i + 1, 1])
        steps = max(abs(r1 - r0), abs(c1 - c0), 1)
        for t in range(steps + 1):
            frac = t / steps
            r = int(round(r0 + frac * (r1 - r0)))
            c = int(round(c0 + frac * (c1 - c0)))
            r = max(0, min(r, rows - 1))
            c = max(0, min(c, cols - 1))
            if not freespace[r, c]:
                return False
    return True


def _save_path_overlay(
    occ_data: "np.ndarray",
    freespace: "np.ndarray",
    path_ij: "np.ndarray",
    start_ij: tuple,
    end_ij: tuple,
    out_path: "Path",
) -> None:
    """Render the planned path over the occupancy grid for visual QA.

    White=freespace, black=occupied (scanned collision), gray=unknown, and a
    translucent red wash = the buffer_m inflation the planner actually routed
    around. The path polyline (blue), start (green) and end (red) let a reviewer
    confirm the camera stays in real free space — the recurring wall/shelf
    clipping shows up here as a path crossing white cells that are actually
    walls the scan missed. Upscaled 4x since maps are small.
    """
    import numpy as _np
    try:
        from PIL import Image, ImageDraw
    except Exception as _e:  # pragma: no cover - QA aid only, never fatal
        print(f"  OMap: path overlay skipped (PIL: {_e})")
        return
    from isaacsim.replicator.mobility_gen.impl.occupancy_map import OccupancyMapDataValue

    rows, cols = occ_data.shape
    rgb = _np.full((rows, cols, 3), 128, dtype=_np.uint8)          # unknown → gray
    rgb[occ_data == OccupancyMapDataValue.FREESPACE] = (255, 255, 255)
    rgb[occ_data == OccupancyMapDataValue.OCCUPIED] = (0, 0, 0)
    # buffer wash: free in raw scan but blocked after inflation
    buffered_only = (occ_data == OccupancyMapDataValue.FREESPACE) & (~freespace)
    rgb[buffered_only] = (255, 170, 170)

    scale = 4
    img = Image.fromarray(rgb, "RGB").resize((cols * scale, rows * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    pts = [(int(c) * scale, int(r) * scale) for r, c in path_ij]     # (col=x, row=y)
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 90, 255), width=2)
    def _dot(ij, color):
        r, c = int(ij[0]) * scale, int(ij[1]) * scale
        draw.ellipse([c - 4, r - 4, c + 4, r + 4], fill=color)
    _dot(start_ij, (0, 200, 0))
    _dot(end_ij, (230, 0, 0))
    img.save(str(out_path))
    print(f"  OMap: path overlay saved → {out_path.name}")


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


def _path_length(waypoints: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(waypoints) - 1):
        dx = waypoints[i + 1][0] - waypoints[i][0]
        dy = waypoints[i + 1][1] - waypoints[i][1]
        dz = waypoints[i + 1][2] - waypoints[i][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _reflect_waypoints(waypoints: list[list[float]], passes: int) -> list[list[float]]:
    """Ping-pong a path: forward, backward, forward, ... for ``passes`` legs.

    The shared turnaround point between consecutive legs is dropped so no
    zero-length segment is produced (which would break arc-length interpolation).
    Traversing a leg in reverse flips the per-segment heading by 180°, so the
    ego camera faces the opposite way and reveals the scene *behind* it — new
    information rather than a re-view.
    """
    seq = [list(w) for w in waypoints]
    out = list(seq)
    for p in range(1, max(1, passes)):
        leg = seq[::-1] if (p % 2 == 1) else list(seq)
        out.extend(leg[1:])
    return out


def _pose_on_leg(
    E: list[list[float]], s0: int, s1: int, arc: float
) -> tuple[float, ...]:
    """Interpolate a pose at ``arc`` metres into the leg spanning E segments s0..s1.

    Returns (x, y, z, roll, pitch, yaw_rel, seg) with ``seg`` the E segment the
    point lies on and ``yaw_rel`` carried straight from the waypoints (0), so the
    caller's per-frame loop derives heading from E[seg+1]-E[seg].
    """
    acc = 0.0
    for s in range(s0, s1 + 1):
        dx = E[s + 1][0] - E[s][0]
        dy = E[s + 1][1] - E[s][1]
        dz = E[s + 1][2] - E[s][2]
        seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        if arc <= acc + seg_len or s == s1:
            local = 0.0 if seg_len < 1e-9 else max(0.0, min(1.0, (arc - acc) / seg_len))
            p0, p1 = E[s], E[s + 1]
            interp = [p0[k] + local * (p1[k] - p0[k]) for k in range(6)]
            return tuple(interp) + (s,)
        acc += seg_len
    p0, p1 = E[s1], E[s1 + 1]
    return tuple(p1[:6]) + (s1,)


def _plan_traversal(
    waypoints: list[list[float]], num_frames: int, traj_cfg: dict[str, Any]
) -> tuple[list[tuple[float, ...]], list[list[float]]]:
    """Spread ``num_frames`` poses along the path, ping-ponging a route that is too
    short to fill the frame budget with meaningfully spaced views, and rotating
    smoothly in place at each turnaround (no 180° heading jump).

    When the planned path is short (a relaxed occupancy path / a small Stage-7
    roam box), spreading every frame along it packs near-duplicate frames a few
    centimetres apart. Instead, reflect the path forward-and-back enough times
    that consecutive frames sit >= ``min_spacing_m`` apart; a reversed leg faces
    the camera the opposite way (new information, not a re-view). To avoid the
    heading snapping 180° between a forward and a reverse leg, the camera pivots
    IN PLACE at each turnaround — a set of frames at the turnaround point whose
    yaw sweeps from the incoming heading to the outgoing (reversed) one — which
    also spends the "spare" frames on rotation as intended. Pivot cost is modelled
    as ``pivot_arc_m`` of virtual arc so frames divide smoothly between travel and
    rotation. Returns (poses, effective_waypoints); the effective (reflected)
    waypoints MUST replace the caller's ``waypoints`` so per-frame heading (indexed
    by segment) stays aligned. No-op (normal interpolation) when the path is
    already long enough, when disabled, or for degenerate inputs.
    """
    fill_cfg = traj_cfg.get("short_path_fill", {}) if isinstance(traj_cfg, dict) else {}
    if not bool(fill_cfg.get("enabled", True)) or num_frames < 3 or len(waypoints) < 2:
        return _interpolate_waypoints(waypoints, num_frames), waypoints

    total = _path_length(waypoints)
    if total < 1e-6:
        return _interpolate_waypoints(waypoints, num_frames), waypoints

    min_spacing = float(fill_cfg.get("min_spacing_m", 0.3))
    max_passes = int(fill_cfg.get("max_passes", 8))
    spacing_actual = total / (num_frames - 1)
    if spacing_actual >= min_spacing or max_passes <= 1:
        return _interpolate_waypoints(waypoints, num_frames), waypoints

    target_len = min_spacing * (num_frames - 1)
    passes = min(max_passes, int(math.ceil(target_len / total)))
    effective = _reflect_waypoints(waypoints, passes)

    seg_per_leg = len(waypoints) - 1
    leg_len = _path_length(waypoints)
    pivot_arc = float(fill_cfg.get("pivot_arc_m", 1.5))

    # Virtual timeline: leg0 (leg_len) | pivot0 (pivot_arc) | leg1 | pivot1 | ...
    spans: list[tuple[str, int, float]] = []
    for i in range(passes):
        spans.append(("leg", i, leg_len))
        if i < passes - 1:
            spans.append(("pivot", i, pivot_arc))
    cum = [0.0]
    for _, _, length in spans:
        cum.append(cum[-1] + length)
    total_v = cum[-1]

    poses: list[tuple[float, ...]] = []
    for f in range(num_frames):
        v = (f / (num_frames - 1)) * total_v
        si = 0
        while si < len(spans) and v > cum[si + 1] + 1e-9:
            si += 1
        si = min(si, len(spans) - 1)
        kind, idx, span_len = spans[si]
        local = 0.0 if span_len < 1e-9 else max(0.0, min(1.0, (v - cum[si]) / span_len))
        if kind == "leg":
            s0 = idx * seg_per_leg
            s1 = s0 + seg_per_leg - 1
            poses.append(_pose_on_leg(effective, s0, s1, local * leg_len))
        else:
            # Pivot at the turnaround vertex between leg idx and idx+1. Anchor to
            # the INCOMING segment so its base heading matches the arriving leg;
            # sweep yaw_rel 0 -> 180 so the view rotates smoothly to the reversed
            # heading the next leg departs on (continuous at both ends).
            v_t = (idx + 1) * seg_per_leg
            in_seg = v_t - 1
            pt = effective[v_t]
            poses.append((pt[0], pt[1], pt[2], pt[3], pt[4], 180.0 * local, in_seg))

    print(f"  Short-path fill: {total:.1f}m path, {num_frames} frames "
          f"(spacing {spacing_actual:.2f}m<{min_spacing}m) -> ping-pong x{passes} "
          f"+ {passes-1} in-place pivot(s)", flush=True)
    return poses, effective


def _build_occupancy_waypoints(
    cfg: dict[str, Any],
    simulation_app,
    nav_dir: Path,
    camera_z: float,
    seed: int,
    sample_poses_n: int | None = None,
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
    buffer_m = float(occ_cfg.get("buffer_m", 0.5))
    min_path_m = float(occ_cfg.get("min_path_m", 3.0))
    max_retries = int(occ_cfg.get("max_retries", 20))
    # Scan a VERTICAL BAND, not a single z-plane. The 2D occupancy projects any
    # geometry within [scan_z_min_m, scan_z_max_m] to occupied. A single thin
    # slice (the old z_slice_m) let pallet racks/shelves — thin posts with gaps
    # at that height — scan as free floor, so the planner drove the camera
    # straight through them. Banding floor→above-camera-height makes racks,
    # shelves and walls solid obstacles. Legacy z_slice_m, if set, seeds the band.
    _legacy_slice = occ_cfg.get("z_slice_m")
    scan_z_min = float(occ_cfg.get("scan_z_min_m", 0.1))
    scan_z_max = float(occ_cfg.get("scan_z_max_m",
                                   (float(_legacy_slice) + 1.0) if _legacy_slice else 2.0))

    np.random.seed(seed)

    print(f"Occupancy map: bounds=({x_min},{y_min})→({x_max},{y_max}) "
          f"z-band=[{scan_z_min},{scan_z_max}]m cell={cell_size}m")
    om_iface = _omap.acquire_omap_interface()
    om_iface.set_cell_size(cell_size)
    om_iface.set_transform(
        (0.0, 0.0, 0.0),
        (x_min, y_min, scan_z_min),
        (x_max, y_max, scan_z_max),
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

    freespace = omap_buffered.freespace_mask()

    if sample_poses_n is not None:
        # Random-frame mode: N INDEPENDENT camera poses in buffered freespace
        # with uniform yaw — disconnected snapshots instead of a connected path.
        sampler = UniformPoseSampler()
        pts = []
        for _ in range(int(sample_poses_n)):
            p = sampler.sample(omap_buffered)
            pts.append([float(p.x), float(p.y), camera_z, 90.0, 0.0,
                        float(np.random.uniform(0.0, 360.0))])
        (nav_dir / "sampled_poses.json").write_text(json.dumps({
            "mode": "random_frames",
            "poses_x_y_yawdeg": [[round(q[0], 3), round(q[1], 3), round(q[5], 1)] for q in pts],
        }, indent=2))
        _omap.release_omap_interface(om_iface)
        print(f"  Random poses: sampled {len(pts)} independent freespace poses")
        return pts

    def _finalize(path_world, path_ij, start_pose, start_ij, end_ij, attempt, path_length, relaxed):
        waypoints = [[float(p[0]), float(p[1]), camera_z, 90.0, 0.0, 0.0] for p in path_world]
        tag = "longest-available" if relaxed else "found"
        print(f"  Path {tag}: {len(waypoints)} pts, {path_length:.1f}m (attempt {attempt+1})")
        path_record = {
            "attempt": attempt + 1,
            "path_length_m": round(path_length, 3),
            "min_path_m": round(min_path_m, 3),
            "relaxed_below_min_path": bool(relaxed),
            "num_waypoints": len(waypoints),
            "start_world_xy": [round(float(start_pose.x), 4), round(float(start_pose.y), 4)],
            "end_world_xy": [round(float(path_world[-1, 0]), 4), round(float(path_world[-1, 1]), 4)],
            "waypoints_world_xy": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in path_world],
        }
        (nav_dir / "planned_path.json").write_text(json.dumps(path_record, indent=2))
        _save_path_overlay(occ_data, freespace, path_ij, start_ij, end_ij,
                           nav_dir / "path_overlay.png")
        # Release the omap interface so its C++ render callbacks are unregistered,
        # preventing debug lines from being redrawn in captured frames.
        _omap.release_omap_interface(om_iface)
        return waypoints

    sampler = UniformPoseSampler()
    # Track the longest valid, obstacle-clear path seen across attempts. When no
    # attempt yields a >= min_path_m route (e.g. a small Stage-7 roam box), fall
    # back to this best path rather than raising — the short clear path is then
    # walked forward AND in reverse (ping-pong) to fill the frame budget with
    # genuinely new views (see _plan_traversal), so a small exploration box
    # degrades gracefully instead of wedging the whole-workflow retry wrapper.
    # generate_paths guarantees the path lies in buffered freespace, so the
    # fallback is always collision-clear.
    best = None  # (path_length, path_world, path_ij, start_pose, start_ij, end_ij, attempt)
    for attempt in range(max_retries):
        start_pose = sampler.sample(omap_buffered)
        start_px = omap.world_to_pixel_numpy(np.array([[start_pose.x, start_pose.y]]))
        # generate_paths expects (row, col) = (y_px, x_px)
        start_ij = (int(start_px[0, 1]), int(start_px[0, 0]))

        result = generate_paths(start_ij, freespace)

        valid = result.get_valid_end_points()
        if len(valid[0]) < 2:
            print(f"  Attempt {attempt+1}: no reachable ends, retry")
            continue

        end_ij = result.sample_random_end_point()
        path_ij = result.unroll_path(end_ij)          # (N,2) [row, col]
        path_ij, _ = compress_path(path_ij)
        # compress_path is purely geometric — verify shortcuts stay in buffered freespace
        if not _compressed_path_is_clear(path_ij, freespace):
            print(f"  Attempt {attempt+1}: compressed path clips obstacle, retry")
            continue
        path_xy_px = path_ij[:, ::-1]                 # → [col, row] = x_px, y_px
        path_world = omap.pixel_to_world_numpy(path_xy_px)  # (N,2) world [x,y]

        diffs = np.diff(path_world, axis=0)
        path_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
        if best is None or path_length > best[0]:
            best = (path_length, path_world, path_ij, start_pose, start_ij, end_ij, attempt)
        if path_length < min_path_m:
            print(f"  Attempt {attempt+1}: path {path_length:.1f}m < {min_path_m}m, retry")
            continue

        return _finalize(path_world, path_ij, start_pose, start_ij, end_ij, attempt, path_length, relaxed=False)

    if best is not None:
        path_length, path_world, path_ij, start_pose, start_ij, end_ij, attempt = best
        print(f"  No path >= {min_path_m}m in {max_retries} tries; "
              f"using longest clear path {path_length:.1f}m (frames top up via reverse traversal)")
        return _finalize(path_world, path_ij, start_pose, start_ij, end_ij, attempt, path_length, relaxed=True)

    raise RuntimeError(f"No valid occupancy path found after {max_retries} attempts")


def apply_roam_bounds(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Derive ``trajectory.bounds_xy`` from a constrained roam reparameterization.

    Stage 7 (exploration-boundary optimization). Rather than let Optuna search the
    four raw ``bounds_xy`` floats — which admits invalid boxes (``x_min > x_max``)
    and confounds box *size* with box *position* — the search space exposes a
    constrained center+extent form under ``trajectory.roam``:

        center_x_frac, center_y_frac  in [-1, 1]   (offset within the envelope)
        width_frac,    height_frac    in (0, 1]     (box size as a fraction)

    interpreted as fractions of the *env envelope*, which here is the config's
    pre-existing ``trajectory.bounds_xy`` (the per-env box the seed author tuned).
    Because the envelope IS the config's own bounds, the box stays env-relative
    (coupling #3 in the plan): the same fractions yield an appropriately sized box
    in ``full_warehouse`` and in the smaller warehouses. At the default
    (center 0, width/height_frac 1.0) the derived box equals the envelope exactly,
    so behavior is unchanged when roam is absent, disabled, or left at defaults.

    The construction keeps the box strictly inside the envelope for any frac in
    range (the center offset scales by the leftover slack, which shrinks to zero
    as width/height_frac -> 1), so no wall-clipping is possible. Object scatter
    reads the same ``bounds_xy`` (coupling #2), so shrinking the roam box also
    insets object placement — objects stay camera-reachable, which is desirable.

    To avoid the occupancy planner wedging on a box too small to contain a
    ``min_path_m`` route (coupling #1 — the ``characters``-style stall), the
    effective ``occupancy.min_path_m`` is scaled down to what the shrunken free
    region can actually yield. Returns a summary dict for event logging, or
    ``None`` when roam is not active.
    """
    import math

    traj_cfg = cfg.get("trajectory")
    if not isinstance(traj_cfg, dict):
        return None
    roam = traj_cfg.get("roam")
    if not isinstance(roam, dict) or not bool(roam.get("enabled", False)):
        return None

    def _clamp(v, lo, hi):
        return max(lo, min(hi, float(v)))

    envelope = [float(v) for v in traj_cfg.get("bounds_xy", [-6.0, 6.0, -6.0, 8.0])]
    x0, x1, y0, y1 = envelope
    ew, eh = (x1 - x0), (y1 - y0)
    cx0, cy0 = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    wf = _clamp(roam.get("width_frac", 1.0), 0.05, 1.0)
    hf = _clamp(roam.get("height_frac", 1.0), 0.05, 1.0)
    cxf = _clamp(roam.get("center_x_frac", 0.0), -1.0, 1.0)
    cyf = _clamp(roam.get("center_y_frac", 0.0), -1.0, 1.0)

    w, h = wf * ew, hf * eh
    slack_x, slack_y = (ew - w) / 2.0, (eh - h) / 2.0
    cx, cy = cx0 + cxf * slack_x, cy0 + cyf * slack_y
    derived = [cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0]
    traj_cfg["bounds_xy"] = [round(v, 4) for v in derived]

    # Feasibility floor vs min_path_m (coupling #1). The planner samples two free
    # cells and returns the path between them; its length is bounded by roughly the
    # free-region diagonal. If min_path_m exceeds that, every attempt is rejected
    # and the whole-workflow retry wrapper loops forever. Scale min_path_m to the
    # box so the study can never wedge on a small proposal.
    occ = traj_cfg.setdefault("occupancy", {})
    inset = float(occ.get("boundary_margin_m", 2.0)) + float(occ.get("buffer_m", 0.5))
    inner_diag = math.hypot(max(0.0, w - 2.0 * inset), max(0.0, h - 2.0 * inset))
    requested_min_path = float(occ.get("min_path_m", 3.0))
    feasible_min_path = inner_diag / 1.15
    adjusted = None
    if requested_min_path > feasible_min_path:
        adjusted = round(max(3.0, feasible_min_path), 3)
        occ["min_path_m"] = adjusted
        if "min_path_m" in traj_cfg:      # keep the mirrored legacy key consistent
            traj_cfg["min_path_m"] = adjusted

    return {
        "envelope_xy": [round(v, 4) for v in envelope],
        "roam_params": {"center_x_frac": cxf, "center_y_frac": cyf,
                        "width_frac": wf, "height_frac": hf},
        "derived_bounds_xy": traj_cfg["bounds_xy"],
        "min_path_m_requested": requested_min_path,
        "min_path_m_effective": adjusted if adjusted is not None else requested_min_path,
    }


def run_stage4(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = load_cfg(config_path)
    apply_cli_overrides(cfg, args)

    # Episode mode (--seeds): per-episode dirs live under --out_root; the legacy
    # per-run data_dir is unused, so session-level artifacts (warmup events,
    # resolved config) go to <out_root>/_session_<config-stem>/ instead of the
    # config's relative default (which may not be writable from Isaac's CWD).
    if getattr(args, "seeds", None) is not None and getattr(args, "out_root", None):
        cfg.setdefault("run", {})["data_dir"] = str(
            Path(args.out_root).resolve() / f"_session_{Path(args.config).stem}"
        )

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

    # Stage 7: derive trajectory.bounds_xy from the searchable roam box (if
    # enabled) BEFORE anything reads bounds_xy (object scatter + occupancy
    # planning both consume it) and before write_run_config dumps the resolved
    # config, so the run_config.yaml records the box actually used.
    roam_info = apply_roam_bounds(cfg)
    if roam_info is not None:
        print(f"Roam bounds: envelope={roam_info['envelope_xy']} "
              f"-> derived={roam_info['derived_bounds_xy']} "
              f"(min_path {roam_info['min_path_m_requested']}->{roam_info['min_path_m_effective']}m)",
              flush=True)
        append_event(events_path, "stage5_roam_bounds_derived", roam_info)

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
    from pxr import Gf, Semantics, Usd, UsdGeom
    from palletjack_sdg.utils.camera import rep_normal

    # Drive Replicator's RNG from the episode seed too. Object/light placement
    # uses rep.distribution.* (see rep_normal / _scatter_position), which is
    # governed by Replicator's global RNG — NOT by random.seed()/np.random.seed()
    # above. Without this, two runs with different --seed re-roll the trajectory
    # path but keep the SAME scene layout, so multi-seed evaluation (and
    # multi-seed dataset generation) would fail to sample the config's layout
    # distribution. Guarded so an older Replicator without the API won't crash.
    if hasattr(rep, "set_global_seed"):
        rep.set_global_seed(seed)

    # CosmosWriter uses OmniGraph script nodes (Canny edge annotator) — without
    # this, the annotator chain silently fails to attach and write() is never called.
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
    # DLSS quality mode recommended by all CosmosWriter reference examples.
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    # Motion blur — using PathTracing mode for higher-quality blur (it samples
    # the shutter window stochastically per ray, instead of the screen-space
    # post-process the RaytracedLighting pipeline uses). Camera shutter attrs
    # + time-sampled xform ops give the renderer the motion samples to
    # integrate. Always enabled at the engine level; shutter_close_fraction
    # controls actual blur amount.
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", 32)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", 32)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", 8)

    # ── Stage 7b-1: enable omni.anim.people so characters idle instead of T-pose.
    # Extensions must be enabled BEFORE Biped_Setup is referenced so the
    # AnimationGraph schema resolves and ApplyAnimationGraphAPICommand exists.
    anim_char_cfg = cfg.get("characters", {}).get("animation", {}) or {}
    anim_enabled_cfg = bool(anim_char_cfg.get("enabled", False))
    anim_extensions_ok = False
    if anim_enabled_cfg:
        try:
            import omni.kit.app as _kit_app
            _s = carb.settings.get_settings()
            # Must be set BEFORE any BehaviorScript is attached — otherwise a
            # security popup gates script execution (headless never shows it,
            # so scripts silently never run).
            _s.set("/app/scripting/ignoreWarningDialog", True)
            _ext_mgr = _kit_app.get_app().get_extension_manager()
            for _ext in (
                "omni.kit.scripting",
                "omni.anim.timeline",
                "omni.anim.graph.core",
                "omni.anim.retarget.core",
                "omni.anim.navigation.core",
                "omni.anim.people",
            ):
                _ext_mgr.set_extension_enabled_immediate(_ext, True)
            # Persistent settings the BehaviorScript / anim graph read on init.
            _s.set("/persistent/exts/omni.anim.people/character_prim_path", "/World/Characters")
            _s.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", False)
            _s.set("/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", False)
            # Needed to avoid see-through characters (DH assets rely on this).
            _s.set("/rtx/raytracing/fractionalCutoutOpacity", True)
            # Multi-frame settle so the extensions register their commands.
            for _ in range(3):
                simulation_app.update()
            anim_extensions_ok = True
            print("Character animation extensions enabled (Stage 7b-1)")
        except Exception as _exc:
            print(f"Character animation extension enable failed (non-fatal): {_exc}")
            anim_extensions_ok = False

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

    # ── Dim/brighten the environment's OWN built-in lights ────────────────────
    # The per-frame replicator randomization further below only drives prims
    # matching "RectLight"; the warehouse USD's dominant ceiling/dome lights are
    # otherwise untouched, so "night" / "very dark" configs still rendered fully
    # lit (IMAGE_REVIEW #3). Scale EVERY UsdLux light's authored intensity by
    # intensity_mean / env_reference_intensity (a ~"normal" brightness) so the
    # existing lighting knob now controls actual scene brightness. Set
    # lighting.env_light_scale to override the auto-derived factor; set it to 1.0
    # (or env_reference_intensity<=0) to disable this pass entirely.
    _lt_cfg = cfg.get("lighting", {})
    if _lt_cfg:
        _ref = float(_lt_cfg.get("env_reference_intensity", 120000.0))
        _scale = _lt_cfg.get("env_light_scale")
        if _scale is None:
            _mean = float(_lt_cfg.get("intensity_mean", _ref))
            _scale = (_mean / _ref) if _ref > 1e-9 else 1.0
        _scale = float(_scale)
        if abs(_scale - 1.0) > 1e-3:
            from pxr import UsdLux
            _n_scaled = 0
            for _prim in stage.Traverse():
                if not (_prim.HasAPI(UsdLux.LightAPI) or "Light" in _prim.GetTypeName()):
                    continue
                _attr = _prim.GetAttribute("inputs:intensity")
                if not (_attr and _attr.IsValid()):
                    _attr = _prim.GetAttribute("intensity")  # pre-UsdLux-2 fallback
                if not (_attr and _attr.IsValid()):
                    continue
                _cur = _attr.Get()
                if _cur is None:
                    continue
                _attr.Set(float(_cur) * _scale)
                _n_scaled += 1
            print(f"Env lighting: scaled {_n_scaled} built-in light(s) by {_scale:.3f} "
                  f"(intensity_mean={_lt_cfg.get('intensity_mean')} / ref={_ref})")
            append_event(events_path, "stage5_env_lights_scaled",
                         {"scale": round(_scale, 4), "num_lights": _n_scaled,
                          "reference_intensity": _ref})

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

    # ── Stage 7: animated human characters ────────────────────────────────────
    # Spawned AFTER _update_semantics() so their `person` class isn't stripped
    # by the keep-list filter. Characters use authored xformOp:translate +
    # xformOp:orient (set, not added — they already exist on the prims).
    spawned_characters: list[dict] = []
    char_cfg = cfg.get("characters", {})
    biped_prim = None
    if char_cfg.get("enabled", False) and int(char_cfg.get("count", 0)) > 0:
        char_count = int(char_cfg["count"])
        char_assets = char_cfg.get("assets", [])
        char_bounds = char_cfg.get("spawn_bounds_xy", [-4.0, 4.0, -4.0, 4.0])
        char_class = str(char_cfg.get("semantic_class", "person"))
        # Parent container so all chars live under one path
        chars_parent = stage.DefinePrim("/World/Characters", "Xform")
        # Biped_Setup is the animation-graph template; characters reference it
        # via their own USDs. Load it once as invisible so animation graphs
        # resolve correctly. Skip silently if it fails — characters still
        # appear in T-pose / default pose, which is fine for OD MVP.
        try:
            biped_url = prefix_with_isaac_asset_server(
                "/Isaac/People/Characters/Biped_Setup.usd"
            )
            biped_prim = stage.DefinePrim("/World/Characters/Biped_Setup", "Xform")
            biped_prim.GetReferences().AddReference(biped_url)
            UsdGeom.Imageable(biped_prim).MakeInvisible()
            # Ensure reference/composition + anim graph schema resolve before
            # ApplyAnimationGraphAPICommand is issued.
            for _ in range(3):
                simulation_app.update()
        except Exception as _exc:
            print(f"  character Biped_Setup load failed (non-fatal): {_exc}")
            biped_prim = None
        print(f"Spawning {char_count} characters from {len(char_assets)} variants...")
        for i in range(char_count):
            asset_rel = random.choice(char_assets) if char_assets else None
            if asset_rel is None:
                break
            usd_url = prefix_with_isaac_asset_server(asset_rel)
            prim_path = f"/World/Characters/person_{i:02d}"
            char_prim = stage.DefinePrim(prim_path, "Xform")
            char_prim.GetReferences().AddReference(usd_url)
            # x/y in spawn_bounds; z on the floor
            cx = random.uniform(char_bounds[0], char_bounds[1])
            cy = random.uniform(char_bounds[2], char_bounds[3])
            cyaw = random.uniform(0.0, 360.0)
            # Characters have authored xformOp:translate + xformOp:orient; set
            # them rather than AddXformOp (which would conflict).
            simulation_app.update()  # resolve the reference so ops exist
            t_attr = char_prim.GetAttribute("xformOp:translate")
            o_attr = char_prim.GetAttribute("xformOp:orient")
            if t_attr and t_attr.IsValid():
                from pxr import Gf as _Gf
                t_attr.Set(_Gf.Vec3d(cx, cy, 0.0))
                rot_quat = _Gf.Rotation(_Gf.Vec3d(0, 0, 1), cyaw).GetQuat()
                if o_attr and o_attr.IsValid():
                    if isinstance(o_attr.Get(), _Gf.Quatf):
                        o_attr.Set(_Gf.Quatf(rot_quat))
                    else:
                        o_attr.Set(rot_quat)
            else:
                # No authored translate — add one
                xformable = UsdGeom.Xformable(char_prim)
                xformable.AddTranslateOp().Set((cx, cy, 0.0))
                xformable.AddRotateXYZOp().Set((0.0, 0.0, cyaw))
            # Apply `person` semantic so bbox writer picks it up
            sem_api = Semantics.SemanticsAPI.Apply(char_prim, "Semantics_class")
            sem_api.CreateSemanticTypeAttr().Set("class")
            sem_api.CreateSemanticDataAttr().Set(char_class)
            spawned_characters.append({
                "prim": prim_path,
                "asset": asset_rel,
                "pos": [round(cx, 3), round(cy, 3), 0.0],
                "yaw_deg": round(cyaw, 1),
            })
            print(f"  person_{i:02d}: {asset_rel.split('/')[-1]} at ({cx:.2f},{cy:.2f}) yaw={cyaw:.0f}°")

        # ── Stage 7b-1: attach the anim graph to each character's SkelRoot ────
        # Without this the character USD renders its skeleton rest pose (T/A).
        # With this + timeline playing, the anim graph enters the "Idle" state
        # (Action="None") and drives the skeleton with an idle clip.
        anim_graph_applied = 0
        if anim_extensions_ok and anim_enabled_cfg and biped_prim is not None:
            try:
                import omni.kit.commands
                from pxr import Sdf as _Sdf
                # Find AnimationGraph inside Biped_Setup.
                anim_graph_prim = None
                for _p in Usd.PrimRange(biped_prim):
                    if _p.GetTypeName() == "AnimationGraph":
                        anim_graph_prim = _p
                        break
                if anim_graph_prim is None:
                    print("  animation graph not found under Biped_Setup — characters stay in T-pose")
                else:
                    ag_path = _Sdf.Path(anim_graph_prim.GetPrimPath())
                    for entry in spawned_characters:
                        char_root = stage.GetPrimAtPath(entry["prim"])
                        skel_root = None
                        for _p in Usd.PrimRange(char_root):
                            if _p.GetTypeName() == "SkelRoot":
                                skel_root = _p
                                break
                        if skel_root is None:
                            print(f"  {entry['prim']}: no SkelRoot, skip anim graph")
                            continue
                        try:
                            omni.kit.commands.execute(
                                "RemoveAnimationGraphAPICommand",
                                paths=[_Sdf.Path(skel_root.GetPrimPath())],
                            )
                        except Exception:
                            pass
                        omni.kit.commands.execute(
                            "ApplyAnimationGraphAPICommand",
                            paths=[_Sdf.Path(skel_root.GetPrimPath())],
                            animation_graph_path=ag_path,
                        )
                        anim_graph_applied += 1
                    print(f"  animation graph applied to {anim_graph_applied}/{len(spawned_characters)} characters")
            except Exception as _exc:
                print(f"  animation graph attach failed (non-fatal, T-pose fallback): {_exc}")

        # ── Stage 7b-2: attach BehaviorScript + emit command file for movement.
        # BehaviorScript ticks each frame while timeline is playing, reads the
        # command list, and drives the character (Idle / LookAround / GoTo).
        # Without navmesh, GoTo walks in a straight line to the target.
        movement_cfg = (anim_char_cfg.get("movement") or {}) if anim_enabled_cfg else {}
        movement_enabled = bool(movement_cfg.get("enabled", False))
        movement_attached = 0
        if (
            movement_enabled
            and anim_extensions_ok
            and biped_prim is not None
            and anim_graph_applied > 0
        ):
            try:
                import omni.kit.commands
                from pxr import Sdf as _Sdf
                # 1) Emit commands.txt from movement_cfg + spawned character positions.
                cmd_lines = _generate_character_commands(
                    spawned_characters, char_bounds, movement_cfg, rng=random
                )
                cmd_dir = output_dir / "characters"
                cmd_dir.mkdir(parents=True, exist_ok=True)
                cmd_file = cmd_dir / "commands.txt"
                cmd_file.write_text("\n".join(cmd_lines) + "\n")
                print(f"  wrote {len(cmd_lines)} character commands → {cmd_file}")

                # 2) Point omni.anim.people at that file + loop indefinitely.
                _s = carb.settings.get_settings()
                _s.set("/exts/omni.anim.people/command_settings/command_file_path", str(cmd_file))
                _s.set("/exts/omni.anim.people/command_settings/number_of_loop", "inf")

                # 3) Attach character_behavior.py BehaviorScript to each SkelRoot.
                import omni.kit.app as _kit_app
                script_path = (
                    _kit_app.get_app()
                    .get_extension_manager()
                    .get_extension_path_by_module("omni.anim.people")
                    + "/omni/anim/people/scripts/character_behavior.py"
                )
                for entry in spawned_characters:
                    char_root = stage.GetPrimAtPath(entry["prim"])
                    skel_root = None
                    for _p in Usd.PrimRange(char_root):
                        if _p.GetTypeName() == "SkelRoot":
                            skel_root = _p
                            break
                    if skel_root is None:
                        continue
                    try:
                        omni.kit.commands.execute(
                            "RemoveScriptingAPICommand",
                            paths=[_Sdf.Path(skel_root.GetPrimPath())],
                        )
                    except Exception:
                        pass
                    omni.kit.commands.execute(
                        "ApplyScriptingAPICommand",
                        paths=[_Sdf.Path(skel_root.GetPrimPath())],
                    )
                    attr = skel_root.GetAttribute("omni:scripting:scripts")
                    if attr and attr.IsValid():
                        attr.Set([script_path])
                        movement_attached += 1
                print(f"  behavior script attached to {movement_attached}/{len(spawned_characters)} characters")
                # Let ScriptManager's async pending-sync task process the new
                # OmniScriptingAPI attachments so the BehaviorScript instances
                # actually get created before timeline.play() fires on_play.
                for _ in range(20):
                    simulation_app.update()
            except Exception as _exc:
                print(f"  behavior script attach failed (non-fatal, characters idle only): {_exc}")

    append_event(events_path, "stage5_scene_spawned", {
        "palletjacks": pj_group is not None,
        "forklifts": fl_group is not None,
        "pallets": pa_group is not None,
        "distractors": dist_group is not None,
        "characters": len(spawned_characters),
        "characters_anim_enabled": anim_enabled_cfg,
        "characters_anim_extensions_ok": anim_extensions_ok,
        "characters_movement_enabled": movement_enabled,
        "characters_movement_attached": movement_attached,
    })

    # ── One-shot scene randomization via on_frame(num_frames=1) ──────────────
    pj_cfg = cfg.get("palletjacks", {})
    fl_cfg = cfg.get("forklifts", {})
    pa_cfg = cfg.get("pallets", {})
    dr_cfg = cfg.get("distractor_randomization", {})
    lt_cfg = cfg.get("lighting", {})
    mat_cfg = cfg.get("materials", {})
    textures = [prefix_with_isaac_asset_server(p) for p in mat_cfg.get("textures", [])]

    # Object XY placement. Default is the original central Gaussian. When an
    # object class sets `scatter: uniform`, positions are drawn UNIFORMLY across
    # the navigable floor (trajectory.bounds_xy, inset off the walls) instead —
    # so targets populate the open/near-wall areas too, and a roaming
    # start-anywhere camera keeps objects in view (fewer empty frames).
    _traj_bounds = cfg.get("trajectory", {}).get("bounds_xy", [-10.0, 10.0, -10.0, 10.0])

    def _scatter_position(obj_cfg):
        if str(obj_cfg.get("scatter", "")).lower() == "uniform":
            b = obj_cfg.get("scatter_bounds_xy")
            if b is None:
                m = float(obj_cfg.get("scatter_inset_m", 2.0))
                b = [_traj_bounds[0] + m, _traj_bounds[1] - m,
                     _traj_bounds[2] + m, _traj_bounds[3] - m]
            z = float(tuple(obj_cfg.get("position_mean", (0.0, 0.0, 0.0)))[2])
            return rep.distribution.uniform(
                (float(b[0]), float(b[2]), z), (float(b[1]), float(b[3]), z)
            )
        return rep_normal(tuple(obj_cfg["position_mean"]), tuple(obj_cfg["position_std"]))

    # Re-fireable via custom event so episode mode can re-roll the scene per
    # seed without restarting Isaac. The warmup below fires it once, matching
    # the old on_frame(num_frames=1) one-shot semantics for single-seed runs.
    with rep.trigger.on_custom_event(event_name="randomize_scene"):
        if pj_group is not None:
            with pj_group:
                rep.modify.pose(
                    position=_scatter_position(pj_cfg),
                    rotation=rep_normal(tuple(pj_cfg["rotation_mean"]), tuple(pj_cfg["rotation_std"])),
                    scale=rep_normal(tuple(pj_cfg["scale_mean"]), tuple(pj_cfg["scale_std"])),
                )
        if pj_group is not None and pj_cfg.get("color_mean") is not None:
            # Palletjack body-color randomization (ported back from the
            # random-frame sampler). color_mean=null disables — the authored
            # asset materials are kept; color_std=[0,0,0] pins the exact color.
            _pj_color_pattern = str(pj_cfg.get("color_prim_pattern", "SteerAxles"))
            print(f"Palletjack color: mean={pj_cfg['color_mean']} "
                  f"std={pj_cfg.get('color_std', [0.0, 0.0, 0.0])} "
                  f"(prims matching '{_pj_color_pattern}')")
            with rep.get.prims(path_pattern=_pj_color_pattern):
                rep.randomizer.color(
                    colors=rep_normal(
                        tuple(pj_cfg["color_mean"]),
                        tuple(pj_cfg.get("color_std", (0.0, 0.0, 0.0))),
                    )
                )
        if fl_group is not None:
            with fl_group:
                rep.modify.pose(
                    position=_scatter_position(fl_cfg),
                    rotation=rep_normal(tuple(fl_cfg["rotation_mean"]), tuple(fl_cfg["rotation_std"])),
                    scale=rep_normal(tuple(fl_cfg["scale_mean"]), tuple(fl_cfg["scale_std"])),
                )
        if pa_group is not None:
            with pa_group:
                rep.modify.pose(
                    position=_scatter_position(pa_cfg),
                    rotation=rep_normal(tuple(pa_cfg["rotation_mean"]), tuple(pa_cfg["rotation_std"])),
                    scale=rep_normal(tuple(pa_cfg["scale_mean"]), tuple(pa_cfg["scale_std"])),
                )
        if dist_group is not None:
            with dist_group:
                # `scatter: uniform` on distractor_randomization spreads distractors
                # UNIFORMLY across the navigable floor (same helper/bounds as the
                # target objects) so a roaming tight-framed camera actually sees
                # them. Without it, _scatter_position falls back to the original
                # central Gaussian (position_mean/std) — backward-compatible.
                rep.modify.pose(
                    position=_scatter_position(dr_cfg),
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
            # Diagnostic: how many prims actually match path_pattern="SM_Wall"?
            # If 0 (or fewer than the visible walls), some wall prims use a
            # different name and keep the default USD concrete — i.e. the
            # per-wall texturing below silently misses them. Surfaced in events.
            try:
                _wall_prims = [
                    str(p.GetPath()) for p in stage.Traverse()
                    if "SM_Wall" in p.GetName()
                ]
                print(f"  wall-texture: {len(_wall_prims)} prims match 'SM_Wall'"
                      f" (e.g. {_wall_prims[:3]})")
                append_event(events_path, "stage5_wall_prims_matched", {
                    "count": len(_wall_prims),
                    "sample_paths": _wall_prims[:5],
                })
            except Exception as _exc:
                print(f"  wall-texture prim scan failed (non-fatal): {_exc}")

            # PER-WALL variety: create a POOL of wall materials (count>1) rather
            # than one. rep.randomizer.materials() then samples a *different*
            # material per matched SM_Wall prim, so walls in a scene no longer all
            # read as the same texture. Diversity of the pool comes from the
            # (now diversified) `textures` palette in the config materials block.
            n_wall_mats = max(8, len(textures))
            wall_mats = rep.create.material_omnipbr(
                diffuse_texture=rep.distribution.choice(textures),
                roughness=rep_normal(mat_cfg["roughness_mean"], mat_cfg["roughness_std"]),
                metallic=rep.distribution.choice(mat_cfg["metallic_choices"]),
                emissive_texture=rep.distribution.choice(textures),
                emissive_intensity=rep_normal(
                    mat_cfg["emissive_intensity_mean"], mat_cfg["emissive_intensity_std"]
                ),
                count=n_wall_mats,
            )
            with rep.get.prims(path_pattern="SM_Wall"):
                rep.randomizer.materials(wall_mats)

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

    horizontal_aperture = float(
        ego_cam_cfg.get(
            "horizontal_aperture_mm",
            legacy_cam.get("horizontal_aperture", 20.955),
        )
    )
    # Focal length: if explicitly set in config, use it (calibrated lens
    # match) and back-solve the effective FOV for logging. Otherwise derive
    # focal_length from FOV + aperture as before.
    focal_length_override = ego_cam_cfg.get("focal_length_mm")
    if focal_length_override is not None:
        focal_length = float(focal_length_override)
        fov_deg = 2.0 * math.degrees(math.atan2(horizontal_aperture / 2.0, focal_length))
    else:
        focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    clipping = legacy_cam.get("clipping_range", [0.1, 1000000.0])

    print(f"Episode camera: fov={fov_deg:.1f}° focal_length={focal_length:.3f}mm aperture={horizontal_aperture:.3f}mm")
    append_event(events_path, "stage5_camera_sampled", {
        "fov_deg": round(fov_deg, 3),
        "focal_length_mm": round(focal_length, 4),
        "resolution": [width, height],
    })

    # ── Static mount + per-frame jitter (simulates handheld / uneven terrain) ──
    ego_projection = str(ego_cam_cfg.get("projection", "perspective"))
    ego_pitch_static = float(ego_cam_cfg.get("pitch_deg", 0.0))
    ego_roll_static = float(ego_cam_cfg.get("roll_deg", 0.0))
    pitch_jit_cfg = ego_cam_cfg.get("pitch_jitter") or {}
    roll_jit_cfg = ego_cam_cfg.get("roll_jitter") or {}
    yaw_jit_cfg = ego_cam_cfg.get("yaw_jitter") or {}
    lat_jit_cfg = ego_cam_cfg.get("lateral_jitter") or {}
    vert_jit_cfg = ego_cam_cfg.get("vertical_jitter") or {}
    pitch_jit_amp = float(pitch_jit_cfg.get("amp_deg", 0.0))
    pitch_jit_hz = float(pitch_jit_cfg.get("hz", 1.5))
    roll_jit_amp = float(roll_jit_cfg.get("amp_deg", 0.0))
    roll_jit_hz = float(roll_jit_cfg.get("hz", 1.2))
    # Yaw jitter = gaze wander around the travel heading (head-turning while
    # walking). Slower default rhythm than the pitch/roll bob.
    yaw_jit_amp = float(yaw_jit_cfg.get("amp_deg", 0.0))
    yaw_jit_hz = float(yaw_jit_cfg.get("hz", 0.9))
    lat_jit_amp = float(lat_jit_cfg.get("amp_m", 0.0))
    lat_jit_hz = float(lat_jit_cfg.get("hz", 1.3))
    vert_jit_amp = float(vert_jit_cfg.get("amp_m", 0.0))
    vert_jit_hz = float(vert_jit_cfg.get("hz", 2.0))
    # Random per-episode phase so each trajectory has its own rhythm.
    pitch_jit_phase = random.uniform(0.0, 2.0 * math.pi)
    roll_jit_phase = random.uniform(0.0, 2.0 * math.pi)
    yaw_jit_phase = random.uniform(0.0, 2.0 * math.pi)
    lat_jit_phase = random.uniform(0.0, 2.0 * math.pi)
    vert_jit_phase = random.uniform(0.0, 2.0 * math.pi)

    # Motion blur — shutter_close_fraction is the fraction of the frame
    # interval the shutter stays open. The ghost camera writes pose as
    # time samples at the open and close moments so RTX can integrate
    # motion across the shutter (gives both camera-sweep blur and, with
    # delta_time>0, dynamic-object blur).
    # IMPORTANT: time samples must be in USD time codes = sim_time_sec * tcps.
    # Stage's tcps is not necessarily 60 — read it once here.
    shutter_close_fraction = float(ego_cam_cfg.get("shutter_close_fraction", 0.0))
    motion_blur_enabled = shutter_close_fraction > 1e-6
    stage_tcps = float(stage.GetTimeCodesPerSecond() or 60.0)
    # capture_dt drives both the orchestrator step interval and the
    # shutter-time-codes math. Read here so the camera setup below can use it.
    _agent_cfg_early = cfg.get("agent", {})
    _agent_type_early = _agent_cfg_early.get("type", "camera_rig")
    physics_dt = float(cfg.get("simulation", {}).get("physics_dt", 1.0 / 60.0))
    if _agent_type_early == "carter":
        capture_dt = float(_agent_cfg_early.get("capture_dt", 0.25))
    else:
        capture_dt = float(_agent_cfg_early.get("capture_dt", 1.0 / 60.0))
    print(f"Stage timeCodesPerSecond: {stage_tcps}, capture_dt: {capture_dt}")
    # Depth of field. f_stop = 0 = pinhole (always sharp). f_stop > 0 = real
    # lens; focus_distance_m sharp, everything else progressively blurred.
    # USD focusDistance is in stage units; warehouse stages are metersPerUnit=1.0
    # so a meters value works directly. If a stage in different units is loaded
    # this scales correctly.
    ego_f_stop = float(ego_cam_cfg.get("f_stop", 0.0))
    _mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    ego_focus_distance_stage = float(ego_cam_cfg.get("focus_distance_m", 5.0)) / _mpu
    print(
        f"Camera mount: pitch={ego_pitch_static:+.1f}° roll={ego_roll_static:+.1f}°, "
        f"jitter pitch={pitch_jit_amp:.1f}°@{pitch_jit_hz:.1f}Hz "
        f"roll={roll_jit_amp:.1f}°@{roll_jit_hz:.1f}Hz, projection={ego_projection}, "
        f"shutter_close={shutter_close_fraction:.2f}"
    )

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

    # ── Agent: ghost camera rig or physical Carter robot ──────────────────────
    agent_cfg = cfg.get("agent", {})
    agent_type = agent_cfg.get("type", "camera_rig")

    # Filled by either branch; consumed by the main loop.
    camera_prim_path: str
    translate_op = None        # ghost camera ops (None for carter)
    rotate_op = None
    robot_prim_path: str | None = None
    robot_translate_op = None
    robot_rotate_op = None

    if agent_type == "carter":
        # Spawn Nova Carter as a USD reference. The Carter USD's root already
        # has translate/orient/scale ops, so we can't AddXformOp directly on
        # the referenced prim. Workaround: empty parent Xform owns our ops;
        # the reference lives on a child prim that keeps its own internal ops.
        robot_prim_path = "/World/Carter"
        robot_body_subpath = "body"
        carter_usd_rel = agent_cfg.get(
            "usd_path", "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        )
        carter_usd_url = prefix_with_isaac_asset_server(carter_usd_rel)
        robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
        robot_body_prim = stage.DefinePrim(
            f"{robot_prim_path}/{robot_body_subpath}", "Xform"
        )
        robot_body_prim.GetReferences().AddReference(carter_usd_url)
        print(f"Spawning Carter at {robot_prim_path}/{robot_body_subpath} from {carter_usd_url}")

        robot_xformable = UsdGeom.Xformable(robot_prim)
        robot_translate_op = robot_xformable.AddTranslateOp()
        robot_rotate_op = robot_xformable.AddRotateXYZOp()

        # Robot forward = local +X. World yaw = atan2(dy, dx).
        if traj_mode == "waypoint_list":
            init_robot_yaw = math.degrees(math.atan2(seg0_dy, seg0_dx))
            robot_translate_op.Set((x0, y0, 0.0))
            robot_rotate_op.Set((0.0, 0.0, init_robot_yaw))
        else:
            # occupancy_path: park Carter far off-map so its footprint doesn't
            # block the omap scan. Repositioned to first waypoint after the
            # path is computed in the main loop.
            robot_translate_op.Set((-20.0, -20.0, 0.0))
            robot_rotate_op.Set((0.0, 0.0, 0.0))

        # Resolve the reference so the camera prim is visible to UsdGeom.Camera.
        simulation_app.update()

        # Mount a fresh perspective camera as a child of chassis_link.
        # Carter's built-in sensors (Hawk, Owl) have lens-housing geometry
        # that shows in the render and Loco doesn't use fisheye anyway.
        # All mount parameters are config-driven so Optuna can tune them.
        mount_cfg = ego_cam_cfg.get("mount", {})
        mount_forward = float(mount_cfg.get("forward_m", 0.30))
        mount_lateral = float(mount_cfg.get("lateral_m", 0.0))
        mount_height = float(mount_cfg.get("height_m", 1.40))
        mount_pitch_deg = float(mount_cfg.get("pitch_deg", 0.0))

        chassis_path = f"{robot_prim_path}/{robot_body_subpath}/chassis_link"
        camera_prim_path = f"{chassis_path}/EgoCamera"
        ego_cam_prim = stage.DefinePrim(camera_prim_path, "Camera")
        ego_cam_schema = UsdGeom.Camera(ego_cam_prim)
        ego_cam_schema.GetHorizontalApertureAttr().Set(horizontal_aperture)
        ego_cam_schema.GetFocalLengthAttr().Set(focal_length)
        ego_cam_schema.GetClippingRangeAttr().Set(
            Gf.Vec2f(float(clipping[0]), float(clipping[1]))
        )
        ego_cam_schema.GetProjectionAttr().Set(ego_projection)
        # Depth of field (also applies on Carter chassis-mounted ego).
        ego_cam_schema.GetFStopAttr().Set(ego_f_stop)
        ego_cam_schema.GetFocusDistanceAttr().Set(ego_focus_distance_stage)
        # Rotation: roll=90 + yaw=-90 makes the camera face Carter's +X
        # (Carter forward). pitch_deg tilts the camera up/down from level.
        ego_xformable = UsdGeom.Xformable(ego_cam_prim)
        ego_translate = ego_xformable.AddTranslateOp()
        ego_rotate = ego_xformable.AddRotateXYZOp()
        ego_translate.Set((mount_forward, mount_lateral, mount_height))
        ego_rotate.Set((90.0 + mount_pitch_deg, 0.0, -90.0))
        print(
            f"Carter ego camera: {camera_prim_path} fov={fov_deg:.1f}° "
            f"mount=(fwd={mount_forward:.2f}, lat={mount_lateral:.2f}, h={mount_height:.2f}, "
            f"pitch={mount_pitch_deg:.1f}°)"
        )
    else:
        # Ghost camera rig (Stages 1-5).
        camera_prim_path = "/World/EgoCamera"
        camera_prim = stage.DefinePrim(camera_prim_path, "Camera")
        cam_schema = UsdGeom.Camera(camera_prim)
        cam_schema.GetHorizontalApertureAttr().Set(horizontal_aperture)
        cam_schema.GetFocalLengthAttr().Set(focal_length)
        cam_schema.GetClippingRangeAttr().Set(Gf.Vec2f(float(clipping[0]), float(clipping[1])))
        cam_schema.GetProjectionAttr().Set(ego_projection)
        # Shutter open/close are in TIME CODES relative to current frame.
        # Convert shutter_close_fraction (fraction of frame interval) into
        # time codes: frame_dt_sec * tcps * fraction.
        ghost_frame_dt = capture_dt if capture_dt > 0 else physics_dt
        shutter_close_tc = shutter_close_fraction * ghost_frame_dt * stage_tcps
        cam_schema.GetShutterOpenAttr().Set(0.0)
        cam_schema.GetShutterCloseAttr().Set(shutter_close_tc)
        # Depth of field. fStop=0 = pinhole (sharp everywhere).
        cam_schema.GetFStopAttr().Set(ego_f_stop)
        cam_schema.GetFocusDistanceAttr().Set(ego_focus_distance_stage)
        print(f"Camera shutter: close_tc={shutter_close_tc:.3f} (frame_dt={ghost_frame_dt:.4f}s × tcps={stage_tcps:.1f} × frac={shutter_close_fraction})")

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

    # Warmup: fire the randomize_scene trigger before writer is attached
    # so the scene settles (objects placed, textures loaded) before capture.
    rep.orchestrator.preview()
    simulation_app.update()
    rep.utils.send_og_event(event_name="randomize_scene")
    simulation_app.update()
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
    simulation_app.update()
    simulation_app.update()

    # ── Episode plan: one Isaac session, many (seed → scene → capture) episodes ──
    # --seeds "s1 s2 ..." renders every seed in THIS process: per episode the
    # randomize_scene graph re-rolls the scene, the occupancy map + path are
    # rebuilt, and writers point at <out_root>/<config-stem>_seed<S>/ — the
    # same dir layout the one-process-per-seed wrapper produced. Legacy
    # single-seed invocations run exactly one episode into --data_dir.
    seeds_mode = getattr(args, "seeds", None) is not None
    capture_mode = str(cfg.get("run", {}).get("capture_mode", "trajectory")).lower()
    if seeds_mode:
        ep_seed_list = [int(t) for t in str(args.seeds).split()]
        if not ep_seed_list:
            raise ValueError("--seeds given but empty")
        if getattr(args, "out_root", None) is None:
            raise ValueError("--seeds episode mode requires --out_root")
        ep_out_root = Path(args.out_root).resolve()
        cfg_stem = Path(args.config).stem
        if agent_type != "camera_rig":
            raise ValueError("--seeds episode mode supports agent.type=camera_rig only")
        if bool(capture_cfg.get("video", False)):
            raise ValueError("--seeds episode mode does not support capture.video")
    else:
        ep_seed_list = [seed]
    if capture_mode == "random" and traj_mode != "occupancy_path":
        raise ValueError("capture_mode=random requires trajectory.mode=occupancy_path")

    # waypoint_list-mode poses were precomputed above; episodes restore them.
    _wl_waypoints, _wl_poses = waypoints, poses

    def _run_episode(
        try_seed: int, label_seed: int, ep_dir: Path, ep_paths: dict, ep_events: Path
    ) -> int:
        """One dataset episode (seed → scene re-roll → plan → capture → manifest)."""
        waypoints, poses = _wl_waypoints, _wl_poses
        random.seed(try_seed)
        if hasattr(rep, "set_global_seed"):
            rep.set_global_seed(try_seed)

        # Re-roll the scene: fire the randomize_scene trigger and settle. Writers
        # from the previous episode are already detached, so nothing is captured.
        rep.utils.send_og_event(event_name="randomize_scene")
        simulation_app.update()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        simulation_app.update()
        simulation_app.update()

        # Per-episode camera intrinsics + jitter phases (same sampling as launch).
        if focal_length_override is not None:
            focal_length = float(focal_length_override)
            fov_deg = 2.0 * math.degrees(math.atan2(horizontal_aperture / 2.0, focal_length))
        else:
            fov_deg = random.gauss(fov_mean, fov_std) if fov_std > 0 else fov_mean
            fov_deg = max(10.0, min(170.0, fov_deg))
            focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path)).GetFocalLengthAttr().Set(focal_length)
        pitch_jit_phase = random.uniform(0.0, 2.0 * math.pi)
        roll_jit_phase = random.uniform(0.0, 2.0 * math.pi)
        yaw_jit_phase = random.uniform(0.0, 2.0 * math.pi)
        lat_jit_phase = random.uniform(0.0, 2.0 * math.pi)
        vert_jit_phase = random.uniform(0.0, 2.0 * math.pi)
        append_event(ep_events, "stage5_camera_sampled", {
            "fov_deg": round(fov_deg, 3),
            "focal_length_mm": round(focal_length, 4),
            "resolution": [width, height],
        })

        ep_motion_blur = motion_blur_enabled and capture_mode != "random"
        _ep_writers: list = []
        try:
            ego_root = ep_paths["ego"]
            if bool(capture_cfg.get("rgb", True)):
                w = rep.WriterRegistry.get("BasicWriter")
                w.initialize(output_dir=str(ego_root / "rgb"), rgb=True)
                w.attach(rp_ego)
                _ep_writers.append(w)
            if bool(capture_cfg.get("bounding_box_2d_tight", False)):
                w = rep.WriterRegistry.get("BasicWriter")
                w.initialize(output_dir=str(ego_root / "bounding_box_2d_tight"), bounding_box_2d_tight=True)
                w.attach(rp_ego)
                _ep_writers.append(w)
            if bool(capture_cfg.get("semantic_segmentation", False)):
                w = rep.WriterRegistry.get("BasicWriter")
                w.initialize(output_dir=str(ego_root / "semantic_segmentation"), semantic_segmentation=True)
                w.attach(rp_ego)
                _ep_writers.append(w)
            if bool(capture_cfg.get("depth", False)):
                w = rep.WriterRegistry.get("BasicWriter")
                w.initialize(output_dir=str(ego_root / "depth"), distance_to_camera=True)
                w.attach(rp_ego)
                _ep_writers.append(w)

            # ── CosmosWriter (optional video) ─────────────────────────────────────────
            # Runs alongside BasicWriter on the same render product. One trajectory =
            # one clip. Produces clip_0000/{rgb,depth,segmentation,shaded_seg,edges}.mp4
            # under video/ after on_final_frame() is called.
            cosmos_writer = None
            if bool(capture_cfg.get("video", False)):
                video_dir = ep_dir / "video"
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
                chase_writer.initialize(output_dir=str(ep_paths["chase"]), rgb=True)
                chase_writer.attach(rp_chase)
                _ep_writers.append(chase_writer)

            append_event(ep_events, "stage5_capture_start", {
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

            poses_path = ep_paths["trajectory"] / "poses.jsonl"
            poses_path.write_text("")

            # ── Distractor props dynamic (cones, barrels, trash) ─────────────────────
            # Path planner can't see things below z=1.6m (cones ~0.7m, barrels ~0.9m),
            # so Carter may run into them. Authored defaults for these USDs are
            # visual-only (no rigid body), which would make Carter stop dead. Give
            # them a light rigid body with gravity ON so they sit on the floor as
            # placed, but get shoved aside when the heavier robot bumps into them.
            # Actor classes (palletjack, forklift, pallet) keep their authored physics.
            if agent_type == "carter":
                from pxr import UsdPhysics
                actor_classes = {"palletjack", "forklift", "pallet"}
                distractor_count = 0
                for prim in stage.Traverse():
                    path_str = str(prim.GetPath())
                    if not path_str.startswith("/Replicator"):
                        continue
                    if not prim.HasAuthoredReferences():
                        continue
                    is_actor = False
                    for prop in prim.GetProperties():
                        if not Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                            continue
                        inst = prop.SplitName()[1]
                        sem = Semantics.SemanticsAPI.Get(prim, inst)
                        if sem.GetSemanticDataAttr().Get() in actor_classes:
                            is_actor = True
                            break
                    if is_actor:
                        continue
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        continue  # respect authored physics
                    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
                    rb.CreateRigidBodyEnabledAttr(True, False)
                    mass = UsdPhysics.MassAPI.Apply(prim)
                    mass.CreateMassAttr(0.3)  # light enough that Carter shoves it
                    distractor_count += 1
                print(f"Distractors made dynamic: {distractor_count} (mass=0.3kg)")

            # CosmosWriter requires the timeline to be playing and pause_timeline=False.
            # The ghost camera is driven by explicit USD ops, so physics running is harmless.
            timeline = omni.timeline.get_timeline_interface()
            if not timeline.is_playing():
                timeline.play()
            # Let BehaviorScripts activate (on_play fires on timeline.play) so their
            # command lists are populated before we start capturing frames.
            if spawned_characters and movement_enabled:
                for _ in range(10):
                    simulation_app.update()

            # Motion-blur time samples must straddle the renderer's CURRENT timeline
            # time, which keeps advancing across in-process episodes.
            ep_t0_sec = float(timeline.get_current_time()) if seeds_mode else 0.0

            # ── Occupancy path (computed here so the settled scene is in collision) ──
            if traj_mode == "occupancy_path":
                nav_dir = ep_dir / "nav"
                if capture_mode == "random":
                    # Random-frame mode: num_frames INDEPENDENT freespace poses (each
                    # frame its own viewpoint); pseudo 2-pt waypoints give the capture
                    # loop a zero heading so yaw_rel carries the absolute sampled yaw.
                    _rand_pts = _build_occupancy_waypoints(
                        cfg, simulation_app, nav_dir, camera_z, try_seed,
                        sample_poses_n=num_frames,
                    )
                    waypoints = [[0.0, 0.0, camera_z, 90.0, 0.0, 0.0],
                                 [0.0, 1.0, camera_z, 90.0, 0.0, 0.0]]
                else:
                    waypoints = _build_occupancy_waypoints(cfg, simulation_app, nav_dir, camera_z, try_seed)
                # DEBUG sanity knob: rigidly translate the planned path in world XY before
                # rendering (occupancy map/overlay are unshifted). Used to test whether the
                # nav map is offset from the rendered warehouse geometry — if shifting the
                # camera moves it from "outside the building" to a proper interior view,
                # the map<->world frames are misaligned. Set env DEBUG_WAYPOINT_SHIFT="dx,dy".
                _dbg_shift = os.environ.get("DEBUG_WAYPOINT_SHIFT")
                if _dbg_shift:
                    _sx, _sy = (float(v) for v in _dbg_shift.split(","))
                    for _wp in waypoints:
                        _wp[0] += _sx
                        _wp[1] += _sy
                    print(f"[DEBUG] shifted {len(waypoints)} waypoints by ({_sx:+.1f},{_sy:+.1f}) m")
                # Spread frames along the path; ping-pong (forward+reverse) a short path so
                # a small roam box / relaxed path fills the frame budget with new views
                # instead of near-duplicate frames, rotating smoothly in place at each
                # turnaround (no 180° heading snap). Reassign waypoints to the effective
                # (reflected) route so per-frame heading stays aligned.
                if capture_mode == "random":
                    poses = [(q[0], q[1], q[2], 90.0, 0.0, q[5], 0) for q in _rand_pts]
                else:
                    poses, waypoints = _plan_traversal(waypoints, num_frames, traj_cfg)
                append_event(ep_events, "stage5_occupancy_path_sampled", {
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

            # ── Physics-driven Carter setup ──────────────────────────────────────────
            # Reset the parent Xform to identity so we can teleport Carter purely via
            # the physics Articulation API. Then drive with wheel velocity targets —
            # PhysX handles collisions naturally: walls stop the robot, light props
            # get pushed, heavy authored assets resist.
            carter_art = None
            carter_left_idx = None
            carter_right_idx = None
            carter_target_idx = 1
            carter_finished = False
            if agent_type == "carter":
                import numpy as np
                from omni.isaac.core.articulations import Articulation
                from omni.isaac.core.utils.types import ArticulationAction

                # Park /World/Carter parent at identity; Articulation owns the pose now.
                robot_translate_op.Set((0.0, 0.0, 0.0))
                robot_rotate_op.Set((0.0, 0.0, 0.0))
                for _ in range(3):
                    simulation_app.update()

                art_path = f"{robot_prim_path}/{robot_body_subpath}"
                carter_art = Articulation(prim_path=art_path, name="carter")
                carter_art.initialize()

                # Diagnose DOFs so we can pick the right wheel joints.
                dof_names = list(carter_art.dof_names) if hasattr(carter_art, "dof_names") else []
                print(f"Carter DOFs ({len(dof_names)}): {dof_names}")
                carter_left_idx = carter_art.get_dof_index("joint_wheel_left")
                carter_right_idx = carter_art.get_dof_index("joint_wheel_right")
                print(f"Carter articulation initialized: wheel_left={carter_left_idx} wheel_right={carter_right_idx}")

                # Teleport Carter to the first waypoint facing the next one.
                wp0 = waypoints[0]
                wp1 = waypoints[1]
                init_yaw_rad = math.atan2(wp1[1] - wp0[1], wp1[0] - wp0[0])
                half = init_yaw_rad / 2.0
                init_orient = np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
                carter_art.set_world_pose(
                    position=np.array([wp0[0], wp0[1], 0.05], dtype=np.float64),
                    orientation=init_orient,
                )
                carter_art.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                carter_art.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))

                # Control gains
                carter_max_speed = float(agent_cfg.get("speed_mps", 1.0))
                carter_max_omega = math.radians(float(agent_cfg.get("turn_rate_dps", 60.0)))
                carter_wheel_radius = 0.14
                carter_wheel_base = 0.413
                carter_end_threshold_m = 0.5
                # capture_dt set earlier (right after stage_tcps). Each captured frame
                # advances ~capture_dt seconds of sim time.
                print(
                    f"Carter physics-driven: max_speed={carter_max_speed:.2f} m/s "
                    f"max_omega={math.degrees(carter_max_omega):.0f} dps "
                    f"capture_dt={capture_dt:.2f}s"
                )

            for frame_index in range(num_frames):
                x_path, y_path, z_path, roll, pitch, yaw_rel, seg_idx = poses[frame_index]
                seg = int(seg_idx)
                dx_seg = waypoints[seg + 1][0] - waypoints[seg][0]
                dy_seg = waypoints[seg + 1][1] - waypoints[seg][1]
                heading_yaw_cam = math.degrees(math.atan2(-dx_seg, dy_seg))
                yaw = heading_yaw_cam + yaw_rel

                if agent_type == "carter":
                    # Read Carter's actual world pose
                    pos_arr, orient_arr = carter_art.get_world_pose()
                    cx = float(pos_arr[0])
                    cy = float(pos_arr[1])
                    cz = float(pos_arr[2])
                    wq, xq, yq, zq = (
                        float(orient_arr[0]),
                        float(orient_arr[1]),
                        float(orient_arr[2]),
                        float(orient_arr[3]),
                    )
                    cyaw = math.atan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))

                    # Advance target waypoint when close enough to the current one.
                    while carter_target_idx < len(waypoints):
                        tx, ty = waypoints[carter_target_idx][0], waypoints[carter_target_idx][1]
                        if math.hypot(tx - cx, ty - cy) > carter_end_threshold_m:
                            break
                        carter_target_idx += 1

                    # Pure-pursuit-lite: head toward current target, turn in place
                    # when off-heading, full speed when roughly aligned.
                    if carter_target_idx >= len(waypoints):
                        v_lin = 0.0
                        v_ang = 0.0
                        carter_finished = True
                    else:
                        tx, ty = waypoints[carter_target_idx][0], waypoints[carter_target_idx][1]
                        target_heading = math.atan2(ty - cy, tx - cx)
                        heading_err = (target_heading - cyaw + math.pi) % (2.0 * math.pi) - math.pi
                        if abs(heading_err) > math.radians(45):
                            v_lin = 0.0
                        else:
                            v_lin = carter_max_speed
                        v_ang = max(-carter_max_omega, min(carter_max_omega, 2.0 * heading_err))

                    # Differential drive → wheel angular velocities (rad/s)
                    v_left = (v_lin - v_ang * carter_wheel_base / 2.0) / carter_wheel_radius
                    v_right = (v_lin + v_ang * carter_wheel_base / 2.0) / carter_wheel_radius
                    carter_art.apply_action(
                        ArticulationAction(
                            joint_velocities=np.array([v_left, v_right], dtype=np.float32),
                            joint_indices=np.array([carter_left_idx, carter_right_idx]),
                        )
                    )

                    # Carter's actual pose drives logging and chase camera.
                    x, y, z = cx, cy, cz
                    robot_yaw = math.degrees(cyaw)
                    heading_yaw_cam = robot_yaw - 90.0  # camera convention for chase
                else:
                    # Ghost camera: write pose as two time samples per frame —
                    # shutterOpen (t = frame_index) and shutterClose (t = frame_index +
                    # shutter_close_fraction). The renderer integrates between them
                    # to produce motion blur from the camera sweep. Both samples
                    # carry their own jitter computed from sim-time so frequency
                    # stays consistent regardless of capture_dt.

                    def _ghost_pose_at(frac_idx: float):
                        # Linear interpolation along the precomputed waypoint poses.
                        # Falls back to endpoints outside [0, num_frames-1].
                        if frac_idx <= 0.0:
                            return poses[0]
                        if frac_idx >= num_frames - 1:
                            return poses[num_frames - 1]
                        i0 = int(frac_idx)
                        a = frac_idx - i0
                        p0 = poses[i0]
                        p1 = poses[i0 + 1]
                        return tuple(p0[k] + a * (p1[k] - p0[k]) for k in range(len(p0)))

                    def _ghost_rot_with_jitter(roll_in: float, pitch_in: float, t_sec: float):
                        pj = pitch_jit_amp * math.sin(2.0 * math.pi * pitch_jit_hz * t_sec + pitch_jit_phase)
                        rj = roll_jit_amp * math.sin(2.0 * math.pi * roll_jit_hz * t_sec + roll_jit_phase)
                        # Pose index-3 ("roll_in") carries the base 90° X-rotation that tilts the
                        # camera from looking-down to looking-horizontal, so the X-rotation channel
                        # is the one that physically pitches the view up/down — the camera PITCH
                        # knob + pitch-jitter belong there (negative = look down, matching the
                        # chase cam's (90 + tilt_down, 0, yaw) and configs like exp12 pitch=-18).
                        # Index-4 ("pitch_in", base 0) feeds the Y-rotation channel, which banks/
                        # ROLLs the image. Previously pitch_deg was fed to Y and rendered as roll
                        # (exp12 "steep downward" showed a horizontal tilt) — see IMAGE_REVIEW #5.
                        x_rot = roll_in + ego_pitch_static + pj    # base horizontal + look up/down
                        y_rot = pitch_in + ego_roll_static + rj    # bank / roll
                        return (x_rot, y_rot)

                    def _ghost_pos_with_jitter(x_in: float, y_in: float, z_in: float, t_sec: float):
                        # Lateral jitter is perpendicular to the camera's heading
                        # direction so a leftward "sway" stays leftward regardless
                        # of which way the camera is travelling.
                        lat = lat_jit_amp * math.sin(2.0 * math.pi * lat_jit_hz * t_sec + lat_jit_phase)
                        vert = vert_jit_amp * math.sin(2.0 * math.pi * vert_jit_hz * t_sec + vert_jit_phase)
                        heading_rad = math.atan2(dy_seg, dx_seg)
                        # Perpendicular to heading (left = +90°)
                        left_x = -math.sin(heading_rad)
                        left_y = math.cos(heading_rad)
                        return (x_in + lat * left_x, y_in + lat * left_y, z_in + vert)

                    frame_dt = capture_dt if capture_dt > 0 else physics_dt
                    t_open_idx = float(frame_index)
                    t_close_idx = float(frame_index) + (
                        shutter_close_fraction if ep_motion_blur else 0.0
                    )
                    t_open_sec = t_open_idx * frame_dt
                    t_close_sec = t_close_idx * frame_dt
                    # Time codes (used as USD time-sample keys) — must match the
                    # renderer's currentTime = sim_time_seconds * stage_tcps.
                    t_open_tc = (ep_t0_sec + t_open_sec) * stage_tcps
                    t_close_tc = (ep_t0_sec + t_close_sec) * stage_tcps

                    xo, yo, zo, ro, po_, yro, _ = _ghost_pose_at(t_open_idx)
                    xc, yc, zc, rc, pc, yrc, _ = _ghost_pose_at(t_close_idx)

                    def _yaw_jit(t_sec: float) -> float:
                        return yaw_jit_amp * math.sin(2.0 * math.pi * yaw_jit_hz * t_sec + yaw_jit_phase)

                    yawo = heading_yaw_cam + yro + _yaw_jit(t_open_sec)
                    yawc = heading_yaw_cam + yrc + _yaw_jit(t_close_sec)
                    ro_t, po_t = _ghost_rot_with_jitter(ro, po_, t_open_sec)
                    rc_t, pc_t = _ghost_rot_with_jitter(rc, pc, t_close_sec)
                    xo, yo, zo = _ghost_pos_with_jitter(xo, yo, zo, t_open_sec)
                    xc, yc, zc = _ghost_pos_with_jitter(xc, yc, zc, t_close_sec)

                    if ep_motion_blur:
                        translate_op.Set((xo, yo, zo), time=Usd.TimeCode(t_open_tc))
                        translate_op.Set((xc, yc, zc), time=Usd.TimeCode(t_close_tc))
                        rotate_op.Set((ro_t, po_t, yawo), time=Usd.TimeCode(t_open_tc))
                        rotate_op.Set((rc_t, pc_t, yawc), time=Usd.TimeCode(t_close_tc))
                    else:
                        translate_op.Set((xo, yo, zo))
                        rotate_op.Set((ro_t, po_t, yawo))

                    # Logged "current" pose is the shutter-open sample.
                    roll = ro_t
                    pitch = po_t
                    yaw = yawo
                    x, y, z = xo, yo, zo

                if chase_enabled and chase_translate_op is not None:
                    if agent_type == "carter":
                        chx = x - math.cos(cyaw) * chase_dist_m
                        chy = y - math.sin(cyaw) * chase_dist_m
                        chz = chase_height_m
                    else:
                        seg_len = math.sqrt(dx_seg * dx_seg + dy_seg * dy_seg)
                        ux, uy = (dx_seg / seg_len, dy_seg / seg_len) if seg_len > 1e-9 else (0.0, 1.0)
                        chx = x - ux * chase_dist_m
                        chy = y - uy * chase_dist_m
                        chz = z + chase_height_m
                    chase_translate_op.Set((chx, chy, chz))
                    chase_rotate_op.Set((90.0 + tilt_down_deg, 0.0, heading_yaw_cam))

                simulation_app.update()
                # Advance time each capture step. Carter uses capture_dt for control
                # integration; ghost uses the frame interval so the renderer sees
                # timeline motion (needed to consume time-sampled xforms for blur).
                step_dt = capture_dt
                rep.orchestrator.step(rt_subframes=4, delta_time=step_dt, pause_timeline=False)

                pose_record = {
                    "agent_type": agent_type,
                    "camera_prim": camera_prim_path,
                    "camera_pos": [x, y, z],
                    "camera_rot_euler_deg": [roll, pitch, yaw],
                    "heading_yaw_deg": round(heading_yaw_cam, 4),
                    "relative_yaw_deg": round(yaw_rel, 4),
                    "fov_deg": round(fov_deg, 3),
                    "frame": frame_index,
                    "sim_time": round(frame_index * (capture_dt if agent_type == "carter" else physics_dt), 6),
                    "waypoint_segment": seg,
                }
                if agent_type == "carter":
                    pose_record["agent_prim"] = robot_prim_path
                    pose_record["agent_pos"] = [x, y, z]
                    pose_record["agent_yaw_deg"] = round(robot_yaw, 4)
                    pose_record["target_idx"] = carter_target_idx
                    pose_record["finished"] = carter_finished
                # Character world positions — for Stage 7b-2 movement validation.
                # Query via ag.get_character() world transform (the source of truth
                # the anim graph writes; USD xformOp is untouched by the anim system).
                if spawned_characters:
                    char_pos = []
                    try:
                        import omni.anim.graph.core as _ag
                        import carb as _carb
                        for entry in spawned_characters:
                            skel_root_path = None
                            cprim = stage.GetPrimAtPath(entry["prim"])
                            for _p in Usd.PrimRange(cprim):
                                if _p.GetTypeName() == "SkelRoot":
                                    skel_root_path = str(_p.GetPrimPath())
                                    break
                            if skel_root_path is None:
                                continue
                            ch = _ag.get_character(skel_root_path)
                            if ch is None:
                                continue
                            pos = _carb.Float3(0, 0, 0)
                            rot = _carb.Float4(0, 0, 0, 0)
                            ch.get_world_transform(pos, rot)
                            char_pos.append({
                                "name": entry["prim"].rsplit("/", 1)[-1],
                                "pos": [round(float(pos.x), 3), round(float(pos.y), 3), round(float(pos.z), 3)],
                            })
                    except Exception:
                        pass
                    pose_record["characters"] = char_pos
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
                video_files = sorted(str(p.relative_to(ep_dir)) for p in (ep_dir / "video").rglob("*.mp4"))
                print(f"Video files written: {video_files}")

            image_count = len(list((ep_paths["ego"] / "rgb").glob("*.png")))

            # Motion-blur diagnostic: dump the camera's authored attrs + a couple of
            # time samples to a file so we can verify the renderer was given what
            # we think we were giving it.
            if agent_type == "camera_rig":
                try:
                    diag_path = ep_dir / "trajectory" / "blur_diag.txt"
                    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_prim_path))
                    t_attr = cam.GetPrim().GetAttribute("xformOp:translate")
                    samples = t_attr.GetTimeSamples() if t_attr.HasAuthoredValue() else []
                    with diag_path.open("w") as f:
                        f.write(f"camera_prim: {camera_prim_path}\n")
                        f.write(f"shutter_open: {cam.GetShutterOpenAttr().Get()}\n")
                        f.write(f"shutter_close: {cam.GetShutterCloseAttr().Get()}\n")
                        f.write(f"projection: {cam.GetProjectionAttr().Get()}\n")
                        f.write(f"focal_length: {cam.GetFocalLengthAttr().Get()}\n")
                        f.write(f"fstop: {cam.GetFStopAttr().Get()}\n")
                        f.write(f"focus_distance: {cam.GetFocusDistanceAttr().Get()}\n")
                        f.write(f"stage_tcps: {stage_tcps}\n")
                        f.write(f"translate time samples ({len(samples)}):\n")
                        for s in samples[:8]:
                            f.write(f"  tc={s} val={t_attr.Get(s)}\n")
                        if len(samples) > 8:
                            f.write(f"  ...({len(samples)-8} more)\n")
                except Exception as exc:
                    print(f"blur_diag write failed: {exc}")

            # ── Fisheye post-render ──────────────────────────────────────────────────
            # USD has no spherical projection, so we render perspective and remap to
            # fisheye via OpenCV. Output sits next to the original perspective frames
            # in Camera/rgb_fisheye/. Bbox labels for OD apply to the perspective
            # frames; if the downstream task needs fisheye labels, they'd be projected
            # through the same remap.
            fisheye_cfg = ego_cam_cfg.get("fisheye") or {}
            if fisheye_cfg.get("enabled", False) and image_count > 0:
                import cv2
                import numpy as np

                rgb_dir = ep_paths["ego"] / "rgb"
                fish_dir = ep_paths["ego"] / "rgb_fisheye"
                fish_dir.mkdir(parents=True, exist_ok=True)

                # Perspective camera intrinsics (pixels)
                fx_p = focal_length * width / horizontal_aperture
                fy_p = fx_p  # assume square pixels
                cx_p = width / 2.0
                cy_p = height / 2.0
                K_persp = np.array([[fx_p, 0, cx_p], [0, fy_p, cy_p], [0, 0, 1]], dtype=np.float64)

                # Fisheye output uses the same focal length and image size; only the
                # distortion differs. Distortion coeffs in OpenCV fisheye order.
                D = np.array([
                    float(fisheye_cfg.get("k1", 0.0)),
                    float(fisheye_cfg.get("k2", 0.0)),
                    float(fisheye_cfg.get("k3", 0.0)),
                    float(fisheye_cfg.get("k4", 0.0)),
                ], dtype=np.float64)

                # Build the remap once: for every output (fisheye) pixel, undistort
                # to a 3D ray, then project that ray through the perspective camera.
                gx, gy = np.meshgrid(np.arange(width), np.arange(height))
                pts_fish = np.stack([gx, gy], axis=-1).reshape(-1, 1, 2).astype(np.float64)
                pts_undist = cv2.fisheye.undistortPoints(pts_fish, K_persp, D)
                x_norm = pts_undist[..., 0].reshape(-1)
                y_norm = pts_undist[..., 1].reshape(-1)
                u_persp = K_persp[0, 0] * x_norm + K_persp[0, 2]
                v_persp = K_persp[1, 1] * y_norm + K_persp[1, 2]
                map_x = u_persp.reshape(height, width).astype(np.float32)
                map_y = v_persp.reshape(height, width).astype(np.float32)

                rgb_files = sorted(rgb_dir.glob("*.png"))
                for rgb_path in rgb_files:
                    img = cv2.imread(str(rgb_path))
                    if img is None:
                        continue
                    warped = cv2.remap(
                        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                    )
                    cv2.imwrite(str(fish_dir / rgb_path.name), warped)
                print(f"Fisheye post-render: wrote {len(rgb_files)} frames to {fish_dir}")

            # ── Sensor-noise + image-augmentation post-render ─────────────────────────
            # USD/RTX renders are clean; real deployment cameras have shot/read noise,
            # JPEG compression, and exposure/colour drift. Reuse the pre-trajectory
            # (mean_std) implementation in palletjack_sdg.utils.image_effects verbatim
            # so noise semantics match the historical datasets exactly:
            #   cameras.ego.dataset_noise.mode: no-noise|gaussian|shot|jpeg|
            #                                    gaussian_jpeg|shot_jpeg
            # Params sampled per-frame (each frame an independent sensor draw, as before
            # — physically correct for shot/read noise). Applied in place to Camera/rgb;
            # bounding boxes are pixel-position invariant so labels stay valid.
            from palletjack_sdg.utils.image_effects import (
                apply_post_write_effects_to_saved_rgb,
                get_dataset_noise_cfg,
                resolve_image_augmentation_cfg,
            )

            noise_cfg = get_dataset_noise_cfg(ego_cam_cfg)
            aug_cfg = resolve_image_augmentation_cfg(
                ego_cam_cfg.get("image_augmentation"), ego_cam_cfg
            )
            noise_active = str(noise_cfg.get("mode", "no-noise")) != "no-noise"
            if (noise_active or aug_cfg.get("enabled", False)) and image_count > 0:
                # Seed noise deterministically off the episode seed unless overridden,
                # so seed42 / seed123 reruns differ but are each reproducible.
                if not int(noise_cfg.get("seed", 0) or 0):
                    noise_cfg["seed"] = try_seed
                apply_post_write_effects_to_saved_rgb(str(ep_dir), noise_cfg, aug_cfg)
                append_event(ep_events, "stage5_dataset_noise_applied", {
                    "mode": noise_cfg.get("mode"),
                    "image_augmentation": bool(aug_cfg.get("enabled", False)),
                    "frames": image_count,
                })

            write_manifest(
                output_dir=ep_dir,
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
                ep_events,
                "stage5_complete",
                {"image_count": image_count, "num_frames": num_frames, "video_files": video_files},
            )
            return image_count
        finally:
            for _w in _ep_writers:
                try:
                    _w.detach()
                except Exception:
                    pass

    # ── Episode driver ────────────────────────────────────────────────────────
    max_ep_retries = int(getattr(args, "max_seed_retries", 4) or 0) if seeds_mode else 0
    ep_ok: list = []
    ep_failed: list = []
    for label_seed in ep_seed_list:
        if seeds_mode:
            ep_dir = ep_out_root / f"{cfg_stem}_seed{label_seed}"
            ep_paths = prepare_output_tree(ep_dir, chase_enabled=chase_enabled)
            ep_events = ep_paths["trajectory"] / "events.jsonl"
            ep_events.write_text("")
        else:
            ep_dir, ep_paths, ep_events = output_dir, paths, events_path
        last_exc = None
        n_img = -1
        for attempt in range(max_ep_retries + 1):
            try_seed = label_seed + attempt * 1000
            if attempt:
                print(f"[episode seed {label_seed}] retry {attempt}/{max_ep_retries} "
                      f"with seed {try_seed}", flush=True)
            cfg.setdefault("simulation", {})["seed"] = try_seed
            if seeds_mode:
                write_run_config(ep_dir, cfg, config_path)
            try:
                n_img = _run_episode(try_seed, label_seed, ep_dir, ep_paths, ep_events)
                break
            except Exception as exc:
                # Layout failures (no navigable freespace / no valid path) are a
                # property of this random layout — resample with a fresh seed,
                # keeping the original seed label for the output dir.
                last_exc = exc
                print(f"[episode seed {label_seed}] attempt {attempt + 1} failed: {exc}", flush=True)
        if n_img >= 0:
            ep_ok.append((label_seed, n_img))
            print(f"[episode seed {label_seed}] OK ({n_img} images)", flush=True)
        else:
            ep_failed.append(label_seed)
            if not seeds_mode:
                raise last_exc if last_exc is not None else RuntimeError("episode failed")
    if seeds_mode:
        print(f"Episodes: {len(ep_ok)}/{len(ep_seed_list)} ok"
              + (f", FAILED seeds: {ep_failed}" if ep_failed else ""), flush=True)
    simulation_app.close()
    if ep_failed:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print(f"Ignoring unknown arguments: {' '.join(unknown_args)}")
    run_stage4(args)


if __name__ == "__main__":
    main()

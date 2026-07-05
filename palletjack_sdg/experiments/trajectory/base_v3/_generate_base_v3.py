"""Generate the base_v3 trajectory experiment configs.

base_v3 is the "tight-framing / de-correlated" proposal, designed from the
v1-vs-v2 finding that trajectory synth saturates ~0.21 mAP@50:95 because its
boxes are tiny (median ~0.0006 rel-area, 88% <1%) and its frames are highly
correlated (consecutive views of one scene). base_v3 attacks both:

  * BIGGER objects  -> lower camera FOV (52-60 vs base_v1's 68) + more hero
    instances (palletjacks count_per_model 3-4) + slightly tighter hero spread.
  * DE-CORRELATED   -> run at 10 frames/seed over a LONGER planned path
    (trajectory.min_path_m 18 vs 6) so the 10 arc-length-spaced frames land far
    apart. NOTE: fps/capture_dt is NOT a spacing knob for camera_rig — ego frames
    are arc-length-interpolated over the path, so spacing = path_length /
    (num_frames-1), independent of fps. Path length + frame count do the work.
  * START ANYWHERE  -> occupancy_path already samples the start uniformly across
    free space each seed; base_v3 widens bounds_xy and trims boundary_margin so
    the start (and path) can reach the whole scene, and sets start_mode: random
    to make the intent explicit.
  * DIVERSITY       -> 10 configs across environments / lighting / height / FOV,
    run at 13 seeds x 10 frames = 130 frames each (1300 total).

Regenerate:  python _generate_base_v3.py
"""

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "base_v1" / "exp01_operator_walk_steady.yaml"

# Overrides applied to EVERY base_v3 config (deep-merged onto the exp01 surface).
COMMON = {
    "run": {"num_frames": 10},
    "agent": {
        "start_mode": "random",   # occupancy_path samples start uniformly in free space
        "speed_mps": 1.5,
        # capture_dt intentionally NOT overridden: fps does not affect camera_rig
        # frame spacing (ego poses are arc-length-interpolated over the path), so it
        # is not a diversity knob here. Uses the exp01 template default.
    },
    "trajectory": {
        # 12m long enough that 10 arc-spaced frames land ~1.3m apart (>=3.5x v2's
        # ~0.38m), but short enough to be reliably plannable — 18m failed the
        # occupancy planner for ~5% of (config, seed) pairs, esp. the smaller
        # plain `warehouse` and tight closeup configs.
        "min_path_m": 12.0,
        "occupancy": {
            "min_path_m": 12.0,
            "boundary_margin_m": 1.5,  # let the start reach nearer the walls
        },
    },
    # Target objects scattered UNIFORMLY across the navigable floor (see the
    # generator's _scatter_position) so open/near-wall areas have targets too.
    # This is the real empties fix (not FOV): a roaming, start-anywhere camera
    # keeps objects in view wherever it points. Counts raised so ~20 hero
    # objects populate the ~26x28m floor.
    "palletjacks": {"count_per_model": 7, "scatter": "uniform"},   # x3 models = 21
    "forklifts": {"count_per_model": 3, "scatter": "uniform"},     # x1 model  = 3
    "pallets": {"count_per_model": 6, "scatter": "uniform"},       # x2 models = 12
}

# Per-experiment overrides. Each dict is deep-merged over COMMON over the template.
EXPERIMENTS = {
    "exp01_tight_eye_level_bright": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 55.0, "height_m": 1.55, "pitch_deg": -3.0}},
        "lighting": {"intensity_mean": 130000.0},
    },
    "exp02_low_pov_closeup": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 56.0, "height_m": 0.95, "pitch_deg": 4.0}},
    },
    "exp03_forklift_yard_tight": {
        "environment": {"name": "warehouse_with_forklifts"},
        "trajectory": {"bounds_xy": [-12.0, 12.0, -12.0, 14.0]},
        "cameras": {"ego": {"fov_mean": 57.0, "height_m": 1.4, "pitch_deg": -3.0}},
        "forklifts": {"count_per_model": 2},
    },
    "exp04_dim_shift_tight": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 54.0, "height_m": 1.45, "pitch_deg": -3.0}},
        "lighting": {"intensity_mean": 55000.0, "intensity_std": 15000.0},
    },
    "exp05_multishelf_aisle_tight": {
        "environment": {"name": "warehouse_multiple_shelves"},
        "trajectory": {"bounds_xy": [-12.0, 12.0, -12.0, 14.0]},
        "cameras": {"ego": {"fov_mean": 56.0, "height_m": 1.5, "pitch_deg": -3.0}},
    },
    "exp06_bright_dense_mixed": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 53.0, "height_m": 1.5, "pitch_deg": -3.0}},
        "palletjacks": {"count_per_model": 4},
        "pallets": {"count_per_model": 3},
        "lighting": {"intensity_mean": 130000.0},
    },
    "exp07_mid_fov_survey": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 60.0, "height_m": 1.6, "pitch_deg": -5.0}},
    },
    "exp08_plain_warehouse_tight": {
        "environment": {"name": "warehouse"},
        "trajectory": {"bounds_xy": [-12.0, 12.0, -12.0, 14.0]},
        "cameras": {"ego": {"fov_mean": 55.0, "height_m": 1.3, "pitch_deg": -3.0}},
    },
    "exp09_low_angle_closeup": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        # DOF removed: f_stop 2.8 + focus 4m produced an all-bokeh, unusable frame.
        "cameras": {"ego": {"fov_mean": 58.0, "height_m": 1.1, "pitch_deg": 0.0, "f_stop": 0.0}},
    },
    "exp10_reference_tight": {
        "environment": {"name": "full_warehouse"},
        "trajectory": {"bounds_xy": [-13.0, 13.0, -13.0, 15.0]},
        "cameras": {"ego": {"fov_mean": 56.0, "height_m": 1.5, "pitch_deg": -3.0}},
    },
}

HEADER = (
    "# base_v3 — tight-framing / de-correlated trajectory proposal.\n"
    "# AUTO-GENERATED by _generate_base_v3.py — do not edit by hand; edit the\n"
    "# generator and re-run. Design rationale in that script's docstring.\n"
    "# Runtime: run at 13 seeds x 10 frames, capture_dt=5.0 (see run_base_v3_train.sh).\n"
    "# data_dir below is a placeholder; the runner overrides it per (config, seed).\n"
)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def main() -> None:
    template = yaml.safe_load(TEMPLATE.read_text())
    for name, over in EXPERIMENTS.items():
        cfg = deep_merge(deep_merge(template, COMMON), over)
        cfg["run"]["data_dir"] = f"/isaac-sim/palletjack_sdg/palletjack_data/trajectory/base_v3/{name}"
        out = HERE / f"{name}.yaml"
        out.write_text(HEADER + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")
    print(f"\n{len(EXPERIMENTS)} base_v3 configs generated.")


if __name__ == "__main__":
    main()

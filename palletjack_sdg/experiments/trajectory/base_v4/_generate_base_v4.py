"""Generate the base_v4 trajectory experiment configs.

base_v4 = base_v3 (tight-framing / de-correlated) PLUS a curated re-import of the
base_v1 variance axes that v3 collapsed. The goal: keep v3's box-size and
de-correlation wins on EVERY config, while restoring the conditions v3 dropped
(overhead / steep pitch, telephoto AND one wide FOV, night + very-dark + bright
lighting, fast + slow motion, dense/sparse density contrast).

The set has two parts:

  * TIGHT (10) — reproduced verbatim from base_v3 (same COMMON + EXPERIMENTS,
    only data_dir repointed to base_v4). See ../base_v3/_generate_base_v3.py.

  * IMPORTS (11) — each based on its REAL base_v1 source yaml (so that config's
    character survives: jitter profile, turn rate, character behaviors, and for
    dense/sparse its clutter_level + characters.count). Onto that we deep-merge:
      - IMPORT_COMMON : the v3 de-correlation + start-anywhere knobs
        (num_frames 10, start_mode random, min_path 12, occupancy margins).
        NOTE: speed / fov / height / pitch / lighting are NOT forced here —
        those ARE the variance we are importing.
      - target counts + uniform scatter : the v3 empties fix, forced on all
        imports EXCEPT dense_clutter / sparse_minimal, which keep their own
        counts so the density-contrast axis survives (they still get uniform
        scatter so their few/many objects are reachable by a roaming camera).
      - per-config policy overrides (the `over` dict): FOV capped <=75 (one
        deliberate wide config kept at 90), f_stop forced to 0 (no bokeh —
        v3 exp09 showed f2.8 produced all-bokeh unusable frames), plus the
        chosen height/pitch/lighting/speed for that axis.

Regenerate:  python _generate_base_v4.py
"""

import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # trajectory/
V1 = ROOT / "base_v1"

# Reuse the base_v3 generator as the single source of truth for the tight set.
_spec = importlib.util.spec_from_file_location(
    "gen_base_v3", ROOT / "base_v3" / "_generate_base_v3.py"
)
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
deep_merge = _v3.deep_merge

DATA_ROOT = "/isaac-sim/palletjack_sdg/palletjack_data/trajectory/base_v4"

# --- v3 de-correlation / start-anywhere knobs applied to every import ---------
# (Deliberately NO speed / fov / height / pitch / lighting — those are variance.)
IMPORT_COMMON = {
    "run": {"num_frames": 10},
    "agent": {"start_mode": "random"},
    "trajectory": {
        "min_path_m": 12.0,
        "occupancy": {"min_path_m": 12.0, "boundary_margin_m": 1.5, "buffer_m": 0.4},
    },
}

# Object placement is inset this far off the walls. The REAL fix for "objects in
# walls / outside" (IMAGE_REVIEW #2) is correct per-env interior bounds_xy (see
# ENV_BOUNDS below) — objects scatter within the true interior now, so only a
# small inset is needed to keep them off the interior walls. A large inset (the
# earlier 3.0) over-concentrated obstacles centrally and starved the occupancy
# planner (4->61 path failures), so keep this modest. _scatter_position() honors it.
SCATTER_INSET_M = 1.5

# v3 empties fix: raised counts + uniform scatter (forced on non-density imports).
FORCE_TARGETS = {
    "palletjacks": {"count_per_model": 7, "scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
    "forklifts": {"count_per_model": 3, "scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
    "pallets": {"count_per_model": 6, "scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
}
# Density imports keep their own counts but still scatter uniformly.
SCATTER_ONLY = {
    "palletjacks": {"scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
    "forklifts": {"scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
    "pallets": {"scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
}

# Distractors were left on the OLD central Gaussian when targets moved to uniform
# scatter, so a roaming tight-framed camera almost never saw them. Mirror the
# objects fix: force uniform scatter (co-locate with targets on the same floor)
# AND raise clutter_level so they're actually present in tight shots. Density
# imports keep their own clutter_level (the dense/sparse contrast) but still
# scatter uniformly. Scatter is honored by _scatter_position() in the SDG script.
FORCE_DISTRACTORS = {
    "distractors": {"clutter_level": 3.0},
    "distractor_randomization": {"scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
}
SCATTER_ONLY_DIST = {
    "distractor_randomization": {"scatter": "uniform", "scatter_inset_m": SCATTER_INSET_M},
}

# Wall/shelf-clipping fix (applied to every config). The occupancy scan now
# spans a vertical band (scan_z_min_m..scan_z_max_m in the SDG script, default
# 0.1..2.0) so pallet racks/shelves/walls become solid obstacles instead of a
# thin z=1.0 slice the camera drove through. buffer_m raised 0.4->0.6 for extra
# clearance so the camera stops staring at / grazing walls.
OCC_TUNE = {
    "trajectory": {"occupancy": {"buffer_m": 0.6}},
}

FULL_BOUNDS = [-13.0, 13.0, -13.0, 15.0]
PLAIN_BOUNDS = [-12.0, 12.0, -12.0, 14.0]

# Per-environment TRUE interior box (occupancy scan + object scatter region).
# Measured 2026-07-07 via the "look straight up" probe (ceiling => inside, black
# void => outside) over a 13x10 grid, then inset ~1m off the shell walls. The old
# symmetric [-13,13,-13,15] guess was the bug: full_warehouse's interior is
# OFFSET toward -X (shell X[-12,7]), so the +X side of the window was exterior
# apron that scans as free floor -> the planner routed the camera OUTSIDE the
# building (and objects scattered out there too). The other three warehouses are
# ~centered (shell X[-10,11]) so they were mostly fine. See base-v4-bounds-offset-bug.
# Refined 2026-07-07 with the "star" step-walk probe (walk out from an interior
# center along +/-X, +/-Y until the view crosses into VOID = the exterior wall;
# interior partitions/shelves don't trigger since interior lies beyond them).
# Measured exterior walls: full_warehouse +X@5.8 (others open past window);
# warehouse-family (shared geometry) +X@9.3, -X@-10.9, -Y@-12.3, +Y open. Bounds
# below are inset ~0.5-1m inside those walls.
ENV_BOUNDS = {
    # full_warehouse: the occupancy-reachable MAIN HALL (green region on the
    # annotated map). x_min=-5 is the partition wall (a separate section lies
    # beyond it that the occupancy scan can't observe and the camera can't reach —
    # scattering objects there would strand them behind the wall). +X=5 sits just
    # inside the wall (apron/exterior beyond ~6, the original escape bug).
    "full_warehouse":             [-5.0, 5.0, -11.0, 13.0],
    "warehouse":                  [-9.5, 8.0, -11.0, 13.0],
    "warehouse_multiple_shelves": [-9.5, 8.0, -11.0, 13.0],
    "warehouse_with_forklifts":   [-9.5, 8.0, -11.0, 13.0],
}

# The tight set (exp01-10) is intentionally consistent framing — the FOV/height
# VARIANCE lives in the imports (exp11-21). But two tight configs carry names
# that promise camera variance the ~55°/eye-level cluster never delivered
# (IMAGE_REVIEW #5/#6). Nudge only those two so their names read, without
# disturbing the other 8. Applied on top of the copied base_v3 config.
TIGHT_OVERRIDES = {
    # "mid-fov": lift out of the 53-58 cluster so the wider framing is visible.
    "exp07_mid_fov_survey": {"cameras": {"ego": {"fov_mean": 70.0}}},
    # "low-angle close-up": drop to a near-floor mount looking slightly up.
    "exp09_low_angle_closeup": {"cameras": {"ego": {"height_m": 0.75, "pitch_deg": 3.0}}},
}


def cam(fov, height, pitch, f_stop=0.0):
    return {"cameras": {"ego": {"fov_mean": fov, "height_m": height,
                                "pitch_deg": pitch, "f_stop": f_stop}}}


def light(mean, std):
    return {"lighting": {"intensity_mean": mean, "intensity_std": std}}


def merged(*ds):
    out = {}
    for d in ds:
        out = deep_merge(out, d)
    return out


# --- The 11 curated imports ---------------------------------------------------
# keep_counts=True  -> preserve the source hero counts (density axis).
IMPORTS = [
    {
        "name": "exp11_overhead_survey",
        "src": "exp04_ceiling_dolly_slow.yaml",  # slow ceiling-dolly character
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 2.6, -22.0), light(90000.0, 30000.0),
                       {"agent": {"speed_mps": 0.5}}),
    },
    {
        "name": "exp12_steep_downward",
        "src": "exp18_steep_downward.yaml",
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 1.2, -18.0), light(100000.0, 25000.0),
                       {"agent": {"speed_mps": 1.2}}),
    },
    {
        "name": "exp13_narrow_telephoto",
        "src": "exp13_narrow_telephoto.yaml",  # 32deg -> BIG boxes, on-thesis
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(32.0, 1.6, 0.0), light(130000.0, 20000.0),
                       {"agent": {"speed_mps": 0.4}}),
    },
    {
        "name": "exp14_night_shift_dim",
        "src": "exp09_night_shift_dim.yaml",
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(70.0, 1.6, -3.0), light(40000.0, 25000.0),
                       {"agent": {"speed_mps": 1.0}}),
    },
    {
        "name": "exp15_very_dark",
        "src": "exp09_night_shift_dim.yaml",  # push the dim base into true low-light
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(70.0, 1.6, -3.0), light(15000.0, 10000.0),
                       {"agent": {"speed_mps": 1.0}}),
    },
    {
        "name": "exp16_bright_daytime",
        "src": "exp10_bright_daytime.yaml",
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(72.0, 1.6, -3.0), light(180000.0, 15000.0),
                       {"agent": {"speed_mps": 1.2}}),
    },
    {
        "name": "exp17_running_operator",
        "src": "exp17_running_operator.yaml",  # fast motion
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 1.55, -4.0), light(115000.0, 25000.0),
                       {"agent": {"speed_mps": 3.5}}),
    },
    {
        "name": "exp18_wide_low_fast",
        "src": "exp02_forklift_driver_low_wide.yaml",  # the ONE kept wide FOV
        # height 0.7m: genuinely near-floor so "low" reads (IMAGE_REVIEW #5); slight
        # up-pitch to keep the horizon/shelves in frame from that low mount.
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(90.0, 0.7, 2.0), light(90000.0, 40000.0),
                       {"agent": {"speed_mps": 3.0}}),
    },
    {
        "name": "exp19_patrol_robot_low",
        "src": "exp08_patrol_robot.yaml",  # low steady wheeled platform, mid speed
        # height 0.65m: near-floor patrol-robot POV (was 1.2m, didn't read as low).
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 0.65, 2.0), light(110000.0, 30000.0),
                       {"agent": {"speed_mps": 2.0}}),
    },
    {
        "name": "exp20_dense_clutter",
        "src": "exp11_dense_clutter.yaml",  # clutter_level 2.4, characters 8, high jitter
        "keep_counts": True,
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 1.55, -5.0), light(100000.0, 30000.0)),
    },
    {
        "name": "exp21_sparse_minimal",
        "src": "exp12_sparse_minimal.yaml",  # clutter_level 0.1, characters 1, low jitter
        "keep_counts": True,
        "over": merged({"environment": {"name": "warehouse"},
                        "trajectory": {"bounds_xy": PLAIN_BOUNDS}},
                       cam(70.0, 1.6, -5.0), light(100000.0, 20000.0)),
    },
]

HEADER = (
    "# base_v4 — v3 tight/de-correlated set PLUS curated base_v1 variance imports.\n"
    "# AUTO-GENERATED by _generate_base_v4.py — do not edit by hand; edit the\n"
    "# generator and re-run. Design rationale in that script's docstring.\n"
    "# Runtime: run at 13 seeds x 10 frames (see run_base_v4_train.sh).\n"
    "# data_dir below is a placeholder; the runner overrides it per (config, seed).\n"
)


def write_cfg(name: str, cfg: dict) -> None:
    cfg["run"]["data_dir"] = f"{DATA_ROOT}/{name}"
    # Override bounds_xy with the measured TRUE interior for this config's env
    # (replaces every source's symmetric guess). This confines BOTH the occupancy
    # camera path and the object scatter region to the real building interior.
    env = cfg.get("environment", {}).get("name")
    if env in ENV_BOUNDS:
        cfg.setdefault("trajectory", {})["bounds_xy"] = list(ENV_BOUNDS[env])
    else:
        print(f"  WARNING: {name} env={env!r} has no ENV_BOUNDS entry — left as-is")
    out = HERE / f"{name}.yaml"
    out.write_text(HEADER + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    print(f"wrote {out.relative_to(ROOT.parent.parent)}")


def main() -> None:
    # --- TIGHT set: copy base_v3's committed 10 configs verbatim (data_dir only) --
    # We copy the committed yamls rather than re-derive from COMMON+EXPERIMENTS
    # because the base_v1/exp01 template drifted after v3 was generated
    # (capture_dt, occupancy.buffer_m), so re-derivation would NOT reproduce v3.
    v3_dir = ROOT / "base_v3"
    for name in _v3.EXPERIMENTS:
        cfg = yaml.safe_load((v3_dir / f"{name}.yaml").read_text())
        # Tight set is all non-density → uniform scatter + raised distractor count.
        # base_v3 already scatters tight-set targets uniformly but without an
        # inset, so they used the SDG 2.0m default and still reached the walls;
        # overlay the wider inset (#2). SCATTER_ONLY = uniform + scatter_inset_m.
        cfg = deep_merge(cfg, SCATTER_ONLY)
        cfg = deep_merge(cfg, FORCE_DISTRACTORS)
        cfg = deep_merge(cfg, OCC_TUNE)
        if name in TIGHT_OVERRIDES:
            cfg = deep_merge(cfg, TIGHT_OVERRIDES[name])
        write_cfg(name, cfg)

    # --- IMPORTS: base on real v1 source + de-correlation + counts + overrides -
    for spec in IMPORTS:
        base = yaml.safe_load((V1 / spec["src"]).read_text())
        cfg = deep_merge(base, IMPORT_COMMON)
        cfg = deep_merge(cfg, SCATTER_ONLY if spec.get("keep_counts") else FORCE_TARGETS)
        cfg = deep_merge(cfg, SCATTER_ONLY_DIST if spec.get("keep_counts") else FORCE_DISTRACTORS)
        cfg = deep_merge(cfg, OCC_TUNE)
        cfg = deep_merge(cfg, spec["over"])
        write_cfg(spec["name"], cfg)

    print(f"\n{len(_v3.EXPERIMENTS)} tight + {len(IMPORTS)} imports = "
          f"{len(_v3.EXPERIMENTS) + len(IMPORTS)} base_v4 configs generated.")


if __name__ == "__main__":
    main()

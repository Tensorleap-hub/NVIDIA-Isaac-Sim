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

# v3 empties fix: raised counts + uniform scatter (forced on non-density imports).
FORCE_TARGETS = {
    "palletjacks": {"count_per_model": 7, "scatter": "uniform"},
    "forklifts": {"count_per_model": 3, "scatter": "uniform"},
    "pallets": {"count_per_model": 6, "scatter": "uniform"},
}
# Density imports keep their own counts but still scatter uniformly.
SCATTER_ONLY = {
    "palletjacks": {"scatter": "uniform"},
    "forklifts": {"scatter": "uniform"},
    "pallets": {"scatter": "uniform"},
}

FULL_BOUNDS = [-13.0, 13.0, -13.0, 15.0]
PLAIN_BOUNDS = [-12.0, 12.0, -12.0, 14.0]


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
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(90.0, 1.1, -5.0), light(90000.0, 40000.0),
                       {"agent": {"speed_mps": 3.0}}),
    },
    {
        "name": "exp19_patrol_robot_low",
        "src": "exp08_patrol_robot.yaml",  # low steady wheeled platform, mid speed
        "over": merged({"trajectory": {"bounds_xy": FULL_BOUNDS}},
                       cam(75.0, 1.2, -2.0), light(110000.0, 30000.0),
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
        write_cfg(name, cfg)

    # --- IMPORTS: base on real v1 source + de-correlation + counts + overrides -
    for spec in IMPORTS:
        base = yaml.safe_load((V1 / spec["src"]).read_text())
        cfg = deep_merge(base, IMPORT_COMMON)
        cfg = deep_merge(cfg, SCATTER_ONLY if spec.get("keep_counts") else FORCE_TARGETS)
        cfg = deep_merge(cfg, spec["over"])
        write_cfg(spec["name"], cfg)

    print(f"\n{len(_v3.EXPERIMENTS)} tight + {len(IMPORTS)} imports = "
          f"{len(_v3.EXPERIMENTS) + len(IMPORTS)} base_v4 configs generated.")


if __name__ == "__main__":
    main()

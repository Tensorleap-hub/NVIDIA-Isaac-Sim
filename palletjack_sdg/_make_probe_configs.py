"""Generate interior-probe configs: a dense snake path over the full scan box,
level camera, lights up, minimal objects. Rendering these lets us classify each
frame inside/outside (by black-void pixel fraction) and read the true per-env
interior extent from the logged camera (x,y). One config per environment."""
import copy, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE / "experiments/trajectory/base_v4"
OUT = HERE / "experiments/trajectory/_probe"
OUT.mkdir(exist_ok=True)

# template config per env (any base_v4 config using that env)
ENVS = {
    "full_warehouse": "exp06_bright_dense_mixed.yaml",
    "warehouse_multiple_shelves": "exp05_multishelf_aisle_tight.yaml",
    "warehouse_with_forklifts": "exp03_forklift_yard_tight.yaml",
    "warehouse": "exp08_plain_warehouse_tight.yaml",
}

# probe grid over the (oversized) symmetric scan box; snake order so it's one path
# Denser X grid to resolve the interior walls precisely.
XS = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12]
YS = [-12, -9, -6, -3, 0, 3, 6, 9, 12, 14]
# Look straight UP via ego.pitch_deg=90 (X-rot 90+90=180): a lit ceiling means
# INSIDE, black void means OUTSIDE. View-independent (unlike a forward-facing
# shot, which reads bright whenever it happens to face a lit wall from outside).
# Waypoint carries base roll=90 (horizontal) + pitch_in=0; the look-up is the
# ego pitch knob.
Z, BASE_ROLL, PITCH_IN, YAWREL = 1.6, 90.0, 0.0, 0.0
EGO_PITCH_UP = 90.0


def snake():
    wps = []
    for j, y in enumerate(YS):
        row = XS if j % 2 == 0 else list(reversed(XS))
        for x in row:
            wps.append([float(x), float(y), Z, BASE_ROLL, PITCH_IN, YAWREL])
    return wps


def main():
    wps = snake()
    for env, tmpl in ENVS.items():
        cfg = yaml.safe_load((V4 / tmpl).read_text())
        cfg["environment"]["name"] = env
        cfg["trajectory"] = {"mode": "waypoint_list", "waypoints": wps}
        ego = cfg.setdefault("cameras", {}).setdefault("ego", {})
        ego.update({"fov_mean": 90.0, "fov_std": 0.0, "height_m": Z,
                    "pitch_deg": EGO_PITCH_UP, "roll_deg": 0.0, "f_stop": 0.0,
                    "shutter_close_fraction": 0.0})
        for k in ("pitch_jitter", "roll_jitter", "lateral_jitter", "vertical_jitter"):
            ego.pop(k, None)
        # lights up so interior is clearly lit and exterior reads as dark void
        cfg.setdefault("lighting", {})["intensity_mean"] = 120000.0
        cfg["lighting"]["env_light_scale"] = 1.0
        # minimal objects so they don't occlude the inside/outside read
        for cls in ("palletjacks", "forklifts", "pallets"):
            if cls in cfg:
                cfg[cls]["count_per_model"] = 1
        if "distractors" in cfg:
            cfg["distractors"]["clutter_level"] = 0.3
        out = OUT / f"probe_{env}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {out}  ({len(wps)} waypoints)")


if __name__ == "__main__":
    main()

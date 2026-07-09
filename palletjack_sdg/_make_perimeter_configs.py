"""Perimeter-loop sanity configs: the camera walks a rectangle at each env's
interior bounds (ENV_BOUNDS), counterclockwise, facing OUTWARD toward the wall
it hugs. If the bounds match the true interior, every frame shows an interior
wall close by; any stretch that shows exterior/void means the bounds are too big
there. waypoint_list mode (no occupancy needed)."""
import importlib.util, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE / "experiments/trajectory/base_v4"
OUT = HERE / "experiments/trajectory/_perimeter"
OUT.mkdir(exist_ok=True)

_spec = importlib.util.spec_from_file_location("gen", V4 / "_generate_base_v4.py")
_gen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gen)
ENV_BOUNDS = _gen.ENV_BOUNDS

TEMPLATES = {
    "full_warehouse": "exp06_bright_dense_mixed.yaml",
    "warehouse_multiple_shelves": "exp05_multishelf_aisle_tight.yaml",
    "warehouse_with_forklifts": "exp03_forklift_yard_tight.yaml",
    "warehouse": "exp08_plain_warehouse_tight.yaml",
}
Z = 1.6
PER_EDGE = 12
# Face the DIRECTION OF TRAVEL (yaw_rel=0) and traverse CLOCKWISE so the exterior
# wall is always on the camera's LEFT and the interior on the right (unambiguous:
# wall surface left + interior right = inside; void on both = outside). Clockwise
# order: (xmin,ymin)->(xmin,ymax)->(xmax,ymax)->(xmax,ymin)->close.
YAW_REL = 0.0


def lerp(a, b, t):
    return a + (b - a) * t


def rect_waypoints(xmin, xmax, ymin, ymax):
    # CLOCKWISE (wall on left when facing travel direction)
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    wps = []
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        for i in range(PER_EDGE):
            t = i / float(PER_EDGE)
            wps.append([lerp(x0, x1, t), lerp(y0, y1, t), Z, 90.0, 0.0, YAW_REL])
    wps.append([xmin, ymin, Z, 90.0, 0.0, YAW_REL])  # close loop
    return wps


def main():
    for env, tmpl in TEMPLATES.items():
        xmin, xmax, ymin, ymax = ENV_BOUNDS[env]
        cfg = yaml.safe_load((V4 / tmpl).read_text())
        cfg["environment"]["name"] = env
        cfg["trajectory"] = {"mode": "waypoint_list", "waypoints": rect_waypoints(xmin, xmax, ymin, ymax)}
        ego = cfg.setdefault("cameras", {}).setdefault("ego", {})
        ego.update({"fov_mean": 95.0, "fov_std": 0.0, "height_m": Z,
                    "pitch_deg": 0.0, "roll_deg": 0.0, "f_stop": 0.0,
                    "shutter_close_fraction": 0.0})
        for k in ("pitch_jitter", "roll_jitter", "lateral_jitter", "vertical_jitter"):
            ego.pop(k, None)
        # Neutral WHITE lighting for clean, judgeable verification frames (the
        # default color randomization tints frames purple/teal — noise for a
        # bounds check).
        lt = cfg.setdefault("lighting", {})
        lt["intensity_mean"] = 130000.0
        lt["intensity_std"] = 0.0
        lt["color_mean"] = [1.0, 1.0, 1.0]
        lt["color_std"] = [0.0, 0.0, 0.0]
        lt["env_light_scale"] = 1.0
        cfg.pop("materials", None)  # no emissive/texture randomization on walls/floor
        for cls in ("palletjacks", "forklifts", "pallets"):
            if cls in cfg:
                cfg[cls]["count_per_model"] = 1
        if "distractors" in cfg:
            cfg["distractors"]["clutter_level"] = 0.3
        out = OUT / f"perim_{env}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {out.name}  bounds={ENV_BOUNDS[env]}  {len(cfg['trajectory']['waypoints'])} wp")


if __name__ == "__main__":
    main()

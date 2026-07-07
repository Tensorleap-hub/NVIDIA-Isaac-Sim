"""Step-walk 'star' probe (user's strategy): from an interior center, walk outward
along +X, -X, +Y, -Y in small steps, camera facing travel direction. Where a
spoke crosses from interior/wall (bright) into VOID (black) is the wall = the
bound for that direction. One render per env; analysis reads the crossing."""
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE / "experiments/trajectory/base_v4"
OUT = HERE / "experiments/trajectory/_star"
OUT.mkdir(exist_ok=True)

TEMPLATES = {
    "full_warehouse": ("exp06_bright_dense_mixed.yaml", (-2.0, -1.0)),
    "warehouse_multiple_shelves": ("exp05_multishelf_aisle_tight.yaml", (0.0, -3.0)),
    "warehouse_with_forklifts": ("exp03_forklift_yard_tight.yaml", (0.0, -3.0)),
    "warehouse": ("exp08_plain_warehouse_tight.yaml", (0.0, -3.0)),
}
Z = 1.6
STEP = 0.4
REACH = 15.0  # walk out to +/-15 m (past any wall, into void)


def spoke(cx, cy, dx, dy):
    n = int(REACH / STEP)
    wps = []
    for i in range(1, n + 1):
        wps.append([cx + dx * STEP * i, cy + dy * STEP * i, Z, 90.0, 0.0, 0.0])
    return wps


def waypoints(cx, cy):
    c = [cx, cy, Z, 90.0, 0.0, 0.0]
    wps = [c]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        wps += spoke(cx, cy, dx, dy)      # walk out (faces +dir)
        wps += [c]                         # return to center
    return wps


def main():
    for env, (tmpl, (cx, cy)) in TEMPLATES.items():
        cfg = yaml.safe_load((V4 / tmpl).read_text())
        cfg["environment"]["name"] = env
        cfg["trajectory"] = {"mode": "waypoint_list", "waypoints": waypoints(cx, cy)}
        ego = cfg.setdefault("cameras", {}).setdefault("ego", {})
        ego.update({"fov_mean": 70.0, "fov_std": 0.0, "height_m": Z,
                    "pitch_deg": 0.0, "roll_deg": 0.0, "f_stop": 0.0,
                    "shutter_close_fraction": 0.0})
        for k in ("pitch_jitter", "roll_jitter", "lateral_jitter", "vertical_jitter"):
            ego.pop(k, None)
        cfg.setdefault("lighting", {})["intensity_mean"] = 120000.0
        cfg["lighting"]["env_light_scale"] = 1.0
        for cls in ("palletjacks", "forklifts", "pallets"):
            if cls in cfg:
                cfg[cls]["count_per_model"] = 0   # no objects: pure geometry probe
        if "distractors" in cfg:
            cfg["distractors"]["clutter_level"] = 0.0
        n = len(cfg["trajectory"]["waypoints"])
        (OUT / f"star_{env}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote star_{env}.yaml  center=({cx},{cy})  {n} waypoints")


if __name__ == "__main__":
    main()

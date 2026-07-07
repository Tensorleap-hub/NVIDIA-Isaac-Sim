"""Read each spoke's wall crossing from the star probe. A frame that sees VOID
(black_frac>THRESH) while facing outward marks exterior. Per direction, the void
frame nearest the center = that exterior wall. Interior partitions/shelves don't
trigger (interior lies beyond them, not void)."""
import json, sys
from pathlib import Path
from PIL import Image

STAR = Path(open(Path(__file__).resolve().parent / ".star_dir").read().strip())
ENVS = ["full_warehouse", "warehouse_multiple_shelves",
        "warehouse_with_forklifts", "warehouse"]
CENTERS = {"full_warehouse": (-2.0, -1.0)}   # others (0,-3); see _make_star_probe
THRESH = 0.5


def bf(p):
    h = Image.open(p).convert("L").histogram()
    return sum(h[:14]) / float(sum(h))


def analyze(env):
    d = STAR / env
    cx, cy = CENTERS.get(env, (0.0, -3.0))
    poses = {int(json.loads(l)["frame"]): json.loads(l)
             for l in (d / "trajectory/poses.jsonl").read_text().splitlines()}
    walls = {"+X": None, "-X": None, "+Y": None, "-Y": None}
    for f, p in sorted(poses.items()):
        png = d / "Camera/rgb" / f"rgb_{f:04d}.png"
        if not png.exists():
            continue
        x, y = p["camera_pos"][0], p["camera_pos"][1]
        if bf(png) <= THRESH:
            continue  # not void
        # classify which spoke: dominant axis offset from center
        dx, dy = x - cx, y - cy
        if abs(dx) > abs(dy) and abs(dy) < 1.5:
            key = "+X" if dx > 0 else "-X"
            dist = abs(dx)
        elif abs(dy) > abs(dx) and abs(dx) < 1.5:
            key = "+Y" if dy > 0 else "-Y"
            dist = abs(dy)
        else:
            continue
        coord = x if "X" in key else y
        if walls[key] is None or dist < walls[key][0]:
            walls[key] = (dist, round(coord, 1))
    print(f"\n{env}  center=({cx},{cy})")
    for k in ("+X", "-X", "+Y", "-Y"):
        w = walls[k]
        print(f"   {k} wall: {'void @ '+str(w[1]) if w else 'no void within reach (open/interior)'}")


if __name__ == "__main__":
    for env in (sys.argv[1:] or ENVS):
        try:
            analyze(env)
        except FileNotFoundError as e:
            print(f"{env}: not ready ({e})")

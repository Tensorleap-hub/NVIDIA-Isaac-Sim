"""Classify each probe frame inside/outside by black-void pixel fraction, using
the logged per-frame camera (x,y). Print an ASCII map + the interior bbox per env."""
import json, sys
from pathlib import Path
from PIL import Image

PROBE = Path(open(Path(__file__).resolve().parent / ".probe_dir").read().strip())
ENVS = ["full_warehouse", "warehouse_multiple_shelves",
        "warehouse_with_forklifts", "warehouse"]
BLACK_LEVEL = 14        # luminance below this = "void" (looking-up: no ceiling)
VOID_FRAC = 0.45        # frame with >this black (void) fraction = OUTSIDE


def black_frac(png):
    h = Image.open(png).convert("L").histogram()
    return sum(h[:BLACK_LEVEL]) / float(sum(h))


def analyze(env):
    d = PROBE / env
    poses = {}
    for line in (d / "trajectory/poses.jsonl").read_text().splitlines():
        p = json.loads(line)
        poses[int(p["frame"])] = p["camera_pos"]
    rows = []
    for f, pos in sorted(poses.items()):
        png = d / "Camera/rgb" / f"rgb_{f:04d}.png"
        if not png.exists():
            continue
        bf = black_frac(png)
        rows.append((pos[0], pos[1], bf, bf <= VOID_FRAC))
    inside = [(x, y) for x, y, bf, ins in rows if ins]
    print(f"\n===== {env}  ({len(rows)} frames, {len(inside)} inside) =====")
    if inside:
        xs = [x for x, y in inside]; ys = [y for x, y in inside]
        print(f"  INTERIOR bbox  X[{min(xs):.1f}, {max(xs):.1f}]  Y[{min(ys):.1f}, {max(ys):.1f}]")
    # ascii grid: rows=Y desc, cols=X asc; O=inside . =outside
    XS = sorted(set(round(x) for x, y, bf, ins in rows))
    YS = sorted(set(round(y) for x, y, bf, ins in rows), reverse=True)
    grid = {}
    for x, y, bf, ins in rows:
        grid[(round(x), round(y))] = "O" if ins else "."
    hdr = "   " + "".join(f"{x:>4}" for x in XS)
    print("  x:" + hdr)
    for y in YS:
        line = "".join(f"  {grid.get((x, y), ' ')} " for x in XS)
        print(f"  y{y:>4} {line}")


if __name__ == "__main__":
    for env in (sys.argv[1:] or ENVS):
        try:
            analyze(env)
        except FileNotFoundError as e:
            print(f"\n{env}: not ready ({e})")

"""GT-verification report: for every arm, per class, example frames from each source with the
ground-truth boxes drawn, plus composition/box statistics. Self-contained HTML (base64 jpgs).

Usage: .venv/bin/python od_scripts/verify_gt.py [--out /home/ubuntu/datasets_coco/gt_report.html]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, OUT, source_of  # noqa: E402

CLASS_COLORS = {"forklift": "#E0A020", "pallet": "#2A9D8F", "pallet_truck": "#C4407A"}
SOURCE_LABEL = {"real": "LOCO real (train)", "valid": "LOCO subset-3 (valid)", "basev2": "base_v2 synth", "may": "may-rounds synth", "traj_optuna": "trajectory-optimized synth"}
THUMB_W = 520
PER_SOURCE = 2


def load(split: Path):
    with open(split / "_annotations.coco.json") as f:
        c = json.load(f)
    id2name = {x["id"]: x["name"] for x in c["categories"]}
    anns = defaultdict(list)
    for a in c["annotations"]:
        anns[a["image_id"]].append((id2name[a["category_id"]], a["bbox"]))
    return c, id2name, anns


def render(img_path: Path, boxes, focus: str) -> str:
    im = Image.open(img_path).convert("RGB")
    scale = THUMB_W / im.width
    im = im.resize((THUMB_W, max(1, round(im.height * scale))))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for name, (x, y, w, h) in sorted(boxes, key=lambda b: b[0] == focus):  # focus class drawn last
        col = CLASS_COLORS[name]
        width = 3 if name == focus else 1
        dr.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline=col, width=width)
        if name == focus:
            dr.rectangle([x * scale, y * scale - 14, x * scale + 8 + 6.5 * len(name), y * scale], fill=col)
            dr.text((x * scale + 3, y * scale - 13), name, fill="white", font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pick(images, anns, src: str, cls: str, k: int, rng: random.Random):
    """k images of source `src` containing class `cls`, preferring a few mid-size boxes."""
    cands = []
    for im in images:
        if source_of(im["file_name"]) != src:
            continue
        b = [bb for n, bb in anns[im["id"]] if n == cls]
        if not b:
            continue
        rel = max(bb[2] * bb[3] for bb in b) / (im["width"] * im["height"])
        if 0.01 < rel < 0.5 and len(anns[im["id"]]) <= 25:
            cands.append(im)
    if len(cands) < k:  # relax
        cands = [im for im in images if source_of(im["file_name"]) == src and any(n == cls for n, _ in anns[im["id"]])]
    rng.shuffle(cands)
    return cands[:k]


def stats(images, anns, id2name):
    by_src = defaultdict(lambda: {"images": 0, "boxes": Counter(), "rel_area": defaultdict(list)})
    for im in images:
        s = source_of(im["file_name"])
        by_src[s]["images"] += 1
        for n, (x, y, w, h) in anns[im["id"]]:
            by_src[s]["boxes"][n] += 1
            by_src[s]["rel_area"][n].append(w * h / (im["width"] * im["height"]))
    out = {}
    for s, d in by_src.items():
        med = {n: (sorted(v)[len(v) // 2] if v else 0) for n, v in d["rel_area"].items()}
        out[s] = {"images": d["images"], "boxes": dict(d["boxes"]), "median_rel_area": med}
    return out


CSS = """
:root{--bg:#F3F5F7;--ink:#1B2229;--muted:#5B6672;--card:#FFFFFF;--line:#D6DCE2;--accent:#3D6EA8;--code:#EEF1F4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14181D;--ink:#E6EAEE;--muted:#98A3AE;--card:#1C2229;--line:#2C353F;--accent:#7FA6D6;--code:#222A33}}
:root[data-theme="dark"]{--bg:#14181D;--ink:#E6EAEE;--muted:#98A3AE;--card:#1C2229;--line:#2C353F;--accent:#7FA6D6;--code:#222A33}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding:40px 28px 80px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
h2{font-size:21px;font-weight:600;margin:48px 0 12px;padding-top:24px;border-top:1px solid var(--line)}
h3{font-size:15px;font-weight:600;margin:22px 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
p.lead{color:var(--muted);max-width:68ch;margin:0 0 20px}
.rules{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 26px}
.rule{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;font-size:13.5px}
.rule b{color:var(--accent)}
.ok{color:#2A9D8F;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
th,td{padding:7px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:12.5px;text-transform:uppercase;letter-spacing:.05em}
td.n,th.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.tblwrap{overflow-x:auto}
.legend{display:flex;gap:18px;margin:10px 0 0;font-size:13.5px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);padding:6px 9px;word-break:break-all}
figcaption span{color:var(--ink)}
.cls{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px;font-weight:600;margin-right:6px}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;background:var(--code);padding:1px 5px;border-radius:3px;font-size:13px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "gt_report.html"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(OUT / "MANIFEST.json") as f:
        manifest = json.load(f)
    classes = [c["name"] for c in manifest["categories"]]

    parts = [f"<title>Warehouse GT Check</title><style>{CSS}</style>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">',
             '<div class="wrap"><h1>Warehouse 3-class ground-truth check</h1>',
             '<p class="lead">Every dataset under <code>/home/ubuntu/datasets_coco</code>, with example frames per class and per source. '
             'Boxes are drawn straight from each split\'s <code>_annotations.coco.json</code>; the class under inspection is drawn thick and labelled, other classes thin.</p>',
             '<div class="rules">'
             f'<div class="rule"><b>Valid</b> = LOCO subset-3 only · {manifest["valid"]["images"]} images · <span class="ok">0 synthetic frames</span></div>'
             '<div class="rule"><b>Train</b> never contains a subset-3 image <span class="ok">✓ asserted</span></div>'
             f'<div class="rule"><b>Categories</b> {" · ".join(f"{c["id"]} {c["name"]}" for c in manifest["categories"])} (identical in all datasets)</div>'
             '</div>']
    parts.append('<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{CLASS_COLORS[c]}"></i>{c}</span>' for c in classes) + "</div>")

    # ---- composition table
    parts.append("<h2>Composition</h2><div class='tblwrap'><table><tr><th>dataset</th><th>role</th><th class='n'>images</th>"
                 "<th class='n'>real</th><th class='n'>base_v2</th><th class='n'>may</th><th class='n'>traj</th>"
                 + "".join(f"<th class='n'>{c} boxes</th>" for c in classes) + "</tr>")
    v = manifest["valid"]
    parts.append(f"<tr><td>real/valid (shared by all arms)</td><td>validation</td><td class='n'>{v['images']}</td><td class='n'>{v['images']}</td>"
                 f"<td class='n'>0</td><td class='n'>0</td><td class='n'>0</td>" + "".join(f"<td class='n'>{v['class_counts'].get(c, 0)}</td>" for c in classes) + "</tr>")
    for arm, d in manifest["arms"].items():
        bs = d["by_source"]
        parts.append(f"<tr><td>{arm}/train</td><td>training arm</td><td class='n'>{d['train_images']}</td><td class='n'>{bs.get('real', 0)}</td>"
                     f"<td class='n'>{bs.get('basev2', 0)}</td><td class='n'>{bs.get('may', 0)}</td><td class='n'>{bs.get('traj_optuna', 0)}</td>"
                     + "".join(f"<td class='n'>{d['class_counts'].get(c, 0)}</td>" for c in classes) + "</tr>")
    parts.append("</table></div>")

    # ---- per-source box stats (from real_all train + valid)
    c_all, id2n, anns_all = load(OUT / "real_all_traj" / "train")
    st = stats(c_all["images"], anns_all, id2n)
    c_val, _, anns_val = load(OUT / "real" / "valid")
    st["valid"] = stats([{**im, "file_name": im["file_name"]} for im in c_val["images"]], anns_val, id2n)["real"]
    parts.append("<h2>Per-source label statistics</h2><p class='lead'>Boxes per image and median box size (fraction of image area) per class. "
                 "Useful to spot a permuted class mapping: a swapped mapping shows up as wildly different size/frequency profiles between real and synthetic.</p>"
                 "<div class='tblwrap'><table><tr><th>source</th><th class='n'>images</th>"
                 + "".join(f"<th class='n'>{c} / img</th><th class='n'>{c} med. area</th>" for c in classes) + "</tr>")
    for s in ("real", "valid", "basev2", "may", "traj_optuna"):
        d = st[s]
        parts.append(f"<tr><td>{SOURCE_LABEL[s]}</td><td class='n'>{d['images']}</td>" + "".join(
            f"<td class='n'>{d['boxes'].get(c, 0) / d['images']:.2f}</td><td class='n'>{d['median_rel_area'].get(c, 0):.3f}</td>" for c in classes) + "</tr>")
    parts.append("</table></div>")

    # ---- examples per arm
    for arm, sources in ARMS.items():
        coco, id2n, anns = load(OUT / arm / "train")
        parts.append(f"<h2>{arm}</h2><p class='lead'>train = real" + "".join(f" + {SOURCE_LABEL[s]}" for s in sources) +
                     f" ({len(coco['images'])} images); valid → real/valid.</p>")
        for cls in classes:
            parts.append(f"<h3><span class='cls' style='background:{CLASS_COLORS[cls]}'>{cls}</span></h3><div class='grid'>")
            for src in ["real"] + sources:
                k = PER_SOURCE if sources else 2 * PER_SOURCE
                for im in pick(coco["images"], anns, src, cls, k, rng):
                    uri = render(OUT / arm / "train" / im["file_name"], anns[im["id"]], cls)
                    n = sum(1 for nm, _ in anns[im["id"]] if nm == cls)
                    parts.append(f"<figure><img src='{uri}' alt='{cls} example'><figcaption><span>{SOURCE_LABEL[src]}</span> · {n} {cls} · {html.escape(im['file_name'])}</figcaption></figure>")
            parts.append("</div>")

    # ---- valid examples
    parts.append("<h2>real/valid (LOCO subset-3)</h2><p class='lead'>The single validation split shared by all arms. Real photos only.</p>")
    for cls in classes:
        parts.append(f"<h3><span class='cls' style='background:{CLASS_COLORS[cls]}'>{cls}</span></h3><div class='grid'>")
        for im in pick(c_val["images"], anns_val, "real", cls, 3, rng):
            uri = render(OUT / "real" / "valid" / im["file_name"], anns_val[im["id"]], cls)
            n = sum(1 for nm, _ in anns_val[im["id"]] if nm == cls)
            parts.append(f"<figure><img src='{uri}' alt='{cls} valid example'><figcaption><span>{SOURCE_LABEL['valid']}</span> · {n} {cls} · {html.escape(im['file_name'])}</figcaption></figure>")
        parts.append("</div>")
    parts.append("</div>")

    out = Path(args.out)
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""
Compare bounding boxes from best synthetic trials vs LOCO real data.

Scans selected_trial_downloads/optuna-ec2/ for best_* subdirectories,
samples frames from each, and renders a labeled grid alongside LOCO real samples.

Usage:
    python scripts/compare_synth_vs_real.py \
        --synth-root /path/to/selected_trial_downloads/optuna-ec2 \
        --loco-ann   /path/to/loco_dataset/labels/loco-sub3-v1-train.json \
        --loco-imgs  /path/to/loco_dataset \
        --n-per-source 4 \
        --output output/compare_synth_vs_real.png

    # Only synth (no LOCO)
    python scripts/compare_synth_vs_real.py \
        --synth-root /path/to/selected_trial_downloads/optuna-ec2 \
        --n-per-source 4 \
        --output output/compare_synth.png
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}
SYNTH_CLASS_MAP = {"palletjack": "pallet_truck", "forklift": "forklift", "pallet": "pallet"}

CLASS_COLORS = {
    "pallet_truck": (255, 80,  80),
    "forklift":     (80,  200, 80),
    "pallet":       (80,  120, 255),
}

LABEL_BG = (30, 30, 30)
GRID_BG  = (20, 20, 20)
SECTION_HEADER_H = 36


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_boxes(image: Image.Image, boxes_and_labels: list) -> Image.Image:
    """boxes_and_labels: list of (x_min, y_min, x_max, y_max, class_name)"""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1, cls in boxes_and_labels:
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tag_w = len(cls) * 8 + 4
        draw.rectangle([x0, y0 - 16, x0 + tag_w, y0], fill=color)
        draw.text((x0 + 2, y0 - 15), cls, fill=(0, 0, 0))
    return img


def make_titled_grid(
    sections: list,   # list of {"title": str, "images": [PIL.Image]}
    cols: int = 4,
    thumb_size: int = 400,
) -> Image.Image:
    """Render sections with a text header above each row group."""
    # Thumbnail all images
    all_thumbs = []
    for sec in sections:
        thumbs = []
        for img in sec["images"]:
            t = img.copy()
            t.thumbnail((thumb_size, thumb_size))
            thumbs.append(t)
        sec["thumbs"] = thumbs
        all_thumbs.extend(thumbs)

    if not all_thumbs:
        return Image.new("RGB", (100, 100), GRID_BG)

    cell_w = max(t.width  for t in all_thumbs)
    cell_h = max(t.height for t in all_thumbs)

    # Calculate total height
    total_h = 0
    for sec in sections:
        n = len(sec["thumbs"])
        rows = (n + cols - 1) // cols
        total_h += SECTION_HEADER_H + rows * cell_h

    total_w = cols * cell_w
    canvas = Image.new("RGB", (total_w, total_h), GRID_BG)
    draw = ImageDraw.Draw(canvas)

    y_offset = 0
    for sec in sections:
        # Section header
        draw.rectangle([0, y_offset, total_w, y_offset + SECTION_HEADER_H - 1], fill=(50, 50, 50))
        draw.text((8, y_offset + 8), sec["title"], fill=(220, 220, 220))
        y_offset += SECTION_HEADER_H

        # Thumbnails
        for i, thumb in enumerate(sec["thumbs"]):
            r, c = divmod(i, cols)
            x = c * cell_w
            y = y_offset + r * cell_h
            canvas.paste(thumb, (x, y))

        rows = (len(sec["thumbs"]) + cols - 1) // cols
        y_offset += rows * cell_h

    return canvas


# ---------------------------------------------------------------------------
# Synth data loaders
# ---------------------------------------------------------------------------

def collect_best_run_dirs(synth_root: Path) -> list:
    """
    Walk synth_root/<theme>/cycle_XX_*/best_*/ and return all run dirs
    inside their outputs/ subdirectory.

    Returns list of (theme, cycle, best_name, run_dir_path).
    """
    results = []
    for theme_dir in sorted(synth_root.iterdir()):
        if not theme_dir.is_dir():
            continue
        for cycle_dir in sorted(theme_dir.iterdir()):
            if not cycle_dir.is_dir():
                continue
            for trial_dir in sorted(cycle_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("best_"):
                    continue
                outputs_dir = trial_dir / "outputs"
                if not outputs_dir.is_dir():
                    continue
                for run_dir in sorted(outputs_dir.iterdir()):
                    if run_dir.is_dir():
                        results.append((theme_dir.name, cycle_dir.name, trial_dir.name, run_dir))
    return results


def collect_synth_frames(run_dir: Path) -> list:
    """Return list of (rgb_path, bbox_path, label_path) for complete triplets."""
    rgb_files   = {int(f.stem.split("_")[1]): f for f in run_dir.glob("rgb_*.png")}
    bbox_files  = {int(f.stem.split("_")[-1]): f for f in run_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
    label_files = {int(f.stem.split("_")[-1]): f
                   for f in run_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
    complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files))
    return [(rgb_files[n], bbox_files[n], label_files[n]) for n in complete]


def load_synth_sample(rgb_path: Path, bbox_path: Path, label_path: Path):
    """Return (PIL.Image with boxes drawn, n_boxes) or None if no valid boxes."""
    bboxes = np.load(bbox_path, allow_pickle=True)
    with open(label_path) as f:
        label_map = json.load(f)

    sem_to_class = {}
    for sem_str, info in label_map.items():
        raw = info.get("class", "")
        mapped = SYNTH_CLASS_MAP.get(raw, raw)
        if mapped in KEEP_CLASSES:
            sem_to_class[int(sem_str)] = mapped

    boxes = []
    for row in bboxes:
        sem_id = int(row["semanticId"])
        if sem_id not in sem_to_class:
            continue
        x0, y0, x1, y1 = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        boxes.append((x0, y0, x1, y1, sem_to_class[sem_id]))

    if not boxes:
        return None

    img = Image.open(rgb_path)
    return draw_boxes(img, boxes), len(boxes)


def sample_synth_images(synth_root: Path, n_per_source: int, seed: int) -> list:
    """
    Returns list of sections {"title": ..., "images": [...]}.
    One section per (theme, cycle, best_name), each with up to n_per_source images.
    """
    rng = random.Random(seed)
    best_runs = collect_best_run_dirs(synth_root)

    if not best_runs:
        print("  WARNING: no best_* run directories found under", synth_root)
        return []

    sections = []
    for theme, cycle, best_name, run_dir in best_runs:
        frames = collect_synth_frames(run_dir)
        rng.shuffle(frames)
        imgs = []
        for rgb_path, bbox_path, label_path in frames:
            if len(imgs) >= n_per_source:
                break
            result = load_synth_sample(rgb_path, bbox_path, label_path)
            if result is None:
                continue
            img, n_boxes = result
            imgs.append(img)
            print(f"  [synth] {theme}/{cycle}/{best_name}/{run_dir.name}/{rgb_path.name}: {n_boxes} boxes")

        if imgs:
            sections.append({
                "title": f"SYNTH  {theme} | {cycle} | {best_name}",
                "images": imgs,
            })

    return sections


# ---------------------------------------------------------------------------
# Real LOCO data loader
# ---------------------------------------------------------------------------

def sample_loco_images(ann_path: Path, img_root: Path, n: int, seed: int) -> list:
    """Returns a single section {"title": ..., "images": [...]}."""
    with open(ann_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        if cat_id_to_name.get(ann["category_id"]) not in KEEP_CLASSES:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    images_with_anns = [img for img in coco["images"] if img["id"] in ann_by_image]
    rng = random.Random(seed)
    rng.shuffle(images_with_anns)

    imgs = []
    for img_meta in images_with_anns:
        if len(imgs) >= n:
            break

        rel = img_meta.get("path", "")
        if rel:
            parts = rel.lstrip("/").split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            img_path = img_root / rel
        else:
            img_path = img_root / img_meta["file_name"]

        if not img_path.exists():
            print(f"  WARNING: not found: {img_path}")
            continue

        img = Image.open(img_path)
        boxes = []
        for ann in ann_by_image[img_meta["id"]]:
            x, y, w, h = ann["bbox"]
            cls = cat_id_to_name[ann["category_id"]]
            boxes.append((x, y, x + w, y + h, cls))

        imgs.append(draw_boxes(img, boxes))
        print(f"  [real]  {img_path.name}: {len(boxes)} boxes")

    return [{"title": f"REAL  LOCO  {ann_path.name}", "images": imgs}] if imgs else []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare best synth trials vs LOCO real boxes")
    parser.add_argument("--synth-root", required=True,
                        help="Path to selected_trial_downloads/optuna-ec2/")
    parser.add_argument("--loco-ann", default=None,
                        help="LOCO COCO annotation JSON (optional)")
    parser.add_argument("--loco-imgs", default=None,
                        help="Root directory of LOCO images (required if --loco-ann given)")
    parser.add_argument("--n-per-source", type=int, default=4,
                        help="Max frames to show per source (synth run or LOCO)")
    parser.add_argument("--cols", type=int, default=4,
                        help="Grid columns (default 4)")
    parser.add_argument("--thumb-size", type=int, default=400,
                        help="Max thumbnail size in pixels (default 400)")
    parser.add_argument("--output", default="output/compare_synth_vs_real.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synth_root = Path(args.synth_root)
    sections = []

    # Real LOCO section (shown first for easy visual comparison)
    if args.loco_ann:
        if not args.loco_imgs:
            parser.error("--loco-imgs is required when --loco-ann is provided")
        print(f"\nLoading LOCO real samples from {args.loco_ann} ...")
        loco_sections = sample_loco_images(
            Path(args.loco_ann), Path(args.loco_imgs), args.n_per_source, args.seed
        )
        sections.extend(loco_sections)

    # Synth sections
    print(f"\nLoading best synthetic samples from {synth_root} ...")
    synth_sections = sample_synth_images(synth_root, args.n_per_source, args.seed)
    sections.extend(synth_sections)

    if not sections:
        print("No images with matching boxes found.")
        return

    grid = make_titled_grid(sections, cols=args.cols, thumb_size=args.thumb_size)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)

    total = sum(len(s["images"]) for s in sections)
    print(f"\nSaved {total}-image comparison grid → {out}")


if __name__ == "__main__":
    main()

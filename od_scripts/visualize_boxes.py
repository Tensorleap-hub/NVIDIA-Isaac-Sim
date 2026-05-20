"""
Sanity-check bounding boxes for real (LOCO) and synthetic (Isaac Sim) data.

Usage:
    # Real LOCO data
    python scripts/visualize_boxes.py real \
        --ann /path/to/loco_dataset/labels/loco-sub3-v1-train.json \
        --img-root /path/to/loco_dataset \
        --n 8 --output output/sanity_real.png

    # Synthetic Isaac Sim data
    python scripts/visualize_boxes.py synth \
        --run-dir /path/to/iter000_run000 \
        --n 8 --output output/sanity_synth.png
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


def draw_boxes(image: Image.Image, boxes_and_labels: list[tuple]) -> Image.Image:
    """boxes_and_labels: list of (x_min, y_min, x_max, y_max, class_name)"""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1, cls in boxes_and_labels:
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.rectangle([x0, y0 - 16, x0 + len(cls) * 8 + 4, y0], fill=color)
        draw.text((x0 + 2, y0 - 15), cls, fill=(0, 0, 0))
    return img


def make_grid(images: list[Image.Image], cols: int = 4, thumb_size: int = 400) -> Image.Image:
    images = [img.copy() for img in images]
    for i, img in enumerate(images):
        img.thumbnail((thumb_size, thumb_size))
        images[i] = img
    rows = (len(images) + cols - 1) // cols
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * cell_w, r * cell_h))
    return grid


def load_real_samples(ann_path: Path, img_root: Path, n: int, seed: int):
    with open(ann_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        if cat_id_to_name.get(ann["category_id"]) not in KEEP_CLASSES:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    images_with_anns = [img for img in coco["images"] if img["id"] in ann_by_image]
    random.seed(seed)
    random.shuffle(images_with_anns)
    samples = images_with_anns[:n]

    results = []
    for img_meta in samples:
        rel = img_meta.get("path", "")
        if rel:
            # Strip leading /dataset/ prefix used in LOCO paths
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

        results.append(draw_boxes(img, boxes))
        print(f"  {img_path.name}: {len(boxes)} boxes")

    return results


def load_synth_samples(run_dir: Path, n: int, seed: int):
    rgb_files = sorted(run_dir.glob("rgb_*.png"))
    random.seed(seed)
    random.shuffle(rgb_files)

    results = []
    for rgb_path in rgb_files:
        if len(results) >= n:
            break
        frame_num = rgb_path.stem.split("_")[1]
        bbox_path = run_dir / f"bounding_box_2d_tight_{frame_num}.npy"
        label_path = run_dir / f"bounding_box_2d_tight_labels_{frame_num}.json"

        if not bbox_path.exists() or not label_path.exists():
            continue

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
            continue

        img = Image.open(rgb_path)
        results.append(draw_boxes(img, boxes))
        print(f"  {rgb_path.name}: {len(boxes)} boxes")

    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize bounding boxes for sanity checking")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_real = sub.add_parser("real", help="LOCO real data")
    p_real.add_argument("--ann", required=True, help="COCO annotation JSON path")
    p_real.add_argument("--img-root", required=True, help="Root directory of LOCO images")
    p_real.add_argument("--n", type=int, default=8, help="Number of images to show")
    p_real.add_argument("--output", default="output/sanity_real.png")
    p_real.add_argument("--seed", type=int, default=42)

    p_synth = sub.add_parser("synth", help="Isaac Sim synthetic data")
    p_synth.add_argument("--run-dir", required=True, help="Isaac Sim run output directory")
    p_synth.add_argument("--n", type=int, default=8, help="Number of frames to show")
    p_synth.add_argument("--output", default="output/sanity_synth.png")
    p_synth.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode == "real":
        print(f"Loading {args.n} real samples from {args.ann}...")
        imgs = load_real_samples(Path(args.ann), Path(args.img_root), args.n, args.seed)
    else:
        print(f"Loading {args.n} synthetic samples from {args.run_dir}...")
        imgs = load_synth_samples(Path(args.run_dir), args.n, args.seed)

    if not imgs:
        print("No images with matching boxes found.")
        return

    grid = make_grid(imgs, cols=4)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"\nSaved {len(imgs)}-image grid → {out}")


if __name__ == "__main__":
    main()

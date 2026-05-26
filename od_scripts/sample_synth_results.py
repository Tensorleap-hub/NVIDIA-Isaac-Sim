"""
Pick N images from each subfolder of a synthetic dataset directory, draw
ground-truth boxes, and save to <dataset_dir>/test/.

Usage:
    python od_scripts/sample_synth_results.py \
        --dataset-dir /path/to/base_v2_final \
        --n 2 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SYNTH_CLASS_MAP = {"palletjack": "pallet_truck", "forklift": "forklift", "pallet": "pallet"}
KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}
CLASS_COLORS = {
    "pallet_truck": (255, 80,  80),
    "forklift":     (80,  200, 80),
    "pallet":       (80,  120, 255),
}


def draw_boxes(image: Image.Image, boxes_and_labels: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1, cls in boxes_and_labels:
        color = CLASS_COLORS.get(cls, (255, 255, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label_w = len(cls) * 8 + 4
        draw.rectangle([x0, y0 - 16, x0 + label_w, y0], fill=color)
        draw.text((x0 + 2, y0 - 15), cls, fill=(0, 0, 0))
    return img


def sample_folder(folder: Path, n: int, seed: int) -> list[tuple[str, Image.Image]]:
    rgb_files = sorted(folder.glob("rgb_*.png"))
    if not rgb_files:
        return []

    rng = random.Random(seed)
    rng.shuffle(rgb_files)

    results = []
    for rgb_path in rgb_files:
        if len(results) >= n:
            break
        frame_num = rgb_path.stem.split("_")[1]
        bbox_path = folder / f"bounding_box_2d_tight_{frame_num}.npy"
        label_path = folder / f"bounding_box_2d_tight_labels_{frame_num}.json"

        if not bbox_path.exists() or not label_path.exists():
            continue

        bboxes = np.load(bbox_path, allow_pickle=True)
        with open(label_path) as f:
            label_map = json.load(f)

        sem_to_class = {}
        for sem_str, info in label_map.items():
            mapped = SYNTH_CLASS_MAP.get(info.get("class", ""), "")
            if mapped in KEEP_CLASSES:
                sem_to_class[int(sem_str)] = mapped

        boxes = []
        for row in bboxes:
            sem_id = int(row["semanticId"])
            if sem_id not in sem_to_class:
                continue
            x0, y0, x1, y1 = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
            if x1 - x0 > 0 and y1 - y0 > 0:
                boxes.append((x0, y0, x1, y1, sem_to_class[sem_id]))

        if not boxes:
            continue

        img = draw_boxes(Image.open(rgb_path), boxes)
        results.append((f"{folder.name}__{rgb_path.stem}", img))

    return results


def main():
    parser = argparse.ArgumentParser(description="Sample and visualize synthetic GT boxes per experiment folder")
    parser.add_argument("--dataset-dir", default="/Users/orram/Tensorleap/data/warehouse/base_v2_final")
    parser.add_argument("--n", type=int, default=2, help="Images to sample per folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = dataset_dir / "test"
    out_dir.mkdir(exist_ok=True)

    folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "test")
    if not folders:
        print(f"No subfolders found in {dataset_dir}")
        return

    total = 0
    for folder in folders:
        samples = sample_folder(folder, args.n, args.seed)
        if not samples:
            print(f"  {folder.name}: no annotated frames found")
            continue
        for name, img in samples:
            out_path = out_dir / f"{name}.jpg"
            img.save(out_path, quality=92)
            print(f"  {out_path.name}")
        total += len(samples)

    print(f"\n{total} images saved to {out_dir}")


if __name__ == "__main__":
    main()

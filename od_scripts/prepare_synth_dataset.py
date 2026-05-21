"""
Convert Isaac Sim BasicWriter output to COCO format for RF-DETR training.

Isaac Sim writes per-frame triplets:
    rgb_XXXX.png
    bounding_box_2d_tight_XXXX.npy   — structured array with x_min/y_min/x_max/y_max/semanticId
    bounding_box_2d_tight_labels_XXXX.json — maps str(semanticId) → {"class": "<name>"}

Class mapping applied:
    palletjack → pallet_truck   (matches LOCO label name)
    forklift   → forklift
    pallet     → pallet

Output layout (COCO format expected by RF-DETR):
    <output_dir>/
        train/
            _annotations.coco.json
            <symlinked rgb images>
        valid/
            _annotations.coco.json
            <symlinked rgb images>

Usage:
    # Convert a single run directory
    python scripts/prepare_synth_dataset.py \\
        --input-dirs /path/to/iter000_run000 /path/to/iter000_run001 ... \\
        --output-dir /data/warehouse3cls_synth \\
        --val-fraction 0.1

    # Merge with existing LOCO dataset (append images into the same split dirs)
    python scripts/prepare_synth_dataset.py \\
        --input-dirs /path/to/synth_outputs/* \\
        --output-dir /data/warehouse3cls \\
        --merge
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

CLASS_MAP = {
    "palletjack": "pallet_truck",
    "forklift": "forklift",
    "pallet": "pallet",
}

CLASSES = ["pallet_truck", "forklift", "pallet"]
CAT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}


def load_existing_coco(ann_path: Path):
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(CLASSES)],
        "images": [],
        "annotations": [],
    }


def collect_frames(input_dir: Path):
    """Return sorted list of frame numbers that have all required files present."""
    rgb_files = {int(f.stem.split("_")[1]): f for f in input_dir.glob("rgb_*.png")}
    bbox_files = {int(f.stem.split("_")[-1]): f for f in input_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
    label_files = {int(f.stem.split("_")[-1]): f
                   for f in input_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
    prim_path_files = {int(f.stem.split("_")[-1]): f
                       for f in input_dir.glob("bounding_box_2d_tight_prim_paths_[0-9]*.json")}
    if prim_path_files:
        complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files) & set(prim_path_files))
        return [(rgb_files[n], bbox_files[n], label_files[n], prim_path_files[n]) for n in complete]
    complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files))
    return [(rgb_files[n], bbox_files[n], label_files[n], None) for n in complete]


def frames_to_coco(
    frames,
    output_img_dir: Path,
    existing_coco: dict,
    run_prefix: str,
) -> dict:
    coco = existing_coco
    next_img_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0) + 1

    for rgb_path, bbox_path, label_path, prim_path_file in frames:
        with open(label_path) as f:
            label_map = json.load(f)  # {"0": {"class": "palletjack"}, ...}

        bboxes = np.load(bbox_path, allow_pickle=True)
        if len(bboxes) == 0:
            continue

        # Map semanticId (int) → canonical class name
        sem_to_class = {}
        for sem_str, info in label_map.items():
            raw = info.get("class", "")
            mapped = CLASS_MAP.get(raw, raw)
            if mapped in CAT_NAME_TO_ID:
                sem_to_class[int(sem_str)] = mapped

        # Deduplicate rows using prim paths: keep only root-level entries (no "/Ref/" in path).
        # Each object instance has exactly one root entry. Child mesh entries (e.g.
        # /Ref/S_ForkliftBody, /Ref/SM_PaletteA_01) are duplicates of the root bbox.
        # Without this, forklifts and pallets are double-counted; with a naive "keep only
        # /Ref/ rows" strategy, palletjacks (which have no child mesh) would be lost entirely.
        if prim_path_file is not None:
            with open(prim_path_file) as f:
                prim_paths = json.load(f)
            keep_indices = {i for i, path in enumerate(prim_paths) if "/Ref/" not in path}
        else:
            keep_indices = set(range(len(bboxes)))

        # Collect valid annotations for this frame
        frame_anns = []
        for i, row in enumerate(bboxes):
            if i not in keep_indices:
                continue
            sem_id = int(row["semanticId"])
            if sem_id not in sem_to_class:
                continue
            x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
            w = x_max - x_min
            h = y_max - y_min
            if w <= 0 or h <= 0:
                continue
            frame_anns.append({
                "category_id": CAT_NAME_TO_ID[sem_to_class[sem_id]],
                "bbox": [x_min, y_min, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })

        if not frame_anns:
            continue

        # Determine image dimensions from filename convention (960×544 default)
        from PIL import Image as PILImage
        with PILImage.open(rgb_path) as img:
            img_w, img_h = img.size

        fname = f"{run_prefix}_{rgb_path.name}"
        dst = output_img_dir / fname
        if not dst.exists():
            os.symlink(rgb_path.resolve(), dst)

        img_id = next_img_id
        next_img_id += 1
        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": img_w,
            "height": img_h,
        })

        for ann in frame_anns:
            ann["id"] = next_ann_id
            ann["image_id"] = img_id
            next_ann_id += 1
            coco["annotations"].append(ann)

    return coco


def main():
    parser = argparse.ArgumentParser(description="Convert Isaac Sim output to COCO for RF-DETR (3 warehouse classes)")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="One or more Isaac Sim run output directories")
    parser.add_argument("--output-dir", required=True,
                        help="Output dataset directory (COCO layout: train/ and valid/)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of frames to put in valid split (default: 0.1)")
    parser.add_argument("--merge", action="store_true",
                        help="Append to existing _annotations.coco.json files instead of overwriting")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_ann_path = train_dir / "_annotations.coco.json"
    valid_ann_path = valid_dir / "_annotations.coco.json"

    train_coco = load_existing_coco(train_ann_path) if args.merge else {
        "info": {},
        "licenses": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(CLASSES)],
        "images": [],
        "annotations": [],
    }
    valid_coco = load_existing_coco(valid_ann_path) if args.merge else {
        "info": {},
        "licenses": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(CLASSES)],
        "images": [],
        "annotations": [],
    }

    for input_dir_str in args.input_dirs:
        input_dir = Path(input_dir_str)
        if not input_dir.is_dir():
            print(f"  SKIP (not a directory): {input_dir}")
            continue

        frames = collect_frames(input_dir)
        if not frames:
            print(f"  SKIP (no complete frame triplets): {input_dir}")
            continue

        random.shuffle(frames)
        n_val = max(1, int(len(frames) * args.val_fraction))
        val_frames = frames[:n_val]
        train_frames = frames[n_val:]

        run_prefix = input_dir.name
        print(f"  {input_dir.name}: {len(train_frames)} train, {len(val_frames)} val frames")

        train_coco = frames_to_coco(train_frames, train_dir, train_coco, run_prefix)
        valid_coco = frames_to_coco(val_frames, valid_dir, valid_coco, run_prefix)

    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(valid_ann_path, "w") as f:
        json.dump(valid_coco, f, indent=2)

    print(f"\nTrain: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"Valid: {len(valid_coco['images'])} images, {len(valid_coco['annotations'])} annotations")
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()

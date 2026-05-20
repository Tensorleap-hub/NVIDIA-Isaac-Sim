"""
Filter a LOCO COCO-format annotation file to the three warehouse classes
(pallet_truck, forklift, pallet) and write a COCO dataset ready for RF-DETR.

Output layout (COCO format expected by RF-DETR):
    <output_dir>/
        train/
            _annotations.coco.json
            <image files (symlinked or copied)>
        valid/
            _annotations.coco.json
            <image files (symlinked or copied)>

Usage:
    python scripts/prepare_loco_dataset.py \
        --train-ann  /path/to/loco_dataset/labels/loco-sub3-v1-train.json \
        --train-imgs /path/to/loco_dataset/subset-3 \
        --val-ann    /path/to/loco_dataset/labels/loco-sub4-v1-val.json \
        --val-imgs   /path/to/loco_dataset/subset-4 \
        --output-dir /data/warehouse3cls
"""

import argparse
import json
import os
import shutil
from pathlib import Path

KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}


def filter_coco(ann_path: str, images_dir: str, output_dir: str, split: str) -> None:
    with open(ann_path) as f:
        coco = json.load(f)

    # Build new category list with contiguous IDs starting at 1
    old_cats = {c["id"]: c["name"] for c in coco["categories"]}
    kept_cats = [(old_id, name) for old_id, name in sorted(old_cats.items()) if name in KEEP_CLASSES]
    if not kept_cats:
        raise ValueError(f"None of {KEEP_CLASSES} found in {ann_path}. Categories: {list(old_cats.values())}")

    old_to_new_cat = {}
    new_categories = []
    for new_id, (old_id, name) in enumerate(kept_cats, start=1):
        old_to_new_cat[old_id] = new_id
        new_categories.append({"id": new_id, "name": name, "supercategory": ""})

    # Filter annotations
    kept_ann_image_ids = set()
    new_annotations = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new_cat:
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = old_to_new_cat[ann["category_id"]]
        new_annotations.append(new_ann)
        kept_ann_image_ids.add(ann["image_id"])

    # Keep only images that have at least one kept annotation
    new_images = [img for img in coco["images"] if img["id"] in kept_ann_image_ids]

    out_split_dir = Path(output_dir) / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # Symlink or copy images
    for img_meta in new_images:
        fname = img_meta["file_name"]
        src = Path(images_dir) / fname
        if not src.exists():
            # Try searching recursively (LOCO stores images in date subfolders)
            matches = list(Path(images_dir).rglob(fname))
            if not matches:
                print(f"  WARNING: image not found: {fname}")
                continue
            src = matches[0]
        dst = out_split_dir / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations,
    }
    ann_out = out_split_dir / "_annotations.coco.json"
    with open(ann_out, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"[{split}] {len(new_images)} images, {len(new_annotations)} annotations "
          f"across {len(new_categories)} classes → {ann_out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LOCO subset for RF-DETR (3 warehouse classes)")
    parser.add_argument("--train-ann", required=True, help="Path to LOCO train annotation JSON")
    parser.add_argument("--train-imgs", required=True, help="Root directory of train images")
    parser.add_argument("--val-ann", required=True, help="Path to LOCO val annotation JSON")
    parser.add_argument("--val-imgs", required=True, help="Root directory of val images")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    args = parser.parse_args()

    print(f"Keeping classes: {sorted(KEEP_CLASSES)}")
    filter_coco(args.train_ann, args.train_imgs, args.output_dir, "train")
    filter_coco(args.val_ann, args.val_imgs, args.output_dir, "valid")
    print(f"\nDataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()

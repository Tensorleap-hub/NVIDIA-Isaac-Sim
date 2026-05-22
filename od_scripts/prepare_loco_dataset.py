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
    # Single train annotation
    python scripts/prepare_loco_dataset.py \
        --train-ann  /path/to/labels/loco-sub2-v1-train.json \
        --train-imgs /path/to/dataset \
        --val-ann    /path/to/labels/loco-sub3-v1-train.json \
        --val-imgs   /path/to/dataset/subset-3 \
        --output-dir /data/warehouse3cls

    # Multiple train annotations (repeat --train-ann for each subset)
    python scripts/prepare_loco_dataset.py \
        --train-ann  loco-sub1-v1-val.json \
        --train-ann  loco-sub2-v1-train.json \
        --train-ann  loco-sub4-v1-val.json \
        --train-ann  loco-sub5-v1-train.json \
        --train-imgs /path/to/dataset \
        --val-ann    loco-sub3-v1-train.json \
        --val-imgs   /path/to/dataset/subset-3 \
        --output-dir /data/warehouse3cls
"""

import argparse
import json
import os
import shutil
from pathlib import Path

KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}


def _build_category_map(coco: dict, ann_path: str) -> tuple[dict, list]:
    old_cats = {c["id"]: c["name"] for c in coco["categories"]}
    kept_cats = [(old_id, name) for old_id, name in sorted(old_cats.items()) if name in KEEP_CLASSES]
    if not kept_cats:
        raise ValueError(f"None of {KEEP_CLASSES} found in {ann_path}. Categories: {list(old_cats.values())}")
    old_to_new = {}
    new_categories = []
    for new_id, (old_id, name) in enumerate(kept_cats, start=1):
        old_to_new[old_id] = new_id
        new_categories.append({"id": new_id, "name": name, "supercategory": ""})
    return old_to_new, new_categories


def filter_coco_into(
    ann_path: str,
    images_dir: str,
    out_split_dir: Path,
    coco_out: dict,
) -> None:
    """Filter one annotation file and merge its images/annotations into coco_out."""
    with open(ann_path) as f:
        coco = json.load(f)

    old_to_new_cat, _ = _build_category_map(coco, ann_path)

    img_id_offset = max((img["id"] for img in coco_out["images"]), default=0)
    ann_id_offset = max((ann["id"] for ann in coco_out["annotations"]), default=0)

    kept_ann_image_ids: set[int] = set()
    new_annotations = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new_cat:
            continue
        new_ann = dict(ann)
        new_ann["id"] = ann["id"] + ann_id_offset
        new_ann["image_id"] = ann["image_id"] + img_id_offset
        new_ann["category_id"] = old_to_new_cat[ann["category_id"]]
        new_annotations.append(new_ann)
        kept_ann_image_ids.add(ann["image_id"])

    new_images = []
    for img_meta in coco["images"]:
        if img_meta["id"] not in kept_ann_image_ids:
            continue
        fname = img_meta["file_name"]
        src = Path(images_dir) / fname
        if not src.exists():
            matches = list(Path(images_dir).rglob(fname))
            if not matches:
                print(f"  WARNING: image not found: {fname}")
                continue
            src = matches[0]
        dst = out_split_dir / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        new_images.append({**img_meta, "id": img_meta["id"] + img_id_offset})

    coco_out["images"].extend(new_images)
    coco_out["annotations"].extend(new_annotations)

    print(f"  +{len(new_images)} images, +{len(new_annotations)} annotations from {Path(ann_path).name}")


def _empty_coco(ann_path: str) -> dict:
    with open(ann_path) as f:
        coco = json.load(f)
    _, new_categories = _build_category_map(coco, ann_path)
    return {"info": {}, "licenses": [], "categories": new_categories, "images": [], "annotations": []}


def main():
    parser = argparse.ArgumentParser(description="Prepare LOCO subset for RF-DETR (3 warehouse classes)")
    parser.add_argument("--train-ann", action="append", dest="train_ann", required=True, metavar="ANN_JSON",
                        help="LOCO train annotation JSON (repeat for multiple subsets)")
    parser.add_argument("--train-imgs", required=True, metavar="IMGS_ROOT",
                        help="Shared root directory containing all train images")
    parser.add_argument("--val-ann", required=True, help="Path to LOCO val annotation JSON")
    parser.add_argument("--val-imgs", required=True, help="Root directory of val images")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory")
    args = parser.parse_args()

    train_imgs_list = [args.train_imgs] * len(args.train_ann)

    print(f"Keeping classes: {sorted(KEEP_CLASSES)}")

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    print("\n[train]")
    train_coco = _empty_coco(args.train_ann[0])
    for ann_path, imgs_dir in zip(args.train_ann, train_imgs_list):
        filter_coco_into(ann_path, imgs_dir, train_dir, train_coco)
    ann_out = train_dir / "_annotations.coco.json"
    with open(ann_out, "w") as f:
        json.dump(train_coco, f, indent=2)
    print(f"  → {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations → {ann_out}")

    print("\n[valid]")
    valid_coco = _empty_coco(args.val_ann)
    filter_coco_into(args.val_ann, args.val_imgs, valid_dir, valid_coco)
    ann_out = valid_dir / "_annotations.coco.json"
    with open(ann_out, "w") as f:
        json.dump(valid_coco, f, indent=2)
    print(f"  → {len(valid_coco['images'])} images, {len(valid_coco['annotations'])} annotations → {ann_out}")

    print(f"\nDataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()

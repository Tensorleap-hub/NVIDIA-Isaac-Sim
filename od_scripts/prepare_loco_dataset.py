"""
Filter LOCO COCO-format annotation files to the three warehouse classes
(pallet_truck, forklift, pallet) and write a COCO dataset ready for RF-DETR.

Multiple annotation files for the same split are merged (IDs are re-assigned
to avoid collisions).  Images are symlinked from their original locations.

Output layout (COCO format expected by RF-DETR):
    <output_dir>/
        train/
            _annotations.coco.json
            <image files (symlinked)>
        valid/
            _annotations.coco.json
            <image files (symlinked)>

Usage:
    # Single train annotation + single val annotation
    python od_scripts/prepare_loco_dataset.py \\
        --train-ann  /path/to/labels/loco-sub2-v1-train.json \\
        --train-imgs /path/to/dataset \\
        --val-ann    /path/to/labels/loco-sub3-v1-train.json \\
        --val-imgs   /path/to/dataset \\
        --output-dir /data/warehouse3cls_real

    # Multiple training subsets merged together
    python od_scripts/prepare_loco_dataset.py \\
        --train-ann  /path/to/labels/loco-sub1-v1-val.json \\
                     /path/to/labels/loco-sub2-v1-train.json \\
                     /path/to/labels/loco-sub4-v1-val.json \\
                     /path/to/labels/loco-sub5-v1-train.json \\
        --train-imgs /path/to/dataset \\
        --val-ann    /path/to/labels/loco-sub3-v1-train.json \\
        --val-imgs   /path/to/dataset \\
        --output-dir /data/warehouse3cls_real
"""

import argparse
import json
import os
from pathlib import Path

KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}
# Fixed output category order so IDs are stable across runs
CLASSES = ["pallet_truck", "forklift", "pallet"]
NEW_CATEGORIES = [{"id": i + 1, "name": c, "supercategory": ""} for i, c in enumerate(CLASSES)]
CAT_NAME_TO_NEW_ID = {c: i + 1 for i, c in enumerate(CLASSES)}


def _load_and_filter(ann_path: Path, images_root: Path) -> tuple[list, list]:
    """Return (images, annotations) filtered to KEEP_CLASSES, IDs not yet remapped."""
    with open(ann_path) as f:
        coco = json.load(f)

    old_cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    old_to_new_cat = {
        old_id: CAT_NAME_TO_NEW_ID[name]
        for old_id, name in old_cat_name.items()
        if name in KEEP_CLASSES
    }
    if not old_to_new_cat:
        raise ValueError(f"None of {KEEP_CLASSES} found in {ann_path}. "
                         f"Categories: {list(old_cat_name.values())}")

    kept_image_ids = set()
    anns = []
    for ann in coco["annotations"]:
        if ann["category_id"] not in old_to_new_cat:
            continue
        a = dict(ann)
        a["category_id"] = old_to_new_cat[ann["category_id"]]
        anns.append(a)
        kept_image_ids.add(ann["image_id"])

    imgs = [img for img in coco["images"] if img["id"] in kept_image_ids]

    # Attach resolved path to each image for symlinking
    for img in imgs:
        rel = img.get("path", "")
        if rel:
            parts = rel.lstrip("/").split("/", 1)
            rel = parts[1] if len(parts) == 2 else parts[0]
            img["_resolved_src"] = images_root / rel
        else:
            fname = img["file_name"]
            matches = list(images_root.rglob(fname))
            img["_resolved_src"] = matches[0] if matches else None

    return imgs, anns


def _merge_and_write(
    split_entries: list[tuple[Path, Path]],   # list of (ann_path, imgs_root)
    out_split_dir: Path,
    split_name: str,
) -> None:
    """Merge multiple (ann, imgs_root) pairs into one COCO JSON, symlinking images."""
    out_split_dir.mkdir(parents=True, exist_ok=True)

    next_img_id = 1
    next_ann_id = 1
    all_images = []
    all_annotations = []

    for ann_path, imgs_root in split_entries:
        imgs, anns = _load_and_filter(ann_path, imgs_root)

        # Remap IDs to be globally unique within the merged set
        old_to_new_img = {}
        for img in imgs:
            new_id = next_img_id
            old_to_new_img[img["id"]] = new_id
            next_img_id += 1
            src = img.pop("_resolved_src", None)
            img = dict(img)
            img["id"] = new_id
            all_images.append(img)

            # Symlink image
            if src and src.exists():
                dst = out_split_dir / img["file_name"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
            else:
                print(f"  WARNING: image not found: {img.get('file_name')}")

        for ann in anns:
            if ann["image_id"] not in old_to_new_img:
                continue
            a = dict(ann)
            a["id"] = next_ann_id
            a["image_id"] = old_to_new_img[ann["image_id"]]
            next_ann_id += 1
            all_annotations.append(a)

    coco_out = {
        "info": {},
        "licenses": [],
        "categories": NEW_CATEGORIES,
        "images": all_images,
        "annotations": all_annotations,
    }
    ann_out = out_split_dir / "_annotations.coco.json"
    with open(ann_out, "w") as f:
        json.dump(coco_out, f, indent=2)

    print(f"[{split_name}] {len(all_images)} images, {len(all_annotations)} annotations "
          f"from {len(split_entries)} annotation file(s) → {ann_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LOCO subsets for RF-DETR (3 warehouse classes). "
                    "Multiple --train-ann files are merged."
    )
    parser.add_argument("--train-ann", nargs="+", required=True,
                        help="One or more LOCO train annotation JSON paths (merged)")
    parser.add_argument("--train-imgs", required=True,
                        help="Root directory containing training images")
    parser.add_argument("--val-ann", nargs="+", required=True,
                        help="One or more LOCO val annotation JSON paths (merged)")
    parser.add_argument("--val-imgs", required=True,
                        help="Root directory containing validation images")
    parser.add_argument("--output-dir", required=True,
                        help="Output dataset directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    train_imgs = Path(args.train_imgs)
    val_imgs = Path(args.val_imgs)

    print(f"Keeping classes: {CLASSES}")
    _merge_and_write(
        [(Path(a), train_imgs) for a in args.train_ann],
        out / "train", "train"
    )
    _merge_and_write(
        [(Path(a), val_imgs) for a in args.val_ann],
        out / "valid", "valid"
    )
    print(f"\nDataset written to: {out}")


if __name__ == "__main__":
    main()

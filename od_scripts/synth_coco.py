"""Isaac Sim BasicWriter output -> COCO annotations (merge into an existing COCO dict).

Per-frame files (flat layout; nested `Camera/` trajectory layout also supported):
    rgb_XXXX.png
    bounding_box_2d_tight_XXXX.npy                 x_min/y_min/x_max/y_max/semanticId
    bounding_box_2d_tight_labels_XXXX.json         {"<semanticId>": {"class": "<name>"}}
    bounding_box_2d_tight_prim_paths_XXXX.json     one prim path per npy row

Class mapping: palletjack -> pallet_truck (LOCO name); forklift, pallet unchanged.

The semanticId -> class mapping is assigned per run in first-seen order, so it is
NOT stable across experiments: always resolve through the per-frame labels json.

Dedup rule (do not "simplify"): the writer emits one row per semantic prim, i.e. the
object root AND its child meshes (/Ref/S_ForkliftBody, /Ref/SM_PaletteA_01, ...).
Keeping every row double-counts forklifts/pallets; keeping only /Ref/ rows drops
palletjacks (no child mesh). We keep rows whose prim path does NOT contain "/Ref/".

Lifted from the previous od_scripts/prepare_synth_dataset.py (behaviour unchanged).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_MAP = {"palletjack": "pallet_truck", "forklift": "forklift", "pallet": "pallet"}


def collect_frames(input_dir: Path):
    """Sorted (rgb, npy, labels_json, prim_paths_json|None) tuples with all files present."""
    rgb_dir = bbox_dir = input_dir
    nested_rgb = input_dir / "Camera" / "rgb"
    if not any(input_dir.glob("rgb_*.png")) and nested_rgb.is_dir():
        rgb_dir = nested_rgb
        bbox_dir = input_dir / "Camera" / "bounding_box_2d_tight"

    rgb = {int(f.stem.split("_")[1]): f for f in rgb_dir.glob("rgb_*.png")}
    bbox = {int(f.stem.split("_")[-1]): f for f in bbox_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
    labels = {int(f.stem.split("_")[-1]): f for f in bbox_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
    prims = {int(f.stem.split("_")[-1]): f for f in bbox_dir.glob("bounding_box_2d_tight_prim_paths_[0-9]*.json")}
    if prims:
        complete = sorted(set(rgb) & set(bbox) & set(labels) & set(prims))
        return [(rgb[n], bbox[n], labels[n], prims[n]) for n in complete]
    complete = sorted(set(rgb) & set(bbox) & set(labels))
    return [(rgb[n], bbox[n], labels[n], None) for n in complete]


def frames_to_coco(frames, output_img_dir: Path, coco: dict, run_prefix: str) -> dict:
    """Append frames (with >=1 kept box) to `coco`, symlinking images as <run_prefix>_<rgb name>.

    Category ids are resolved from `coco["categories"]` by NAME, so the synth labels
    follow whatever id order the (real-seeded) dataset already uses.
    """
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    next_img_id = max((im["id"] for im in coco["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    for rgb_path, bbox_path, label_path, prim_path_file in frames:
        with open(label_path) as f:
            label_map = json.load(f)
        bboxes = np.load(bbox_path, allow_pickle=True)
        if len(bboxes) == 0:
            continue

        sem_to_class = {}
        for sem_str, info in label_map.items():
            mapped = CLASS_MAP.get(info.get("class", ""), info.get("class", ""))
            if mapped in name_to_id:
                sem_to_class[int(sem_str)] = mapped

        if prim_path_file is not None:
            with open(prim_path_file) as f:
                prim_paths = json.load(f)
            keep = {i for i, p in enumerate(prim_paths) if "/Ref/" not in p}
        else:
            keep = set(range(len(bboxes)))

        anns = []
        for i, row in enumerate(bboxes):
            if i not in keep:
                continue
            sem_id = int(row["semanticId"])
            if sem_id not in sem_to_class:
                continue
            x0, y0, x1, y1 = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            anns.append({"category_id": name_to_id[sem_to_class[sem_id]],
                         "bbox": [x0, y0, w, h], "area": float(w * h), "iscrowd": 0})
        if not anns:
            continue

        with Image.open(rgb_path) as im:
            img_w, img_h = im.size

        fname = f"{run_prefix}_{rgb_path.name}"
        dst = output_img_dir / fname
        if not dst.exists():
            os.symlink(rgb_path.resolve(), dst)

        img_id = next_img_id
        next_img_id += 1
        coco["images"].append({"id": img_id, "file_name": fname, "width": img_w, "height": img_h})
        for a in anns:
            a["id"] = next_ann_id
            a["image_id"] = img_id
            next_ann_id += 1
            coco["annotations"].append(a)
    return coco

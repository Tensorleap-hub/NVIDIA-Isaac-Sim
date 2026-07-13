"""
Convert Cosmos-Transfer2.5 stylized videos into COCO frames for RF-DETR training.

Cosmos-Transfer restyles the *appearance* of an Isaac Sim trajectory-SDG clip while
preserving its exact geometry (it's conditioned on that same clip's depth/edge/
segmentation control videos), so the stylized output video is frame-for-frame aligned
with the original render: same frame count, same resolution, same object layout. That
means the original per-frame bounding boxes are still valid ground truth for the new,
stylized pixels — nothing needs to be re-annotated.

For each Cosmos output sample this script:
  1. reads the output's <sample>.json sidecar and follows its "video_path" field back
     to the source experiment dir (<exp_dir>/video/clip_0000/rgb.mp4)
  2. loads that experiment's per-frame ground truth from
     <exp_dir>/Camera/bounding_box_2d_tight/bounding_box_2d_tight_XXXX.npy (+ labels json)
  3. decodes the stylized <sample>.mp4 frame by frame and pairs frame XXXX with the
     matching annotation, writing the frame out as a PNG and a COCO record

Class mapping matches prepare_synth_dataset.py: palletjack -> pallet_truck, forklift,
pallet (other classes, e.g. "person", are dropped — not in the 3-class warehouse set).

Train/valid split is done per *sample* (whole video), not per frame: adjacent frames of
the same clip are near-duplicates, so a frame-level split would leak near-identical
images across train/valid and inflate validation metrics.

Output layout (COCO format expected by RF-DETR, same as prepare_synth_dataset.py):
    <output_dir>/
        train/
            _annotations.coco.json
            <extracted stylized frames, *.png>
        valid/
            _annotations.coco.json
            <extracted stylized frames, *.png>

Usage:
    # Discover every finished (non-control) sample under one or more Cosmos output batch dirs
    python od_scripts/prepare_cosmos_dataset.py \\
        --cosmos-output-dirs /mnt/cosmos/output/batch1_new_themes_2seeds /mnt/cosmos/output/all_seeds_2prompts \\
        --output-dir od_scripts/data/warehouse3cls_cosmos \\
        --val-fraction 0.1

    # Re-run later as more batches finish, merging into the same prepared dataset
    python od_scripts/prepare_cosmos_dataset.py \\
        --cosmos-output-dirs /mnt/cosmos/output/batchB_optuna_23runs \\
        --output-dir od_scripts/data/warehouse3cls_cosmos \\
        --merge

Requires opencv-python-headless (already in tensorleap_requirements.txt) for video decode.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_MAP = {
    "palletjack": "pallet_truck",
    "forklift": "forklift",
    "pallet": "pallet",
}

CLASSES = ["pallet_truck", "forklift", "pallet"]
CAT_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}


def empty_coco() -> dict:
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(CLASSES)],
        "images": [],
        "annotations": [],
    }


def load_existing_coco(ann_path: Path) -> dict:
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)
    return empty_coco()


def find_samples(cosmos_output_dirs: list[str]) -> list[Path]:
    """Return sorted list of finished, non-control Cosmos output mp4s that have a JSON sidecar."""
    samples = []
    for d in cosmos_output_dirs:
        out_dir = Path(d)
        if not out_dir.is_dir():
            print(f"  SKIP (not a directory): {out_dir}")
            continue
        for mp4 in sorted(out_dir.glob("*.mp4")):
            if "_control_" in mp4.name:
                continue
            if mp4.with_suffix(".json").exists():
                samples.append(mp4)
            else:
                print(f"  SKIP (no .json sidecar): {mp4}")
    return samples


def resolve_source_exp_dir(sample_json: dict) -> Path | None:
    """The output sidecar's video_path points at <exp_dir>/video/clip_XXXX/rgb.mp4."""
    video_path = sample_json.get("video_path")
    if not video_path:
        return None
    return Path(video_path).parents[2]


def load_frame_annotations(bbox_dir: Path, frame_idx: int):
    """Return list of COCO-style annotation dicts (bbox/area/category_id) for one frame, or None if missing."""
    bbox_path = bbox_dir / f"bounding_box_2d_tight_{frame_idx:04d}.npy"
    label_path = bbox_dir / f"bounding_box_2d_tight_labels_{frame_idx:04d}.json"
    if not bbox_path.exists() or not label_path.exists():
        return None

    with open(label_path) as f:
        label_map = json.load(f)  # {"0": {"class": "palletjack"}, ...}

    sem_to_class = {}
    for sem_str, info in label_map.items():
        mapped = CLASS_MAP.get(info.get("class", ""), info.get("class", ""))
        if mapped in CAT_NAME_TO_ID:
            sem_to_class[int(sem_str)] = mapped

    bboxes = np.load(bbox_path, allow_pickle=True)
    anns = []
    for row in bboxes:
        sem_id = int(row["semanticId"])
        if sem_id not in sem_to_class:
            continue
        x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        w, h = x_max - x_min, y_max - y_min
        if w <= 0 or h <= 0:
            continue
        anns.append({
            "category_id": CAT_NAME_TO_ID[sem_to_class[sem_id]],
            "bbox": [x_min, y_min, w, h],
            "area": float(w * h),
            "iscrowd": 0,
        })
    return anns


def process_sample(mp4_path: Path, coco: dict, output_img_dir: Path, frame_stride: int) -> int:
    """Decode mp4_path, pair each frame with its source annotations, append to coco. Returns frames added."""
    sample_json = json.loads(mp4_path.with_suffix(".json").read_text())
    exp_dir = resolve_source_exp_dir(sample_json)
    if exp_dir is None or not exp_dir.is_dir():
        print(f"  SKIP (can't resolve source experiment dir): {mp4_path.name}")
        return 0

    bbox_dir = exp_dir / "Camera" / "bounding_box_2d_tight"
    if not bbox_dir.is_dir():
        print(f"  SKIP (no Camera/bounding_box_2d_tight under {exp_dir}): {mp4_path.name}")
        return 0

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  SKIP (failed to open video): {mp4_path.name}")
        return 0

    next_img_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0) + 1

    sample_name = mp4_path.stem
    frame_idx = 0
    n_added = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            anns = load_frame_annotations(bbox_dir, frame_idx)
            if anns:
                fname = f"{sample_name}_{frame_idx:04d}.png"
                dst = output_img_dir / fname
                if not dst.exists():
                    cv2.imwrite(str(dst), frame)

                img_id = next_img_id
                next_img_id += 1
                coco["images"].append({
                    "id": img_id,
                    "file_name": fname,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                })
                for ann in anns:
                    ann["id"] = next_ann_id
                    ann["image_id"] = img_id
                    next_ann_id += 1
                    coco["annotations"].append(ann)
                n_added += 1
        frame_idx += 1
    cap.release()
    return n_added


def main():
    parser = argparse.ArgumentParser(description="Convert Cosmos-Transfer2.5 output videos to COCO for RF-DETR (3 warehouse classes)")
    parser.add_argument("--cosmos-output-dirs", nargs="+", required=True,
                        help="One or more Cosmos-Transfer2.5 output directories (e.g. /mnt/cosmos/output/<run_name>)")
    parser.add_argument("--output-dir", required=True,
                        help="Output dataset directory (COCO layout: train/ and valid/)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of samples (whole videos) to put in the valid split (default: 0.1)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Only keep every Nth frame of each video (default: 1, i.e. keep all frames)")
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

    train_coco = load_existing_coco(train_ann_path) if args.merge else empty_coco()
    valid_coco = load_existing_coco(valid_ann_path) if args.merge else empty_coco()

    samples = find_samples(args.cosmos_output_dirs)
    if not samples:
        print("No finished Cosmos output samples found.")
        return

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_fraction))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Found {len(samples)} samples: {len(train_samples)} train, {len(val_samples)} valid")

    for mp4_path in train_samples:
        n = process_sample(mp4_path, train_coco, train_dir, args.frame_stride)
        print(f"  [train] {mp4_path.name}: {n} frames")
    for mp4_path in val_samples:
        n = process_sample(mp4_path, valid_coco, valid_dir, args.frame_stride)
        print(f"  [valid] {mp4_path.name}: {n} frames")

    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f, indent=2)
    with open(valid_ann_path, "w") as f:
        json.dump(valid_coco, f, indent=2)

    print(f"\nTrain: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"Valid: {len(valid_coco['images'])} images, {len(valid_coco['annotations'])} annotations")
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()

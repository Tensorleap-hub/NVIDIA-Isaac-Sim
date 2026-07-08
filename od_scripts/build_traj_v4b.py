"""Build warehouse3cls_traj_v4b: LOCO real train + valid (symlinked from
warehouse3cls_real) + trajectory SDG synth from the 2026-07-07 full re-run dump
(train_v4_20260707_181129, 21 cfg x 20 seeds x 10 frames = 4200 rendered frames).

All synth frames go into train/. valid = 858 real only (shared eval set).
Mirrors od_scripts/convert_trajectory_synth_v4.py but self-contained (also seeds
the real split) and pointed at the new dump + a fresh output dir.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/NVIDIA-Isaac-Sim/od_scripts")
from prepare_synth_dataset import frames_to_coco, load_existing_coco  # noqa: E402

DUMP = Path("/home/ubuntu/NVIDIA-Isaac-Sim/palletjack_sdg/palletjack_data/trajectory/train_v4_20260707_181129")
REAL = Path("/home/ubuntu/warehouse3cls_real")
OUT = Path("/home/ubuntu/warehouse3cls_traj_v4b")


def link_realpath(src_file: Path, dst_file: Path):
    real = os.path.realpath(src_file)
    if os.path.lexists(dst_file):
        os.remove(dst_file)
    os.symlink(real, dst_file)


def seed_split(split: str):
    """Symlink real images + copy real annotations for train/ or valid/."""
    src_dir = REAL / split
    dst_dir = OUT / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    ann = json.load(open(src_dir / "_annotations.coco.json"))
    json.dump(ann, open(dst_dir / "_annotations.coco.json", "w"))
    for im in ann["images"]:
        link_realpath(src_dir / im["file_name"], dst_dir / im["file_name"])
    print(f"seed {split}: {len(ann['images'])} real imgs, {len(ann['annotations'])} anns")
    return len(ann["images"]), len(ann["annotations"])


def collect_trajectory_frames(camera_dir: Path):
    rgb_dir = camera_dir / "rgb"
    bbox_dir = camera_dir / "bounding_box_2d_tight"
    rgb_files = {int(f.stem.split("_")[1]): f for f in rgb_dir.glob("rgb_*.png")}
    bbox_files = {int(f.stem.split("_")[-1]): f for f in bbox_dir.glob("bounding_box_2d_tight_[0-9]*.npy")}
    label_files = {int(f.stem.split("_")[-1]): f
                   for f in bbox_dir.glob("bounding_box_2d_tight_labels_[0-9]*.json")}
    prim_path_files = {int(f.stem.split("_")[-1]): f
                       for f in bbox_dir.glob("bounding_box_2d_tight_prim_paths_[0-9]*.json")}
    complete = sorted(set(rgb_files) & set(bbox_files) & set(label_files))
    return [(rgb_files[n], bbox_files[n], label_files[n], prim_path_files.get(n)) for n in complete]


def main():
    real_train_imgs, _ = seed_split("train")
    seed_split("valid")

    train_dir = OUT / "train"
    train_ann_path = train_dir / "_annotations.coco.json"
    train_coco = load_existing_coco(train_ann_path)

    seed_dirs = sorted(d for d in DUMP.iterdir() if d.is_dir() and d.name.startswith("exp"))
    print(f"\nFound {len(seed_dirs)} synth seed dirs")

    rgb_available = 0
    frames_added = 0
    skipped = []
    for seed_dir in seed_dirs:
        camera_dir = seed_dir / "Camera"
        if not camera_dir.is_dir():
            skipped.append(seed_dir.name)
            continue
        frames = collect_trajectory_frames(camera_dir)
        if not frames:
            skipped.append(seed_dir.name)
            continue
        rgb_available += len(frames)
        before = len(train_coco["images"])
        train_coco = frames_to_coco(frames, train_dir, train_coco, seed_dir.name)
        frames_added += len(train_coco["images"]) - before

    with open(train_ann_path, "w") as f:
        json.dump(train_coco, f)

    synth_imgs = len(train_coco["images"]) - real_train_imgs
    print(f"\nSynth RGB frames available (complete triplets): {rgb_available}")
    print(f"Synth frames kept (>=1 in-class annotation): {frames_added}")
    print(f"Synth frames dropped (no in-class object): {rgb_available - frames_added}")
    if skipped:
        print(f"Skipped {len(skipped)} empty/no-Camera dirs: {skipped}")
    print(f"\nFinal train: {len(train_coco['images'])} imgs "
          f"({real_train_imgs} real + {synth_imgs} synth), {len(train_coco['annotations'])} anns")
    print(f"Dataset -> {OUT}")


if __name__ == "__main__":
    main()

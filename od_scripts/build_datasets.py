"""Build every COCO dataset for the real/synth arm study under /home/ubuntu/datasets_coco.

    real/           train = LOCO sub1+2+4+5 (3 classes)      valid = LOCO subset-3 ONLY
    real_basev2/    train = real + base_v2_final              valid -> ../real/valid (dir symlink)
    real_may/       train = real + top-runs-may-ok            valid -> ../real/valid
    real_all/       train = real + base_v2 + may              valid -> ../real/valid
    real_traj/      train = real + trajectory-optimized       valid -> ../real/valid
    real_basev4/    train = real + base_v4 (trajectory+random) valid -> ../real/valid
    real_traj_basev4/ train = real + traj + base_v4            valid -> ../real/valid
    real_all_traj/  train = real + base_v2 + may + traj + v4  valid -> ../real/valid
    evalsets/       eval-only sets whose valid/ IS a training subset (train-fit diagnostics):
        train_real/valid   -> ../../real/train
        train_basev2/valid = base_v2 synth frames only
        train_may/valid    = may synth frames only
        train_traj_optuna/valid = trajectory-optimized synth frames only
        train_basev4/valid = base_v4 synth frames only
        train_optuna_rand/valid = optuna_rand (random-frame render of the 24 traj_optuna configs)

Invariants asserted at the end (hard failure):
  * every valid/ image is a subset-3 image, none of them appears in any train/
  * valid/ contains no synthetic frame (no .png)
  * identical 3-category table (ids+names) across all datasets
  * zero broken symlinks
All images are symlinks to the realpath of the source file. Writes MANIFEST.json.

Usage:  .venv/bin/python od_scripts/build_datasets.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ARMS, ARM_SUBSAMPLE, SUBSAMPLE_SEED, EVALSETS, SYNTH_SOURCES, KEEP_CLASSES, LOCO, LOCO_TRAIN_ANNS, LOCO_VAL_ANN,  # noqa: E402
                    LOCO_VAL_IMGS, OUT, source_of, synth_run_dirs)
from synth_coco import collect_frames, frames_to_coco  # noqa: E402


# ----------------------------------------------------------------------------- LOCO
def _category_map(coco: dict):
    old = {c["id"]: c["name"] for c in coco["categories"]}
    kept = [(oid, n) for oid, n in sorted(old.items()) if n in KEEP_CLASSES]
    assert len(kept) == 3, f"expected 3 kept classes, got {kept}"
    old_to_new = {oid: i for i, (oid, _) in enumerate(kept, start=1)}
    cats = [{"id": i, "name": n, "supercategory": ""} for i, (_, n) in enumerate(kept, start=1)]
    return old_to_new, cats


def _empty_coco(categories):
    return {"info": {}, "licenses": [], "categories": categories, "images": [], "annotations": []}


def loco_into(ann_path: Path, images_root: Path, split_dir: Path, coco: dict) -> tuple[int, int]:
    with open(ann_path) as f:
        src = json.load(f)
    old_to_new, cats = _category_map(src)
    assert cats == coco["categories"], "category table differs between LOCO files"
    img_off = max((i["id"] for i in coco["images"]), default=0)
    ann_off = max((a["id"] for a in coco["annotations"]), default=0)

    keep_img_ids, anns = set(), []
    for a in src["annotations"]:
        if a["category_id"] not in old_to_new:
            continue
        anns.append({"id": a["id"] + ann_off, "image_id": a["image_id"] + img_off,
                     "category_id": old_to_new[a["category_id"]], "bbox": a["bbox"],
                     "area": a.get("area", a["bbox"][2] * a["bbox"][3]), "iscrowd": a.get("iscrowd", 0)})
        keep_img_ids.add(a["image_id"])

    # index images under the root once (LOCO json 'path' is rooted at /dataset/..., so use names)
    by_name = {}
    for p in images_root.rglob("*.jpg"):
        by_name.setdefault(p.name, p)
    imgs = []
    for im in src["images"]:
        if im["id"] not in keep_img_ids:
            continue
        s = by_name.get(Path(im["file_name"]).name)
        if s is None:
            print(f"  WARNING missing image {im['file_name']}")
            continue
        dst = split_dir / s.name
        if not dst.exists():
            os.symlink(s.resolve(), dst)
        imgs.append({"id": im["id"] + img_off, "file_name": s.name, "width": im["width"], "height": im["height"]})
    coco["images"].extend(imgs)
    coco["annotations"].extend(anns)
    print(f"  +{len(imgs)} images, +{len(anns)} anns from {ann_path.name}")
    return len(imgs), len(anns)


def build_real(force: bool) -> Path:
    d = OUT / "real"
    if d.exists() and not force:
        print(f"[real] exists, skip ({d})")
        return d
    if d.exists():
        shutil.rmtree(d)
    (d / "train").mkdir(parents=True)
    (d / "valid").mkdir(parents=True)
    with open(LOCO_TRAIN_ANNS[0]) as f:
        _, cats = _category_map(json.load(f))
    print(f"[real] categories: {cats}")
    train = _empty_coco(cats)
    print("[real/train]")
    for ann in LOCO_TRAIN_ANNS:
        loco_into(ann, LOCO, d / "train", train)
    print("[real/valid]  (LOCO subset-3 only)")
    valid = _empty_coco(cats)
    loco_into(LOCO_VAL_ANN, LOCO_VAL_IMGS, d / "valid", valid)
    json.dump(train, open(d / "train" / "_annotations.coco.json", "w"))
    json.dump(valid, open(d / "valid" / "_annotations.coco.json", "w"))
    return d


# ----------------------------------------------------------------------------- synth
def _run_md5(run_dir: Path) -> str:
    """Content key for run-level dedup: md5 of the run's LARGEST rgb png. (Frame 0 is sometimes a
    blank/uninitialised render identical across unrelated runs, so it must not be the key.)"""
    pngs = sorted(run_dir.glob("rgb_*.png")) or sorted(run_dir.glob("Camera/rgb/rgb_*.png"))
    return hashlib.md5(max(pngs, key=lambda p: p.stat().st_size).read_bytes()).hexdigest()


def build_synth_only(source: str, categories, force: bool) -> Path:
    """evalsets/train_<source>/valid = all labeled frames of that synthetic source."""
    d = EVALSETS / f"train_{source}"
    if d.exists() and not force:
        print(f"[{source}] exists, skip ({d})")
        return d
    if d.exists():
        shutil.rmtree(d)
    (d / "valid").mkdir(parents=True)
    coco = _empty_coco(categories)
    seen_md5 = {}
    skipped_dup = 0
    for rd in synth_run_dirs(source):
        h = _run_md5(rd)
        if h in seen_md5:
            print(f"  DUP run (same content as {seen_md5[h]}): {rd} -> skipped")
            skipped_dup += 1
            continue
        seen_md5[h] = rd.name
        before = len(coco["images"])
        frames_to_coco(collect_frames(rd), d / "valid", coco, run_prefix=rd.name)
        print(f"  {rd.name}: {len(coco['images']) - before} labeled frames")
    json.dump(coco, open(d / "valid" / "_annotations.coco.json", "w"))
    # train/ -> valid/ so load_class_names() works on eval-only sets
    os.symlink("valid", d / "train")
    print(f"[{source}] {len(coco['images'])} images, {len(coco['annotations'])} anns, {skipped_dup} dup runs skipped")
    return d


def merge_split(src_split: Path, dst_split: Path, coco: dict, limit: int | None = None) -> int:
    """Append src_split's images/annotations into coco, symlinking to realpath.
    limit: seeded uniform subsample of that many images (size-matched control arms)."""
    with open(src_split / "_annotations.coco.json") as f:
        s = json.load(f)
    assert s["categories"] == coco["categories"]
    if limit is not None and limit < len(s["images"]):
        keep_imgs = random.Random(SUBSAMPLE_SEED).sample(s["images"], limit)
        keep_ids = {im["id"] for im in keep_imgs}
        s = {**s, "images": keep_imgs, "annotations": [a for a in s["annotations"] if a["image_id"] in keep_ids]}
    img_off = max((i["id"] for i in coco["images"]), default=0)
    ann_off = max((a["id"] for a in coco["annotations"]), default=0)
    names = {i["file_name"] for i in coco["images"]}
    n = 0
    for im in s["images"]:
        assert im["file_name"] not in names, f"filename collision {im['file_name']}"
        dst = dst_split / im["file_name"]
        if not dst.exists():
            os.symlink((src_split / im["file_name"]).resolve(strict=True), dst)
        coco["images"].append({**im, "id": im["id"] + img_off})
        n += 1
    for a in s["annotations"]:
        coco["annotations"].append({**a, "id": a["id"] + ann_off, "image_id": a["image_id"] + img_off})
    return n


def build_arm(arm: str, sources: list[str], force: bool):
    if arm == "real":          # the real arm IS the base dataset built by build_real()
        return
    d = OUT / arm
    if d.exists() and not force:
        print(f"[{arm}] exists, skip")
        return
    if d.exists():
        shutil.rmtree(d)
    (d / "train").mkdir(parents=True)
    real_train = OUT / "real" / "train"
    with open(real_train / "_annotations.coco.json") as f:
        cats = json.load(f)["categories"]
    coco = _empty_coco(cats)
    n_real = merge_split(real_train, d / "train", coco)
    counts = {"real": n_real}
    limits = ARM_SUBSAMPLE.get(arm, {})
    for s in sources:
        counts[s] = merge_split(EVALSETS / f"train_{s}" / "valid", d / "train", coco, limit=limits.get(s))
    json.dump(coco, open(d / "train" / "_annotations.coco.json", "w"))
    os.symlink("../real/valid", d / "valid")          # ONE shared real-only valid
    print(f"[{arm}] train={len(coco['images'])} {counts}  valid -> real/valid")


# ----------------------------------------------------------------------------- checks
def _names(split: Path) -> set[str]:
    with open(split / "_annotations.coco.json") as f:
        return {i["file_name"] for i in json.load(f)["images"]}


def _cats(split: Path):
    with open(split / "_annotations.coco.json") as f:
        return json.load(f)["categories"]


def _class_counts(split: Path) -> dict:
    with open(split / "_annotations.coco.json") as f:
        c = json.load(f)
    id2n = {x["id"]: x["name"] for x in c["categories"]}
    return dict(Counter(id2n[a["category_id"]] for a in c["annotations"]))


def verify_and_manifest():
    with open(LOCO_VAL_ANN) as f:
        sub3 = {Path(i["file_name"]).name for i in json.load(f)["images"]}
    valid_names = _names(OUT / "real" / "valid")
    assert valid_names <= sub3, "valid contains a non-subset-3 image"
    assert not any(n.endswith(".png") for n in valid_names), "synthetic frame in valid!"
    ref_cats = _cats(OUT / "real" / "train")
    manifest = {"categories": ref_cats, "valid": {"images": len(valid_names), "source": "LOCO subset-3 only",
                                                  "class_counts": _class_counts(OUT / "real" / "valid")},
                "arms": {}, "evalsets": {}}
    for arm in ARMS:
        tr = OUT / arm / "train"
        names = _names(tr)
        assert not (names & sub3), f"{arm}: subset-3 image leaked into train"
        assert _cats(tr) == ref_cats, f"{arm}: category table differs"
        assert _names(OUT / arm / "valid") == valid_names, f"{arm}: valid differs from real valid"
        manifest["arms"][arm] = {"train_images": len(names), "by_source": dict(Counter(map(source_of, names))),
                                 "class_counts": _class_counts(tr)}
    for es in sorted(EVALSETS.iterdir()):
        v = es / "valid"
        assert _cats(v) == ref_cats
        manifest["evalsets"][es.name] = {"images": len(_names(v)), "class_counts": _class_counts(v)}
    broken = [p for p in OUT.rglob("*") if p.is_symlink() and not p.exists()]
    assert not broken, f"{len(broken)} broken symlinks, e.g. {broken[:3]}"
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== ALL INVARIANTS OK ===")
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild everything from scratch")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    EVALSETS.mkdir(parents=True, exist_ok=True)

    real = build_real(args.force)
    with open(real / "train" / "_annotations.coco.json") as f:
        cats = json.load(f)["categories"]
    for s in SYNTH_SOURCES:
        build_synth_only(s, cats, args.force)
    tr_real = EVALSETS / "train_real"
    if args.force and tr_real.exists():
        shutil.rmtree(tr_real)
    if not tr_real.exists():
        tr_real.mkdir()
        os.symlink("../../real/train", tr_real / "valid")
        os.symlink("../../real/train", tr_real / "train")
    for arm, sources in ARMS.items():
        build_arm(arm, sources, args.force)
    verify_and_manifest()


if __name__ == "__main__":
    main()

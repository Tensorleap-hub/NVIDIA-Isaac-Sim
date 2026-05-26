"""
Evaluate real / base_synth / opt0 checkpoints on LOCO subset-3 and export a CSV
with per-sample metrics.  Highlights images where opt0 beats both baselines.

Usage:
    python3 od_scripts/compare_models_subset3.py
    python3 od_scripts/compare_models_subset3.py --output results_subset3.csv --score-thresh 0.3 --iou-thresh 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "rf-detr" / "src"))

from rfdetr import RFDETR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/Users/orram/Tensorleap/data/warehouse")
ANN_FILE  = DATA_ROOT / "dataset/labels/loco-sub3-v1-train.json"
IMG_ROOT  = DATA_ROOT

CHECKPOINTS = {
    "real":       DATA_ROOT / "training/real/checkpoint_best_ema.pth",
    "base_synth": DATA_ROOT / "training/base_synth/checkpoint_best_ema.pth",
    "opt0":       DATA_ROOT / "training/opt0/checkpoint_best_ema.pth",
}

# COCO category id → model class index
# Models trained on: pallet_truck(0), forklift(1), pallet(2)
COCO_ID_TO_IDX = {11: 0, 5: 1, 7: 2}
CLASS_NAMES    = {0: "pallet_truck", 1: "forklift", 2: "pallet"}

# base_synth and opt0 have forklift(1) and pallet(2) indices swapped vs real
PRED_CLASS_REMAP = {
    "real":       {},
    "base_synth": {1: 2, 2: 1},
    "opt0":       {1: 2, 2: 1},
}


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes (xyxy).  Returns [N, M] matrix."""
    ax1, ay1, ax2, ay2 = box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3]
    bx1, by1, bx2, by2 = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_detections(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    iou_thresh: float,
) -> tuple[int, int, int, list[float]]:
    """Return (TP, FP, FN, matched_ious)."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0, []
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    iou_mat = box_iou(pred_boxes, gt_boxes)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_matched = np.zeros(len(pred_boxes), dtype=bool)
    matched_ious: list[float] = []

    # Greedy matching sorted by IoU descending
    order = np.dstack(np.unravel_index(np.argsort(-iou_mat, axis=None), iou_mat.shape))[0]
    for pi, gi in order:
        if iou_mat[pi, gi] < iou_thresh:
            break
        if pred_matched[pi] or gt_matched[gi]:
            continue
        if pred_classes[pi] != gt_classes[gi]:
            continue
        pred_matched[pi] = True
        gt_matched[gi] = True
        matched_ious.append(float(iou_mat[pi, gi]))

    tp = int(pred_matched.sum())
    fp = int((~pred_matched).sum())
    fn = int((~gt_matched).sum())
    return tp, fp, fn, matched_ious


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="od_scripts/results_subset3.csv")
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--iou-thresh",   type=float, default=0.5)
    p.add_argument("--min-objects",  type=int,   default=4,
                   help="Min GT objects for the 'top candidates' table")
    p.add_argument("--top-n",        type=int,   default=10,
                   help="How many top candidates to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Load annotations
    # -----------------------------------------------------------------------
    with open(ANN_FILE) as f:
        coco = json.load(f)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    print(f"Subset-3: {len(images)} images")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    models: dict[str, RFDETR] = {}
    for name, ckpt in CHECKPOINTS.items():
        print(f"Loading {name} from {ckpt} …")
        models[name] = RFDETR.from_checkpoint(str(ckpt))

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    rows = []
    model_names = list(CHECKPOINTS.keys())

    for img_meta in images:
        img_path = IMG_ROOT / img_meta["path"].lstrip("/")
        if not img_path.exists():
            continue

        gt_anns   = anns_by_image.get(img_meta["id"], [])
        valid_gt  = [a for a in gt_anns if a["category_id"] in COCO_ID_TO_IDX]
        num_gt    = len(valid_gt)

        gt_boxes_xywh = np.array([a["bbox"] for a in valid_gt], dtype=np.float32) if valid_gt else np.zeros((0, 4))
        gt_boxes = np.column_stack([
            gt_boxes_xywh[:, 0],
            gt_boxes_xywh[:, 1],
            gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2],
            gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3],
        ]) if num_gt > 0 else np.zeros((0, 4))
        gt_classes = np.array([COCO_ID_TO_IDX[a["category_id"]] for a in valid_gt], dtype=np.int32)

        row: dict = {
            "image_id":   img_meta["id"],
            "image_path": str(img_path),
            "width":      img_meta["width"],
            "height":     img_meta["height"],
            "num_gt":     num_gt,
        }

        # Per-class GT counts
        for cls_idx, cls_name in CLASS_NAMES.items():
            row[f"gt_{cls_name}"] = int((gt_classes == cls_idx).sum()) if num_gt > 0 else 0

        for name, model in models.items():
            dets = model.predict(str(img_path), threshold=args.score_thresh)
            pred_boxes   = dets.xyxy   if len(dets) > 0 else np.zeros((0, 4))
            pred_classes = dets.class_id.copy() if len(dets) > 0 else np.zeros(0, dtype=np.int32)
            pred_scores  = dets.confidence if len(dets) > 0 else np.zeros(0)

            remap = PRED_CLASS_REMAP[name]
            if remap and len(pred_classes) > 0:
                for src, dst in remap.items():
                    pred_classes[dets.class_id == src] = dst

            tp, fp, fn, matched_ious = match_detections(
                pred_boxes, pred_classes, gt_boxes, gt_classes, args.iou_thresh
            )
            precision, recall, f1 = compute_f1(tp, fp, fn)
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

            row[f"{name}_tp"]         = tp
            row[f"{name}_fp"]         = fp
            row[f"{name}_fn"]         = fn
            row[f"{name}_precision"]  = round(precision, 4)
            row[f"{name}_recall"]     = round(recall, 4)
            row[f"{name}_f1"]         = round(f1, 4)
            row[f"{name}_mean_iou"]   = round(mean_iou, 4)
            row[f"{name}_num_preds"]  = len(pred_boxes)
            row[f"{name}_max_score"]  = round(float(pred_scores.max()), 4) if len(pred_scores) > 0 else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Derived columns
    # -----------------------------------------------------------------------
    df["opt0_beats_real"]       = df["opt0_f1"] > df["real_f1"]
    df["opt0_beats_base_synth"] = df["opt0_f1"] > df["base_synth_f1"]
    df["opt0_beats_both"]       = df["opt0_beats_real"] & df["opt0_beats_base_synth"]
    df["opt0_f1_gain_vs_real"]       = (df["opt0_f1"] - df["real_f1"]).round(4)
    df["opt0_f1_gain_vs_base_synth"] = (df["opt0_f1"] - df["base_synth_f1"]).round(4)
    df["opt0_f1_gain_min"]           = df[["opt0_f1_gain_vs_real", "opt0_f1_gain_vs_base_synth"]].min(axis=1).round(4)

    # -----------------------------------------------------------------------
    # Save full CSV
    # -----------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows → {out_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    beats_both = df[df["opt0_beats_both"]]
    print(f"\nopt0 beats both:  {len(beats_both)} / {len(df)} images")
    print(f"opt0 mean F1:     {df['opt0_f1'].mean():.4f}")
    print(f"real mean F1:     {df['real_f1'].mean():.4f}")
    print(f"base_synth mean F1: {df['base_synth_f1'].mean():.4f}")

    # -----------------------------------------------------------------------
    # Top candidates: beats both, >= min_objects, largest gain
    # -----------------------------------------------------------------------
    candidates = (
        beats_both[beats_both["num_gt"] >= args.min_objects]
        .sort_values("opt0_f1_gain_min", ascending=False)
        .head(args.top_n)
    )

    print(f"\n── Top {args.top_n} candidates (≥{args.min_objects} objects, opt0 beats both) ──")
    display_cols = [
        "image_path", "num_gt",
        "real_f1", "base_synth_f1", "opt0_f1",
        "opt0_f1_gain_min",
        "opt0_recall", "opt0_precision", "opt0_mean_iou",
    ]
    print(candidates[display_cols].to_string(index=False))

    top_path = out_path.with_name(out_path.stem + "_top_candidates.csv")
    candidates.to_csv(top_path, index=False)
    print(f"\nWrote top candidates → {top_path}")


if __name__ == "__main__":
    main()

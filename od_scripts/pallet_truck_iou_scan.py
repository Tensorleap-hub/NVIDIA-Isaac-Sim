"""
For every image in subset-3 that has LOCO pallet_truck (cat id=11) GT boxes,
run opt0 inference and compute class-agnostic IoU between each pallet_truck GT
box and the best-matching predicted box.

Reports:
  - mean_pt_iou  : mean of (max IoU over all preds) for each pallet_truck GT box
  - pt_recall_50 : fraction of pallet_truck GT boxes matched at IoU >= 0.5
  - num_pt       : number of pallet_truck GT boxes

Sorted ascending by mean_pt_iou (lowest first).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "rf-detr" / "src"))

from rfdetr import RFDETR  # noqa: E402

DATA_ROOT = Path("/Users/orram/Tensorleap/data/warehouse")
ANN_FILE  = DATA_ROOT / "dataset/labels/loco-sub3-v1-train.json"
CKPT      = DATA_ROOT / "training/opt0/checkpoint_best_ema.pth"
SCORE_THR = 0.3


def max_iou_per_gt(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """For each GT box return the max IoU against any pred box. Shape: [N_gt]."""
    if len(pred_boxes) == 0:
        return np.zeros(len(gt_boxes))
    ax1, ay1, ax2, ay2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    bx1, by1, bx2, by2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    union = aa[:, None] + ab[None, :] - inter
    iou_mat = np.where(union > 0, inter / union, 0.0)
    return iou_mat.max(axis=1)


def main() -> None:
    with open(ANN_FILE) as f:
        coco = json.load(f)

    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only images that have at least one pallet_truck (id=11)
    pt_images = [
        im for im in coco["images"]
        if any(a["category_id"] == 11 for a in anns_by_image.get(im["id"], []))
    ]
    print(f"Images with pallet_truck GT: {len(pt_images)}")

    print("Loading opt0…")
    model = RFDETR.from_checkpoint(str(CKPT))

    rows = []
    for i, meta in enumerate(pt_images):
        img_path = DATA_ROOT / meta["path"].lstrip("/")
        if not img_path.exists():
            continue

        dets = model.predict(str(img_path), threshold=SCORE_THR)
        pred_boxes = dets.xyxy if len(dets) > 0 else np.zeros((0, 4))

        pt_anns = [a for a in anns_by_image.get(meta["id"], []) if a["category_id"] == 11]
        xywh = np.array([a["bbox"] for a in pt_anns], dtype=np.float32)
        gt_boxes = np.column_stack([xywh[:, 0], xywh[:, 1],
                                    xywh[:, 0] + xywh[:, 2],
                                    xywh[:, 1] + xywh[:, 3]])

        per_box_iou = max_iou_per_gt(gt_boxes, pred_boxes)
        mean_iou    = float(per_box_iou.mean())
        recall_50   = float((per_box_iou >= 0.5).mean())

        rows.append({
            "path":         meta["path"],
            "stem":         Path(meta["path"]).stem,
            "num_pt":       len(pt_anns),
            "num_preds":    len(pred_boxes),
            "mean_pt_iou":  round(mean_iou, 4),
            "pt_recall_50": round(recall_50, 4),
            "per_box_ious": [round(v, 3) for v in per_box_iou.tolist()],
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pt_images)} done…")

    df = pd.DataFrame(rows).sort_values("mean_pt_iou", ascending=True)

    print("\n── 5 lowest mean pallet_truck IoU (opt0 misses them most) ──")
    for _, r in df.head(5).iterrows():
        print(f"  {r['stem']:<40}  pt={r['num_pt']}  preds={r['num_preds']}"
              f"  mean_iou={r['mean_pt_iou']:.3f}  recall@0.5={r['pt_recall_50']:.2f}"
              f"  per_box={r['per_box_ious']}")

    print("\n── 5 highest mean pallet_truck IoU (opt0 best localizes them) ──")
    for _, r in df.tail(5).iterrows():
        print(f"  {r['stem']:<40}  pt={r['num_pt']}  preds={r['num_preds']}"
              f"  mean_iou={r['mean_pt_iou']:.3f}  recall@0.5={r['pt_recall_50']:.2f}"
              f"  per_box={r['per_box_ious']}")

    out = REPO_ROOT / "od_scripts/pt_iou_scan.csv"
    df.to_csv(out, index=False)
    print(f"\nFull table → {out}")


if __name__ == "__main__":
    main()

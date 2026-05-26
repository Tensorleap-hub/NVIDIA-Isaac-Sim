"""
Render GT + model predictions for pallet_truck candidate images.
Saves 4-panel strips (GT | Real | Base Synth | Opt-0) to outputs/loco-labels/.

Usage:
    python3 od_scripts/render_pt_candidates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "rf-detr" / "src"))

from rfdetr import RFDETR  # noqa: E402

DATA_ROOT  = Path("/Users/orram/Tensorleap/data/warehouse")
ANN_FILE   = DATA_ROOT / "dataset/labels/loco-sub3-v1-train.json"
TRAIN_ROOT = DATA_ROOT / "training"
OUT_DIR    = REPO_ROOT / "outputs" / "loco-labels"

# All 5 LOCO categories with distinct colors
LOCO_CATS = {
    3:  ("small_load_carrier", "#FF1744"),
    5:  ("forklift",           "#FF9100"),
    7:  ("pallet",             "#00E5FF"),
    10: ("stillage",           "#D500F9"),
    11: ("pallet_truck",       "#EEFF41"),   # bright yellow-green — easy to spot
}

COCO_ID_TO_IDX = {3: 0, 5: 1, 7: 2}
CLASS_NAMES    = ["small_load_carrier", "forklift", "pallet"]
PRED_COLORS    = ["#FF1744", "#FF9100", "#00E5FF"]  # match GT palette for model classes
PRED_CLASS_REMAP = {
    "real":       {},
    "base_synth": {1: 2, 2: 1},
    "opt0":       {1: 2, 2: 1},
}
CHECKPOINTS = {
    "real":       TRAIN_ROOT / "real/checkpoint_best_ema.pth",
    "base_synth": TRAIN_ROOT / "base_synth/checkpoint_best_ema.pth",
    "opt0":       TRAIN_ROOT / "opt0/checkpoint_best_ema.pth",
}
MODEL_LABELS = {"real": "Real", "base_synth": "Base Synth", "opt0": "Opt-0 (TL)"}
MODEL_COLORS = {"real": "#1976D2", "base_synth": "#F57C00", "opt0": "#69FF47"}

# Candidate image stems to render (best pallet_truck, opt0 > real & base)
CANDIDATES = [
    "1576592663.3978252",  # pt=1, 8 GT, opt0=0.462, margin=0.062, pt_iou=0.689
    "1576596185.4985235",  # pt=2, 42 GT, opt0=0.561, margin=0.049, pt_iou=0.567
]


def apply_remap(class_ids: np.ndarray, remap: dict) -> np.ndarray:
    if not remap or len(class_ids) == 0:
        return class_ids
    out = class_ids.copy()
    for src, dst in remap.items():
        out[class_ids == src] = dst
    return out


def draw_panel(ax, img_rgb: np.ndarray,
               gt_anns: list[dict],
               pred_boxes: np.ndarray, pred_classes: np.ndarray, pred_scores: np.ndarray,
               title: str, title_color: str) -> None:
    ax.imshow(img_rgb)
    ax.axis("off")

    for ann in gt_anns:
        cat_id = ann["category_id"]
        if cat_id not in LOCO_CATS:
            continue
        name, color = LOCO_CATS[cat_id]
        x, y, w, h = ann["bbox"]
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            linewidth=2.0, edgecolor=color, facecolor="none",
            linestyle="--", boxstyle="square,pad=0")
        ax.add_patch(rect)
        ax.text(x + 2, y + h - 3, name,
                color=color, fontsize=5, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))

    for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
        x1, y1, x2, y2 = box
        color = PRED_COLORS[int(cls) % len(PRED_COLORS)]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none",
            boxstyle="square,pad=0")
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 + 10, f"{CLASS_NAMES[int(cls)]} {score:.2f}",
                color=color, fontsize=5, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))

    ax.set_title(title, color=title_color, fontsize=8, fontweight="bold", pad=5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANN_FILE) as f:
        coco = json.load(f)
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Build stem → image_meta lookup
    stem_to_meta = {}
    for im in coco["images"]:
        stem = Path(im["path"]).stem
        stem_to_meta[stem] = im

    print("Loading models…")
    models = {name: RFDETR.from_checkpoint(str(ckpt)) for name, ckpt in CHECKPOINTS.items()}

    for stem in CANDIDATES:
        meta = stem_to_meta.get(stem)
        if meta is None:
            print(f"  [SKIP] {stem} not found in annotations")
            continue

        img_path = DATA_ROOT / meta["path"].lstrip("/")
        if not img_path.exists():
            print(f"  [SKIP] {img_path} missing")
            continue

        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_anns = anns_by_image.get(meta["id"], [])

        # Count by category for title
        from collections import Counter
        cat_counts = Counter(LOCO_CATS[a["category_id"]][0] for a in gt_anns if a["category_id"] in LOCO_CATS)
        cat_str = "  ".join(f"{v}×{k}" for k, v in sorted(cat_counts.items()))

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.patch.set_facecolor("#1a1a2e")

        # GT panel
        draw_panel(axes[0], img_rgb, gt_anns,
                   np.zeros((0, 4)), np.zeros(0), np.zeros(0),
                   f"GT ({cat_str})", "#FFFFFF")

        # Model prediction panels
        for ax, (name, model) in zip(axes[1:], models.items()):
            dets = model.predict(str(img_path), threshold=0.3)
            pred_boxes   = dets.xyxy       if len(dets) > 0 else np.zeros((0, 4))
            raw_classes  = dets.class_id   if len(dets) > 0 else np.zeros(0, dtype=np.int32)
            pred_scores  = dets.confidence if len(dets) > 0 else np.zeros(0)
            pred_classes = apply_remap(raw_classes, PRED_CLASS_REMAP[name])

            # Compute F1 (class-aware, model classes only)
            valid_gt = [a for a in gt_anns if a["category_id"] in COCO_ID_TO_IDX]
            gt_cls   = np.array([COCO_ID_TO_IDX[a["category_id"]] for a in valid_gt], dtype=np.int32)
            n_gt = len(valid_gt)

            if n_gt > 0 and len(pred_boxes) > 0:
                gt_xywh   = np.array([a["bbox"] for a in valid_gt], dtype=np.float32)
                gt_xyxy   = np.column_stack([gt_xywh[:, 0], gt_xywh[:, 1],
                                             gt_xywh[:, 0] + gt_xywh[:, 2],
                                             gt_xywh[:, 1] + gt_xywh[:, 3]])
                iou = np.zeros((len(pred_boxes), len(gt_xyxy)))
                for pi, pb in enumerate(pred_boxes):
                    for gi, gb in enumerate(gt_xyxy):
                        ix1, iy1 = max(pb[0], gb[0]), max(pb[1], gb[1])
                        ix2, iy2 = min(pb[2], gb[2]), min(pb[3], gb[3])
                        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
                        ua = (pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter
                        iou[pi, gi] = inter / ua if ua > 0 else 0.0
                tp = 0
                gt_matched = np.zeros(len(gt_xyxy), bool)
                pred_matched = np.zeros(len(pred_boxes), bool)
                order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
                for pi, gi in order:
                    if iou[pi, gi] < 0.5:
                        break
                    if pred_matched[pi] or gt_matched[gi]:
                        continue
                    if pred_classes[pi] != gt_cls[gi]:
                        continue
                    pred_matched[pi] = True
                    gt_matched[gi] = True
                    tp += 1
                fp = int((~pred_matched).sum())
                fn = int((~gt_matched).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            else:
                f1 = 0.0

            draw_panel(ax, img_rgb, [], pred_boxes, pred_classes, pred_scores,
                       f"{MODEL_LABELS[name]}  F1={f1:.2f}", MODEL_COLORS[name])

        plt.suptitle(Path(img_path).name, color="#80CBC4", fontsize=9, y=1.01)
        plt.tight_layout(pad=0.5)

        out_path = OUT_DIR / f"candidate_pt_{stem}.jpg"
        plt.savefig(str(out_path), format="jpg", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

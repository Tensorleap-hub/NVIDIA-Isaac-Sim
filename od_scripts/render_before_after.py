"""
Render before/after comparison for two candidate images.
Before = current wrong mapping (small_load_carrier GT + wrong class names).
After  = corrected mapping (pallet_truck GT id=11 + CLASS_NAMES from training config).
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "rf-detr" / "src"))
from rfdetr import RFDETR  # noqa: E402

DATA_ROOT  = Path("/Users/orram/Tensorleap/data/warehouse")
ANN_FILE   = DATA_ROOT / "dataset/labels/loco-sub3-v1-train.json"
TRAIN_ROOT = DATA_ROOT / "training"
OUT_DIR    = REPO_ROOT / "outputs" / "loco-labels"

CHECKPOINTS = {
    "real":       TRAIN_ROOT / "real/checkpoint_best_ema.pth",
    "base_synth": TRAIN_ROOT / "base_synth/checkpoint_best_ema.pth",
    "opt0":       TRAIN_ROOT / "opt0/checkpoint_best_ema.pth",
}
MODEL_LABELS = {"real": "Real", "base_synth": "Base Synth", "opt0": "Opt-0 (TL)"}
MODEL_COLORS = {"real": "#1976D2", "base_synth": "#F57C00", "opt0": "#69FF47"}
PRED_CLASS_REMAP = {"real": {}, "base_synth": {1:2, 2:1}, "opt0": {1:2, 2:1}}

# ── BEFORE (wrong) ───────────────────────────────────────────────────────────
BEFORE_COCO_ID_TO_IDX = {3: 0, 5: 1, 7: 2}
BEFORE_CLASS_NAMES    = ["small_load_carrier", "forklift", "pallet"]

# ── AFTER (correct) ──────────────────────────────────────────────────────────
AFTER_COCO_ID_TO_IDX  = {11: 0, 5: 1, 7: 2}
AFTER_CLASS_NAMES     = ["pallet_truck", "forklift", "pallet"]

# All LOCO category colors for GT display
LOCO_CATS = {
    3:  ("small_load_carrier", "#FF1744"),
    5:  ("forklift",           "#FF9100"),
    7:  ("pallet",             "#00E5FF"),
    10: ("stillage",           "#D500F9"),
    11: ("pallet_truck",       "#EEFF41"),
}

PRED_COLORS = ["#EEFF41", "#FF9100", "#00E5FF"]  # pallet_truck, forklift, pallet

CANDIDATES = ["1576596185.4985235", "1576591740.1848662"]


def apply_remap(cls_ids, remap):
    if not remap or len(cls_ids) == 0:
        return cls_ids
    out = cls_ids.copy()
    for s, d in remap.items():
        out[cls_ids == s] = d
    return out


def compute_f1(pred_boxes, pred_classes, gt_xyxy, gt_classes, iou_thr=0.5):
    if len(gt_xyxy) == 0 and len(pred_boxes) == 0:
        return 0.0
    if len(gt_xyxy) == 0 or len(pred_boxes) == 0:
        return 0.0
    ax1,ay1,ax2,ay2 = pred_boxes[:,0],pred_boxes[:,1],pred_boxes[:,2],pred_boxes[:,3]
    bx1,by1,bx2,by2 = gt_xyxy[:,0],gt_xyxy[:,1],gt_xyxy[:,2],gt_xyxy[:,3]
    ix1=np.maximum(ax1[:,None],bx1[None,:]); iy1=np.maximum(ay1[:,None],by1[None,:])
    ix2=np.minimum(ax2[:,None],bx2[None,:]); iy2=np.minimum(ay2[:,None],by2[None,:])
    inter=np.maximum(ix2-ix1,0)*np.maximum(iy2-iy1,0)
    aa=(ax2-ax1)*(ay2-ay1); ab=(bx2-bx1)*(by2-by1)
    union=aa[:,None]+ab[None,:]-inter
    iou=np.where(union>0,inter/union,0.)
    gt_matched=np.zeros(len(gt_xyxy),bool); pred_matched=np.zeros(len(pred_boxes),bool)
    for pi,gi in np.dstack(np.unravel_index(np.argsort(-iou,axis=None),iou.shape))[0]:
        if iou[pi,gi]<iou_thr: break
        if pred_matched[pi] or gt_matched[gi]: continue
        if pred_classes[pi]!=gt_classes[gi]: continue
        pred_matched[pi]=True; gt_matched[gi]=True
    tp=pred_matched.sum(); fp=(~pred_matched).sum(); fn=(~gt_matched).sum()
    p=tp/(tp+fp) if tp+fp>0 else 0.; r=tp/(tp+fn) if tp+fn>0 else 0.
    return 2*p*r/(p+r) if p+r>0 else 0.


def draw_strip(ax, img_rgb, gt_anns, pred_boxes, pred_classes, pred_scores,
               class_names, coco_id_to_idx, title, title_color):
    ax.imshow(img_rgb); ax.axis("off")
    # GT
    for ann in gt_anns:
        if ann["category_id"] not in LOCO_CATS: continue
        name, color = LOCO_CATS[ann["category_id"]]
        x,y,w,h = ann["bbox"]
        ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,
            linewidth=2,edgecolor=color,facecolor="none",
            linestyle="--",boxstyle="square,pad=0"))
        ax.text(x+2,y+h-3,name,color=color,fontsize=5,fontweight="bold",
            bbox=dict(facecolor="black",alpha=0.6,pad=1,edgecolor="none"))
    # Preds
    for box,cls,score in zip(pred_boxes,pred_classes,pred_scores):
        x1,y1,x2,y2=box; color=PRED_COLORS[int(cls)%3]
        ax.add_patch(mpatches.FancyBboxPatch((x1,y1),x2-x1,y2-y1,
            linewidth=2.5,edgecolor=color,facecolor="none",boxstyle="square,pad=0"))
        ax.text(x1+2,y1+10,f"{class_names[int(cls)]} {score:.2f}",
            color=color,fontsize=5,fontweight="bold",
            bbox=dict(facecolor="black",alpha=0.6,pad=1,edgecolor="none"))
    ax.set_title(title,color=title_color,fontsize=7.5,fontweight="bold",pad=4)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANN_FILE) as f:
        coco = json.load(f)
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    stem_to_meta = {Path(im["path"]).stem: im for im in coco["images"]}

    print("Loading models…")
    models = {n: RFDETR.from_checkpoint(str(p)) for n,p in CHECKPOINTS.items()}

    for stem in CANDIDATES:
        meta = stem_to_meta.get(stem)
        img_path = DATA_ROOT / meta["path"].lstrip("/")
        img_bgr  = cv2.imread(str(img_path))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_anns  = anns_by_image.get(meta["id"], [])

        # Run inference once per model
        inferences = {}
        for name, model in models.items():
            dets = model.predict(str(img_path), threshold=0.3)
            pb   = dets.xyxy       if len(dets)>0 else np.zeros((0,4))
            rc   = dets.class_id   if len(dets)>0 else np.zeros(0,dtype=np.int32)
            ps   = dets.confidence if len(dets)>0 else np.zeros(0)
            pc   = apply_remap(rc, PRED_CLASS_REMAP[name])
            inferences[name] = (pb, pc, ps)

        # 2 rows (before / after), 4 cols (GT + 3 models)
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.patch.set_facecolor("#1a1a2e")

        for row_i, (coco_map, cnames, row_label) in enumerate([
            (BEFORE_COCO_ID_TO_IDX, BEFORE_CLASS_NAMES, "BEFORE  (wrong GT: small_load_carrier as class-0)"),
            (AFTER_COCO_ID_TO_IDX,  AFTER_CLASS_NAMES,  "AFTER   (correct GT: pallet_truck as class-0)"),
        ]):
            valid_gt   = [a for a in gt_anns if a["category_id"] in coco_map]
            gt_xywh    = np.array([a["bbox"] for a in valid_gt],dtype=np.float32) if valid_gt else np.zeros((0,4))
            gt_xyxy    = np.column_stack([gt_xywh[:,0],gt_xywh[:,1],
                                          gt_xywh[:,0]+gt_xywh[:,2],
                                          gt_xywh[:,1]+gt_xywh[:,3]]) if len(valid_gt)>0 else np.zeros((0,4))
            gt_classes = np.array([coco_map[a["category_id"]] for a in valid_gt],dtype=np.int32)

            from collections import Counter
            cat_map = {c["id"]:c["name"] for c in coco["categories"]}
            all_cats = Counter(cat_map[a["category_id"]] for a in gt_anns
                               if a["category_id"] in cat_map)
            model_cats = Counter(cnames[coco_map[a["category_id"]]] for a in valid_gt)
            gt_title = f"GT  ({len(valid_gt)} model-class boxes)\n" + \
                       "  ".join(f"{v}×{k}" for k,v in sorted(model_cats.items()))

            # GT panel
            draw_strip(axes[row_i,0], img_rgb, gt_anns,
                       np.zeros((0,4)), np.zeros(0,dtype=np.int32), np.zeros(0),
                       cnames, coco_map, gt_title, "#FFFFFF")

            for col_i, (name, (pb, pc, ps)) in enumerate(inferences.items(), start=1):
                f1 = compute_f1(pb, pc, gt_xyxy, gt_classes)
                draw_strip(axes[row_i,col_i], img_rgb, [],
                           pb, pc, ps, cnames, coco_map,
                           f"{MODEL_LABELS[name]}  F1={f1:.2f}", MODEL_COLORS[name])

            # Row label on leftmost axis
            axes[row_i,0].set_ylabel(row_label, color="#80CBC4",
                                     fontsize=8, fontweight="bold", labelpad=8)

        plt.suptitle(Path(img_path).name, color="#90CAF9", fontsize=10, y=1.01)
        plt.tight_layout(pad=0.5, h_pad=1.5)
        out = OUT_DIR / f"before_after_{stem}.jpg"
        plt.savefig(str(out), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), format="jpg")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()

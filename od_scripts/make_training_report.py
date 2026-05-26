"""
Build od_scripts/training_report.html — training summary + best opt0 sample detections.

Usage:
    python3 od_scripts/make_training_report.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models" / "rf-detr" / "src"))

from rfdetr import RFDETR  # noqa: E402

DATA_ROOT   = Path("/Users/orram/Tensorleap/data/warehouse")
ANN_FILE    = DATA_ROOT / "dataset/labels/loco-sub3-v1-train.json"
TRAIN_ROOT  = DATA_ROOT / "training"
AGNOSTIC_CSV  = REPO_ROOT / "od_scripts/opt0_agnostic_eval.csv"
COMPARE_CSV   = REPO_ROOT / "od_scripts/results_subset3.csv"
OUT_HTML    = REPO_ROOT / "od_scripts/training_report.html"

# Models trained on: pallet_truck(0), forklift(1), pallet(2)
COCO_ID_TO_IDX = {11: 0, 5: 1, 7: 2}
CLASS_NAMES    = ["pallet_truck", "forklift", "pallet"]
PRED_COLORS    = ["#EEFF41", "#FF6D00", "#00E5FF"]   # yellow-green, orange, cyan
GT_COLOR       = "#FF1744"

# base_synth and opt0 have forklift(1) and pallet(2) indices swapped vs real
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
MODEL_LINE_COLORS = {"real": "#1976D2", "base_synth": "#F57C00", "opt0": "#388E3C"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def img_to_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def box_iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(a[:, 0, None], b[None, :, 0])
    iy1 = np.maximum(a[:, 1, None], b[None, :, 1])
    ix2 = np.minimum(a[:, 2, None], b[None, :, 2])
    iy2 = np.minimum(a[:, 3, None], b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def build_training_figure() -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    metrics = [
        ("val/ema_mAP_50",     "mAP@50",          0, 0.75),
        ("val/ema_mAP_50_95",  "mAP@50-95",       0, 0.45),
        ("val/AP/pallet_truck","AP pallet_truck", 0, 0.6),
    ]

    for ax, (col, title, ymin, ymax) in zip(axes, metrics):
        ax.set_facecolor("#0f0e17")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel(col.split("/")[-1], color="#aaa")
        ax.tick_params(colors="#aaa")
        ax.spines[:].set_color("#333")
        ax.set_ylim(ymin, ymax)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        for name in ["real", "base_synth", "opt0"]:
            df = pd.read_csv(TRAIN_ROOT / name / "metrics.csv")
            sub = df[["epoch", col]].dropna()
            sub = sub[sub["epoch"] <= 35]
            ax.plot(sub["epoch"], sub[col],
                    color=MODEL_LINE_COLORS[name],
                    label=MODEL_LABELS[name], linewidth=1.8, alpha=0.9)

        ax.set_xlim(0, 35)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    plt.tight_layout(pad=1.5)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# 2. Stats table HTML
# ---------------------------------------------------------------------------

def build_stats_table() -> str:
    rows_html = ""
    for name, label in MODEL_LABELS.items():
        df = pd.read_csv(TRAIN_ROOT / name / "metrics.csv")
        best = df.loc[df["val/ema_mAP_50"].idxmax()]
        color = MODEL_LINE_COLORS[name]
        best_f1 = df["val/F1"].max()
        rows_html += f"""
        <tr>
          <td><span style="color:{color};font-weight:bold">{label}</span></td>
          <td>{best['val/ema_mAP_50']:.3f}</td>
          <td>{best_f1:.3f}</td>
          <td>{best['val/ema_mAR']:.3f}</td>
          <td>{best['val/AP/pallet_truck']:.3f}</td>
          <td>{best['val/AP/forklift']:.3f}</td>
          <td>{best['val/AP/pallet']:.3f}</td>
        </tr>"""

    return f"""
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>mAP@50</th><th>F1</th><th>mAR</th>
          <th>AP pallet_truck</th><th>AP forklift</th><th>AP pallet</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# 3. Sample images
# ---------------------------------------------------------------------------

def draw_single_model(img_bgr: np.ndarray, gt_boxes: np.ndarray, gt_classes: np.ndarray,
                      pred_boxes: np.ndarray, pred_classes: np.ndarray,
                      pred_scores: np.ndarray, model_label: str,
                      model_color: str) -> str:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0f0e17")
    ax.imshow(img)
    ax.axis("off")

    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.8, edgecolor=GT_COLOR, facecolor="none",
            linestyle="--", boxstyle="square,pad=0")
        ax.add_patch(rect)
        ax.text(x1 + 2, y2 - 3, CLASS_NAMES[int(cls)],
                color=GT_COLOR, fontsize=5.5, fontweight="bold",
                bbox=dict(facecolor="#0f0e17", alpha=0.65, pad=1, edgecolor="none"))

    for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
        x1, y1, x2, y2 = box
        color = PRED_COLORS[int(cls) % len(PRED_COLORS)]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none",
            boxstyle="square,pad=0")
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 + 9, f"{CLASS_NAMES[int(cls)]} {score:.2f}",
                color=color, fontsize=5.5, fontweight="bold",
                bbox=dict(facecolor="#0f0e17", alpha=0.65, pad=1, edgecolor="none"))

    legend_handles = [
        mpatches.Patch(edgecolor=GT_COLOR, facecolor="none", linestyle="--",
                       linewidth=2, label="GT (dashed)"),
        *[mpatches.Patch(color=PRED_COLORS[i], label=f"pred: {CLASS_NAMES[i]}")
          for i in range(3)],
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    ax.set_title(
        f"{model_label}  |  GT: {len(gt_boxes)}  |  Preds: {len(pred_boxes)}",
        color=model_color, fontsize=9, fontweight="bold", pad=6)

    plt.tight_layout(pad=0.3)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def _apply_remap(class_ids: np.ndarray, remap: dict) -> np.ndarray:
    if not remap or len(class_ids) == 0:
        return class_ids
    out = class_ids.copy()
    for src, dst in remap.items():
        out[class_ids == src] = dst
    return out


def _pick_candidates() -> pd.DataFrame:
    """Pick the three curated scenes from the corrected compare CSV."""
    if not COMPARE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(COMPARE_CSV)

    # Pinned by filename stem — top margin, opt0 beats both (corrected GT mapping)
    PINNED = [
        "1576593123.2732508",   # margin 0.238, 14 GT
        "1576594751.9590912",   # margin 0.236,  7 GT
        "1576592652.96357",     # margin 0.222,  9 GT
    ]
    mask = df["image_path"].apply(lambda p: any(s in p for s in PINNED))
    return df[mask].reset_index(drop=True)


def build_samples_section() -> str:
    with open(ANN_FILE) as f:
        coco = json.load(f)
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    id_to_meta = {im["id"]: im for im in coco["images"]}
    path_to_id = {(DATA_ROOT / im["path"].lstrip("/")).as_posix(): im["id"] for im in coco["images"]}

    candidates = _pick_candidates()
    if candidates.empty:
        return "<p style='color:#f88'>results_subset3.csv not found — run compare_models_subset3.py first.</p>"

    print(f"Candidate scenes: {len(candidates)}")
    for _, r in candidates.iterrows():
        print(f"  {Path(r['image_path']).name}  opt0_f1={r['opt0_f1']:.3f}  "
              f"real_f1={r['real_f1']:.3f}  base_f1={r['base_synth_f1']:.3f}  "
              f"pt={r['gt_pallet_truck']}")

    print("Loading all 3 models for sample rendering…")
    models = {name: RFDETR.from_checkpoint(str(ckpt)) for name, ckpt in CHECKPOINTS.items()}

    html_parts = []
    for _, row in candidates.iterrows():
        img_path = row["image_path"]
        img_id   = path_to_id.get(Path(img_path).as_posix())
        if img_id is None:
            # fallback: match by filename stem
            img_id = next(
                (im["id"] for im in coco["images"] if Path(img_path).name in im["path"]),
                None,
            )
        if img_id is None:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        valid_gt = [a for a in anns_by_image.get(img_id, []) if a["category_id"] in COCO_ID_TO_IDX]
        gt_xywh  = np.array([a["bbox"] for a in valid_gt], dtype=np.float32)
        gt_boxes = np.column_stack([gt_xywh[:, 0], gt_xywh[:, 1],
                                    gt_xywh[:, 0] + gt_xywh[:, 2],
                                    gt_xywh[:, 1] + gt_xywh[:, 3]])
        gt_classes = np.array([COCO_ID_TO_IDX[a["category_id"]] for a in valid_gt])

        fname = Path(img_path).name
        slc_note = f" · pallet_truck GT: {int(row['gt_pallet_truck'])}" if row["gt_pallet_truck"] > 0 else ""
        img_cards = ""
        for name, model in models.items():
            dets = model.predict(img_path, threshold=0.3)
            pred_boxes   = dets.xyxy       if len(dets) > 0 else np.zeros((0, 4))
            raw_classes  = dets.class_id   if len(dets) > 0 else np.zeros(0, dtype=np.int32)
            pred_scores  = dets.confidence if len(dets) > 0 else np.zeros(0)
            pred_classes = _apply_remap(raw_classes, PRED_CLASS_REMAP[name])

            f1_val = row[f"{name}_f1"]
            label_str = f"{MODEL_LABELS[name]}  F1={f1_val:.2f}"
            b64 = draw_single_model(img_bgr, gt_boxes, gt_classes,
                                    pred_boxes, pred_classes, pred_scores,
                                    label_str, MODEL_LINE_COLORS[name])
            img_cards += f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:5px;margin-bottom:6px">\n'

        html_parts.append(f"""
        <div class="scene-block">
          <div class="scene-title">{fname} &nbsp;·&nbsp; GT: {int(row['num_gt'])} objects{slc_note} &nbsp;·&nbsp;
            opt-0 F1: <b style="color:#AED581">{row['opt0_f1']:.2f}</b> &nbsp;vs&nbsp;
            real: <b style="color:#1976D2">{row['real_f1']:.2f}</b> &nbsp;/&nbsp;
            base: <b style="color:#F57C00">{row['base_synth_f1']:.2f}</b>
          </div>
          <div class="model-row">
            {img_cards}
          </div>
        </div>""")

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# 4. Assemble HTML
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building training curves…")
    curves_b64 = build_training_figure()
    print("Building stats table…")
    stats_html = build_stats_table()
    print("Rendering sample images…")
    samples_html = build_samples_section()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Warehouse OD — Model Training Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1a1a2e; color: #e0e0e0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      padding: 32px 48px;
    }}
    h1 {{ font-size: 1.8rem; color: #90CAF9; margin-bottom: 8px; }}
    h2 {{ font-size: 1.2rem; color: #80CBC4; margin: 32px 0 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
    p.subtitle {{ color: #aaa; font-size: 0.9rem; margin-bottom: 24px; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th {{ background: #16213e; color: #90CAF9; padding: 10px 14px; text-align: left; border-bottom: 2px solid #333; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #222; }}
    tr:hover td {{ background: #0f3460; }}

    .curves-img {{ width: 100%; border-radius: 8px; margin-top: 8px; }}

    .note-box {{
      background: #0f3460; border-left: 4px solid #F57C00;
      padding: 14px 18px; border-radius: 4px; margin: 20px 0;
      font-size: 0.88rem; line-height: 1.6;
    }}

    .scene-block {{ background: #0f0e17; border-radius: 8px; padding: 14px; margin-bottom: 28px; }}
    .scene-title {{ font-size: 0.82rem; color: #80CBC4; margin-bottom: 10px; font-weight: bold; }}
    .model-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
  </style>
</head>
<body>

<h1>Warehouse OD — Model Training Report</h1>
<p class="subtitle">
  Three RF-DETR checkpoints trained on different data mixes.
  Validation on each model's own held-out split (real or synthetic).
</p>

<h2>Best Checkpoint Stats</h2>
{stats_html}

<h2>Training Curves</h2>
<img class="curves-img" src="data:image/png;base64,{curves_b64}">

<h2>Subset-3 Evaluation Note</h2>
<div class="note-box">
  Evaluated on <b>LOCO subset-3</b> (873 real warehouse images).
  GT classes: <b>pallet_truck</b> (LOCO id 11), <b>forklift</b> (LOCO id 5), <b>pallet</b> (LOCO id 7).
  LOCO small_load_carrier (id 3) and stillage (id 10) are not model classes and are excluded from evaluation.<br><br>
  Note: <b>base_synth</b> and <b>opt-0</b> have forklift/pallet class indices swapped vs real — a remapping is applied before matching.
</div>

<h2>Best Scenes — opt-0 Outperforms Both Baselines</h2>
<p style="font-size:0.82rem;color:#aaa;margin-bottom:12px">
  Red dashed = GT &nbsp;·&nbsp; Solid = predictions (yellow=pallet_truck, orange=forklift, cyan=pallet) &nbsp;·&nbsp; Each panel: Real / Base Synth / Opt-0 (TL) &nbsp;·&nbsp; Forklift↔pallet swap applied for synth models
</p>
{samples_html}

</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nWrote → {OUT_HTML}")


if __name__ == "__main__":
    main()

"""Summarise the 4-arm study: training curves (metrics.csv) + post-hoc evals (eval/*.json) -> markdown.

Usage: .venv/bin/python od_scripts/results.py [> /home/ubuntu/datasets_coco/RESULTS.md]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, OUT, arm_output_dir  # noqa: E402

CLASSES = ["forklift", "pallet", "pallet_truck"]
SPLIT_ORDER = ["valid_real", "train_real", "train_basev2", "train_may", "train_traj_optuna", "train_basev4", "train_combined"]
SPLIT_LABEL = {"valid_real": "**valid** (subset-3, real)", "train_real": "train: real", "train_basev2": "train: base_v2 synth",
               "train_may": "train: may synth", "train_traj_optuna": "train: trajectory-optimized synth", "train_basev4": "train: base_v4 synth", "train_combined": "train: combined"}


def training_summary(arm: str):
    mcsv = arm_output_dir(arm) / "metrics.csv"
    if not mcsv.exists():
        return None
    all_rows = list(csv.DictReader(open(mcsv)))
    rows = [r for r in all_rows if r.get("val/ema_mAP_50_95") not in (None, "")]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val/ema_mAP_50_95"]))
    lr_key = next((k for k in ("train/lr_max", "train/lr") if k in all_rows[0]), None)
    lrs = sorted({round(float(r[lr_key]), 9) for r in all_rows if r.get(lr_key)}, reverse=True) if lr_key else []
    return {"epochs": len(rows), "best_epoch": int(float(best["epoch"])), "best_5095": float(best["val/ema_mAP_50_95"]),
            "best_50": float(best.get("val/ema_mAP_50", "nan")), "lr_levels": len(lrs), "done": (arm_output_dir(arm) / "checkpoint_best_ema.pth").exists()}


def evals(arm: str) -> dict:
    out = {}
    for p in sorted((arm_output_dir(arm) / "eval").glob("*.json")) if (arm_output_dir(arm) / "eval").exists() else []:
        out[p.stem] = json.load(open(p))["metrics"]
    return out


def g(m: dict, *keys, default=float("nan")):
    for k in keys:
        if k in m:
            return m[k]
    return default


def main():
    print("# Real/synth arm study — results\n")
    print("All arms: RF-DETR Base, COCO pretrain, ReduceLROnPlateau recipe (`od_scripts/train.py`). "
          "Selection metric = EMA mAP@50:95 on **real/valid = LOCO subset-3 only**. Train-split numbers are fit diagnostics, never used for selection.\n")
    print("## Validation (selection metric, best EMA epoch)\n")
    print("| arm | train imgs | synth sources | epochs run | best ep | LR levels | mAP@50 | mAP@50:95 |\n|---|---:|---|---:|---:|---:|---:|---:|")
    man = json.load(open(OUT / "MANIFEST.json"))
    for arm, src in ARMS.items():
        t = training_summary(arm)
        n = man["arms"][arm]["train_images"]
        if not t:
            print(f"| {arm} | {n} | {', '.join(src) or '—'} | — | — | — | — | — |")
            continue
        flag = "" if t["done"] else " (running)"
        print(f"| {arm}{flag} | {n} | {', '.join(src) or '—'} | {t['epochs']} | {t['best_epoch']} | {t['lr_levels']} | {t['best_50']:.4f} | {t['best_5095']:.4f} |")

    print("\n## Post-hoc evals of `checkpoint_best_ema.pth` (per split)\n")
    print("| arm | split | mAP@50 | mAP@50:95 | " + " | ".join(f"AP50:95 {c}" for c in CLASSES) + " |\n|---|---|---:|---:|" + "---:|" * len(CLASSES))
    for arm in ARMS:
        ev = evals(arm)
        for s in SPLIT_ORDER:
            if s not in ev:
                continue
            m = ev[s]
            per = " | ".join(f"{g(m, f'val/AP/{c}', f'val/AP/{i}'):.4f}" for i, c in enumerate(CLASSES))  # validate() keys per-class AP by index
            print(f"| {arm} | {SPLIT_LABEL[s]} | {g(m, 'val/mAP_50'):.4f} | {g(m, 'val/mAP_50_95'):.4f} | {per} |")

    print("\n## Generalisation gap (train-real fit − valid-real), mAP@50:95\n")
    print("| arm | train_real | valid_real | gap | synth fit (per source) |\n|---|---:|---:|---:|---|")
    for arm, src in ARMS.items():
        ev = evals(arm)
        if "valid_real" not in ev or "train_real" not in ev:
            continue
        tr, va = g(ev["train_real"], "val/mAP_50_95"), g(ev["valid_real"], "val/mAP_50_95")
        synth = ", ".join(f"{s}: {g(ev[f'train_{s}'], 'val/mAP_50_95'):.4f}" for s in src if f"train_{s}" in ev) or "—"
        print(f"| {arm} | {tr:.4f} | {va:.4f} | {tr - va:+.4f} | {synth} |")


if __name__ == "__main__":
    main()

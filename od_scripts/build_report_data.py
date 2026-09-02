"""Gather every arm's full epoch curve + post-hoc eval metrics into one JSON for the
summary report (verify_gt.py's sibling: that one checks labels, this one checks results).

Usage: .venv/bin/python od_scripts/build_report_data.py [--out /home/ubuntu/datasets_coco/report_data.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARMS, OUT, arm_output_dir  # noqa: E402

CLASSES = ["forklift", "pallet", "pallet_truck"]
SOURCE_LABEL = {"basev2": "base_v2", "may": "may (Optuna rounds)", "traj_optuna": "traj-optuna",
                "basev4": "base_v4", "optuna_rand": "optuna-rand"}


def epoch_curve(arm: str):
    mcsv = arm_output_dir(arm) / "metrics.csv"
    if not mcsv.exists():
        return []
    curve = []
    for r in csv.DictReader(open(mcsv)):
        if not r.get("val/ema_mAP_50_95"):
            continue
        curve.append({
            "epoch": int(float(r["epoch"])),
            "map50": round(float(r["val/ema_mAP_50"]), 4),
            "map5095": round(float(r["val/ema_mAP_50_95"]), 4),
            "ap": [round(float(r.get(f"val/AP/{i}", "nan") or "nan"), 4) if r.get(f"val/AP/{i}") else None
                   for i in range(3)],
        })
    return curve


def lr_levels(arm: str):
    mcsv = arm_output_dir(arm) / "metrics.csv"
    if not mcsv.exists():
        return []
    levels, prev = [], None
    for r in csv.DictReader(open(mcsv)):
        key = next((k for k in ("train/lr_max", "train/lr") if r.get(k)), None)
        if not key:
            continue
        lr = round(float(r[key]), 9)
        ep = int(float(r["epoch"]))
        if lr != prev:
            levels.append({"epoch": ep, "lr": lr})
            prev = lr
    return levels


def eval_metrics(arm: str, split: str):
    p = arm_output_dir(arm) / "eval" / f"{split}.json"
    if not p.exists():
        return None
    m = json.load(open(p))["metrics"]
    return {
        "map50": round(m.get("val/mAP_50", float("nan")), 4),
        "map5095": round(m.get("val/mAP_50_95", float("nan")), 4),
        "ap": [round(m.get(f"val/AP/{i}", float("nan")), 4) for i in range(3)],
        "f1": round(m.get("val/F1", float("nan")), 4),
        "precision": round(m.get("val/precision", float("nan")), 4),
        "recall": round(m.get("val/recall", float("nan")), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "report_data.json"))
    args = ap.parse_args()

    manifest = json.load(open(OUT / "MANIFEST.json"))
    arms_out = []
    for arm, sources in ARMS.items():
        curve = epoch_curve(arm)
        if not curve:
            continue
        best = max(curve, key=lambda c: c["map5095"])
        d = manifest["arms"][arm]
        valid = eval_metrics(arm, "valid_real")
        train_real = eval_metrics(arm, "train_real")
        train_combined = eval_metrics(arm, "train_combined")
        synth_fit = {SOURCE_LABEL.get(s, s): eval_metrics(arm, f"train_{s}") for s in sources}
        synth_fit = {k: v for k, v in synth_fit.items() if v is not None}
        arms_out.append({
            "arm": arm,
            "sources": [SOURCE_LABEL.get(s, s) for s in sources],
            "train_images": d["train_images"],
            "by_source": d["by_source"],
            "epochs_run": len(curve),
            "lr_levels": lr_levels(arm),
            "curve": curve,
            "best_epoch": best["epoch"],
            "best_train": {"map50": best["map50"], "map5095": best["map5095"]},
            "valid": valid,
            "train_real_fit": train_real,
            "train_combined_fit": train_combined,
            "synth_fit": synth_fit,
        })

    out = {
        "classes": CLASSES,
        "real_valid_images": manifest["valid"]["images"],
        "real_train_images": manifest["arms"]["real"]["train_images"],
        "arms": arms_out,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out} ({len(arms_out)} arms)")


if __name__ == "__main__":
    main()

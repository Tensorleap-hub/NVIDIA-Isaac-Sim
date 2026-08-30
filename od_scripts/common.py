"""Shared constants/helpers for the warehouse-3cls RF-DETR study (od_scripts v2).

Rules baked in here:
  * validation = LOCO subset-3 ONLY (real photos); never train on it.
  * never validate on synthetic frames (all synth goes to train/).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/ubuntu/NVIDIA-Isaac-Sim")
PY = REPO / ".venv" / "bin" / "python"
OD = REPO / "od_scripts"

RAW = Path("/home/ubuntu/datasets")
LOCO = RAW / "loco_dataset"
LOCO_LABELS = LOCO / "labels"
BASEV2 = RAW / "base_v2_final"
MAY = RAW / "top-runs-may-ok"
TRAJ_OPTUNA = RAW / "trajectory-optimized"
BASEV4_TRAJ = RAW / "base_v4_trajectory"
BASEV4_RAND = RAW / "base_v4_random"

OUT = Path("/home/ubuntu/datasets_coco")   # built COCO datasets live here
LOGS = OUT / "logs"
EVALSETS = OUT / "evalsets"

KEEP_CLASSES = {"pallet_truck", "forklift", "pallet"}

# LOCO subsets -> role. Subset-3 is the one and only validation set.
LOCO_TRAIN_ANNS = [
    LOCO_LABELS / "loco-sub1-v1-val.json",
    LOCO_LABELS / "loco-sub2-v1-train.json",
    LOCO_LABELS / "loco-sub4-v1-val.json",
    LOCO_LABELS / "loco-sub5-v1-train.json",
]
LOCO_VAL_ANN = LOCO_LABELS / "loco-sub3-v1-train.json"
LOCO_VAL_IMGS = LOCO / "subset-3"

# Training arms: name -> synthetic sources merged into train/ (real is always included).
ARMS: dict[str, list[str]] = {
    "real": [],
    "real_basev2": ["basev2"],
    "real_may": ["may"],
    "real_all": ["basev2", "may"],
    "real_traj": ["traj_optuna"],
    "real_basev4": ["basev4"],
    "real_traj_basev4": ["traj_optuna", "basev4"],
    "real_all_traj": ["basev2", "may", "traj_optuna", "basev4"],
}
SYNTH_SOURCES = ["basev2", "may", "traj_optuna", "basev4"]
RUN_NAME = "rfdetr_reducelr"


def synth_run_dirs(source: str) -> list[Path]:
    """Isaac BasicWriter run directories for a synthetic source (sorted, unique names)."""
    if source == "basev2":
        dirs = sorted(p for p in BASEV2.iterdir() if p.is_dir())
    elif source == "may":
        dirs = sorted(MAY.glob("*/trial_*/outputs/*/"))
    elif source == "traj_optuna":
        # nested Camera/rgb layout (trajectory-SDG runs), not flat rgb_*.png
        dirs = sorted(p for p in TRAJ_OPTUNA.iterdir() if p.is_dir())
    elif source == "basev4":
        # ONE dataset in two render modes: base_v4_trajectory (exp01-06, v4t_ prefix, ~5 frames/seed)
        # + base_v4_random (exp07-32, v4r_ prefix, 1 frame/seed). Nested Camera/rgb layout.
        # Prefixes were added on copy so names don't collide with base_v2_final's expNN_ in source_of().
        dirs = sorted(p for p in list(BASEV4_TRAJ.iterdir()) + list(BASEV4_RAND.iterdir()) if p.is_dir())
    else:
        raise ValueError(source)
    dirs = [d for d in dirs if any(d.glob("rgb_*.png")) or any(d.glob("Camera/rgb/rgb_*.png"))]
    names = [d.name for d in dirs]
    assert len(names) == len(set(names)), f"{source}: duplicate run-dir names would collide as run_prefix"
    return dirs


def arm_dir(arm: str) -> Path:
    return OUT / arm


def arm_output_dir(arm: str) -> Path:
    return arm_dir(arm) / "output" / RUN_NAME


def load_class_names(dataset_dir: str | Path) -> list[str]:
    """Class names ordered by ascending category_id.

    RF-DETR's roboflow loader maps sorted(cat_ids) -> 0..N-1, so class_names must be
    given in ascending-id order or every per-class metric is silently permuted.
    """
    with open(Path(dataset_dir) / "train" / "_annotations.coco.json") as f:
        cats = json.load(f)["categories"]
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def source_of(file_name: str) -> str:
    """Classify a train image by its filename (real jpgs vs prefixed synth pngs)."""
    if file_name.endswith(".jpg"):
        return "real"
    if file_name.startswith(("v4t_", "v4r_")):
        return "basev4"
    if file_name.startswith("exp"):
        return "basev2"
    if file_name.startswith(("top", "dp", "dl")):
        return "traj_optuna"
    if file_name.startswith("iter"):
        return "may"
    return "unknown"

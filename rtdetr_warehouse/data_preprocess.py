import json
import os
import random
import re
from collections import Counter
from typing import List

import cv2
import numpy as np
import yaml

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_preprocess,
)

from rtdetr_warehouse.config import COCO_ID_TO_IDX, CONFIG

IMAGE_SIZE = int(CONFIG["image_size"])
MAX_DETS = int(CONFIG["max_num_of_objects"])


def _validate_unique_sample_ids(sample_ids: list[str], label: str) -> None:
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(
            f"Duplicate sample ids found in {label}: {preview}"
        )


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    """
    Load LOCO COCO annotations and return train + val PreprocessResponse objects.

    preprocess.data  : list of record dicts, indexed by integer idx
    preprocess.length: len(records)

    Each record dict:
        path    : absolute image path
        width   : original image width (pixels)
        height  : original image height (pixels)
        subset  : 'subset-1' … 'subset-5'
        anns    : list of COCO annotation dicts for this image
    """
    data_path = CONFIG["data"]["data_path"]
    ann_file = os.path.join(data_path, CONFIG["data"]["annotations_file"])

    with open(ann_file, "r") as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    train_subsets = set(CONFIG["split"]["train_subsets"])
    val_subsets = set(CONFIG["split"]["val_subsets"])

    train_records, val_records = [], []

    for img in coco["images"]:
        parts = img["path"].lstrip("/").split("/")
        subset = parts[1] if len(parts) > 1 else ""
        full_path = os.path.join(data_path, *parts)
        record = {
            "image_id": img["id"],
            "path": full_path,
            "width": img["width"],
            "height": img["height"],
            "subset": subset,
            "anns": anns_by_image.get(img["id"], []),
        }
        if subset in train_subsets:
            train_records.append(record)
        elif subset in val_subsets:
            val_records.append(record)

    synth_records = _load_synth_records()
    extended_records = _load_extended_records()
    optuna_records = _load_optuna_records()

    max_samples = CONFIG.get("max_samples")
    if max_samples is not None:
        train_records = train_records[:max_samples]
        val_records   = val_records[:max_samples]

    synth_records.sort(key=lambda r: r["run_number"])
    extended_records.sort(key=lambda r: (r["run_number"], r["experiment"]))
    optuna_records.sort(key=lambda r: (r["iteration"], r["run_number"], r["experiment"]))

    additional_records = synth_records + extended_records + optuna_records

    train_ids = [str(r["image_id"]) for r in train_records]
    val_ids   = [str(r["image_id"]) for r in val_records]
    additional_ids = []
    for r in additional_records:
        if r["subset"] == "synth":
            sample_id = f"run{r['run_number']}_{r['experiment']}_frame{r['image_id']}"
        elif r["subset"] == "extended":
            sample_id = f"ext{r['run_number']}_{r['experiment']}_frame{r['image_id']}"
        elif r["subset"] == "optuna":
            sample_id = (
                f"optuna_iter{r['iteration']}_run{r['run_number']}_"
                f"{r['experiment']}_frame{r['image_id']}"
            )
        else:
            raise ValueError(f"Unsupported additional subset {r['subset']!r}")
        additional_ids.append(sample_id)
    _validate_unique_sample_ids(train_ids, "training split")
    _validate_unique_sample_ids(val_ids, "validation split")
    _validate_unique_sample_ids(additional_ids, "additional split")
    _validate_unique_sample_ids(train_ids + val_ids + additional_ids, "all splits")
    if len(additional_ids) > 0:
        return [
            PreprocessResponse(data={sid: r for sid, r in zip(train_ids, train_records)}, sample_ids=train_ids, state=DataStateType.training),
            PreprocessResponse(data={sid: r for sid, r in zip(val_ids,   val_records)},   sample_ids=val_ids,   state=DataStateType.validation),
            PreprocessResponse(data={sid: r for sid, r in zip(additional_ids, additional_records)}, sample_ids=additional_ids, state=DataStateType.additional),
        ]
    else:
        return [
            PreprocessResponse(data={sid: r for sid, r in zip(train_ids, train_records)}, sample_ids=train_ids,
                               state=DataStateType.training),
            PreprocessResponse(data={sid: r for sid, r in zip(val_ids, val_records)}, sample_ids=val_ids,
                               state=DataStateType.validation),
            ]
            # ---------------------------------------------------------------------------
# Synthetic data loading (KITTI annotations, Isaac Sim)
# ---------------------------------------------------------------------------

# KITTI class name → LOCO category index
_SYNTH_CLASS_TO_IDX = {"palletjack": 4}  # pallet_truck


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_sdg_base_config() -> dict:
    sdg_path = os.path.join(os.path.dirname(__file__), "..", "palletjack_sdg", "sdg_config.yaml")
    sdg_path = os.path.normpath(sdg_path)
    if not os.path.isfile(sdg_path):
        return {}
    with open(sdg_path, "r") as f:
        return yaml.safe_load(f)


_SDG_BASE_CONFIG = _load_sdg_base_config()


def _load_synth_records() -> list:
    """
    Load synthetic frames from all palletjack_run_*/exp* directories.

    Each experiment's run_config.yaml is deep-merged on top of the base
    sdg_config.yaml so missing fields always have the sim default.
    Each record includes 'run_number' and 'experiment' for metadata.
    """
    synth_cfg = CONFIG.get("synth_data", {})
    if not synth_cfg.get("additional", True):
        return []

    base = synth_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    allowed_runs = synth_cfg.get("run_numbers")  # None, int, or list of ints
    if isinstance(allowed_runs, int):
        allowed_runs = [allowed_runs]

    records = []
    run_dirs = sorted(
        d for d in os.listdir(base)
        if d.startswith("palletjack_run_") and os.path.isdir(os.path.join(base, d))
    )
    if allowed_runs is not None:
        allowed_runs = set(allowed_runs)
        available_runs = {int(d.split("_")[-1]) for d in run_dirs}
        missing = allowed_runs - available_runs
        if missing:
            raise ValueError(
                f"synth_data.run_numbers: desired {sorted(allowed_runs)}, "
                f"data has {sorted(available_runs)}, "
                f"missing {sorted(missing)}"
            )
        run_dirs = [d for d in run_dirs if int(d.split("_")[-1]) in allowed_runs]

    for run_dir in run_dirs:
        run_number = int(run_dir.split("_")[-1])
        run_path = os.path.join(base, run_dir)

        exp_dirs = sorted(
            d for d in os.listdir(run_path)
            if os.path.isdir(os.path.join(run_path, d))
        )

        for exp_dir in exp_dirs:
            exp_path = os.path.join(run_path, exp_dir)
            run_config_path = os.path.join(exp_path, "run_config.yaml")
            if not os.path.isfile(run_config_path):
                continue

            with open(run_config_path, "r") as f:
                exp_config = yaml.safe_load(f)
            run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

            rgb_dir = os.path.join(exp_path, "Camera", "rgb")
            ann_dir = os.path.join(exp_path, "Camera", "object_detection")
            num_frames = int(run_config.get("run", {}).get("num_frames", 0))
            orig_w = int(run_config.get("render", {}).get("width", 960))
            orig_h = int(run_config.get("render", {}).get("height", 544))

            for i in range(num_frames):
                img_path = os.path.join(rgb_dir, f"{i}.png")
                if not os.path.isfile(img_path):
                    continue
                ann_path = os.path.join(ann_dir, f"{i}.txt")

                anns = []
                if os.path.isfile(ann_path):
                    with open(ann_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 8:
                                continue
                            if parts[0].lower() not in _SYNTH_CLASS_TO_IDX:
                                continue
                            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                            anns.append({
                                "category_id": 11,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                            })

                records.append({
                    "image_id": i,
                    "path": img_path,
                    "width": orig_w,
                    "height": orig_h,
                    "subset": "synth",
                    "anns": anns,
                    "run_config": run_config,
                    "run_number": run_number,
                    "experiment": exp_dir,
                })

    num_samples = synth_cfg.get("num_samples")
    if num_samples is not None:
        by_run = {}
        for r in records:
            by_run.setdefault(r["run_number"], []).append(r)
        sampled = []
        rng = random.Random(42)
        for run_records in by_run.values():
            if len(run_records) > num_samples:
                rng.shuffle(run_records)
                sampled.extend(run_records[:num_samples])
            else:
                sampled.extend(run_records)
        records = sampled

    return records


def _load_extended_records() -> list:
    """
    Load frames from extended/{run_number}/{exp_dir}/Camera/ directories.

    Same KITTI format as synth data; run dirs are plain integers (0, 1, ...).
    Records are tagged subset='extended' and carry run_config for metadata.
    """
    ext_cfg = CONFIG.get("extended_data", {})
    if not ext_cfg.get("additional", True):
        return []

    base = ext_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    allowed_runs = ext_cfg.get("extended_numbers")  # None, int, or list of ints
    if isinstance(allowed_runs, int):
        allowed_runs = [allowed_runs]

    run_dirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.isdigit()
    )
    if allowed_runs is not None:
        allowed_runs = set(allowed_runs)
        available_runs = {int(d) for d in run_dirs}
        missing = allowed_runs - available_runs
        if missing:
            raise ValueError(
                f"extended_data.extended_numbers: desired {sorted(allowed_runs)}, "
                f"data has {sorted(available_runs)}, "
                f"missing {sorted(missing)}"
            )
        run_dirs = [d for d in run_dirs if int(d) in allowed_runs]

    records = []
    for run_dir in run_dirs:
        run_number = int(run_dir)
        run_path = os.path.join(base, run_dir)

        exp_dirs = sorted(
            d for d in os.listdir(run_path)
            if os.path.isdir(os.path.join(run_path, d))
        )

        for exp_dir in exp_dirs:
            exp_path = os.path.join(run_path, exp_dir)
            run_config_path = os.path.join(exp_path, "run_config.yaml")
            if not os.path.isfile(run_config_path):
                continue

            with open(run_config_path, "r") as f:
                exp_config = yaml.safe_load(f)
            run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

            rgb_dir = os.path.join(exp_path, "Camera", "rgb")
            ann_dir = os.path.join(exp_path, "Camera", "object_detection")
            num_frames = int(run_config.get("run", {}).get("num_frames", 0))
            orig_w = int(run_config.get("render", {}).get("width", 960))
            orig_h = int(run_config.get("render", {}).get("height", 544))

            for i in range(num_frames):
                img_path = os.path.join(rgb_dir, f"{i}.png")
                if not os.path.isfile(img_path):
                    continue
                ann_path = os.path.join(ann_dir, f"{i}.txt")

                anns = []
                if os.path.isfile(ann_path):
                    with open(ann_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 8:
                                continue
                            if parts[0].lower() not in _SYNTH_CLASS_TO_IDX:
                                continue
                            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                            anns.append({
                                "category_id": 11,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                            })

                records.append({
                    "image_id": i,
                    "path": img_path,
                    "width": orig_w,
                    "height": orig_h,
                    "subset": "extended",
                    "anns": anns,
                    "run_config": run_config,
                    "run_number": run_number,
                    "experiment": exp_dir,
                })

    num_samples = ext_cfg.get("num_samples")
    if num_samples is not None:
        by_run = {}
        for r in records:
            by_run.setdefault(r["run_number"], []).append(r)
        sampled = []
        rng = random.Random(42)
        for run_records in by_run.values():
            if len(run_records) > num_samples:
                rng.shuffle(run_records)
                sampled.extend(run_records[:num_samples])
            else:
                sampled.extend(run_records)
        records = sampled

    return records


_OPTUNA_DIR_RE = re.compile(r"^iter(?P<iteration>\d+)_run(?P<run>\d+)$")


def _load_optuna_records() -> list:
    """
    Load frames from optuna-ec2/iterXXX_runYYY/Camera/ directories.

    Each top-level directory is treated as one experiment. Records are tagged
    subset='optuna' and carry parsed iteration/run identifiers for metadata.
    """
    optuna_cfg = CONFIG.get("optuna_data", {})
    if not optuna_cfg.get("additional", True):
        return []

    base = optuna_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    exp_dirs = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and _OPTUNA_DIR_RE.match(d)
    )

    records = []
    for exp_dir in exp_dirs:
        match = _OPTUNA_DIR_RE.match(exp_dir)
        if match is None:
            continue

        iteration = int(match.group("iteration"))
        run_number = int(match.group("run"))
        exp_path = os.path.join(base, exp_dir)
        run_config_path = os.path.join(exp_path, "run_config.yaml")
        if not os.path.isfile(run_config_path):
            continue

        with open(run_config_path, "r") as f:
            exp_config = yaml.safe_load(f)
        run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

        rgb_dir = os.path.join(exp_path, "Camera", "rgb")
        ann_dir = os.path.join(exp_path, "Camera", "object_detection")
        num_frames = int(run_config.get("run", {}).get("num_frames", 0))
        orig_w = int(run_config.get("render", {}).get("width", 960))
        orig_h = int(run_config.get("render", {}).get("height", 544))

        for i in range(num_frames):
            img_path = os.path.join(rgb_dir, f"{i}.png")
            if not os.path.isfile(img_path):
                continue
            ann_path = os.path.join(ann_dir, f"{i}.txt")

            anns = []
            if os.path.isfile(ann_path):
                with open(ann_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 8:
                            continue
                        if parts[0].lower() not in _SYNTH_CLASS_TO_IDX:
                            continue
                        x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                        anns.append({
                            "category_id": 11,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                        })

            records.append({
                "image_id": i,
                "path": img_path,
                "width": orig_w,
                "height": orig_h,
                "subset": "optuna",
                "anns": anns,
                "run_config": run_config,
                "run_number": run_number,
                "iteration": iteration,
                "experiment": exp_dir,
            })

    num_samples = optuna_cfg.get("num_samples")
    if num_samples is not None:
        by_experiment = {}
        for record in records:
            by_experiment.setdefault(record["experiment"], []).append(record)
        sampled = []
        rng = random.Random(42)
        for experiment_records in by_experiment.values():
            if len(experiment_records) > num_samples:
                rng.shuffle(experiment_records)
                sampled.extend(experiment_records[:num_samples])
            else:
                sampled.extend(experiment_records)
        records = sampled

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_chw(path: str) -> np.ndarray:
    """Load image as CHW float32 normalized to [0, 1]."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0  # CHW


def _build_padded_gt(record: dict) -> np.ndarray:
    """
    Build padded GT array of shape (MAX_DETS, 5).

    Columns: [class_idx, cx, cy, w, h] normalized in model input space.
    Padding rows filled with -1.
    """
    gt = np.full((MAX_DETS, 5), -1.0, dtype=np.float32)
    valid_anns = [a for a in record["anns"] if a["category_id"] in COCO_ID_TO_IDX]
    n = min(len(valid_anns), MAX_DETS)
    if n == 0:
        return gt

    orig_w, orig_h = record["width"], record["height"]
    x_scale = IMAGE_SIZE / orig_w
    y_scale = IMAGE_SIZE / orig_h

    for i, ann in enumerate(valid_anns[:n]):
        x_min, y_min, bw, bh = ann["bbox"]
        cx = (x_min + bw / 2) * x_scale / IMAGE_SIZE
        cy = (y_min + bh / 2) * y_scale / IMAGE_SIZE
        w_n = bw * x_scale / IMAGE_SIZE
        h_n = bh * y_scale / IMAGE_SIZE
        gt[i, 0] = float(COCO_ID_TO_IDX[ann["category_id"]])
        gt[i, 1] = cx
        gt[i, 2] = cy
        gt[i, 3] = w_n
        gt[i, 4] = h_n

    return gt


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Returns CHW float32 image normalized to [0, 1].
    Shape: (3, 640, 640)
    """
    return _load_image_chw(preprocess.data[idx]["path"])


@tensorleap_input_encoder("orig_size", channel_dim=1)
def input_size_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Returns [H, W] as float32 — cast to int64 in integration_test before inference.
    The RT-DETR model uses this to scale box outputs to pixel space.
    """
    return np.array([IMAGE_SIZE, IMAGE_SIZE], dtype=np.float32)


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Full GT tensor: (MAX_DETS, 5) float32 — [cls, cx, cy, w, h], -1 = padding.
    """
    return _build_padded_gt(preprocess.data[idx])


@tensorleap_gt_encoder("gt_boxes")
def gt_boxes_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """GT boxes only: (MAX_DETS, 4) float32 — [cx, cy, w, h], 0 for padded rows."""
    gt = _build_padded_gt(preprocess.data[idx])
    boxes = gt[:, 1:5].copy()
    boxes[gt[:, 0] < 0] = 0.0
    return boxes


@tensorleap_gt_encoder("gt_labels")
def gt_labels_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """GT class indices: (MAX_DETS,) float32, -1 for padded rows."""
    return _build_padded_gt(preprocess.data[idx])[:, 0]


@tensorleap_gt_encoder("gt_valid_mask")
def gt_valid_mask_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """Binary mask: (MAX_DETS,) float32 — 1 for valid GT rows, 0 for padding."""
    gt = _build_padded_gt(preprocess.data[idx])
    return (gt[:, 0] >= 0).astype(np.float32)

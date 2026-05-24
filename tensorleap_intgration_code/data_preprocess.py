import json
import os
import random
import re
from collections import Counter
from pathlib import Path
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

from tensorleap_intgration_code.config import COCO_ID_TO_IDX, CONFIG, abs_path_from_root

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
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_additional_sample_id(r: dict) -> str:
    if r["subset"] == "synth":
        return f"run{r['run_number']}_{r['experiment']}_frame{r['image_id']}"
    elif r["subset"] == "extended":
        return f"ext_iter{r.get('iteration', 0)}_run{r['run_number']}_{r['experiment']}_frame{r['image_id']}"
    elif r["subset"] == "Tensorleap-Optimized":
        if r.get("optuna_selected_cycle") is not None:
            return (
                f"optuna_selected_{r.get('optuna_theme', 'flat')}_"
                f"cycle{int(r['optuna_selected_cycle']):02d}_"
                f"{r.get('optuna_selected_timestamp', 'unknown')}_"
                f"{r.get('optuna_selected_kind', 'unknown')}_"
                f"{r.get('optuna_selected_label', 'unknown')}_"
                f"iter{r['iteration']}_run{r['run_number']}_"
                f"{r['experiment']}_frame{r['image_id']}"
            )
        else:
            repetition_part = f"{r.get('optuna_repetition', '')}_" if r.get("optuna_repetition") else ""
            return (
                f"optuna_{r.get('optuna_bucket', 'regular')}_{r.get('optuna_theme', 'flat')}_"
                f"{repetition_part}"
                f"iter{r['iteration']}_run{r['run_number']}_"
                f"{r['experiment']}_frame{r['image_id']}"
            )
    elif r["subset"] == "optuna_tests":
        return (
            f"optuna_tests_{r.get('optuna_test_name', 'unknown')}_"
            f"{r.get('run_name', 'unknown')}_{r['experiment']}_frame{r['image_id']}"
        )
    elif r["subset"] == "Synthetic_Base":
        return f"base_synth_{r['experiment']}_frame{r['image_id']}"
    else:
        raise ValueError(f"Unsupported additional subset {r['subset']!r}")


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
    custom_cfg = CONFIG.get("custom_latent_space") or {}
    if custom_cfg.get("real_cache_manifest") or custom_cfg.get("base_cache_manifest") or custom_cfg.get("runs_root"):
        return _preprocess_custom_latent_space(custom_cfg)

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
    base_synth_records = _load_base_synth_records()
    extended_records = _load_extended_records()
    optuna_records = _load_optuna_records()
    optuna_test_records = _load_optuna_test_records()

    max_samples = CONFIG.get("max_samples")
    if max_samples is not None:
        train_records = train_records[:max_samples]
        val_records   = val_records[:max_samples]

    synth_records.sort(key=lambda r: r["run_number"])
    base_synth_records.sort(key=lambda r: (r["experiment"], r["image_id"]))
    extended_records.sort(key=lambda r: (r.get("iteration", 0), r["run_number"], r["experiment"]))
    optuna_records.sort(
        key=lambda r: (
            str(r.get("optuna_bucket", "")),
            str(r.get("optuna_theme", "")),
            str(r.get("optuna_repetition", "")),
            r["iteration"],
            r["run_number"],
            r["experiment"],
        )
    )
    optuna_test_records.sort(
        key=lambda r: (
            str(r.get("optuna_test_name", "")),
            str(r.get("run_name", "")),
            r["experiment"],
            r["image_id"],
        )
    )

    additional_records = synth_records + base_synth_records + extended_records + optuna_records + optuna_test_records

    train_ids = [str(r["image_id"]) for r in train_records]
    val_ids   = [str(r["image_id"]) for r in val_records]
    additional_ids = [_make_additional_sample_id(r) for r in additional_records]
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
# Custom latent-space loading (images cached by visualize_population.py)
# ---------------------------------------------------------------------------

def _load_manifest_image_records(manifest_path: str, subset_name: str) -> list:
    if not os.path.isabs(manifest_path):
        manifest_path = abs_path_from_root(manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    image_paths = manifest.get("image_paths", [])
    if not image_paths:
        raise ValueError(f"No image_paths in manifest: {manifest_path}")
    return [
        {
            "image_id": i,
            "path": path,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "subset": subset_name,
            "anns": [],
        }
        for i, path in enumerate(image_paths)
    ]


def _load_flat_run_records(runs_path: Path) -> list:
    """Handle flat layout: {runs_root}/{category}/{trial_id}/outputs/{iter_run__{fp}/"""
    records: list = []
    for category_dir in sorted(p for p in runs_path.iterdir() if p.is_dir()):
        category = category_dir.name
        for trial_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            outputs_root = trial_dir / "outputs"
            if not outputs_root.is_dir():
                continue
            for experiment_dir in sorted(
                p for p in outputs_root.iterdir()
                if p.is_dir() and _OPTUNA_DIR_RE.match(p.name)
            ):
                match = _OPTUNA_DIR_RE.match(experiment_dir.name)
                iteration = int(match.group("iteration"))
                run_number = int(match.group("run"))
                yaml_name = f"iter{iteration:03d}_run{run_number:03d}.yaml"
                run_config_path = trial_dir / "yamls" / yaml_name
                if not run_config_path.is_file():
                    continue
                with run_config_path.open("r") as f:
                    exp_config = yaml.safe_load(f)
                run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)
                orig_w = int(run_config.get("render", {}).get("width", 960))
                orig_h = int(run_config.get("render", {}).get("height", 544))
                frame_records = _read_supported_frame_records(experiment_dir, run_config)
                for image_id, img_path, anns in frame_records:
                    records.append({
                        "image_id": image_id,
                        "path": str(img_path),
                        "width": orig_w,
                        "height": orig_h,
                        "subset": "Tensorleap-Optimized",
                        "anns": anns,
                        "run_config": run_config,
                        "run_number": run_number,
                        "iteration": iteration,
                        "experiment": experiment_dir.name,
                        "optuna_bucket": "selected",
                        "optuna_theme": category,
                        "optuna_repetition": trial_dir.name,
                        "trial_number": None,
                        "optuna_rank": None,
                        "optuna_objective_value": None,
                        "optuna_selected_kind": "",
                        "optuna_selected_label": trial_dir.name,
                        "optuna_selected_cycle": None,
                        "optuna_selected_timestamp": "",
                    })
    return records


def _load_custom_run_records(runs_root: str) -> list:
    if not os.path.isabs(runs_root):
        runs_root = abs_path_from_root(runs_root)
    runs_path = Path(runs_root)
    if not runs_path.is_dir():
        raise ValueError(f"custom_latent_space.runs_root does not exist: {runs_root}")
    has_cycles = any(_OPTUNA_CYCLE_RE.match(p.name) for p in runs_path.rglob("*") if p.is_dir())
    if not has_cycles:
        return _load_flat_run_records(runs_path)
    records: list = []
    for cycle_dir in sorted(p for p in runs_path.rglob("*") if p.is_dir() and _OPTUNA_CYCLE_RE.match(p.name)):
        cycle_match = _OPTUNA_CYCLE_RE.match(cycle_dir.name)
        cycle_index = int(cycle_match.group("cycle"))
        timestamp = cycle_match.group("timestamp")
        category = cycle_dir.parent.name
        for selected_trial_dir in sorted(p for p in cycle_dir.iterdir() if p.is_dir()):
            _append_optuna_selected_trial_records(
                records=records,
                selected_trial_dir=selected_trial_dir,
                category=category,
                cycle_index=cycle_index,
                timestamp=timestamp,
            )
    return records


def _preprocess_custom_latent_space(custom_cfg: dict) -> List[PreprocessResponse]:
    real_manifest = custom_cfg.get("real_cache_manifest")
    base_manifest = custom_cfg.get("base_cache_manifest")
    runs_root = custom_cfg.get("runs_root")

    real_records = _load_manifest_image_records(real_manifest, "custom_real") if real_manifest else []
    base_records = _load_manifest_image_records(base_manifest, "custom_base") if base_manifest else []
    run_records = _load_custom_run_records(runs_root) if runs_root else []

    real_ids = [f"custom_real_{i}" for i in range(len(real_records))]
    base_ids = [f"custom_base_{i}" for i in range(len(base_records))]
    run_ids = [_make_additional_sample_id(r) for r in run_records]

    _validate_unique_sample_ids(real_ids, "custom_latent_space real")
    _validate_unique_sample_ids(base_ids, "custom_latent_space base")
    _validate_unique_sample_ids(run_ids, "custom_latent_space runs")
    _validate_unique_sample_ids(real_ids + base_ids + run_ids, "custom_latent_space all")

    subsets = []
    if real_records:
        subsets.append(PreprocessResponse(
            data={sid: r for sid, r in zip(real_ids, real_records)},
            sample_ids=real_ids,
            state=DataStateType.training,
        ))
    if base_records:
        subsets.append(PreprocessResponse(
            data={sid: r for sid, r in zip(base_ids, base_records)},
            sample_ids=base_ids,
            state=DataStateType.validation,
        ))
    if run_records:
        subsets.append(PreprocessResponse(
            data={sid: r for sid, r in zip(run_ids, run_records)},
            sample_ids=run_ids,
            state=DataStateType.additional,
        ))
    return subsets


# ---------------------------------------------------------------------------
# Synthetic data loading (KITTI annotations, Isaac Sim)
# ---------------------------------------------------------------------------

# Synthetic class name → COCO category_id (matched to the 3-class warehouse config)
_SYNTH_CLASS_TO_IDX = {
    "palletjack": 3,   # small_load_carrier (idx 0)
    "forklift":    5,  # forklift           (idx 1)
    "pallet":      7,  # pallet             (idx 2)
}


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
    selected_runs = set(allowed_runs) if allowed_runs is not None else None

    records = []
    base_path = Path(base)
    run_dirs = sorted(
        path for path in base_path.iterdir()
        if path.name.startswith("palletjack_run_") and path.is_dir()
    )
    if selected_runs is not None:
        available_runs = {int(path.name.split("_")[-1]) for path in run_dirs}
        missing = selected_runs - available_runs
        if missing:
            raise ValueError(
                f"synth_data.run_numbers: desired {sorted(selected_runs)}, "
                f"data has {sorted(available_runs)}, "
                f"missing {sorted(missing)}"
            )
        run_dirs = [path for path in run_dirs if int(path.name.split("_")[-1]) in selected_runs]

    for run_path in run_dirs:
        run_number = int(run_path.name.split("_")[-1])
        run_records_start = len(records)

        exp_dirs = sorted(
            path for path in run_path.iterdir()
            if path.is_dir()
        )

        for exp_path in exp_dirs:
            exp_dir = exp_path.name
            run_config_path = exp_path / "run_config.yaml"
            if not run_config_path.is_file():
                continue

            with run_config_path.open("r") as f:
                exp_config = yaml.safe_load(f)
            run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

            orig_w = int(run_config.get("render", {}).get("width", 960))
            orig_h = int(run_config.get("render", {}).get("height", 544))
            frame_records = _read_supported_frame_records(exp_path, run_config)

            for image_id, img_path, anns in frame_records:
                records.append({
                    "image_id": image_id,
                    "path": str(img_path),
                    "width": orig_w,
                    "height": orig_h,
                    "subset": "synth",
                    "anns": anns,
                    "run_config": run_config,
                    "run_number": run_number,
                    "experiment": exp_dir,
                })

        if selected_runs is not None and len(records) == run_records_start:
            _raise_empty_selected_data_error(
                "synth_data.run_numbers",
                run_number,
                run_path,
            )

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


def _load_base_synth_records() -> list:
    """
    Load frames from a flat directory of named experiments:
      base_path/
        exp01_clean_overhead_low_noise/
          run_config.yaml
          rgb_0000.png
          bounding_box_2d_tight_0000.npy
          bounding_box_2d_tight_labels_0000.json
        exp02_full_warehouse_bright/ ...

    Configured via `base_synth_data` in project_config.yaml.
    """
    cfg = CONFIG.get("base_synth_data", {})
    if not cfg.get("additional", True):
        return []

    base = cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    base_path = Path(base)
    records = []

    for exp_path in sorted(path for path in base_path.iterdir() if path.is_dir()):
        run_config_path = exp_path / "run_config.yaml"
        if not run_config_path.is_file():
            continue

        with run_config_path.open("r") as f:
            exp_config = yaml.safe_load(f)
        run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

        orig_w = int(run_config.get("render", {}).get("width", 960))
        orig_h = int(run_config.get("render", {}).get("height", 544))
        frame_records = _read_supported_frame_records(exp_path, run_config)

        for image_id, img_path, anns in frame_records:
            records.append({
                "image_id": image_id,
                "path": str(img_path),
                "width": orig_w,
                "height": orig_h,
                "subset": "Synthetic_Base",
                "anns": anns,
                "run_config": run_config,
                "experiment": exp_path.name,
            })

    num_samples = cfg.get("num_samples")
    if num_samples is not None:
        by_exp = {}
        for r in records:
            by_exp.setdefault(r["experiment"], []).append(r)
        sampled = []
        rng = random.Random(42)
        for exp_records in by_exp.values():
            if len(exp_records) > num_samples:
                rng.shuffle(exp_records)
                sampled.extend(exp_records[:num_samples])
            else:
                sampled.extend(exp_records)
        records = sampled

    return records


def _load_extended_records() -> list:
    """
    Load frames from extended/{name}-{iter}/{run_name}/ directories.

    Top-level folders are named like 'camera-01'; the trailing number is the
    iteration index.  Each subfolder (e.g. 'dist_29') is a run; the number
    after the last '_' is the run_number.  Frames use the flat basic-writer
    format (rgb_XXXX.png + bounding_box_2d_tight_XXXX.npy/json).
    """
    ext_cfg = CONFIG.get("extended_data", {})
    if not ext_cfg.get("additional", True):
        return []

    base = ext_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    allowed_iters = ext_cfg.get("extended_numbers")
    if isinstance(allowed_iters, int):
        allowed_iters = [allowed_iters]
    selected_iters = set(allowed_iters) if allowed_iters is not None else None

    base_path = Path(base)
    iter_dirs = sorted(
        path for path in base_path.iterdir()
        if path.is_dir() and _EXTENDED_ITER_DIR_RE.match(path.name)
    )
    if not iter_dirs:
        raise FileNotFoundError(f"no iteration dirs found in {base}")

    if selected_iters is not None:
        available_iters = {int(_EXTENDED_ITER_DIR_RE.match(p.name).group(1)) for p in iter_dirs}
        missing = selected_iters - available_iters
        if missing:
            raise ValueError(
                f"extended_data.extended_numbers: desired {sorted(selected_iters)}, "
                f"data has {sorted(available_iters)}, "
                f"missing {sorted(missing)}"
            )
        iter_dirs = [p for p in iter_dirs if int(_EXTENDED_ITER_DIR_RE.match(p.name).group(1)) in selected_iters]

    records = []
    for iter_path in iter_dirs:
        iteration = int(_EXTENDED_ITER_DIR_RE.match(iter_path.name).group(1))

        for run_path in sorted(path for path in iter_path.iterdir() if path.is_dir()):
            run_number_str = run_path.name.split("_")[-1]
            if not run_number_str.isdigit():
                continue
            run_number = int(run_number_str)

            run_config_path = run_path / "run_config.yaml"
            if not run_config_path.is_file():
                continue

            with run_config_path.open("r") as f:
                exp_config = yaml.safe_load(f)
            run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

            orig_w = int(run_config.get("render", {}).get("width", 960))
            orig_h = int(run_config.get("render", {}).get("height", 544))
            frame_records = _read_supported_frame_records(run_path, run_config)

            for image_id, img_path, anns in frame_records:
                records.append({
                    "image_id": image_id,
                    "path": str(img_path),
                    "width": orig_w,
                    "height": orig_h,
                    "subset": "extended",
                    "anns": anns,
                    "run_config": run_config,
                    "iteration": iteration,
                    "run_number": run_number,
                    "experiment": run_path.name,
                })

    num_samples = ext_cfg.get("num_samples")
    if num_samples is not None:
        by_run = {}
        for r in records:
            by_run.setdefault((r["iteration"], r["run_number"]), []).append(r)
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


_EXTENDED_ITER_DIR_RE = re.compile(r"^.+-(\d+)$")
_OPTUNA_DIR_RE = re.compile(r"^iter(?P<iteration>\d+)_run(?P<run>\d+)(?:__[0-9a-f]+)?$")
_OPTUNA_TRIAL_DIR_RE = re.compile(r"^trial_(?P<trial>\d+)$")
_OPTUNA_FLAT_RGB_RE = re.compile(r"^rgb_(?P<frame>\d+)\.png$")
_OPTUNA_CYCLE_RE = re.compile(r"^cycle_(?P<cycle>\d+)_(?P<timestamp>\d{8}T\d{6}Z)$")
_OPTUNA_SELECTED_TRIAL_RE = re.compile(r"^(?P<kind>best|worst)_(?P<label>.+)$")


def _parse_kitti_annotation_file(annotation_path: str) -> list[dict]:
    anns = []
    if not os.path.isfile(annotation_path):
        return anns

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            class_name = parts[0].lower()
            if class_name not in _SYNTH_CLASS_TO_IDX:
                continue
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            anns.append({
                "category_id": _SYNTH_CLASS_TO_IDX[class_name],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            })
    return anns


def _parse_basic_writer_bbox_annotations(annotation_path: Path, label_path: Path) -> list[dict]:
    anns = []
    if not annotation_path.is_file() or not label_path.is_file():
        return anns

    with label_path.open("r") as f:
        label_payload = json.load(f)
    labels_by_id = label_payload if isinstance(label_payload, dict) else {}
    boxes = np.load(annotation_path, allow_pickle=False)

    for box in boxes:
        semantic_id = int(box["semanticId"])
        label_entry = labels_by_id.get(str(semantic_id), {})
        class_name = str(label_entry.get("class", "")).lower()
        if class_name not in _SYNTH_CLASS_TO_IDX:
            continue

        x1 = float(box["x_min"])
        y1 = float(box["y_min"])
        x2 = float(box["x_max"])
        y2 = float(box["y_max"])
        if x2 <= x1 or y2 <= y1:
            continue

        anns.append({
            "category_id": _SYNTH_CLASS_TO_IDX[class_name],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        })
    return anns


def _read_camera_kitti_frame_records(
    experiment_dir: Path,
    run_config: dict,
) -> list[tuple[int, Path, list[dict]]]:
    rgb_dir = experiment_dir / "Camera" / "rgb"
    ann_dir = experiment_dir / "Camera" / "object_detection"
    if not rgb_dir.is_dir():
        return []

    num_frames = int(run_config.get("run", {}).get("num_frames", 0))
    frame_ids = list(range(num_frames))
    if not frame_ids:
        frame_ids = sorted(
            int(path.stem)
            for path in rgb_dir.iterdir()
            if path.is_file() and path.suffix == ".png" and path.stem.isdigit()
        )

    frame_records = []
    for frame_id in frame_ids:
        img_path = rgb_dir / f"{frame_id}.png"
        if not img_path.is_file():
            continue
        ann_path = ann_dir / f"{frame_id}.txt"
        frame_records.append(
            (frame_id, img_path, _parse_kitti_annotation_file(str(ann_path)))
        )
    return frame_records


def _read_basic_writer_frame_records(
    experiment_dir: Path,
    run_config: dict,
) -> list[tuple[int, Path, list[dict]]]:
    flat_rgb_paths = sorted(
        path for path in experiment_dir.iterdir()
        if path.is_file() and _OPTUNA_FLAT_RGB_RE.match(path.name)
    )

    frame_records = []
    for img_path in flat_rgb_paths:
        frame_match = _OPTUNA_FLAT_RGB_RE.match(img_path.name)
        if frame_match is None:
            continue
        frame_id = int(frame_match.group("frame"))
        ann_path = experiment_dir / f"bounding_box_2d_tight_{frame_id:04d}.npy"
        label_path = experiment_dir / f"bounding_box_2d_tight_labels_{frame_id:04d}.json"
        frame_records.append(
            (frame_id, img_path, _parse_basic_writer_bbox_annotations(ann_path, label_path))
        )
    return frame_records


def _read_supported_frame_records(
    experiment_dir: Path,
    run_config: dict,
) -> list[tuple[int, Path, list[dict]]]:
    frame_records = _read_camera_kitti_frame_records(experiment_dir, run_config)
    if frame_records:
        return frame_records
    return _read_basic_writer_frame_records(experiment_dir, run_config)


def _raise_empty_selected_data_error(config_key: str, selected_id: int, data_path: Path) -> None:
    raise ValueError(
        f"{config_key} includes {selected_id}, but no supported image frames were found "
        f"under {data_path}. Expected either Camera/rgb/<frame>.png with "
        f"Camera/object_detection/<frame>.txt annotations, or flat rgb_####.png "
        f"with bounding_box_2d_tight_####.npy/json annotations."
    )


def _load_optuna_summary(summary_path: Path) -> dict:
    if not summary_path.is_file():
        return {}
    with summary_path.open("r") as f:
        summary = json.load(f)
    return summary if isinstance(summary, dict) else {}


def _append_optuna_experiment_records(
    *,
    records: list,
    experiment_dir: Path,
    experiment_name: str,
    optuna_bucket: str,
    optuna_theme: str,
    optuna_repetition: str,
    trial_number: int | None,
    summary: dict | None,
    trial_dir: Path | None,
) -> None:
    match = _OPTUNA_DIR_RE.match(experiment_name)
    if match is None:
        return

    run_config_path = experiment_dir / "run_config.yaml"
    if not run_config_path.is_file() and trial_dir is not None:
        fallback_yaml_path = trial_dir / "yamls" / f"{experiment_name}.yaml"
        if fallback_yaml_path.is_file():
            run_config_path = fallback_yaml_path
    if not run_config_path.is_file():
        return

    with run_config_path.open("r") as f:
        exp_config = yaml.safe_load(f)
    run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

    orig_w = int(run_config.get("render", {}).get("width", 960))
    orig_h = int(run_config.get("render", {}).get("height", 544))
    iteration = int(match.group("iteration"))
    run_number = int(match.group("run"))
    summary = summary or {}

    rank_value = summary.get("rank")
    objective_value = summary.get("objective_value")
    frame_records = _read_supported_frame_records(experiment_dir, run_config)

    for image_id, img_path, anns in frame_records:
        records.append({
            "image_id": image_id,
            "path": str(img_path),
            "width": orig_w,
            "height": orig_h,
            "subset": "Tensorleap-Optimized",
            "anns": anns,
            "run_config": run_config,
            "run_number": run_number,
            "iteration": iteration,
            "experiment": experiment_name,
            "optuna_bucket": optuna_bucket,
            "optuna_theme": optuna_theme,
            "optuna_repetition": optuna_repetition,
            "trial_number": trial_number,
            "optuna_rank": int(rank_value) if rank_value is not None else None,
            "optuna_objective_value": float(objective_value) if objective_value is not None else None,
        })


def _append_optuna_trial_records(
    *,
    records: list,
    trial_dir: Path,
    optuna_bucket: str,
    optuna_theme: str,
    optuna_repetition: str,
) -> None:
    trial_match = _OPTUNA_TRIAL_DIR_RE.match(trial_dir.name)
    if trial_match is None:
        return

    trial_number = int(trial_match.group("trial"))
    outputs_root = trial_dir / "outputs"
    if not outputs_root.is_dir():
        return

    for experiment_dir in sorted(
        path for path in outputs_root.iterdir()
        if path.is_dir() and _OPTUNA_DIR_RE.match(path.name)
    ):
        _append_optuna_experiment_records(
            records=records,
            experiment_dir=experiment_dir,
            experiment_name=experiment_dir.name,
            optuna_bucket=optuna_bucket,
            optuna_theme=optuna_theme,
            optuna_repetition=optuna_repetition,
            trial_number=trial_number,
            summary=None,
            trial_dir=trial_dir,
        )


def _append_optuna_run_wrapper_records(
    *,
    records: list,
    run_dir: Path,
    optuna_bucket: str,
    optuna_theme: str,
    optuna_repetition: str,
) -> None:
    if _OPTUNA_DIR_RE.match(run_dir.name) is None:
        return

    experiment_dir = run_dir / "outputs" / run_dir.name
    if not experiment_dir.is_dir():
        experiment_dir = run_dir

    _append_optuna_experiment_records(
        records=records,
        experiment_dir=experiment_dir,
        experiment_name=run_dir.name,
        optuna_bucket=optuna_bucket,
        optuna_theme=optuna_theme,
        optuna_repetition=optuna_repetition,
        trial_number=None,
        summary=None,
        trial_dir=run_dir,
    )


def _append_optuna_selected_trial_records(
    *,
    records: list,
    selected_trial_dir: Path,
    category: str,
    cycle_index: int,
    timestamp: str,
) -> None:
    selected_match = _OPTUNA_SELECTED_TRIAL_RE.match(selected_trial_dir.name)
    if selected_match is None:
        return

    outputs_root = selected_trial_dir / "outputs"
    if not outputs_root.is_dir():
        return

    trial_kind = selected_match.group("kind")
    trial_label = selected_match.group("label")

    for experiment_dir in sorted(
        path for path in outputs_root.iterdir()
        if path.is_dir() and _OPTUNA_DIR_RE.match(path.name)
    ):
        experiment_name = experiment_dir.name
        match = _OPTUNA_DIR_RE.match(experiment_name)
        if match is None:
            continue

        run_config_path = selected_trial_dir / "yamls" / f"{experiment_name}.yaml"
        if not run_config_path.is_file():
            run_config_path = experiment_dir / "run_config.yaml"
        if not run_config_path.is_file():
            continue

        with run_config_path.open("r") as f:
            exp_config = yaml.safe_load(f)
        run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

        orig_w = int(run_config.get("render", {}).get("width", 960))
        orig_h = int(run_config.get("render", {}).get("height", 544))
        iteration = int(match.group("iteration"))
        run_number = int(match.group("run"))
        frame_records = _read_supported_frame_records(experiment_dir, run_config)

        for image_id, img_path, anns in frame_records:
            records.append({
                "image_id": image_id,
                "path": str(img_path),
                "width": orig_w,
                "height": orig_h,
                "subset": "Tensorleap-Optimized-01",
                "anns": anns,
                "run_config": run_config,
                "run_number": run_number,
                "iteration": iteration,
                "experiment": experiment_name,
                "optuna_bucket": "selected",
                "optuna_theme": category,
                "optuna_repetition": f"cycle_{cycle_index:02d}_{timestamp}",
                "trial_number": None,
                "optuna_rank": None,
                "optuna_objective_value": None,
                "optuna_selected_kind": trial_kind,
                "optuna_selected_label": trial_label,
                "optuna_selected_cycle": cycle_index,
                "optuna_selected_timestamp": timestamp,
            })


def _load_optuna_selected_records() -> list:
    """
    Load frames from the warehouse tree shaped like:
      optuna-ec2/<category>/cycle_XX_<timestamp>/<best|worst>_*/
    """
    optuna_cfg = CONFIG.get("Tensorleap-Optimized", {})
    if not optuna_cfg.get("additional", True):
        return []

    base = optuna_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    records = []
    base_path = Path(base)

    for category_dir in sorted(path for path in base_path.iterdir() if path.is_dir()):
        for cycle_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            cycle_match = _OPTUNA_CYCLE_RE.match(cycle_dir.name)
            if cycle_match is None:
                continue
            cycle_index = int(cycle_match.group("cycle"))
            timestamp = cycle_match.group("timestamp")
            for selected_trial_dir in sorted(path for path in cycle_dir.iterdir() if path.is_dir()):
                _append_optuna_selected_trial_records(
                    records=records,
                    selected_trial_dir=selected_trial_dir,
                    category=category_dir.name,
                    cycle_index=cycle_index,
                    timestamp=timestamp,
                )

    num_samples = optuna_cfg.get("num_samples")
    if num_samples is not None:
        by_cycle = {}
        for record in records:
            cycle_key = (
                record.get("optuna_theme"),
                record.get("optuna_selected_cycle"),
                record.get("optuna_selected_label"),
            )
            by_cycle.setdefault(cycle_key, []).append(record)
        sampled = []
        rng = random.Random(42)
        for cycle_records in by_cycle.values():
            if len(cycle_records) > num_samples:
                rng.shuffle(cycle_records)
                sampled.extend(cycle_records[:num_samples])
            else:
                sampled.extend(cycle_records)
        records = sampled

    return records


def _load_optuna_records() -> list:
    """
    Load frames from optuna trees, including:
      - flat top-level iterXXX_runYYY/Camera directories
      - themed trial folders like camera/trial_147/outputs/iter016_run007
      - themed repetition folders like camera-color/repetition_0/iter000_run000/outputs/iter000_run000
      - themed repetition trial folders like camera-color/repetition_0/trial_28/outputs/iter001_run000
      - worst folders like worst/camera/iter004_run007/outputs/iter004_run007
    """
    optuna_cfg = CONFIG.get("Tensorleap-Optimized", {})
    if not optuna_cfg.get("additional", True):
        return []

    base = optuna_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    records = []
    base_path = Path(base)
    selected_records = _load_optuna_selected_records()
    if selected_records:
        records.extend(selected_records)

    flat_experiment_dirs = sorted(
        path for path in base_path.iterdir()
        if path.is_dir() and _OPTUNA_DIR_RE.match(path.name)
    )
    for experiment_dir in flat_experiment_dirs:
        _append_optuna_experiment_records(
            records=records,
            experiment_dir=experiment_dir,
            experiment_name=experiment_dir.name,
            optuna_bucket="flat",
            optuna_theme="flat",
            optuna_repetition="",
            trial_number=None,
            summary=None,
            trial_dir=None,
        )

    regular_theme_dirs = sorted(
        path for path in base_path.iterdir()
        if path.is_dir() and path.name not in {"worst"} and not _OPTUNA_DIR_RE.match(path.name)
    )
    for theme_dir in regular_theme_dirs:
        for trial_dir in sorted(path for path in theme_dir.iterdir() if path.is_dir()):
            if _OPTUNA_TRIAL_DIR_RE.match(trial_dir.name) is None:
                continue
            _append_optuna_trial_records(
                records=records,
                trial_dir=trial_dir,
                optuna_bucket="regular",
                optuna_theme=theme_dir.name,
                optuna_repetition="",
            )

        repetition_dirs = sorted(
            path for path in theme_dir.iterdir()
            if path.is_dir()
            and path.name not in {"cache", "outputs", "yamls"}
            and _OPTUNA_DIR_RE.match(path.name) is None
            and _OPTUNA_TRIAL_DIR_RE.match(path.name) is None
        )
        for repetition_dir in repetition_dirs:
            for run_dir in sorted(path for path in repetition_dir.iterdir() if path.is_dir()):
                if _OPTUNA_DIR_RE.match(run_dir.name) is None:
                    continue
                _append_optuna_run_wrapper_records(
                    records=records,
                    run_dir=run_dir,
                    optuna_bucket="regular",
                    optuna_theme=theme_dir.name,
                    optuna_repetition=repetition_dir.name,
                )

            for trial_dir in sorted(path for path in repetition_dir.iterdir() if path.is_dir()):
                if _OPTUNA_TRIAL_DIR_RE.match(trial_dir.name) is None:
                    continue
                _append_optuna_trial_records(
                    records=records,
                    trial_dir=trial_dir,
                    optuna_bucket="regular",
                    optuna_theme=theme_dir.name,
                    optuna_repetition=repetition_dir.name,
                )

    worst_root = base_path / "worst"
    if worst_root.is_dir():
        for theme_dir in sorted(path for path in worst_root.iterdir() if path.is_dir()):
            for run_dir in sorted(
                path for path in theme_dir.iterdir()
                if path.is_dir() and _OPTUNA_DIR_RE.match(path.name)
            ):
                experiment_dir = run_dir / "outputs" / run_dir.name
                if not experiment_dir.is_dir():
                    continue
                _append_optuna_experiment_records(
                    records=records,
                    experiment_dir=experiment_dir,
                    experiment_name=run_dir.name,
                    optuna_bucket="worst",
                    optuna_theme=theme_dir.name,
                    optuna_repetition="",
                    trial_number=None,
                    summary=_load_optuna_summary(run_dir / "summary.json"),
                    trial_dir=None,
                )

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


def _load_optuna_test_records() -> list:
    """
    Load frames from optuna_tests trees shaped like:
      - test_{name}/{run_name}/rgb_XXXX.png
      - test_{name}/{run_name}/.../rgb_XXXX.png

    `run_config.yaml` is resolved from the frame directory first and then the
    run root as a fallback.
    """
    optuna_tests_cfg = CONFIG.get("optuna_tests_data", {})
    if not optuna_tests_cfg.get("additional", True):
        return []

    base = optuna_tests_cfg.get("base_path", "")
    if not base or not os.path.isdir(base):
        return []

    base_path = Path(base)
    records = []
    allowed_tests = optuna_tests_cfg.get("tests")
    if isinstance(allowed_tests, str):
        allowed_tests = [allowed_tests]
    allowed_tests = set(allowed_tests) if allowed_tests is not None else None

    for test_dir in sorted(path for path in base_path.iterdir() if path.is_dir()):
        test_name = test_dir.name
        test_suffix = test_name[len("test_"):] if test_name.startswith("test_") else test_name
        if allowed_tests is not None and test_suffix not in allowed_tests:
            continue
        for run_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
            _append_optuna_test_run_records(
                records=records,
                test_name=test_name,
                run_name=run_dir.name,
                run_dir=run_dir,
            )

    num_samples = optuna_tests_cfg.get("num_samples")
    if num_samples is not None:
        by_run = {}
        for record in records:
            run_key = (record.get("optuna_test_name"), record.get("run_name"))
            by_run.setdefault(run_key, []).append(record)
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


def _append_optuna_test_run_records(
    *,
    records: list,
    test_name: str,
    run_name: str,
    run_dir: Path,
) -> None:
    frame_dirs = _discover_optuna_test_frame_dirs(run_dir)
    for frame_dir in frame_dirs:
        run_config_path = frame_dir / "run_config.yaml"
        if not run_config_path.is_file():
            run_config_path = run_dir / "run_config.yaml"
        if not run_config_path.is_file():
            continue

        with run_config_path.open("r") as f:
            exp_config = yaml.safe_load(f)
        run_config = _deep_merge(_SDG_BASE_CONFIG, exp_config)

        orig_w = int(run_config.get("render", {}).get("width", 960))
        orig_h = int(run_config.get("render", {}).get("height", 544))
        relative_experiment = frame_dir.relative_to(run_dir)
        experiment_name = run_name if str(relative_experiment) == "." else f"{run_name}__{relative_experiment.as_posix().replace('/', '__')}"
        frame_records = _read_supported_frame_records(frame_dir, run_config)

        for image_id, img_path, anns in frame_records:
            records.append({
                "image_id": image_id,
                "path": str(img_path),
                "width": orig_w,
                "height": orig_h,
                "subset": "optuna_tests",
                "anns": anns,
                "run_config": run_config,
                "run_number": -1,
                "run_name": run_name,
                "experiment": experiment_name,
                "optuna_test_name": test_name,
            })


def _discover_optuna_test_frame_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "Camera" / "rgb").is_dir():
        return [run_dir]

    direct_rgb_paths = sorted(
        path for path in run_dir.iterdir()
        if path.is_file() and _OPTUNA_FLAT_RGB_RE.match(path.name)
    )
    if direct_rgb_paths:
        return [run_dir]

    camera_frame_dirs = {
        path.parent.parent
        for path in run_dir.rglob("Camera/rgb")
        if path.is_dir()
    }
    basic_writer_frame_dirs = {
        path.parent
        for path in run_dir.rglob("rgb_*.png")
        if path.is_file() and _OPTUNA_FLAT_RGB_RE.match(path.name)
    }
    return sorted(camera_frame_dirs | basic_writer_frame_dirs)


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

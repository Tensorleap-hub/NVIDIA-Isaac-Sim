import cv2
import numpy as np

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from rtdetr_warehouse.config import CONFIG, CLASS_NAMES, COCO_ID_TO_IDX


def _safe_stat(values: np.ndarray, reducer) -> float:
    return float(np.nan) if len(values) == 0 else float(reducer(values))


@tensorleap_metadata("data_type", DatasetMetadataType.string)
def data_type_metadata(idx: int, preprocessing: PreprocessResponse) -> str:
    record = preprocessing.data[idx]
    if not isinstance(record, dict):
        return "real"
    if "run_config" not in record:
        return "real"
    return str(record.get("subset", "synth"))


@tensorleap_metadata("metadata")
def sample_metadata(idx: int, preprocessing: PreprocessResponse) -> dict:
    record = preprocessing.data[idx]
    image = cv2.imread(record["path"])
    image = cv2.resize(image, (CONFIG["image_size"], CONFIG["image_size"]))

    valid_anns = [a for a in record["anns"] if a["category_id"] in COCO_ID_TO_IDX]
    gt_classes = np.array([COCO_ID_TO_IDX[a["category_id"]] for a in valid_anns], dtype=np.float32)

    if len(valid_anns) > 0:
        orig_w, orig_h = record["width"], record["height"]
        x_scale = CONFIG["image_size"] / orig_w
        y_scale = CONFIG["image_size"] / orig_h
        raw_boxes = np.array([a["bbox"] for a in valid_anns], dtype=np.float32)
        # cx, cy (normalized) from pixel xywh
        bbox_cx = (raw_boxes[:, 0] + raw_boxes[:, 2] / 2) * x_scale / CONFIG["image_size"]
        bbox_cy = (raw_boxes[:, 1] + raw_boxes[:, 3] / 2) * y_scale / CONFIG["image_size"]
        bbox_areas = (raw_boxes[:, 2] * x_scale / CONFIG["image_size"]) * \
                     (raw_boxes[:, 3] * y_scale / CONFIG["image_size"])
    else:
        bbox_cx = bbox_cy = bbox_areas = np.array([])

    unique_classes, class_counts = np.unique(gt_classes, return_counts=True)
    class_count_map = {int(cls): int(cnt) for cls, cnt in zip(unique_classes, class_counts)}
    per_label_counts = {
        f"# of {name}": float(class_count_map.get(i, np.nan))
        for i, name in enumerate(CLASS_NAMES)
    }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "image_sharpness": sharpness,
        "subset": record["subset"],
        "optuna_bucket": str(record.get("optuna_bucket", "")),
        "optuna_theme": str(record.get("optuna_theme", "")),
        "optuna_trial_number": float(record["trial_number"]) if record.get("trial_number") is not None else float(np.nan),
        "optuna_rank": float(record["optuna_rank"]) if record.get("optuna_rank") is not None else float(np.nan),
        "optuna_objective_value": float(record["optuna_objective_value"]) if record.get("optuna_objective_value") is not None else float(np.nan),
        "# of objects": len(valid_anns),
        "# of unique classes": int(len(unique_classes)),
        "bbox area mean":   _safe_stat(bbox_areas, np.mean),
        "bbox area median": _safe_stat(bbox_areas, np.median),
        "bbox area min":    _safe_stat(bbox_areas, np.min),
        "bbox area max":    _safe_stat(bbox_areas, np.max),
        "bbox area var":    _safe_stat(bbox_areas, np.var),
        "bbox cx mean":     _safe_stat(bbox_cx, np.mean),
        "bbox cx median":   _safe_stat(bbox_cx, np.median),
        "bbox cy mean":     _safe_stat(bbox_cy, np.mean),
        "bbox cy median":   _safe_stat(bbox_cy, np.median),
        "bbox center var":  _safe_stat(bbox_cx, np.var) + _safe_stat(bbox_cy, np.var),
        **per_label_counts,
    }

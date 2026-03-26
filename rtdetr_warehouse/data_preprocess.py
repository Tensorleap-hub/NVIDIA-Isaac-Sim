import json
import os
from typing import List

import cv2
import numpy as np

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

    return [
        PreprocessResponse(data=train_records, length=len(train_records), state=DataStateType.training),
        PreprocessResponse(data=val_records, length=len(val_records), state=DataStateType.validation),
    ]


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

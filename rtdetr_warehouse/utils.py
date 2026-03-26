import os
import numpy as np
import cv2

from rtdetr_warehouse.config import CONFIG, COCO_ID_TO_IDX, INPUT_SIZE, MAX_DETS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load image from disk, resize to INPUT_SIZE x INPUT_SIZE,
    apply ImageNet normalization and convert to CHW float32.

    Returns:
        np.ndarray: shape (3, INPUT_SIZE, INPUT_SIZE), float32
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)  # HWC -> CHW


def preprocess_image_for_viz(image_path: str) -> np.ndarray:
    """
    Load and resize image for visualization (HWC float32, un-normalized).

    Returns:
        np.ndarray: shape (INPUT_SIZE, INPUT_SIZE, 3), float32 in [0, 1]
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    return img.astype(np.float32) / 255.0


def coco_boxes_to_model_norm(boxes_xywh, orig_w: int, orig_h: int) -> np.ndarray:
    """
    Convert COCO pixel boxes [x_min, y_min, w, h] to normalized [cx, cy, w, h]
    in model input space (INPUT_SIZE x INPUT_SIZE).

    Args:
        boxes_xywh: list of [x_min, y_min, w, h] in original pixel coords
        orig_w: original image width
        orig_h: original image height

    Returns:
        np.ndarray: shape (N, 4) float32, normalized [cx, cy, w, h] in [0, 1]
    """
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.array(boxes_xywh, dtype=np.float32)
    # Scale to model input size
    x_scale = INPUT_SIZE / orig_w
    y_scale = INPUT_SIZE / orig_h

    x_min = boxes[:, 0] * x_scale
    y_min = boxes[:, 1] * y_scale
    w = boxes[:, 2] * x_scale
    h = boxes[:, 3] * y_scale

    cx = (x_min + w / 2) / INPUT_SIZE
    cy = (y_min + h / 2) / INPUT_SIZE
    w_n = w / INPUT_SIZE
    h_n = h / INPUT_SIZE

    return np.stack([cx, cy, w_n, h_n], axis=1)


def build_gt_tensor(anns, orig_w: int, orig_h: int) -> np.ndarray:
    """
    Build padded GT tensor of shape (MAX_DETS, 5).
    Columns: [class_idx, cx, cy, w, h] normalized in model input space.
    Padding rows filled with -1.

    Args:
        anns: list of annotation dicts with 'category_id' and 'bbox' keys
        orig_w: original image width
        orig_h: original image height

    Returns:
        np.ndarray: shape (MAX_DETS, 5), float32
    """
    gt = np.full((MAX_DETS, 5), -1.0, dtype=np.float32)
    valid = [a for a in anns if a['category_id'] in COCO_ID_TO_IDX]
    n = min(len(valid), MAX_DETS)
    if n == 0:
        return gt

    boxes = [a['bbox'] for a in valid[:n]]
    class_idxs = [COCO_ID_TO_IDX[a['category_id']] for a in valid[:n]]

    norm_boxes = coco_boxes_to_model_norm(boxes, orig_w, orig_h)
    gt[:n, 0] = np.array(class_idxs, dtype=np.float32)
    gt[:n, 1:] = norm_boxes
    return gt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] normalized to [x1, y1, x2, y2] normalized."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes in [cx, cy, w, h] normalized format.

    Returns:
        np.ndarray: IoU matrix of shape (N, M)
    """
    a = cxcywh_to_xyxy(boxes_a)  # (N, 4)
    b = cxcywh_to_xyxy(boxes_b)  # (M, 4)

    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)

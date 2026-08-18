"""YOLO11 detection metrics for Tensorleap.

Metrics are class-agnostic (IoU-based) because YOLO11 predicts COCO classes
while GT labels are LOCO-specific (5 categories).
"""
import numpy as np
import torch

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric

from tensorleap_intgration_code.common import label_names, xywh2xyxy
from tensorleap_intgration_code.config import CONFIG
from tensorleap_intgration_code.yolo_common import decode_yolo_output


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """(N,4) xyxy × (M,4) xyxy → (N,M) IoU."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt    = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(1e-6)


def _prf(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, iou_threshold: float = 0.1):
    iou_mat = _box_iou(gt_boxes, pred_boxes)
    matched_gt, matched_pred, tp = set(), set(), 0
    for pred_idx in range(iou_mat.shape[1]):
        gt_idx = int(iou_mat[:, pred_idx].argmax())
        max_iou = float(iou_mat[gt_idx, pred_idx])
        if max_iou >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            tp += 1
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, fp, tp, fn


@tensorleap_custom_metric(
    name="yolo_per_sample_metrics",
    direction={
        "precision": MetricDirection.Upward,
        "recall":    MetricDirection.Upward,
        "f1":        MetricDirection.Upward,
        "FP":        MetricDirection.Downward,
        "TP":        MetricDirection.Upward,
        "FN":        MetricDirection.Downward,
        "iou":       MetricDirection.Upward,
    },
)
def yolo_per_sample_metrics(output0: np.ndarray, classes: np.ndarray):
    image_size = float(CONFIG["image_size"])
    labels_arr, boxes_xyxy_pix, scores_arr = decode_yolo_output(output0)

    gt = np.asarray(classes)
    if gt.ndim == 3:
        gt = gt[0]
    mask = ~(gt == -1).any(axis=1)
    gt = gt[mask]

    metrics = {
        "precision": np.array([], dtype=np.float32),
        "recall":    np.array([], dtype=np.float32),
        "f1":        np.array([], dtype=np.float32),
        "iou":       np.array([], dtype=np.float32),
        "FP":        np.array([], dtype=np.int32),
        "TP":        np.array([], dtype=np.int32),
        "FN":        np.array([], dtype=np.int32),
    }

    def _append(p, r, f1, fp, tp, fn, iou):
        metrics["precision"] = np.append(metrics["precision"], np.float32(p))
        metrics["recall"]    = np.append(metrics["recall"],    np.float32(r))
        metrics["f1"]        = np.append(metrics["f1"],        np.float32(f1))
        metrics["FP"]        = np.append(metrics["FP"],        np.int32(fp))
        metrics["TP"]        = np.append(metrics["TP"],        np.int32(tp))
        metrics["FN"]        = np.append(metrics["FN"],        np.int32(fn))
        metrics["iou"]       = np.append(metrics["iou"],       np.float32(iou))

    n_gt   = gt.shape[0]
    n_pred = len(labels_arr)

    if n_gt == 0 and n_pred == 0:
        _append(np.nan, np.nan, 0, 0, 0, 0, 1)
        return metrics
    if n_pred == 0:
        _append(np.nan, 0, 0, 0, 0, n_gt, 0)
        return metrics
    if n_gt == 0:
        _append(0, np.nan, 0, n_pred, 0, 0, 0)
        return metrics

    # GT: normalized cxcywh → normalized xyxy
    gt_boxes_norm   = torch.from_numpy(xywh2xyxy(gt[:, 1:].astype(np.float32)))
    # Pred: pixel xyxy → normalized xyxy
    pred_boxes_norm = torch.from_numpy(boxes_xyxy_pix / image_size)

    iou_mat = _box_iou(gt_boxes_norm, pred_boxes_norm)
    p, r, f1, fp, tp, fn = _prf(gt_boxes_norm, pred_boxes_norm)
    iou_mean = float(iou_mat.max(dim=0).values.mean()) if iou_mat.numel() > 0 else 0.0
    _append(p, r, f1, fp, tp, fn, iou_mean)
    return metrics


@tensorleap_custom_metric("yolo_confusion_matrix", compute_insights=False)
def yolo_confusion_matrix(output0: np.ndarray, classes: np.ndarray):
    image_size = float(CONFIG["image_size"])
    labels_arr, boxes_xyxy_pix, scores_arr = decode_yolo_output(output0)

    gt = np.asarray(classes)
    if gt.ndim == 3:
        gt = gt[0]
    mask = ~(gt == -1).any(axis=1)
    gt   = gt[mask]

    names     = label_names()
    threshold = 0.1

    gt_labels_arr   = gt[:, 0] if gt.shape[0] > 0 else np.array([])
    gt_boxes_norm   = torch.from_numpy(xywh2xyxy(gt[:, 1:].astype(np.float32))) if gt.shape[0] > 0 else torch.zeros((0, 4))
    pred_boxes_norm = torch.from_numpy(boxes_xyxy_pix / image_size) if len(labels_arr) > 0 else torch.zeros((0, 4))

    elements = []

    if len(labels_arr) > 0 and gt.shape[0] > 0:
        ious          = _box_iou(gt_boxes_norm, pred_boxes_norm).numpy().T   # (N_pred, N_gt)
        pred_detected = np.any(ious > threshold, axis=1)
        max_iou_ind   = np.argmax(ious, axis=1)
        for i, detected in enumerate(pred_detected):
            gt_idx     = int(gt_labels_arr[max_iou_ind[i]])
            class_name = names[gt_idx] if 0 <= gt_idx < len(names) else "Unknown"
            conf       = float(scores_arr[i])
            if detected:
                elements.append(ConfusionMatrixElement(class_name, ConfusionMatrixValue.Positive, conf))
            else:
                pred_cls   = int(labels_arr[i])
                elements.append(ConfusionMatrixElement(f"coco{pred_cls}", ConfusionMatrixValue.Negative, conf))
        gts_detected = np.any(ious > threshold, axis=0)
    else:
        gts_detected = np.zeros(gt.shape[0], dtype=bool)

    for k, det in enumerate(gts_detected):
        if not det:
            class_idx  = int(gt_labels_arr[k]) if k < len(gt_labels_arr) else -1
            class_name = names[class_idx] if 0 <= class_idx < len(names) else "Unknown"
            elements.append(ConfusionMatrixElement(class_name, ConfusionMatrixValue.Positive, 0.0))

    if len(gts_detected) == 0 or bool(np.all(~gts_detected)):
        elements.append(ConfusionMatrixElement("background", ConfusionMatrixValue.Positive, 0.0))

    return [elements]

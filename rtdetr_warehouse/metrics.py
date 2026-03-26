import numpy as np
import torch

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue, MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric

from rtdetr_warehouse.common import format_rtdetr_predictions, label_names, prediction_rows, xywh2xyxy
from rtdetr_warehouse.config import CONFIG


# ---------------------------------------------------------------------------
# Box utilities (torch tensors, xyxy format)
# ---------------------------------------------------------------------------

def _box_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of xyxy boxes. Returns (N, M) matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _compute_iou_mean(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> float:
    iou_mat = _box_iou_torch(gt_boxes, pred_boxes)
    if iou_mat.numel() == 0:
        return 0.0
    max_iou = iou_mat.max(dim=0).values
    filtered = iou_mat * iou_mat.eq(iou_mat.max(dim=0, keepdim=True).values)
    return float(filtered.max(dim=1).values.mean())


def _compute_accuracy(
    gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor, pred_labels: torch.Tensor,
) -> float:
    iou_mat = _box_iou_torch(gt_boxes, pred_boxes)
    if iou_mat.numel() == 0:
        return 0.0
    max_iou = iou_mat.max(dim=0, keepdim=True).values
    filtered = iou_mat * iou_mat.eq(max_iou)
    succ = (pred_labels[filtered.max(dim=1)[1]] == gt_labels).numpy()
    return float(succ.mean())


def _compute_prf(
    gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, iou_threshold: float = 0.1
):
    iou_mat = _box_iou_torch(gt_boxes, pred_boxes)
    matched_gt, matched_pred, tp = set(), set(), 0
    for pred_idx in range(iou_mat.shape[1]):
        gt_idx = int(iou_mat[:, pred_idx].argmax().item())
        max_iou = float(iou_mat[gt_idx, pred_idx].item())
        if max_iou >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            tp += 1
    fp = pred_boxes.shape[0] - tp
    fn = gt_boxes.shape[0] - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, fp, tp, fn


def _batched_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    return targets[None, ...] if targets.ndim == 2 else targets


# ---------------------------------------------------------------------------
# Per-sample detection metrics
# ---------------------------------------------------------------------------

@tensorleap_custom_metric(
    name="per_sample_metrics",
    direction={
        "precision": MetricDirection.Upward,
        "recall": MetricDirection.Upward,
        "f1": MetricDirection.Upward,
        "FP": MetricDirection.Downward,
        "TP": MetricDirection.Upward,
        "FN": MetricDirection.Downward,
        "iou": MetricDirection.Upward,
        "accuracy": MetricDirection.Upward,
    },
)
def get_per_sample_metrics(
    labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, targets: np.ndarray
):
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)

    metrics = {k: np.array([], dtype=np.float32) for k in
               ["precision", "recall", "f1", "iou", "accuracy"]}
    metrics.update({k: np.array([], dtype=np.int32) for k in ["FP", "TP", "FN"]})

    def _append(m, p, r, f1, fp, tp, fn, iou, acc):
        m["precision"] = np.append(m["precision"], np.float32(p))
        m["recall"]    = np.append(m["recall"],    np.float32(r))
        m["f1"]        = np.append(m["f1"],        np.float32(f1))
        m["FP"]        = np.append(m["FP"],        np.int32(fp))
        m["TP"]        = np.append(m["TP"],        np.int32(tp))
        m["FN"]        = np.append(m["FN"],        np.int32(fn))
        m["iou"]       = np.append(m["iou"],       np.float32(iou))
        m["accuracy"]  = np.append(m["accuracy"],  np.float32(acc))

    image_size = float(CONFIG["image_size"])
    preds = prediction_rows(y_preds)
    for pred_t, gt in zip(preds, _batched_targets(targets)):
        mask = ~(gt == -1).any(axis=1)
        gt = torch.from_numpy(gt[mask])
        pred = pred_t.numpy()

        if gt.shape[0] == 0 and pred.shape[0] == 0:
            _append(metrics, np.nan, np.nan, 0, 0, 0, 0, 1, 1); continue
        if pred.shape[0] == 0:
            _append(metrics, np.nan, 0, 0, 0, 0, gt.shape[0], 0, 0); continue
        if gt.shape[0] == 0:
            _append(metrics, 0, np.nan, 0, pred.shape[0], 0, 0, 0, 0); continue

        # pred boxes are in xyxy pixel space (image_size units)
        pred_boxes = torch.from_numpy(pred[:, :4] / image_size)
        pred_labels = torch.from_numpy(pred[:, 5])

        # gt boxes are in cxcywh normalized → convert to xyxy
        gt_boxes = torch.from_numpy(xywh2xyxy(gt[:, 1:].numpy()))
        gt_labels = gt[:, 0]

        p, r, f1, fp, tp, fn = _compute_prf(gt_boxes, pred_boxes, iou_threshold=0.1)
        iou = _compute_iou_mean(gt_boxes, pred_boxes)
        acc = _compute_accuracy(gt_boxes, gt_labels, pred_boxes, pred_labels)
        _append(metrics, p, r, f1, fp, tp, fn, iou, acc)

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

@tensorleap_custom_metric("Confusion Matrix")
def confusion_matrix_metric(
    labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, targets: np.ndarray
):
    y_preds = format_rtdetr_predictions(labels, boxes_xyxy, scores)
    threshold = 0.1
    confusion_matrices = []
    names = label_names()
    image_size = float(CONFIG["image_size"])
    preds = prediction_rows(y_preds)

    for pred_t, gt in zip(preds, _batched_targets(targets)):
        confusion_matrix_elements = []
        mask = ~(gt == -1).any(axis=1)
        gt = torch.from_numpy(gt[mask])
        gt_bbox = torch.from_numpy(xywh2xyxy(gt[:, 1:].numpy())) if gt.shape[0] > 0 else torch.zeros((0, 4))
        gt_labels = gt[:, 0]

        pred = pred_t.numpy()
        pred_boxes = torch.from_numpy(pred[:, :4] / image_size) if pred.shape[0] > 0 else torch.zeros((0, 4))

        if pred.shape[0] != 0 and gt_bbox.shape[0] != 0:
            ious = _box_iou_torch(gt_bbox, pred_boxes).numpy().T  # (num_pred, num_gt)
            prediction_detected = np.any(ious > threshold, axis=1)
            max_iou_ind = np.argmax(ious, axis=1)
            for i, detected in enumerate(prediction_detected):
                gt_idx = int(gt_labels[max_iou_ind[i]])
                class_name = names[gt_idx] if 0 <= gt_idx < len(names) else "Unknown"
                confidence = float(pred[i, 4])
                if detected:
                    confusion_matrix_elements.append(
                        ConfusionMatrixElement(class_name, ConfusionMatrixValue.Positive, confidence)
                    )
                else:
                    pred_idx = int(pred[i, 5])
                    pred_class = names[pred_idx] if 0 <= pred_idx < len(names) else "Unknown"
                    confusion_matrix_elements.append(
                        ConfusionMatrixElement(pred_class, ConfusionMatrixValue.Negative, confidence)
                    )
            ious_for_gt = ious
        else:
            ious_for_gt = np.zeros((max(pred.shape[0], 1), max(gt_labels.shape[0], 1)))

        gts_detected = np.any(ious_for_gt > threshold, axis=0)
        for k, gt_det in enumerate(gts_detected):
            if not gt_det:
                class_idx = int(gt_labels[k]) if k < gt_labels.shape[0] else -1
                class_name = names[class_idx] if 0 <= class_idx < len(names) else "Unknown"
                confusion_matrix_elements.append(
                    ConfusionMatrixElement(class_name, ConfusionMatrixValue.Positive, 0.0)
                )
        if all(~gts_detected):
            confusion_matrix_elements.append(
                ConfusionMatrixElement("background", ConfusionMatrixValue.Positive, 0.0)
            )
        confusion_matrices.append(confusion_matrix_elements)

    return confusion_matrices

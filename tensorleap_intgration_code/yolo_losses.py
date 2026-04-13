"""
YOLO11 loss functions for Tensorleap.

output0 shape: (1, 84, 8400)
  [:4, :]  = cx, cy, w, h  in pixel coordinates (already DFL-decoded)
  [4:, :]  = 80 COCO class scores (already sigmoid-activated)

GT boxes are normalized cxcywh; GT labels are LOCO indices (0-4).
Box loss : mean (1 - GIoU) between GT and best-matched predictions.
Cls loss : BCE on matched predictions' max-class confidence (target = 1.0).
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_metric,
)

from tensorleap_intgration_code.config import CONFIG


def _giou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between matched pairs (N, 4) xyxy → (N,)."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1 + a2 - inter
    iou = inter / union.clamp(1e-6)
    cx1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    c_area = (cx2 - cx1).clamp(0) * (cy2 - cy1).clamp(0)
    return iou - (c_area - union) / c_area.clamp(1e-6)


def compute_yolo_losses(
    output0: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute YOLO box + cls losses against GT.

    output0      : (1, 84, 8400)
    gt_boxes     : (MAX_DETS, 4) or (1, MAX_DETS, 4) — normalized cxcywh
    gt_labels    : (MAX_DETS,) or (1, MAX_DETS)      — LOCO class indices
    gt_valid_mask: (MAX_DETS,) or (1, MAX_DETS)      — 1=valid, 0=pad
    """
    image_size = float(CONFIG["image_size"])

    pred = output0[0]                    # (84, 8400)
    all_boxes_xywh = pred[:4].T          # (8400, 4) pixel cx,cy,w,h
    all_cls_scores = pred[4:].T          # (8400, 80) sigmoid

    # Predicted boxes: pixel xywh → pixel xyxy
    all_boxes_xyxy = np.empty_like(all_boxes_xywh)
    all_boxes_xyxy[:, 0] = all_boxes_xywh[:, 0] - all_boxes_xywh[:, 2] / 2
    all_boxes_xyxy[:, 1] = all_boxes_xywh[:, 1] - all_boxes_xywh[:, 3] / 2
    all_boxes_xyxy[:, 2] = all_boxes_xywh[:, 0] + all_boxes_xywh[:, 2] / 2
    all_boxes_xyxy[:, 3] = all_boxes_xywh[:, 1] + all_boxes_xywh[:, 3] / 2

    # Valid GT boxes
    boxes_raw = gt_boxes[0] if gt_boxes.ndim == 3 else gt_boxes
    valid_raw = gt_valid_mask[0] if gt_valid_mask.ndim == 2 else gt_valid_mask
    keep = valid_raw > 0.5
    gt_norm = boxes_raw[keep].astype(np.float32)

    if len(gt_norm) == 0:
        return {"loss_box": 0.0, "loss_cls": 0.0, "total": 0.0}

    # GT: normalized cxcywh → pixel xyxy
    gt_xyxy = np.empty_like(gt_norm)
    gt_xyxy[:, 0] = (gt_norm[:, 0] - gt_norm[:, 2] / 2) * image_size
    gt_xyxy[:, 1] = (gt_norm[:, 1] - gt_norm[:, 3] / 2) * image_size
    gt_xyxy[:, 2] = (gt_norm[:, 0] + gt_norm[:, 2] / 2) * image_size
    gt_xyxy[:, 3] = (gt_norm[:, 1] + gt_norm[:, 3] / 2) * image_size

    gt_t   = torch.from_numpy(gt_xyxy)                             # (M, 4)
    pred_t = torch.from_numpy(all_boxes_xyxy.astype(np.float32))  # (8400, 4)
    cls_t  = torch.from_numpy(all_cls_scores.astype(np.float32))  # (8400, 80)

    # IoU matrix (M, 8400) for GT→pred matching
    area_gt   = (gt_t[:, 2] - gt_t[:, 0]) * (gt_t[:, 3] - gt_t[:, 1])
    area_pred = (pred_t[:, 2] - pred_t[:, 0]) * (pred_t[:, 3] - pred_t[:, 1])
    lt    = torch.max(gt_t[:, None, :2], pred_t[None, :, :2])
    rb    = torch.min(gt_t[:, None, 2:], pred_t[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = area_gt[:, None] + area_pred[None, :] - inter
    iou_mat = inter / union.clamp(1e-6)        # (M, 8400)

    best_idx = iou_mat.argmax(dim=1)           # (M,) best pred for each GT

    # Box loss: mean (1 - GIoU) on best-matched pairs
    matched_pred = pred_t[best_idx]            # (M, 4)
    giou         = _giou_pairwise(gt_t, matched_pred)
    loss_box     = float((1.0 - giou).mean().item())

    # Cls loss: BCE on max-class confidence of matched preds (target = 1.0)
    matched_cls = cls_t[best_idx]              # (M, 80)
    max_conf    = matched_cls.max(dim=1).values
    loss_cls    = float(F.binary_cross_entropy(
        max_conf, torch.ones_like(max_conf), reduction="mean"
    ).item())

    total = loss_box + loss_cls
    return {"loss_box": loss_box, "loss_cls": loss_cls, "total": total}


@tensorleap_custom_loss("yolo_total_loss")
def yolo_total_loss(
    output0: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> np.ndarray:
    losses = compute_yolo_losses(output0, gt_boxes, gt_labels, gt_valid_mask)
    return np.array([losses["total"]], dtype=np.float32)


@tensorleap_custom_metric(
    "yolo_loss_components",
    direction={
        "loss_box": MetricDirection.Downward,
        "loss_cls": MetricDirection.Downward,
        "total":    MetricDirection.Downward,
    },
)
def yolo_loss_components(
    output0: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    losses = compute_yolo_losses(output0, gt_boxes, gt_labels, gt_valid_mask)
    return {k: np.array([v], dtype=np.float32) for k, v in losses.items()}

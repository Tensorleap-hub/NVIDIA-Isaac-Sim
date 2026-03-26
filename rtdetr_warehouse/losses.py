from typing import Dict, List

import numpy as np
import torch

from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_metric,
)
from rtdetr_native.criterion import RTDETRCriterionv2
from rtdetr_native.matcher import HungarianMatcher

from rtdetr_warehouse.config import CONFIG


def _loss_cfg() -> Dict:
    loss_cfg = CONFIG.get("loss", {})
    matcher_cfg = loss_cfg.get("matcher", {})
    weight_cfg = loss_cfg.get("weight_dict", {})
    return {
        "alpha": float(loss_cfg.get("alpha", 0.75)),
        "gamma": float(loss_cfg.get("gamma", 2.0)),
        "matcher": {
            "cost_class": float(matcher_cfg.get("cost_class", 2.0)),
            "cost_bbox": float(matcher_cfg.get("cost_bbox", 5.0)),
            "cost_giou": float(matcher_cfg.get("cost_giou", 2.0)),
            "alpha": float(matcher_cfg.get("alpha", 0.25)),
            "gamma": float(matcher_cfg.get("gamma", 2.0)),
        },
        "weight_dict": {
            "loss_vfl": float(weight_cfg.get("loss_vfl", 1.0)),
            "loss_bbox": float(weight_cfg.get("loss_bbox", 5.0)),
            "loss_giou": float(weight_cfg.get("loss_giou", 2.0)),
        },
    }


def _extract_targets(
    gt_boxes: np.ndarray, gt_labels: np.ndarray, gt_valid_mask: np.ndarray
) -> List[Dict[str, torch.Tensor]]:
    boxes = gt_boxes[0] if gt_boxes.ndim == 3 else gt_boxes
    labels = gt_labels[0] if gt_labels.ndim == 2 else gt_labels
    valid = gt_valid_mask[0] if gt_valid_mask.ndim == 2 else gt_valid_mask

    keep = valid > 0.5
    boxes = boxes[keep].astype(np.float32)
    labels = labels[keep].astype(np.int64)

    return [{"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels)}]


def compute_rtdetr_native_losses(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, float]:
    logits = pred_logits if pred_logits.ndim == 3 else np.expand_dims(pred_logits, 0)
    boxes = pred_boxes if pred_boxes.ndim == 3 else np.expand_dims(pred_boxes, 0)

    outputs = {
        "pred_logits": torch.from_numpy(logits.astype(np.float32)),
        "pred_boxes": torch.from_numpy(boxes.astype(np.float32)),
    }
    targets = _extract_targets(gt_boxes, gt_labels, gt_valid_mask)

    cfg = _loss_cfg()
    matcher = HungarianMatcher(
        weight_dict={
            "cost_class": cfg["matcher"]["cost_class"],
            "cost_bbox": cfg["matcher"]["cost_bbox"],
            "cost_giou": cfg["matcher"]["cost_giou"],
        },
        use_focal_loss=True,
        alpha=cfg["matcher"]["alpha"],
        gamma=cfg["matcher"]["gamma"],
    )
    criterion = RTDETRCriterionv2(
        matcher=matcher,
        weight_dict=cfg["weight_dict"],
        losses=["vfl", "boxes"],
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        num_classes=int(outputs["pred_logits"].shape[-1]),
    )
    loss_tensors = criterion(outputs, targets)
    scalar_losses = {
        k: float(v.detach().cpu().item())
        for k, v in loss_tensors.items()
        if isinstance(v, torch.Tensor)
    }
    scalar_losses["total"] = float(sum(scalar_losses.values()))
    return scalar_losses


@tensorleap_custom_loss("rtdetr_total_loss")
def rtdetr_total_loss_native(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> np.ndarray:
    losses = compute_rtdetr_native_losses(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid_mask)
    return np.array([losses["total"]], dtype=np.float32)


@tensorleap_custom_metric(
    "rtdetr_loss_components",
    direction={
        "loss_vfl": MetricDirection.Downward,
        "loss_bbox": MetricDirection.Downward,
        "loss_giou": MetricDirection.Downward,
        "total": MetricDirection.Downward,
    },
)
def rtdetr_loss_components_native(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    losses = compute_rtdetr_native_losses(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid_mask)
    return {k: np.array([v], dtype=np.float32) for k, v in losses.items()}

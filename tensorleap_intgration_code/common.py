from typing import List

import numpy as np
import torch

from tensorleap_intgration_code.config import CLASS_NAMES, CONFIG


def label_names() -> List[str]:
    return CLASS_NAMES


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert boxes from [cx,cy,w,h] normalized to [x1,y1,x2,y2] normalized."""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1,y1,x2,y2] to [cx,cy,w,h]."""
    y = np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def format_rtdetr_predictions(
    labels: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """
    Filter model outputs by score threshold and pack into (1, N, 6) array
    [x1, y1, x2, y2, score, label_idx].
    RT-DETR already applies NMS internally, so no extra NMS needed.
    """
    labels = np.asarray(labels).squeeze()
    boxes_xyxy = np.asarray(boxes_xyxy).squeeze()
    scores = np.asarray(scores).squeeze()

    if labels.ndim == 0:
        labels = np.array([labels], dtype=np.float32)
    if scores.ndim == 0:
        scores = np.array([scores], dtype=np.float32)
    if boxes_xyxy.ndim == 1:
        boxes_xyxy = boxes_xyxy.reshape(1, -1)

    score_threshold = float(CONFIG.get("score_threshold", 0.3))
    max_detections = int(CONFIG.get("max_detections", 300))

    keep = scores >= score_threshold
    labels = labels[keep]
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    if scores.size == 0:
        return np.zeros((1, 0, 6), dtype=np.float32)

    order = np.argsort(-scores)[:max_detections]
    labels = labels[order]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]

    pred = np.concatenate(
        [boxes_xyxy, scores[:, None], labels[:, None]], axis=1
    ).astype(np.float32)
    return pred[None, ...]  # (1, N, 6)


def prediction_rows(y_preds: np.ndarray) -> List[torch.Tensor]:
    """Return list of (N, 6) tensors — one per batch item."""
    y_preds = np.asarray(y_preds)
    # format_rtdetr_predictions already produced (1, N, 6)
    if y_preds.ndim == 3 and y_preds.shape[-1] == 6:
        return [torch.from_numpy(y_preds[0].astype(np.float32))]
    # Fallback: treat as raw and return as single batch item
    return [torch.from_numpy(y_preds.astype(np.float32))]


def decode_rfdetr_outputs(
    dets: np.ndarray, labels_logits: np.ndarray
) -> tuple:
    """
    Decode RF-DETR ONNX outputs into labels, boxes_xyxy (pixels), scores.

    dets:          (..., 300, 4) cxcywh normalized
    labels_logits: (..., 300, 4) raw logits — 3 foreground + 1 background
    Returns: (class_indices (N,), boxes_xyxy_pixels (N, 4), scores (N,))
    """
    image_size = float(CONFIG["image_size"])
    dets_2d = np.asarray(dets).reshape(-1, 4)         # (300, 4)
    logits_2d = np.asarray(labels_logits).reshape(-1, 4)  # (300, 4)

    # Numerically-stable softmax
    shifted = logits_2d - logits_2d.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

    # Last logit is background; first (num_classes) are foreground
    num_fg = probs.shape[-1] - 1
    fg_probs = probs[:, :num_fg]
    scores = fg_probs.max(axis=-1).astype(np.float32)
    class_indices = fg_probs.argmax(axis=-1).astype(np.float32)

    boxes_xyxy = xywh2xyxy(dets_2d) * image_size  # pixel space

    return class_indices, boxes_xyxy, scores

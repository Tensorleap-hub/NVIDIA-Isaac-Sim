"""
Shared YOLO11 post-processing utilities.

output0 shape: (1, 84, 8400)
  [:4, :]  = cx, cy, w, h  in pixel coordinates (already decoded via DFL)
  [4:, :]  = 80 COCO class scores (already sigmoid-activated)
"""
import numpy as np
import torch
import torchvision

from tensorleap_intgration_code.config import CONFIG


def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """(N, 4) cx,cy,w,h → x1,y1,x2,y2  (same coordinate space)."""
    out = np.empty_like(boxes_xywh)
    out[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    out[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    out[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    out[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return out


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """(N,4) xyxy × (M,4) xyxy → (N,M) IoU."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = np.maximum(rb - lt, 0).prod(axis=-1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def decode_yolo_output(output0: np.ndarray) -> tuple:
    """
    NMS-decode YOLO11 output0 (1, 84, 8400).

    Returns:
        labels     (N,)    float32 – COCO class index (0-79)
        boxes_xyxy (N, 4)  float32 – pixel coords [x1, y1, x2, y2]
        scores     (N,)    float32 – max class confidence
    """
    conf_thres  = float(CONFIG.get("score_threshold", 0.25))
    iou_thres   = float(CONFIG.get("nms_iou_threshold", 0.45))
    max_det     = int(CONFIG.get("max_detections", 300))

    pred         = output0[0]           # (84, 8400)
    boxes_xywh   = pred[:4].T.copy()   # (8400, 4) pixel cx,cy,w,h
    class_scores = pred[4:].T          # (8400, 80) sigmoid

    scores = class_scores.max(axis=1)
    labels = class_scores.argmax(axis=1)

    keep = scores >= conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores     = scores[keep]
    labels     = labels[keep]

    if len(scores) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return np.array([], dtype=np.float32), empty, np.array([], dtype=np.float32)

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    keep_idx = torchvision.ops.nms(
        torch.from_numpy(boxes_xyxy.astype(np.float32)),
        torch.from_numpy(scores.astype(np.float32)),
        iou_thres,
    ).numpy()[:max_det]

    return (
        labels[keep_idx].astype(np.float32),
        boxes_xyxy[keep_idx].astype(np.float32),
        scores[keep_idx].astype(np.float32),
    )

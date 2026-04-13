"""YOLO11 visualizers for Tensorleap."""
import numpy as np

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer
from code_loader.visualizers.default_visualizers import LeapImage

from tensorleap_intgration_code.common import label_names
from tensorleap_intgration_code.yolo_common import decode_yolo_output


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype == np.uint8:
        return image
    return (image * 255).astype(np.uint8)


def _pred_bboxes(boxes_xyxy: np.ndarray, labels_arr: np.ndarray, scores_arr: np.ndarray, h: int, w: int) -> list:
    bboxes = []
    for i in range(len(labels_arr)):
        cx = (boxes_xyxy[i, 0] + boxes_xyxy[i, 2]) / 2 / w
        cy = (boxes_xyxy[i, 1] + boxes_xyxy[i, 3]) / 2 / h
        bw = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) / w
        bh = (boxes_xyxy[i, 3] - boxes_xyxy[i, 1]) / h
        bboxes.append(BoundingBox(
            x=float(np.clip(cx, 0, 1)),
            y=float(np.clip(cy, 0, 1)),
            width=float(np.clip(bw, 0, 1)),
            height=float(np.clip(bh, 0, 1)),
            confidence=float(scores_arr[i]),
            label=f"cls{int(labels_arr[i])}_PRED",
        ))
    return bboxes


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    return LeapImage(_to_uint8(image), compress=False)


@tensorleap_custom_visualizer("yolo_pred_bb_decoder", LeapDataType.ImageWithBBox)
def yolo_pred_bb_decoder(image: np.ndarray, output0: np.ndarray) -> LeapImageWithBBox:
    """Overlay YOLO11 predictions (after NMS) on image."""
    image_data = _to_uint8(image)
    h, w = image_data.shape[:2]
    labels_arr, boxes_xyxy, scores_arr = decode_yolo_output(output0)
    return LeapImageWithBBox(
        data=image_data,
        bounding_boxes=_pred_bboxes(boxes_xyxy, labels_arr, scores_arr, h, w),
    )


@tensorleap_custom_visualizer("yolo_bb_decoder", LeapDataType.ImageWithBBox)
def yolo_bb_decoder(image: np.ndarray, classes: np.ndarray, output0: np.ndarray) -> LeapImageWithBBox:
    """Overlay GT boxes (LOCO) and YOLO11 predictions on image."""
    image_data = _to_uint8(image)
    h, w = image_data.shape[:2]

    # GT boxes — stored as [cls, cx, cy, bw, bh] normalized
    gt = np.asarray(classes)
    if gt.ndim == 3:
        gt = gt[0]
    mask = ~(gt == -1).any(axis=1)
    gt = gt[mask]

    names = label_names()
    bboxes = []
    for row in gt:
        label_idx = int(row[0]) if not np.isnan(row[0]) else -1
        label = (names[label_idx] + "_GT") if 0 <= label_idx < len(names) else "Unknown_GT"
        bboxes.append(BoundingBox(
            x=float(row[1]),
            y=float(row[2]),
            width=float(row[3]),
            height=float(row[4]),
            confidence=1.0,
            label=label,
        ))

    labels_arr, boxes_xyxy, scores_arr = decode_yolo_output(output0)
    bboxes += _pred_bboxes(boxes_xyxy, labels_arr, scores_arr, h, w)
    return LeapImageWithBBox(data=image_data, bounding_boxes=bboxes)

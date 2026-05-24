import numpy as np

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer
from code_loader.visualizers.default_visualizers import LeapImage

from tensorleap_intgration_code.common import (
    decode_rfdetr_outputs,
    format_rtdetr_predictions,
    label_names,
    prediction_rows,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    # CHW → HWC if needed
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.dtype == np.uint8:
        return image
    # Denormalize ImageNet normalization before converting to uint8
    image = image * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(image * 255, 0, 255).astype(np.uint8)


def _squeeze_boxes(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes)
    return boxes[0] if boxes.ndim == 3 else boxes


@tensorleap_custom_visualizer("image_visualizer", LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    return LeapImage(_image_to_uint8(image), compress=False)


@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(
    image: np.ndarray,
    classes: np.ndarray,
    dets: np.ndarray,
    labels: np.ndarray,
) -> LeapImageWithBBox:
    """Overlay both ground-truth (suffix _GT) and predictions (suffix _PRED) on image."""
    image_data = _image_to_uint8(image)
    bb_gt = _squeeze_boxes(classes)
    mask = ~(bb_gt == -1).any(axis=1)
    bb_gt = bb_gt[mask]

    names = label_names()
    bboxes = []

    # GT boxes — stored as [cls, cx, cy, w, h] normalized
    for row in bb_gt:
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

    pred_bboxes = _make_pred_bboxes(image_data, dets, labels)
    return LeapImageWithBBox(data=image_data, bounding_boxes=bboxes + pred_bboxes)


@tensorleap_custom_visualizer("pred_bb_decoder", LeapDataType.ImageWithBBox)
def pred_bb_decoder(
    image: np.ndarray,
    dets: np.ndarray,
    labels: np.ndarray,
) -> LeapImageWithBBox:
    """Overlay predictions only on image."""
    image_data = _image_to_uint8(image)
    return LeapImageWithBBox(
        data=image_data,
        bounding_boxes=_make_pred_bboxes(image_data, dets, labels),
    )


def _make_pred_bboxes(
    image_data: np.ndarray,
    dets: np.ndarray,
    labels: np.ndarray,
) -> list:
    """
    Build BoundingBox list from RF-DETR raw ONNX outputs.
    dets: (..., 300, 4) cxcywh normalized
    labels: (..., 300, 4) class logits
    """
    decoded_labels, boxes_xyxy, scores = decode_rfdetr_outputs(dets, labels)
    predictions = format_rtdetr_predictions(decoded_labels, boxes_xyxy, scores)
    pred_rows = prediction_rows(predictions)
    preds = pred_rows[0].numpy() if len(pred_rows) > 0 else np.zeros((0, 6), dtype=np.float32)

    h, w = image_data.shape[:2]
    names = label_names()
    bboxes = []
    for pred in preds:
        # pred: [x1, y1, x2, y2, score, label_idx] in pixel coords
        cx = (pred[0] + pred[2]) / 2 / w
        cy = (pred[1] + pred[3]) / 2 / h
        bw = (pred[2] - pred[0]) / w
        bh = (pred[3] - pred[1]) / h
        label_idx = int(pred[5]) if not np.isnan(pred[5]) else -1
        label = (names[label_idx] + "_PRED") if 0 <= label_idx < len(names) else "Unknown_PRED"
        bboxes.append(BoundingBox(
            x=float(np.clip(cx, 0, 1)),
            y=float(np.clip(cy, 0, 1)),
            width=float(np.clip(bw, 0, 1)),
            height=float(np.clip(bh, 0, 1)),
            confidence=float(pred[4]),
            label=label,
        ))
    return bboxes

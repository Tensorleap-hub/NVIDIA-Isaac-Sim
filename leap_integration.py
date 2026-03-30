"""
Tensorleap integration — YOLO11s on LOCO Warehouse dataset.
Model : yolo11s.onnx  — output0 shape (1, 84, anchors): 4 bbox + 80 COCO classes, pixel coords
Data  : LOCO warehouse (COCO format) — 5 classes
        small_load_carrier | forklift | pallet | stillage | pallet_truck
"""
import numpy as np
import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler, PreprocessResponse
from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_visualizer,
    tensorleap_input_encoder,
    tensorleap_integration_test,
    tensorleap_load_model,
)

from rtdetr_warehouse import (
    data_type_metadata,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    preprocess_func_leap,
    sample_metadata,
    synth_metadata,
)
from rtdetr_warehouse.config import CONFIG, abs_path_from_root

prediction_type = PredictionTypeHandler(name="raw_output", labels=[str(i) for i in range(84)], channel_dim=1)


@tensorleap_load_model([prediction_type])
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])


@tensorleap_input_encoder("yolo_image", channel_dim=1)
def yolo_image_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    """Return image with batch dimension added for YOLO11: (1, 3, H, W)."""
    import cv2
    img_size = int(CONFIG.get("image_size", 640))
    path = preprocess.data[idx]["path"]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    chw = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[np.newaxis]   # (1, 3, H, W)


@tensorleap_custom_visualizer("yolo_detections", LeapDataType.ImageWithBBox)
def yolo_detections_visualizer(yolo_image: np.ndarray, raw_output: np.ndarray) -> LeapImageWithBBox:
    """Post-process YOLO11 output0 (1, 84, anchors) and draw predicted bounding boxes."""
    pred = raw_output[0]              # (84, anchors)
    boxes_xywh  = pred[:4].T         # (anchors, 4) cx,cy,w,h pixels
    class_scores = pred[4:].T        # (anchors, 80)

    scores = class_scores.max(axis=1)
    labels = class_scores.argmax(axis=1)

    keep = scores >= float(CONFIG.get("score_threshold", 0.3))
    boxes_xywh = boxes_xywh[keep]
    scores     = scores[keep]
    labels     = labels[keep]

    img_size = float(CONFIG.get("image_size", 640))
    order = np.argsort(-scores)[:int(CONFIG.get("max_detections", 300))]
    boxes_xywh = boxes_xywh[order]
    scores     = scores[order]
    labels     = labels[order]

    img = yolo_image[0] if yolo_image.ndim == 4 else yolo_image   # (3, H, W)
    img_uint8 = (img.transpose(1, 2, 0) * 255).astype(np.uint8)   # (H, W, 3)

    bboxes = []
    for i in range(len(scores)):
        cx, cy, w, h = boxes_xywh[i]
        bboxes.append(BoundingBox(
            x=float(np.clip((cx - w / 2) / img_size, 0, 1)),
            y=float(np.clip((cy - h / 2) / img_size, 0, 1)),
            width=float(np.clip(w / img_size, 0, 1)),
            height=float(np.clip(h / img_size, 0, 1)),
            confidence=float(scores[i]),
            label=f"class_{int(labels[i])}_PRED",
        ))
    return LeapImageWithBBox(data=img_uint8, bounding_boxes=bboxes)


@tensorleap_custom_loss("yolo_dummy_loss")
def yolo_dummy_loss(
    raw_output: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    """Placeholder loss — returns zero. Inference runtime test only."""
    return np.array([0.0], dtype=np.float32)


@tensorleap_integration_test()
def check_integration(idx, subset):
    model      = load_model()
    yolo_image = yolo_image_encoder(idx, subset)
    gt         = gt_encoder(idx, subset)
    gt_boxes   = gt_boxes_encoder(idx, subset)
    gt_labels  = gt_labels_encoder(idx, subset)
    gt_valid   = gt_valid_mask_encoder(idx, subset)

    raw = model.run(["output0"], {"images": yolo_image})

    _ = image_visualizer(yolo_image)
    _ = yolo_detections_visualizer(yolo_image, raw[0])
    _ = yolo_dummy_loss(raw[0], gt_boxes, gt_labels, gt_valid)
    _ = data_type_metadata(idx, subset)
    _ = sample_metadata(idx, subset)
    _ = synth_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    sample_idx = subsets[subset_idx].sample_ids[0]
    check_integration(sample_idx, subsets[subset_idx])
    print("Integration test passed.")

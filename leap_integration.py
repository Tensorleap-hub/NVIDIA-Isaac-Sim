"""
Tensorleap integration — YOLO11s on LOCO Warehouse dataset.
Model : yolo11s.onnx  — output0 shape (1, 84, anchors): 4 bbox + 80 COCO classes, pixel coords
Data  : LOCO warehouse (COCO format) — 5 classes
        small_load_carrier | forklift | pallet | stillage | pallet_truck
"""
import numpy as np
import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from rtdetr_warehouse import (
    bb_decoder,
    confusion_matrix_metric,
    data_type_metadata,
    get_per_sample_metrics,
    gt_boxes_encoder,
    gt_encoder,
    gt_labels_encoder,
    gt_valid_mask_encoder,
    image_visualizer,
    input_encoder,
    input_size_encoder,
    pred_bb_decoder,
    preprocess_func_leap,
    rtdetr_total_loss_native,
    sample_metadata,
    synth_metadata,
)
from rtdetr_warehouse.config import CLASS_NAMES, CONFIG, abs_path_from_root

prediction_type = PredictionTypeHandler(name="raw_output", labels=[str(i) for i in range(84)], channel_dim=1)


@tensorleap_load_model([prediction_type])
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()

    image    = input_encoder(idx, subset)
    gt       = gt_encoder(idx, subset)
    gt_boxes = gt_boxes_encoder(idx, subset)
    gt_labels = gt_labels_encoder(idx, subset)
    gt_valid  = gt_valid_mask_encoder(idx, subset)

    # ── YOLO11 inference ──────────────────────────────────────────────────────
    img_batch = image[np.newaxis] if image.ndim == 3 else image   # (1, 3, H, W)
    raw = model.run(["output0"], {"images": img_batch})[0]         # (1, 84, anchors)

    pred = raw[0]                      # (84, anchors)
    boxes_xywh  = pred[:4].T          # (anchors, 4) cx,cy,w,h pixels
    class_scores = pred[4:].T         # (anchors, 80)

    scores = class_scores.max(axis=1)
    labels = class_scores.argmax(axis=1).astype(np.float32)

    keep = scores >= float(CONFIG.get("score_threshold", 0.3))
    boxes_xywh = boxes_xywh[keep]
    scores     = scores[keep]
    labels     = labels[keep]

    img_size = float(CONFIG.get("image_size", 640))
    boxes_xyxy = np.empty_like(boxes_xywh)
    boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2) / img_size
    boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2) / img_size
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2) / img_size
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2) / img_size

    order      = np.argsort(-scores)[:int(CONFIG.get("max_detections", 300))]
    labels     = labels[order][np.newaxis]      # (1, N)
    boxes_xyxy = boxes_xyxy[order][np.newaxis]  # (1, N, 4)
    scores     = scores[order][np.newaxis]      # (1, N)

    # Dummy RT-DETR-specific outputs (losses accept but produce zeros)
    pred_logits = np.zeros((1, 300, len(CONFIG["categories"])), dtype=np.float32)
    pred_boxes  = np.zeros((1, 300, 4), dtype=np.float32)

    # ── Visualizers ───────────────────────────────────────────────────────────
    vis_image = image_visualizer(image)
    vis_gt    = bb_decoder(image, gt, labels, boxes_xyxy, scores)
    vis_pred  = pred_bb_decoder(image, labels, boxes_xyxy, scores)

    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt,    title="GT + predictions")
        visualize(vis_pred,  title="Predictions only")

    # ── Metrics ───────────────────────────────────────────────────────────────
    _ = get_per_sample_metrics(labels, boxes_xyxy, scores, gt)
    _ = confusion_matrix_metric(labels, boxes_xyxy, scores, gt)

    # ── Loss (dummy inputs) ───────────────────────────────────────────────────
    _ = rtdetr_total_loss_native(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid)

    # ── Metadata ──────────────────────────────────────────────────────────────
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

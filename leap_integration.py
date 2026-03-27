"""
Tensorleap integration — RT-DETR v2 on LOCO Warehouse dataset.
Model : rtdetrv2_r18vd_120e_raw_outputs.onnx  (standard ONNX, no TRT ops)
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
    rtdetr_loss_components_native,
    rtdetr_total_loss_native,
    sample_metadata,
    synth_metadata,
)
from rtdetr_warehouse.config import CLASS_NAMES, CONFIG, abs_path_from_root

OUTPUT_INDICES = {
    "labels":      int(CONFIG["output_indices"]["labels"]),
    "boxes":       int(CONFIG["output_indices"]["boxes"]),
    "scores":      int(CONFIG["output_indices"]["scores"]),
    "pred_logits": int(CONFIG["output_indices"]["pred_logits"]),
    "pred_boxes":  int(CONFIG["output_indices"]["pred_boxes"]),
}

prediction_type  = PredictionTypeHandler(name="labels",      labels=CLASS_NAMES,               channel_dim=-1)
prediction_type1 = PredictionTypeHandler(name="boxes",       labels=["x1","y1","x2","y2"],     channel_dim=-1)
prediction_type2 = PredictionTypeHandler(name="confidence",  labels=["score"],                 channel_dim=-1)
prediction_type3 = PredictionTypeHandler(name="pred_logits", labels=CLASS_NAMES,               channel_dim=-1)
prediction_type4 = PredictionTypeHandler(name="pred_boxes",  labels=["cx","cy","w","h"],       channel_dim=-1)


@tensorleap_load_model([prediction_type, prediction_type1, prediction_type2, prediction_type3, prediction_type4])
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    if not model_path.endswith(".onnx"):
        raise ValueError("Only ONNX is supported.")
    return ort.InferenceSession(model_path)


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()

    image      = input_encoder(idx, subset)
    gt         = gt_encoder(idx, subset)
    gt_boxes   = gt_boxes_encoder(idx, subset)
    gt_labels  = gt_labels_encoder(idx, subset)
    gt_valid   = gt_valid_mask_encoder(idx, subset)
    orig_sizes = input_size_encoder(idx, subset)

    predictions = model.run(None, {
        "images": image,
        "orig_target_sizes": orig_sizes,
    })

    labels    = predictions[OUTPUT_INDICES["labels"]]
    boxes_xyxy = predictions[OUTPUT_INDICES["boxes"]]
    scores    = predictions[OUTPUT_INDICES["scores"]]
    pred_logits = predictions[OUTPUT_INDICES["pred_logits"]]
    pred_boxes  = predictions[OUTPUT_INDICES["pred_boxes"]]

    # Visualizers
    vis_image = image_visualizer(image)
    vis_gt    = bb_decoder(image, gt, labels, boxes_xyxy, scores)
    vis_pred  = pred_bb_decoder(image, labels, boxes_xyxy, scores)

    if bool(CONFIG.get("plot_visualizers", False)):
        visualize(vis_image, title="Input image")
        visualize(vis_gt,    title="GT + predictions")
        visualize(vis_pred,  title="Predictions only")

    # Metrics
    _ = get_per_sample_metrics(labels, boxes_xyxy, scores, gt)
    _ = confusion_matrix_metric(labels, boxes_xyxy, scores, gt)

    # Loss
    _ = rtdetr_total_loss_native(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid)
    _ = rtdetr_loss_components_native(pred_logits, pred_boxes, gt_boxes, gt_labels, gt_valid)

    # Metadata
    _ = data_type_metadata(idx, subset)
    _ = sample_metadata(idx, subset)

    _ = synth_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx  = int(CONFIG.get("check_subset_index", 0))
    sample_idx  = int(CONFIG.get("check_sample_index", 0))
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    check_integration(sample_idx, subsets[subset_idx])
    print("Integration test passed.")

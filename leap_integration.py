"""
Tensorleap integration — RF-DETR on LOCO Warehouse dataset.
Model : rfdetr-base.onnx — outputs:
  dets:   (1, 300, 4) — cxcywh normalized bounding boxes
  labels: (1, 300, 4) — class logits (3 foreground + 1 background)
Data  : LOCO warehouse (COCO format) — 3 classes
        small_load_carrier | forklift | pallet
"""
import numpy as np
import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from tensorleap_intgration_code import (
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
    pred_bb_decoder,
    preprocess_func_leap,
    rtdetr_loss_components_native,
    rtdetr_total_loss_native,
    sample_metadata,
    # synth_metadata_mean_std,
)
from tensorleap_intgration_code.config import CONFIG, abs_path_from_root

prediction_type_dets   = PredictionTypeHandler(name="dets",   labels=[str(i) for i in range(4)], channel_dim=2)
prediction_type_labels = PredictionTypeHandler(name="labels", labels=[str(i) for i in range(4)], channel_dim=2)


@tensorleap_load_model([prediction_type_dets, prediction_type_labels])
def load_model():
    model_path = abs_path_from_root(CONFIG["model_path"])
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])


@tensorleap_integration_test()
def check_integration(idx, subset):
    model     = load_model()
    image     = input_encoder(idx, subset)
    gt        = gt_encoder(idx, subset)
    gt_boxes  = gt_boxes_encoder(idx, subset)
    gt_labels = gt_labels_encoder(idx, subset)
    gt_valid  = gt_valid_mask_encoder(idx, subset)

    raw = model.run(None, {"input": image})
    dets_out   = raw[0]  # (1, 300, 4)
    labels_out = raw[1]  # (1, 300, 4)

    _ = image_visualizer(image)
    _ = pred_bb_decoder(image, dets_out, labels_out)
    _ = bb_decoder(image, gt, dets_out, labels_out)
    _ = rtdetr_total_loss_native(labels_out, dets_out, gt_boxes, gt_labels, gt_valid)
    _ = rtdetr_loss_components_native(labels_out, dets_out, gt_boxes, gt_labels, gt_valid)
    _ = get_per_sample_metrics(dets_out, labels_out, gt)
    _ = confusion_matrix_metric(dets_out, labels_out, gt)
    _ = data_type_metadata(idx, subset)
    _ = sample_metadata(idx, subset)
    # _ = synth_metadata_mean_std(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    subset_idx = int(CONFIG.get("check_subset_index", 0))
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    sample_idx = subsets[subset_idx].sample_ids[0]
    check_integration(sample_idx, subsets[subset_idx])
    print("Integration test passed.")

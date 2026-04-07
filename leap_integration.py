"""
Tensorleap integration — YOLO11s on LOCO Warehouse dataset.
Model : yolo11s.onnx — outputs:
  output0: (1, 84, 8400) — 4 bbox (cx,cy,w,h pixels, DFL-decoded) + 80 COCO class scores (sigmoid)
  output1/2/3: (1, 144, H, W) — raw feature maps at strides 8/16/32
Data  : LOCO warehouse (COCO format) — 5 classes
        small_load_carrier | forklift | pallet | stillage | pallet_truck
"""
import numpy as np
import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
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
    input_encoder,
    preprocess_func_leap,
    sample_metadata,
    synth_metadata,
    yolo_bb_decoder,
    yolo_confusion_matrix,
    yolo_loss_components,
    yolo_per_sample_metrics,
    yolo_pred_bb_decoder,
    yolo_total_loss,
)
from rtdetr_warehouse.config import CONFIG, abs_path_from_root

prediction_type0 = PredictionTypeHandler(name="output0",  labels=[str(i) for i in range(84)],  channel_dim=1)
prediction_type1 = PredictionTypeHandler(name="output1",  labels=[str(i) for i in range(144)], channel_dim=1)
prediction_type2 = PredictionTypeHandler(name="output2",  labels=[str(i) for i in range(144)], channel_dim=1)
prediction_type3 = PredictionTypeHandler(name="output3",  labels=[str(i) for i in range(144)], channel_dim=1)


@tensorleap_load_model([prediction_type0, prediction_type1, prediction_type2, prediction_type3])
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
    raw       = model.run(None, {"images": image})

    _ = image_visualizer(image)
    # _ = yolo_pred_bb_decoder(image, raw[0])
    # _ = yolo_bb_decoder(image, gt, raw[0])
    _ = yolo_total_loss(raw[0], gt_boxes, gt_labels, gt_valid)
    # _ = yolo_loss_components(raw[0], gt_boxes, gt_labels, gt_valid)
    _ = yolo_per_sample_metrics(raw[0], gt)
    # _ = yolo_confusion_matrix(raw[0], gt)
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

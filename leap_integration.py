"""
Tensorleap integration — YOLO11s on LOCO Warehouse dataset.
Model : yolo11s.onnx  — output0 shape (1, 84, anchors): 4 bbox + 80 COCO classes, pixel coords
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
    image_visualizer,
    input_encoder,
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


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)

    model.run(["output0"], {"images": image})

    _ = image_visualizer(image)
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

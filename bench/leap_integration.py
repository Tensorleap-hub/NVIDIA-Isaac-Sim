from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_integration_test,
    tensorleap_load_model,
)

from benchmark_integration import (
    data_type_metadata,
    input_encoder,
    preprocess_func_leap,
    simulation_type_metadata,
    theta_metadata,
)

_ONNX_PATH = Path(__file__).parent / "convergence" / "dinov2_vits14.onnx"
_EMBEDDING_DIM = 384

prediction_embedding = PredictionTypeHandler(
    name="embedding",
    labels=[str(i) for i in range(_EMBEDDING_DIM)],
    channel_dim=0,
)


@tensorleap_load_model([prediction_embedding])
def load_model():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(_ONNX_PATH),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    _ = model.run(None, {"img": image})[0]
    _ = data_type_metadata(idx, subset)
    _ = simulation_type_metadata(idx, subset)
    _ = theta_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    subset = subsets[0]
    sample_idx = subset.sample_ids[0]
    check_integration(sample_idx, subset)
    print("Integration test passed.")

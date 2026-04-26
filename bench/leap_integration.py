from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from code_loader.contract.datasetclasses import PreprocessResponse, PredictionTypeHandler
from code_loader.contract.enums import DatasetMetadataType, DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_integration_test,
    tensorleap_load_model,
    tensorleap_metadata,
    tensorleap_preprocess,
)

_BENCH_DIR = Path(__file__).parent
_DATA_ROOT = Path.home() / "tensorleap" / "data" / "synth-data-benchmark"
_REAL_DIR = _DATA_ROOT / "real"
_METADATA_PATH = _DATA_ROOT / "tl_seed" / "metadata.csv"
_ONNX_PATH = _BENCH_DIR / "convergence" / "dinov2_vits14.onnx"

_THETA_KEYS = [
    "blur_sigma", "noise_std", "brightness_shift",
    "color_shift_r", "color_shift_g", "color_shift_b",
    "clutter_count", "background_id",
]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_EMBEDDING_DIM = 384

prediction_embedding = PredictionTypeHandler(
    name="embedding",
    labels=[str(i) for i in range(_EMBEDDING_DIM)],
    channel_dim=0,
)


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), Image.BICUBIC)
    img = img.crop((16, 16, 240, 240))  # center crop 224
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - _MEAN) / _STD).transpose(2, 0, 1)  # CHW


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    real_records = [
        {"image_path": str(p), "data_type": "real", "simulation_type": ""}
        for p in sorted(_REAL_DIR.glob("*.png"))
    ]
    meta_df = pd.read_csv(_METADATA_PATH)
    synth_records = [
        {"image_path": str(row["image_path"]), "data_type": "synth", "simulation_type": "simulation_1",
         **{k: float(row[k]) for k in _THETA_KEYS}}
        for _, row in meta_df.iterrows()
    ]
    split = int(len(real_records) * 0.8)
    train, val = real_records[:split], real_records[split:]
    train_ids = [f"real_{i:04d}" for i in range(len(train))]
    val_ids   = [f"real_{i:04d}" for i in range(len(train), len(real_records))]
    synth_ids = [f"synth_{i:06d}" for i in range(len(synth_records))]
    return [
        PreprocessResponse(data=dict(zip(train_ids, train)), sample_ids=train_ids, state=DataStateType.training),
        PreprocessResponse(data=dict(zip(val_ids, val)),     sample_ids=val_ids,   state=DataStateType.validation),
        PreprocessResponse(data=dict(zip(synth_ids, synth_records)), sample_ids=synth_ids, state=DataStateType.additional),
    ]


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return _load_image(preprocess.data[idx]["image_path"])


@tensorleap_gt_encoder("domain")
def domain_gt_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return np.asarray(1.0 if preprocess.data[idx]["data_type"] == "synth" else 0.0, dtype=np.float32)


@tensorleap_custom_loss("embedding_l2")
def embedding_l2_loss(embedding: np.ndarray, domain: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(embedding ** 2), dtype=np.float32)


@tensorleap_metadata("data_type", DatasetMetadataType.string)
def data_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx]["data_type"]


@tensorleap_metadata("simulation_type", DatasetMetadataType.string)
def simulation_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx].get("simulation_type", "")


@tensorleap_metadata("theta")
def theta_metadata(idx: str, preprocess: PreprocessResponse) -> dict:
    record = preprocess.data[idx]
    if record["data_type"] == "real":
        return {k: float("nan") for k in _THETA_KEYS}
    return {k: float(record[k]) for k in _THETA_KEYS}


@tensorleap_load_model([prediction_embedding])
def load_model():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(_ONNX_PATH), sess_options=sess_options, providers=["CPUExecutionProvider"])


@tensorleap_integration_test()
def check_integration(idx, subset):
    model = load_model()
    image = input_encoder(idx, subset)
    domain = domain_gt_encoder(idx, subset)
    raw = model.run(None, {"img": image})[0]
    _ = embedding_l2_loss(raw, domain)
    _ = data_type_metadata(idx, subset)
    _ = simulation_type_metadata(idx, subset)
    _ = theta_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    subset = subsets[0]
    check_integration(subset.sample_ids[0], subset)
    print("Integration test passed.")

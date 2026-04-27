from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from code_loader.contract.datasetclasses import PreprocessResponse, PredictionTypeHandler
from code_loader.contract.enums import DatasetMetadataType, DataStateType, LeapDataType
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_visualizer,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_integration_test,
    tensorleap_load_model,
    tensorleap_metadata,
    tensorleap_preprocess,
)

import os

_BENCH_DIR = Path(__file__).parent
_REL = "synth-data-benchmark"


def _get_data_root() -> Path:
    if "GENERIC_HOST_PATH" in os.environ:
        return Path(os.environ["GENERIC_HOST_PATH"]) / _REL
    fallback = Path("/home/ssm-user/tensorleap/data")
    if fallback.exists():
        return fallback / _REL
    return Path("~/tensorleap/data").expanduser() / _REL


_DATA_ROOT = _get_data_root()
_REAL_DIR = _DATA_ROOT / "real"
_METADATA_PATH = _DATA_ROOT / "tl_seed" / "metadata.csv"
_ONNX_PATH = _BENCH_DIR / "convergence" / "dinov2_vits14.onnx"


def _parse_synth_iters() -> set | None:
    raw = os.environ.get("SYNTH_ITERS", "all").strip()
    if raw.lower() == "all":
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


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
    synth_iters = _parse_synth_iters()
    tagged = []
    if _METADATA_PATH.exists() and (synth_iters is None or 0 in synth_iters):
        df0 = pd.read_csv(_METADATA_PATH)
        df0["_iter"] = 0
        tagged.append(df0)
    for d in sorted(_DATA_ROOT.glob("tl_iter_*")):
        iter_num = int(d.name.split("_")[-1])
        if synth_iters is not None and iter_num not in synth_iters:
            continue
        p = d / "metadata.csv"
        if p.exists():
            df_i = pd.read_csv(p)
            df_i["_iter"] = iter_num
            tagged.append(df_i)
    meta_df = pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame(columns=["image_path"] + _THETA_KEYS + ["_iter"])
    synth_records = [
        {"image_path": str(row["image_path"]), "data_type": "synth", "simulation_type": "simulation_1",
         "tl_iter": int(row["_iter"]),
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


@tensorleap_custom_visualizer(name="image", visualizer_type=LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    if image.ndim == 4:
        image = image[0]
    img = (image.transpose(1, 2, 0) * _STD + _MEAN).clip(0, 1)
    return LeapImage(data=(img * 255).astype(np.uint8))


@tensorleap_custom_loss("embedding_l2")
def embedding_l2_loss(embedding: np.ndarray, domain: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(embedding ** 2), dtype=np.float32)


@tensorleap_metadata("data_type", DatasetMetadataType.string)
def data_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx]["data_type"]


@tensorleap_metadata("simulation_type", DatasetMetadataType.string)
def simulation_type_metadata(idx: str, preprocess: PreprocessResponse) -> str:
    return preprocess.data[idx].get("simulation_type", "")


@tensorleap_metadata("tl_iter", DatasetMetadataType.float)
def tl_iter_metadata(idx: str, preprocess: PreprocessResponse) -> float:
    val = preprocess.data[idx].get("tl_iter")
    return float("nan") if val is None else float(val)


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
    _ = image_visualizer(image)
    _ = data_type_metadata(idx, subset)
    _ = simulation_type_metadata(idx, subset)
    _ = theta_metadata(idx, subset)


if __name__ == "__main__":
    subsets = preprocess_func_leap()
    print(f"Subsets: {[len(s.data) for s in subsets]}")
    subset = subsets[0]
    check_integration(subset.sample_ids[0], subset)
    print("Integration test passed.")

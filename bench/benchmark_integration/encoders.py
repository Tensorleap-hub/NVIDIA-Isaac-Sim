from __future__ import annotations

import numpy as np
from PIL import Image

from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_gt_encoder, tensorleap_input_encoder

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CROP = 224
_RESIZE = 256


def _preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((_RESIZE, _RESIZE), Image.BICUBIC)
    left = (_RESIZE - _CROP) // 2
    img = img.crop((left, left, left + _CROP, left + _CROP))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)  # CHW (3, 224, 224)


@tensorleap_input_encoder("image", channel_dim=1)
def input_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    return _preprocess_image(preprocess.data[idx]["image_path"])


@tensorleap_gt_encoder("domain")
def domain_gt_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    is_synth = preprocess.data[idx]["data_type"] == "synth"
    return np.asarray(1.0 if is_synth else 0.0, dtype=np.float32)

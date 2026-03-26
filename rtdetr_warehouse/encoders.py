import numpy as np
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
)
from code_loader.contract.datasetclasses import PreprocessResponse

from rtdetr_warehouse.utils import load_and_preprocess_image, build_gt_tensor


@tensorleap_input_encoder('image', channel_dim=-1)
def encode_image(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Load and preprocess image for Tensorleap.

    Returns:
        np.ndarray: shape (640, 640, 3), float32 HWC, ImageNet-normalized.
        Transposed to CHW inside integration_test before passing to the model.
    """
    img_id = int(sample_id)
    img_info = preprocess.data['images'][img_id]
    return load_and_preprocess_image(img_info['path']).transpose(1, 2, 0)  # CHW -> HWC


@tensorleap_gt_encoder('gt_boxes')
def encode_gt_boxes(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Encode ground-truth boxes as padded tensor.

    Returns:
        np.ndarray: shape (MAX_DETS, 5), float32
            col 0:   class index (0–4), or -1 for padding
            cols 1–4: [cx, cy, w, h] normalized in model input space (640×640)
    """
    img_id = int(sample_id)
    img_info = preprocess.data['images'][img_id]
    anns = preprocess.data['anns'].get(img_id, [])
    return build_gt_tensor(anns, img_info['width'], img_info['height'])

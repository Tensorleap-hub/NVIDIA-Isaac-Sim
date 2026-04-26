from __future__ import annotations

import numpy as np

from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss


@tensorleap_custom_loss("embedding_l2")
def embedding_l2_loss(embedding: np.ndarray, domain: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(embedding ** 2), dtype=np.float32)

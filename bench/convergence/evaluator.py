from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from calibration_optuna.metrics import DistributionMetrics

_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

DEFAULT_ONNX_PATH = Path(__file__).parent / "dinov2_vitb14_reg.onnx"


def mmd_rbf(X: np.ndarray, Y: np.ndarray, max_samples: int = 1000) -> float:
    rng = np.random.RandomState(0)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    return DistributionMetrics.mmd(X, Y, kernel="rbf")


class Embedder:
    def __init__(
        self,
        onnx_path: Path | None = DEFAULT_ONNX_PATH,
        device: str = "cpu",
    ):
        if onnx_path is not None and Path(onnx_path).exists():
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(onnx_path))
            self._input_name = self._session.get_inputs()[0].name
            self._backend = "onnx"
        else:
            import torch
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
            self._model.eval()
            self._device = torch.device(device)
            self._model.to(self._device)
            self._backend = "torch"

    def embed(self, images: list, batch_size: int = 32) -> np.ndarray:
        import torch
        tensors = [_TRANSFORM(img.convert("RGB")) for img in images]
        batches = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size])
            if self._backend == "onnx":
                out = self._session.run(None, {self._input_name: batch.numpy()})[0]
            else:
                with torch.inference_mode():
                    out = self._model(batch.to(self._device)).cpu().numpy()
            batches.append(out)
        return np.concatenate(batches, axis=0)

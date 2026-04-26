import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from convergence.evaluator import mmd_rbf, Embedder


def test_mmd_zero_for_identical():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 32).astype(np.float32)
    assert mmd_rbf(X, X) < 1e-5


def test_mmd_positive_for_separated():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 32).astype(np.float32)
    Y = (rng.randn(50, 32) + 5.0).astype(np.float32)
    assert mmd_rbf(X, Y) > 0.5


def test_mmd_symmetric():
    rng = np.random.RandomState(7)
    X = rng.randn(40, 16).astype(np.float32)
    Y = (rng.randn(40, 16) + 1.0).astype(np.float32)
    assert abs(mmd_rbf(X, Y) - mmd_rbf(Y, X)) < 1e-5


def test_mmd_handles_large_inputs_via_subsampling():
    rng = np.random.RandomState(2)
    X = rng.randn(2000, 8).astype(np.float32)
    Y = rng.randn(2000, 8).astype(np.float32)
    result = mmd_rbf(X, Y, max_samples=100)
    assert isinstance(result, float)
    assert result >= 0.0


def test_embedder_init_torch_hub():
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    with patch("torch.hub.load", return_value=mock_model):
        emb = Embedder(onnx_path=None)
    assert emb._backend == "torch"


def test_embedder_init_onnx(tmp_path):
    fake_onnx = tmp_path / "dinov2.onnx"
    fake_onnx.write_bytes(b"fake")
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="pixel_values")]
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        emb = Embedder(onnx_path=fake_onnx)
    assert emb._backend == "onnx"

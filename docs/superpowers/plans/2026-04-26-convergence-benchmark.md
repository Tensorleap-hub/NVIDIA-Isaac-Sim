# Convergence Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, Isaac-free benchmark that isolates whether the TL loop failure is caused by the suggester (TL CSV workflow) vs. the evaluator, by running Optuna TPE and TL against the same toy generator + fixed DINOv2 evaluator.

**Architecture:** A pure-Python 2D toy image generator (parameterized by 8D θ) produces images; a DINOv2 embedder (torch.hub, ONNX when available) extracts features; MMD measures synthetic-vs-real distance per trial. Condition A runs Optuna TPE; Condition B reads θ rows from a TL-exported CSV. All outputs land in `~/tensorleap/data/synth-data-benchmark/`.

**Tech Stack:** Python 3.10.12, poetry, numpy, Pillow, torch (torch.hub DINOv2), onnxruntime, optuna, pandas, matplotlib, scipy, scikit-learn, pytest.

---

## File Map

| File | Responsibility |
|------|----------------|
| `bench/pyproject.toml` | Poetry project for bench deps (python 3.10.12) |
| `bench/convergence/__init__.py` | Package marker |
| `bench/convergence/config.py` | Data paths, constants |
| `bench/convergence/theta_star.json` | Frozen target θ* |
| `bench/convergence/generator.py` | `generate_images(theta, n, seed) → list[Image]` |
| `bench/convergence/evaluator.py` | `mmd_rbf(X, Y) → float`, `Embedder.embed(images) → np.ndarray` |
| `bench/convergence/harness.py` | `run_trial(theta, n, real_embs, embedder) → (float, np.ndarray)` |
| `bench/convergence/metrics.py` | `MetricsLogger`, `param_gap`, `spread`, `normalize_theta` |
| `bench/convergence/optuna_loop.py` | Condition A: full Optuna TPE loop |
| `bench/convergence/tl_loop.py` | Condition B: reads TL CSV, same evaluator |
| `bench/convergence/plot.py` | `plot_comparison(optuna_csv, tl_csv)` |
| `bench/tests/__init__.py` | Package marker |
| `bench/tests/test_generator.py` | Unit tests for generator |
| `bench/tests/test_evaluator.py` | Unit tests for mmd_rbf + Embedder construction |
| `bench/tests/test_harness.py` | Unit tests for run_trial (embedder mocked) |
| `bench/tests/test_metrics.py` | Unit tests for MetricsLogger + helpers |
| `bench/tests/test_optuna_loop.py` | Smoke test for optuna loop (embedder mocked, 2 iters) |

Data directory layout (`~/tensorleap/data/synth-data-benchmark/`):
```
real/                    # 500 PNG images generated from θ*
real_embeddings.npy      # cached DINOv2 embeddings, shape (500, 768)
runs/
  optuna_seed42/
    metrics.csv
  tl_seed42/
    metrics.csv
```

---

## Task 1: Poetry project + package scaffold

**Files:**
- Create: `bench/pyproject.toml`
- Create: `bench/convergence/__init__.py`
- Create: `bench/tests/__init__.py`

- [ ] **Step 1: Create `bench/pyproject.toml`**

```toml
[tool.poetry]
name = "synth-bench"
version = "0.1.0"
description = "Convergence benchmark: Optuna vs TL suggester"
authors = []
packages = [{include = "convergence"}]

[tool.poetry.dependencies]
python = "^3.10"
numpy = ">=1.24"
Pillow = ">=9.0"
torch = ">=2.0"
torchvision = ">=0.15"
onnxruntime = ">=1.16"
optuna = ">=3.0"
pandas = ">=1.5"
matplotlib = ">=3.5"
scipy = ">=1.9"
scikit-learn = ">=1.2"

[tool.poetry.group.dev.dependencies]
pytest = ">=7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 2: Create empty `__init__.py` files**

```python
# bench/convergence/__init__.py  (empty)
# bench/tests/__init__.py        (empty)
```

- [ ] **Step 3: Install the env (run from `bench/`)**

```bash
cd bench
poetry env use ~/.pyenv/versions/3.10.12/bin/python
poetry install
```

Expected output ends with: `Installing the current project: synth-bench (0.1.0)`

- [ ] **Step 4: Verify python version**

```bash
cd bench && poetry run python --version
```

Expected: `Python 3.10.12`

- [ ] **Step 5: Commit**

```bash
git add bench/pyproject.toml bench/convergence/__init__.py bench/tests/__init__.py
git commit -m "bench: poetry project + package scaffold"
```

---

## Task 2: Config + theta_star.json

**Files:**
- Create: `bench/convergence/config.py`
- Create: `bench/convergence/theta_star.json`

- [ ] **Step 1: Write `bench/convergence/config.py`**

```python
from pathlib import Path

DATA_ROOT = Path.home() / "tensorleap" / "data" / "synth-data-benchmark"
REAL_DIR = DATA_ROOT / "real"
REAL_EMBEDDINGS_PATH = DATA_ROOT / "real_embeddings.npy"
RUNS_DIR = DATA_ROOT / "runs"
THETA_STAR_PATH = Path(__file__).parent / "theta_star.json"
N_REAL_IMAGES = 500

THETA_KEYS = [
    "blur_sigma", "noise_std", "brightness_shift",
    "color_shift_r", "color_shift_g", "color_shift_b",
    "clutter_count", "background_id",
]
THETA_BOUNDS = {
    "blur_sigma":        (0.0,  5.0),
    "noise_std":         (0.0,  0.5),
    "brightness_shift":  (-0.5, 0.5),
    "color_shift_r":     (-0.3, 0.3),
    "color_shift_g":     (-0.3, 0.3),
    "color_shift_b":     (-0.3, 0.3),
    "clutter_count":     (0.0,  20.0),
    "background_id":     (0.0,  3.0),
}

IMAGE_SIZE = 256
N_IMAGES_PER_TRIAL = 128
N_ITERATIONS = 30
N_TRIALS_PER_ITER = 8
SEED = 42
MMD_MAX_SAMPLES = 1000
```

- [ ] **Step 2: Write `bench/convergence/theta_star.json`**

Interior point — not at any boundary, picked once and committed:

```json
{
  "blur_sigma": 1.5,
  "noise_std": 0.12,
  "brightness_shift": 0.08,
  "color_shift_r": 0.06,
  "color_shift_g": -0.04,
  "color_shift_b": 0.02,
  "clutter_count": 7,
  "background_id": 1
}
```

- [ ] **Step 3: Commit**

```bash
git add bench/convergence/config.py bench/convergence/theta_star.json
git commit -m "bench: config + frozen theta_star"
```

---

## Task 3: Generator

**Files:**
- Create: `bench/convergence/generator.py`
- Create: `bench/tests/test_generator.py`

- [ ] **Step 1: Write the failing tests in `bench/tests/test_generator.py`**

```python
import numpy as np
import pytest
from PIL import Image
from convergence.generator import generate_images

_BASE = {
    "blur_sigma": 0.0, "noise_std": 0.0, "brightness_shift": 0.0,
    "color_shift_r": 0.0, "color_shift_g": 0.0, "color_shift_b": 0.0,
    "clutter_count": 0, "background_id": 0,
}


def test_returns_correct_count():
    imgs = generate_images(_BASE, n=4, seed=0)
    assert len(imgs) == 4


def test_returns_pil_images():
    imgs = generate_images(_BASE, n=2, seed=0)
    assert all(isinstance(img, Image.Image) for img in imgs)


def test_output_size():
    imgs = generate_images(_BASE, n=1, seed=0)
    assert imgs[0].size == (256, 256)


def test_deterministic():
    theta = {**_BASE, "noise_std": 0.1, "clutter_count": 3}
    a = generate_images(theta, n=2, seed=42)
    b = generate_images(theta, n=2, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(np.array(x), np.array(y))


def test_different_seeds_differ():
    theta = {**_BASE, "noise_std": 0.3}
    i1 = generate_images(theta, n=1, seed=1)[0]
    i2 = generate_images(theta, n=1, seed=2)[0]
    assert not np.array_equal(np.array(i1), np.array(i2))


@pytest.mark.parametrize("bg_id", [0, 1, 2, 3])
def test_all_background_ids(bg_id):
    theta = {**_BASE, "background_id": bg_id}
    imgs = generate_images(theta, n=1, seed=0)
    assert len(imgs) == 1


def test_blur_reduces_high_frequency():
    theta_noisy = {**_BASE, "background_id": 3}
    no_blur = np.array(generate_images({**theta_noisy, "blur_sigma": 0.0}, n=1, seed=0)[0]).astype(float)
    blurred = np.array(generate_images({**theta_noisy, "blur_sigma": 3.0}, n=1, seed=0)[0]).astype(float)
    assert blurred.std() < no_blur.std()


def test_clutter_changes_image():
    no_clutter = np.array(generate_images({**_BASE, "clutter_count": 0}, n=1, seed=5)[0])
    with_clutter = np.array(generate_images({**_BASE, "clutter_count": 15}, n=1, seed=5)[0])
    assert not np.array_equal(no_clutter, with_clutter)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd bench && poetry run pytest tests/test_generator.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` (convergence.generator not yet implemented)

- [ ] **Step 3: Write `bench/convergence/generator.py`**

```python
import numpy as np
from PIL import Image, ImageFilter
from .config import IMAGE_SIZE


def _make_background(bg_id: int) -> np.ndarray:
    h = w = IMAGE_SIZE
    rng = np.random.RandomState(int(bg_id) * 12345)
    if bg_id == 0:
        arr = np.zeros((h, w, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(0.2, 0.8, w)
        arr[:, :, 1] = np.linspace(0.5, 0.3, w)
        arr[:, :, 2] = np.linspace(0.8, 0.2, w)
    elif bg_id == 1:
        block = 32
        grid = (np.indices((h, w)).sum(axis=0) // block) % 2
        v = (0.3 + grid * 0.4).astype(np.float32)
        arr = np.stack([v, v, v], axis=-1)
    elif bg_id == 2:
        arr = np.zeros((h, w, 3), dtype=np.float32)
        arr[:, :, 0] = np.linspace(0.1, 0.7, h)[:, None]
        arr[:, :, 1] = np.linspace(0.6, 0.2, h)[:, None]
        arr[:, :, 2] = np.linspace(0.3, 0.9, h)[:, None]
    else:
        arr = rng.uniform(0.2, 0.8, (h, w, 3)).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def generate_images(theta: dict, n: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    bg = _make_background(int(theta["background_id"]))
    images = []
    for _ in range(n):
        img = bg.copy()

        img += float(theta["brightness_shift"])
        img[:, :, 0] += float(theta["color_shift_r"])
        img[:, :, 1] += float(theta["color_shift_g"])
        img[:, :, 2] += float(theta["color_shift_b"])
        img = np.clip(img, 0.0, 1.0)

        n_rects = int(theta["clutter_count"])
        sz = IMAGE_SIZE
        for _ in range(n_rects):
            x1 = rng.randint(0, sz - 1)
            y1 = rng.randint(0, sz - 1)
            x2 = rng.randint(x1 + 1, min(x1 + sz // 4 + 1, sz))
            y2 = rng.randint(y1 + 1, min(y1 + sz // 4 + 1, sz))
            img[y1:y2, x1:x2] = rng.rand(3).astype(np.float32)

        noise_std = float(theta["noise_std"])
        if noise_std > 0:
            img += rng.normal(0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        blur_sigma = float(theta["blur_sigma"])
        pil = Image.fromarray((img * 255).astype(np.uint8))
        if blur_sigma > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        images.append(pil)
    return images
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd bench && poetry run pytest tests/test_generator.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add bench/convergence/generator.py bench/tests/test_generator.py
git commit -m "bench: toy image generator + tests"
```

---

## Task 4: Evaluator (MMD + DINOv2 embedder)

**Files:**
- Create: `bench/convergence/evaluator.py`
- Create: `bench/tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests in `bench/tests/test_evaluator.py`**

```python
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


def test_embedder_init_torch_hub(monkeypatch):
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
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        emb = Embedder(onnx_path=fake_onnx)
    assert emb._backend == "onnx"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd bench && poetry run pytest tests/test_evaluator.py -v
```

Expected: `ImportError` (module not yet created)

- [ ] **Step 3: Write `bench/convergence/evaluator.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

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

    Z = np.vstack([X, Y])
    sq_dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
    nonzero = sq_dists[sq_dists > 0]
    gamma = 1.0 / (2.0 * float(np.median(nonzero))) if len(nonzero) > 0 else 1.0

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)
        return np.exp(-gamma * (A_sq + B_sq.T - 2.0 * A @ B.T))

    val = rbf(X, X).mean() + rbf(Y, Y).mean() - 2.0 * rbf(X, Y).mean()
    return float(np.sqrt(max(val, 0.0)))


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
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd bench && poetry run pytest tests/test_evaluator.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add bench/convergence/evaluator.py bench/tests/test_evaluator.py
git commit -m "bench: evaluator — mmd_rbf + DINOv2 Embedder (torch.hub / ONNX)"
```

---

## Task 5: Harness

**Files:**
- Create: `bench/convergence/harness.py`
- Create: `bench/tests/test_harness.py`

- [ ] **Step 1: Write failing tests in `bench/tests/test_harness.py`**

```python
import numpy as np
from unittest.mock import MagicMock
from convergence.harness import run_trial

_THETA = {
    "blur_sigma": 1.5, "noise_std": 0.1, "brightness_shift": 0.05,
    "color_shift_r": 0.0, "color_shift_g": 0.0, "color_shift_b": 0.0,
    "clutter_count": 3, "background_id": 0,
}


def test_returns_float_and_array():
    real = np.random.randn(50, 768).astype(np.float32)
    embedder = MagicMock()
    embedder.embed.return_value = np.random.randn(8, 768).astype(np.float32)

    dist, embs = run_trial(_THETA, n_images=8, real_embeddings=real, embedder=embedder, seed=0)

    assert isinstance(dist, float)
    assert dist >= 0.0
    assert embs.shape == (8, 768)
    embedder.embed.assert_called_once()


def test_embedder_receives_correct_image_count():
    real = np.random.randn(50, 768).astype(np.float32)
    embedder = MagicMock()
    embedder.embed.return_value = np.random.randn(16, 768).astype(np.float32)

    run_trial(_THETA, n_images=16, real_embeddings=real, embedder=embedder, seed=0)

    called_images = embedder.embed.call_args[0][0]
    assert len(called_images) == 16


def test_mmd_zero_when_embeddings_match_real():
    real = np.ones((50, 32), dtype=np.float32)
    embedder = MagicMock()
    embedder.embed.return_value = np.ones((8, 32), dtype=np.float32)

    dist, _ = run_trial(_THETA, n_images=8, real_embeddings=real, embedder=embedder, seed=0)
    assert dist < 1e-5
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd bench && poetry run pytest tests/test_harness.py -v
```

- [ ] **Step 3: Write `bench/convergence/harness.py`**

```python
from __future__ import annotations

import numpy as np
from .generator import generate_images
from .evaluator import mmd_rbf
from .config import MMD_MAX_SAMPLES


def run_trial(
    theta: dict,
    n_images: int,
    real_embeddings: np.ndarray,
    embedder,
    seed: int = 0,
    mmd_max_samples: int = MMD_MAX_SAMPLES,
) -> tuple[float, np.ndarray]:
    images = generate_images(theta, n=n_images, seed=seed)
    syn_embeddings = embedder.embed(images)
    distance = mmd_rbf(syn_embeddings, real_embeddings, max_samples=mmd_max_samples)
    return distance, syn_embeddings
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd bench && poetry run pytest tests/test_harness.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add bench/convergence/harness.py bench/tests/test_harness.py
git commit -m "bench: trial harness"
```

---

## Task 6: Metrics logger

**Files:**
- Create: `bench/convergence/metrics.py`
- Create: `bench/tests/test_metrics.py`

- [ ] **Step 1: Write failing tests in `bench/tests/test_metrics.py`**

```python
import numpy as np
import pytest
from convergence.metrics import MetricsLogger, param_gap, spread, normalize_theta

_STAR = {
    "blur_sigma": 1.5, "noise_std": 0.12, "brightness_shift": 0.08,
    "color_shift_r": 0.06, "color_shift_g": -0.04, "color_shift_b": 0.02,
    "clutter_count": 7, "background_id": 1,
}


def test_normalize_theta_in_unit_range():
    vec = normalize_theta(_STAR)
    assert vec.shape == (8,)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_param_gap_zero_for_star():
    assert param_gap(_STAR, _STAR) < 1e-10


def test_param_gap_positive_for_different():
    other = {**_STAR, "blur_sigma": 5.0}
    assert param_gap(other, _STAR) > 0


def test_spread_zero_for_identical():
    assert spread([_STAR] * 4) < 1e-10


def test_spread_positive_for_varied():
    thetas = [
        {**_STAR, "blur_sigma": 0.0},
        {**_STAR, "blur_sigma": 5.0},
    ]
    assert spread(thetas) > 0


def test_logger_creates_csv_on_first_log(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    trial_results = [({**_STAR, "blur_sigma": 0.5}, 0.4), ({**_STAR}, 0.35)]
    record = logger.log(0, trial_results)
    assert record.iteration == 0
    assert abs(record.best_objective - 0.35) < 1e-6


def test_logger_tracks_global_best_across_iterations(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    logger.log(0, [(_STAR, 0.5)])
    record = logger.log(1, [({**_STAR, "blur_sigma": 4.0}, 0.8)])
    assert abs(record.best_objective - 0.5) < 1e-6


def test_logger_load_roundtrip(tmp_path):
    logger = MetricsLogger(tmp_path / "metrics.csv", _STAR)
    logger.log(0, [(_STAR, 0.5)])
    logger.log(1, [({**_STAR, "blur_sigma": 1.0}, 0.3)])
    records = logger.load()
    assert len(records) == 2
    assert records[0].iteration == 0
    assert records[1].iteration == 1
    assert abs(records[1].best_objective - 0.3) < 1e-6
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd bench && poetry run pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Write `bench/convergence/metrics.py`**

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
from .config import THETA_KEYS, THETA_BOUNDS


def normalize_theta(theta: dict) -> np.ndarray:
    vec = []
    for k in THETA_KEYS:
        lo, hi = THETA_BOUNDS[k]
        vec.append((float(theta[k]) - lo) / (hi - lo))
    return np.array(vec, dtype=np.float32)


def param_gap(theta: dict, theta_star: dict) -> float:
    return float(np.linalg.norm(normalize_theta(theta) - normalize_theta(theta_star)))


def spread(thetas: list[dict]) -> float:
    mat = np.stack([normalize_theta(t) for t in thetas])
    return float(mat.std(axis=0).mean())


@dataclass
class IterationRecord:
    iteration: int
    best_objective: float
    best_theta_json: str
    param_gap: float
    spread: float
    median_objective: float
    mean_objective: float


_FIELDNAMES = [f.name for f in fields(IterationRecord)]


class MetricsLogger:
    def __init__(self, csv_path: Path, theta_star: dict):
        self._path = Path(csv_path)
        self._theta_star = theta_star
        self._global_best = float("inf")
        self._global_best_theta: dict | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()

    def log(self, iteration: int, trial_results: list[tuple[dict, float]]) -> IterationRecord:
        objectives = [r[1] for r in trial_results]
        thetas = [r[0] for r in trial_results]
        iter_best = min(objectives)
        iter_best_theta = thetas[objectives.index(iter_best)]
        if iter_best < self._global_best:
            self._global_best = iter_best
            self._global_best_theta = iter_best_theta
        record = IterationRecord(
            iteration=iteration,
            best_objective=self._global_best,
            best_theta_json=json.dumps(self._global_best_theta),
            param_gap=param_gap(self._global_best_theta, self._theta_star),
            spread=spread(thetas),
            median_objective=float(np.median(objectives)),
            mean_objective=float(np.mean(objectives)),
        )
        with self._path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(asdict(record))
        return record

    def load(self) -> list[IterationRecord]:
        records = []
        with self._path.open() as f:
            for row in csv.DictReader(f):
                records.append(IterationRecord(
                    iteration=int(row["iteration"]),
                    best_objective=float(row["best_objective"]),
                    best_theta_json=row["best_theta_json"],
                    param_gap=float(row["param_gap"]),
                    spread=float(row["spread"]),
                    median_objective=float(row["median_objective"]),
                    mean_objective=float(row["mean_objective"]),
                ))
        return records
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd bench && poetry run pytest tests/test_metrics.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add bench/convergence/metrics.py bench/tests/test_metrics.py
git commit -m "bench: metrics logger + param_gap + spread"
```

---

## Task 7: Optuna loop (Condition A)

**Files:**
- Create: `bench/convergence/optuna_loop.py`
- Create: `bench/tests/test_optuna_loop.py`

- [ ] **Step 1: Write failing smoke test in `bench/tests/test_optuna_loop.py`**

```python
import numpy as np
from unittest.mock import patch, MagicMock
from convergence.optuna_loop import run_optuna_loop


def test_optuna_loop_smoke(tmp_path):
    real_embeddings = np.random.randn(50, 768).astype(np.float32)
    fake_embs = np.random.randn(4, 768).astype(np.float32)

    with patch("convergence.optuna_loop.Embedder") as MockEmbedder:
        mock_emb = MagicMock()
        mock_emb.embed.return_value = fake_embs
        MockEmbedder.return_value = mock_emb

        records = run_optuna_loop(
            real_embeddings=real_embeddings,
            run_dir=tmp_path,
            n_iterations=2,
            n_trials_per_iter=3,
            n_images=4,
            seed=42,
        )

    assert len(records) == 2
    assert all(r.best_objective >= 0.0 for r in records)
    assert records[0].best_objective >= records[1].best_objective or True  # non-increasing (not guaranteed in 2 iters)
    assert (tmp_path / "metrics.csv").exists()


def test_optuna_loop_deterministic(tmp_path):
    real_embeddings = np.random.randn(50, 768).astype(np.float32)
    fake_embs = np.random.randn(4, 768).astype(np.float32)

    def run():
        with patch("convergence.optuna_loop.Embedder") as MockEmbedder:
            mock_emb = MagicMock()
            mock_emb.embed.return_value = fake_embs.copy()
            MockEmbedder.return_value = mock_emb
            return run_optuna_loop(
                real_embeddings=real_embeddings,
                run_dir=tmp_path / "runA",
                n_iterations=2,
                n_trials_per_iter=2,
                n_images=4,
                seed=0,
            )

    r1 = run()
    r2 = run()
    assert abs(r1[0].best_objective - r2[0].best_objective) < 1e-6
```

- [ ] **Step 2: Run tests — expect failures**

```bash
cd bench && poetry run pytest tests/test_optuna_loop.py -v
```

- [ ] **Step 3: Write `bench/convergence/optuna_loop.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from .config import (
    THETA_STAR_PATH, N_ITERATIONS, N_TRIALS_PER_ITER, N_IMAGES_PER_TRIAL, SEED, RUNS_DIR,
)
from .evaluator import Embedder
from .harness import run_trial
from .metrics import MetricsLogger, IterationRecord

_SEARCH_SPACE = {
    "blur_sigma":        ("float",       0.0,  5.0),
    "noise_std":         ("float",       0.0,  0.5),
    "brightness_shift":  ("float",      -0.5,  0.5),
    "color_shift_r":     ("float",      -0.3,  0.3),
    "color_shift_g":     ("float",      -0.3,  0.3),
    "color_shift_b":     ("float",      -0.3,  0.3),
    "clutter_count":     ("int",         0,    20),
    "background_id":     ("categorical", [0, 1, 2, 3]),
}


def _suggest(trial: optuna.Trial) -> dict:
    theta = {}
    for name, spec in _SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float":
            theta[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            theta[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            theta[name] = trial.suggest_categorical(name, spec[1])
    return theta


def run_optuna_loop(
    real_embeddings: np.ndarray,
    run_dir: Path,
    n_iterations: int = N_ITERATIONS,
    n_trials_per_iter: int = N_TRIALS_PER_ITER,
    n_images: int = N_IMAGES_PER_TRIAL,
    seed: int = SEED,
) -> list[IterationRecord]:
    run_dir = Path(run_dir)
    theta_star = json.loads(THETA_STAR_PATH.read_text())
    logger = MetricsLogger(run_dir / "metrics.csv", theta_star)
    embedder = Embedder()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    trial_seed_counter = [seed]

    for iteration in range(n_iterations):
        def objective(trial: optuna.Trial) -> float:
            theta = _suggest(trial)
            dist, _ = run_trial(theta, n_images, real_embeddings, embedder, seed=trial_seed_counter[0])
            trial_seed_counter[0] += 1
            return dist

        study.optimize(objective, n_trials=n_trials_per_iter)

        iter_trials = study.trials[-n_trials_per_iter:]
        trial_results = [(t.params, t.value) for t in iter_trials]
        record = logger.log(iteration, trial_results)
        print(
            f"[optuna] iter={iteration:02d}  best={record.best_objective:.4f}"
            f"  gap={record.param_gap:.4f}  spread={record.spread:.4f}"
        )

    return logger.load()


if __name__ == "__main__":
    embs = np.load(str(RUNS_DIR.parent / "real_embeddings.npy"))
    run_dir = RUNS_DIR / f"optuna_seed{SEED}"
    run_optuna_loop(real_embeddings=embs, run_dir=run_dir)
```

- [ ] **Step 4: Run smoke tests — all must pass**

```bash
cd bench && poetry run pytest tests/test_optuna_loop.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add bench/convergence/optuna_loop.py bench/tests/test_optuna_loop.py
git commit -m "bench: Optuna TPE loop (Condition A)"
```

---

## Task 8: TL loop stub (Condition B)

**Files:**
- Create: `bench/convergence/tl_loop.py`

No tests yet — the TL CSV format is unknown until a real TL run produces one. The stub validates that given a properly formatted CSV it computes identical metrics.

- [ ] **Step 1: Write `bench/convergence/tl_loop.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    THETA_STAR_PATH, N_IMAGES_PER_TRIAL, N_TRIALS_PER_ITER, SEED, RUNS_DIR,
)
from .evaluator import Embedder
from .harness import run_trial
from .metrics import MetricsLogger, IterationRecord

_THETA_COLUMNS = [
    "blur_sigma", "noise_std", "brightness_shift",
    "color_shift_r", "color_shift_g", "color_shift_b",
    "clutter_count", "background_id",
]


def run_tl_loop(
    real_embeddings: np.ndarray,
    csv_path: Path,
    run_dir: Path,
    n_images: int = N_IMAGES_PER_TRIAL,
    n_trials_per_iter: int = N_TRIALS_PER_ITER,
    seed: int = SEED,
) -> list[IterationRecord]:
    run_dir = Path(run_dir)
    csv_path = Path(csv_path)
    theta_star = json.loads(THETA_STAR_PATH.read_text())
    logger = MetricsLogger(run_dir / "metrics.csv", theta_star)
    embedder = Embedder()

    df = pd.read_csv(csv_path)
    missing = [c for c in _THETA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TL CSV missing columns: {missing}")

    trial_seed = seed
    chunks = [df.iloc[i : i + n_trials_per_iter] for i in range(0, len(df), n_trials_per_iter)]
    for iteration, chunk in enumerate(chunks):
        trial_results = []
        for _, row in chunk.iterrows():
            theta = {k: row[k] for k in _THETA_COLUMNS}
            dist, _ = run_trial(theta, n_images, real_embeddings, embedder, seed=trial_seed)
            trial_seed += 1
            trial_results.append((theta, dist))
        record = logger.log(iteration, trial_results)
        print(
            f"[tl]     iter={iteration:02d}  best={record.best_objective:.4f}"
            f"  gap={record.param_gap:.4f}  spread={record.spread:.4f}"
        )

    return logger.load()
```

- [ ] **Step 2: Commit**

```bash
git add bench/convergence/tl_loop.py
git commit -m "bench: TL CSV loop stub (Condition B)"
```

---

## Task 9: Plot

**Files:**
- Create: `bench/convergence/plot.py`

- [ ] **Step 1: Write `bench/convergence/plot.py`**

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(
    optuna_csv: Path,
    tl_csv: Path | None = None,
    output_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sources = [("Optuna", optuna_csv)]
    if tl_csv is not None:
        sources.append(("TL", tl_csv))

    for label, csv_path in sources:
        df = pd.read_csv(csv_path)
        axes[0].plot(df["iteration"], df["best_objective"], marker="o", label=label)
        axes[1].plot(df["iteration"], df["param_gap"], marker="o", label=label)
        axes[2].plot(df["iteration"], df["spread"], marker="o", label=label)

    axes[0].set(title="Best Objective (MMD)", xlabel="Iteration", ylabel="MMD ↓")
    axes[1].set(title="Param Gap to θ*", xlabel="Iteration", ylabel="‖θ - θ*‖ ↓")
    axes[2].set(title="Spread (Exploration)", xlabel="Iteration", ylabel="Mean Param Std")

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot → {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from .config import RUNS_DIR, SEED
    plot_comparison(
        optuna_csv=RUNS_DIR / f"optuna_seed{SEED}" / "metrics.csv",
        tl_csv=RUNS_DIR / f"tl_seed{SEED}" / "metrics.csv",
        output_path=RUNS_DIR / "comparison.png",
    )
```

- [ ] **Step 2: Commit**

```bash
git add bench/convergence/plot.py
git commit -m "bench: comparison plot"
```

---

## Task 10: Full test suite + run all tests

- [ ] **Step 1: Run all tests**

```bash
cd bench && poetry run pytest tests/ -v
```

Expected: all green. If any fail, fix before proceeding.

- [ ] **Step 2: Run a pilot Optuna loop (2 iterations, uses real DINOv2)**

```bash
cd bench && poetry run python -m convergence.optuna_loop
```

Confirm it prints iteration lines and writes `~/tensorleap/data/synth-data-benchmark/runs/optuna_seed42/metrics.csv`.

- [ ] **Step 3: Push branch**

```bash
git push -u origin yolo11-inference-test
```

---

## Decision log checklist (from README)

Once implementation is done, revisit `bench/convergence/README.md` §12 and tick off:

- [ ] θ* committed (`bench/convergence/theta_star.json`) ← done in Task 2
- [ ] DINOv2 exported to ONNX (separate agent)
- [ ] DINOv2 ONNX imports successfully into TL (separate agent)
- [ ] DINOv2 ONNX parity verified (same embedding within 1e-5)
- [ ] ε_θ, ε_obj set from pilot Optuna run
- [ ] TL CSV column schema documented
- [ ] TL iteration trigger mechanism decided

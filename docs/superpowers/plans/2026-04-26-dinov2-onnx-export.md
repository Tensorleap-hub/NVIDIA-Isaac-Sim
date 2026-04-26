# DINOv2 ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export `dinov2_vitb14_reg` to ONNX with parity verification, so both the local Optuna loop and Tensorleap share an identical evaluator.

**Architecture:** Single script (`export_dinov2.py`) that loads the model via `torch.hub`, exports to ONNX opset 14 (preprocessed float32 tensors in, CLS-token embeddings out), verifies numerical parity against `onnxruntime`, and writes a SHA256 hash. The ONNX file is gitignored; only the hash is committed.

**Tech Stack:** Python 3.10.12, Poetry, PyTorch 2.1.2, torchvision 0.16.2, onnxruntime ≥1.17, numpy <2

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `bench/convergence/pyproject.toml` | Create | Poetry env declaration |
| `bench/convergence/.gitignore` | Create | Ignore `*.onnx` |
| `bench/convergence/export_dinov2.py` | Create | Load → export → verify → hash |
| `bench/convergence/dinov2_onnx_hash.txt` | Generated | SHA256 written by script at runtime |
| `.gitignore` | Modify | Add `*.onnx` at root level too |

---

### Task 1: Poetry environment

**Files:**
- Create: `bench/convergence/pyproject.toml`

- [ ] **Step 1: Initialise poetry project**

```bash
cd bench/convergence && poetry init \
  --name dinov2-bench \
  --python "~3.10" \
  --no-interaction
```

- [ ] **Step 2: Add dependencies**

```bash
cd bench/convergence && \
  poetry add "torch==2.1.2" "torchvision==0.16.2" "onnxruntime>=1.17,<2" "numpy<2" "Pillow>=10.0"
```

Expected: `poetry.lock` created, no errors.

- [ ] **Step 3: Verify Python version**

```bash
cd bench/convergence && poetry run python --version
```

Expected output: `Python 3.10.12`

- [ ] **Step 4: Commit**

```bash
git add bench/convergence/pyproject.toml bench/convergence/poetry.lock
git commit -m "add poetry env for convergence bench"
```

---

### Task 2: Gitignore

**Files:**
- Create: `bench/convergence/.gitignore`
- Modify: `.gitignore` (root)

- [ ] **Step 1: Create bench-level gitignore**

Create `bench/convergence/.gitignore` with content:
```
*.onnx
```

- [ ] **Step 2: Add *.onnx to root .gitignore**

Append `*.onnx` to the existing `.gitignore` at the repo root.

- [ ] **Step 3: Verify the ONNX path would be ignored**

```bash
git check-ignore -v bench/convergence/dinov2_vitb14_reg.onnx
```

Expected: a line showing the matching gitignore rule (e.g. `.gitignore:4:*.onnx`). If output is empty, the ignore is not active — check the file was saved correctly.

- [ ] **Step 4: Commit**

```bash
git add bench/convergence/.gitignore .gitignore
git commit -m "gitignore onnx files"
```

---

### Task 3: Export script

**Files:**
- Create: `bench/convergence/export_dinov2.py`

- [ ] **Step 1: Write the script**

Create `bench/convergence/export_dinov2.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


OPSET = 14
INPUT_SHAPE = (1, 3, 224, 224)
PARITY_TOL = 1e-5
DEFAULT_OUT = Path(__file__).parent / "dinov2_vitb14_reg.onnx"
HASH_FILE = Path(__file__).parent / "dinov2_onnx_hash.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export dinov2_vitb14_reg to ONNX and verify parity")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_model(device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    model.eval()
    model.to(torch.device(device))
    return model


def export(model: torch.nn.Module, out: Path, device: str) -> None:
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=torch.device(device))
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=OPSET,
            export_params=True,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["embeddings"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "embeddings": {0: "batch"},
            },
        )
    print(f"Exported → {out}")


def verify(model: torch.nn.Module, out: Path, device: str) -> None:
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=torch.device(device))
    with torch.inference_mode():
        pt_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["embeddings"], {"pixel_values": dummy.cpu().numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    print(f"Parity max |pt − onnx| = {max_diff:.2e}")
    if max_diff >= PARITY_TOL:
        print(f"FAIL: max diff {max_diff:.2e} ≥ tolerance {PARITY_TOL:.2e}", file=sys.stderr)
        sys.exit(1)
    print("Parity OK")


def write_hash(out: Path) -> None:
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    HASH_FILE.write_text(f"{sha256}  {out.name}\n")
    print(f"SHA256 written → {HASH_FILE}")


def main() -> None:
    args = parse_args()
    print(f"Loading dinov2_vitb14_reg on {args.device}…")
    model = load_model(args.device)
    print("Exporting to ONNX…")
    export(model, args.out, args.device)
    print("Verifying parity…")
    verify(model, args.out, args.device)
    write_hash(args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the script**

```bash
git add bench/convergence/export_dinov2.py
git commit -m "add dinov2 onnx export script"
```

---

### Task 4: Run the export

**Files:**
- Generated: `bench/convergence/dinov2_vitb14_reg.onnx` (gitignored)
- Generated: `bench/convergence/dinov2_onnx_hash.txt` (to be committed)

- [ ] **Step 1: Run the export script**

```bash
cd bench/convergence && poetry run python export_dinov2.py --device cpu
```

Expected output (approximate):
```
Loading dinov2_vitb14_reg on cpu…
Using cache found in …
Exporting to ONNX…
Exported → bench/convergence/dinov2_vitb14_reg.onnx
Verifying parity…
Parity max |pt − onnx| = <value less than 1e-05>
Parity OK
SHA256 written → bench/convergence/dinov2_onnx_hash.txt
```

If parity fails (exits non-zero), check the torch version matches 2.1.2 and that no grad is flowing. Do not proceed to the next step if this fails.

- [ ] **Step 2: Confirm hash file was written**

```bash
cat bench/convergence/dinov2_onnx_hash.txt
```

Expected: one line like `a3f1…  dinov2_vitb14_reg.onnx`

- [ ] **Step 3: Confirm ONNX file is gitignored**

```bash
git status bench/convergence/
```

Expected: `dinov2_vitb14_reg.onnx` does NOT appear in untracked files. `dinov2_onnx_hash.txt` appears as untracked.

- [ ] **Step 4: Commit the hash file**

```bash
git add bench/convergence/dinov2_onnx_hash.txt
git commit -m "add dinov2 onnx sha256 hash"
```

---

## Post-plan manual step (out of scope for this plan)

Upload `bench/convergence/dinov2_vitb14_reg.onnx` to Tensorleap and confirm it parses without errors. Mark the checkbox in `bench/convergence/README.md` §12: "DINOv2 ONNX imports successfully into TL".

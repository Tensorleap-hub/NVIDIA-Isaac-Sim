from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


OPSET = 14
INPUT_SHAPE = (1, 3, 224, 224)
PARITY_TOL = 1e-4
DEFAULT_OUT = Path(__file__).parent / "dinov2_vits14.onnx"
HASH_FILE = Path(__file__).parent / "dinov2_onnx_hash.txt"


class _DinoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export dinov2_vits14 to ONNX and verify parity")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_model(device: str) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(torch.device(device))
    return model


def export(model: torch.nn.Module, out: Path, device: str) -> None:
    wrapper = _DinoWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=torch.device(device))
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
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
    print(f"Loading dinov2_vits14 on {args.device}…")
    model = load_model(args.device)
    print("Exporting to ONNX…")
    export(model, args.out, args.device)
    print("Verifying parity…")
    verify(model, args.out, args.device)
    write_hash(args.out)


if __name__ == "__main__":
    main()

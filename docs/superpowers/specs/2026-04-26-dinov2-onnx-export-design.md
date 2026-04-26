# DINOv2 ONNX Export — Design Spec

**Date:** 2026-04-26  
**Status:** Approved  
**Branch:** dinov2-benchmark

## Goal

Export `dinov2_vitb14_reg` to ONNX so both the local Optuna loop and Tensorleap can use the identical evaluator. This is the prerequisite blocker for Condition B of the convergence benchmark (§4.3, §12 of `bench/convergence/README.md`).

## Scope

Single export + verify script. No preprocessing baked in — ONNX input is already-normalized float32 tensors. No git-tracked ONNX artifact (too large); only the SHA256 hash is committed.

## Design

### Script: `bench/convergence/export_dinov2.py`

Three sequential steps:

1. **Load** — `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")` in eval mode. Default device: CPU. Optional `--device` flag accepts `cpu`, `mps`, `cuda`.

2. **Export** — `torch.onnx.export` at opset 14.
   - Input: `pixel_values`, shape `[N, 3, 224, 224]`, dtype `float32`
   - Output: `embeddings`, shape `[N, 768]`, dtype `float32`
   - Batch dimension dynamic (`N` unconstrained)
   - Dummy input: `[1, 3, 224, 224]` random float32

3. **Verify + hash**
   - Load exported ONNX with `onnxruntime.InferenceSession`
   - Run the same dummy input through both PyTorch and ONNX
   - Assert `max(|pt_out - onnx_out|) < 1e-5`; print actual max diff
   - Compute SHA256 of the `.onnx` file; write to `dinov2_onnx_hash.txt`
   - Script exits non-zero if parity fails

### File layout

```
bench/convergence/
  export_dinov2.py       ← script (committed)
  dinov2_onnx_hash.txt   ← SHA256 of the .onnx (committed)
  pyproject.toml         ← poetry env (committed)
  .gitignore             ← *.onnx (not committed)
  dinov2_vitb14_reg.onnx ← generated locally, gitignored
```

### Python environment

Poetry project at `bench/convergence/`, python 3.10.12.

| Package | Pin |
|---|---|
| torch | 2.1.2 |
| torchvision | 0.16.2 |
| onnxruntime | ^1.17 |
| numpy | <2 |
| Pillow | ^10.0 |

### ONNX export parameters

| Parameter | Value | Rationale |
|---|---|---|
| opset_version | 14 | Broad compatibility; supported by onnxruntime ≥1.14 and most TL parsers |
| export_params | True | Bake weights into the file |
| do_constant_folding | True | Reduce graph size |
| input_names | `["pixel_values"]` | Matches DINOv2 HuggingFace convention |
| output_names | `["embeddings"]` | CLS token output |
| dynamic_axes | batch dim on both I/O | Allow variable batch sizes at inference |

## Success criteria

- Script runs to completion without error
- Parity check passes: `max(|pt - onnx|) < 1e-5`
- `dinov2_onnx_hash.txt` written
- ONNX loads in Tensorleap without parse errors (manual step, post-export)

## Out of scope

- Preprocessing baked into ONNX
- TL upload automation
- git-lfs or S3 artifact storage

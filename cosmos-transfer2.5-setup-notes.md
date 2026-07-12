# Cosmos-Transfer2.5 Inference — Setup Notes

Learnings from getting `nvidia-cosmos/cosmos-transfer2.5` running end-to-end on a
4x NVIDIA A10G (24GB) instance, generating a stylized video from Isaac Sim
trajectory SDG output (`palletjack_sdg` / `standalone_palletjack_trajectory_sdg.py`
clips under `video/clip_0000/{rgb,depth,edges,segmentation,shaded_seg}.mp4`).

Repo used: `/home/ubuntu/cosmos-work/repo/cosmos-transfer2.5` (this session's box).
Data used: `/mnt/cosmos/input/<experiment>/video/clip_0000/*.mp4`.

## 1. Storage gotcha: `/mnt/cosmos` is ephemeral

On this instance, the large data disk mounted at `/mnt/cosmos` is **NVMe
instance store** (`Amazon EC2 NVMe Instance Storage`, mounted via a systemd
unit `mnt-cosmos.mount`, not `/etc/fstab`) — **not EBS**. It survives a plain
reboot but is wiped on stop/terminate or host replacement. Only `/dev/root`
(the actual EBS root volume) is durable. If input/output data here matters,
sync it to S3 or a real EBS volume before tearing down the instance.

The HF checkpoint cache (`HF_HOME`) was pointed at
`/home/ubuntu/cosmos-work/hf-cache`, which is on the durable root volume —
checkpoints don't need re-downloading across restarts of the same instance,
even though `/mnt/cosmos` data would be lost.

## 2. Environment setup

- Repo comes with a `.venv` (uv-managed) but **the CUDA extra must be synced
  explicitly**: `uv sync --extra=<cuda_name> --active --inexact`.
- **Python 3.13 + `cu128` extra fails**: `flash-attn==2.7.3+cu128.torch27` only
  ships a `cp310` wheel. Use `--extra=cu130` instead — that release has a
  `cp313` wheel (`flash_attn-...cu130.torch29-cp313-...whl`, v1.5.0 release).
  Check `.python-version` and the installed driver/CUDA version before
  picking; `cu130` needs a reasonably new driver (595.x here) which was fine.
- After `uv sync --extra=cu130`, `cosmos_transfer2/__init__.py`'s
  `_check_cuda_extra()` needs the `cosmos_cuda` package importable — this
  comes from the sync, not a separate step.
- **Missing `libcudart.so` (unversioned)**: pip's `nvidia-cuda-runtime` /
  `nvidia-cu13` packages only ship versioned `.so.13`/`.so.12` files, but
  `cosmos_transfer2/_src/imaginaire/utils/distributed.py` does
  `ctypes.CDLL("libcudart.so")` (no version). Fix: symlink it and put all
  `nvidia/*/lib` site-package dirs on `LD_LIBRARY_PATH` before running:
  ```bash
  ln -sf libcudart.so.13 .venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so
  export LD_LIBRARY_PATH=$(python3 -c "
  import glob, os
  base = '<venv>/lib/python3.13/site-packages/nvidia'
  print(':'.join(d for d in glob.glob(base + '/*/lib') if os.path.isdir(d)))
  ")
  ```
- **Missing ffmpeg/libavformat**: video decode fails with
  `libavformat.so.60: cannot open shared object file`. The setup guide's
  system-dependency line was never run. Fix: `sudo apt update && sudo apt -y
  install curl ffmpeg libx11-dev tree wget`.
- `uv`/`hf` CLIs resolve fine in a shell that inherits the venv-activated
  `PATH` (e.g. the same shell Claude Code was launched from), but a fresh
  interactive shell (or `!` passthrough) may not have that PATH — you may
  need to `source .venv/bin/activate` explicitly there.

## 3. Hugging Face auth

- `hf auth login` (interactive) fails in non-TTY contexts with a `getpass`
  `EOFError`. Use `hf auth login --token <token>` instead.
- Three separate **gated** repos need their license individually accepted by
  the HF account (accepting one does not imply the others):
  `nvidia/Cosmos-Guardrail1`, `nvidia/Cosmos-Transfer2.5-2B`,
  `nvidia/Cosmos-Predict2.5-2B` (the last one is a hidden dependency — the
  base model shares Predict2.5's VAE/tokenizer — and only surfaces as an
  `Access denied` error mid-run, not upfront).
- `HfApi.model_info(repo)` succeeding does **not** confirm download access
  for gated repos — it only confirms the repo/metadata is visible. Verify
  with an actual `hf_hub_download(repo, some_file)` call before trusting it.

## 4. Guardrail false positive: "jerry"

The text-prompt guardrail runs a plain keyword blocklist *before* any ML
classifier (`cosmos_transfer2/_src/imaginaire/auxiliary/guardrail/blocklist/blocklist.py`).
It does whole-word exact matching with no context awareness. Our prompt's
"**jerry cans**" hard-matched a blocklist entry `Jerry` (line 596 of
`blocklist/exact_match/blocked` in the downloaded `Cosmos-Guardrail1`
snapshot) — almost certainly meant to block the "Tom and Jerry" cartoon
character, not fuel containers. No CLI-level word-specific override exists;
options are: reword the prompt (e.g. "fuel canisters"), or pass
`--disable-guardrails` to skip both the blocklist and the downstream
Qwen3Guard ML classifier entirely for local/offline runs.

## 5. GPU memory on 4x A10G (24GB each)

The documented VRAM requirement (65.4GB) is for **single-control** on one
GPU; it is *not* representative of multicontrol or of a context-parallel
split across smaller GPUs. Observed on this hardware:

- **Multicontrol** (edge+depth+seg combined spec) loads all 4 modality
  branches (edge/blur/depth/seg) regardless of which hint keys you actually
  use, and OOMs immediately at model-load time even split via
  `context_parallel_size=4` (~22GB/GPU used just loading, before sampling).
- **Single-control** (`seg` only) fits the model-load phase (~20GB/GPU) but
  still OOM'd during the actual diffusion sampling step at the full 128-frame
  clip length — activation memory for a 960x544x128 clip exceeds headroom.
- **Capping `max_frames` to 93 (the model's native chunk size) did NOT fix
  the OOM.** Tested and confirmed: identical failure signature at 93 frames
  as at 128 (`Tried to allocate 2.58 GiB`, ~19-20GiB already in use, out of
  22.06GiB total). This means the bottleneck on this hardware is the
  **fixed model-weight footprint (~19-20GB per GPU)**, not activation memory
  that scales with frame count — trimming frames doesn't help. A10G's 24GB
  is simply undersized for this model even at single-control, 1 chunk,
  `context_parallel_size=4`.
- On a bigger/more-VRAM machine (A100 40/80GB, H100, etc.) both multicontrol
  and full frame count should be fine without any of these workarounds —
  they were purely a consequence of the A10G's 24GB ceiling, not a repo
  limitation. **Recommendation for the next machine**: check
  `nvidia-smi` VRAM per GPU; if ≥40GB per GPU, single-GPU inference
  (no torchrun/context-parallel needed) should work per the documented
  65.4GB-total / reasonable-per-GPU-share math. If still on ~24GB-class
  GPUs, budget for ≥5-6 GPUs of headroom or investigate model sharding
  (FSDP) beyond context-parallel, which this repo's CLI doesn't expose
  directly.
- `--offload-guardrail-models` (moves Qwen2.5-VL-7B / Cosmos-Reason1(.1)-7B
  guardrail models to CPU) is worth keeping regardless of GPU size — it's
  "free" VRAM headroom for the main diffusion model with no quality cost.

## 6. Spec JSON field gotchas

- The per-sample JSON schema (`InferenceArguments` in `cosmos_transfer2/config.py`)
  uses `pydantic.ConfigDict(extra="forbid")` — an unrecognized key fails the
  *entire run* at startup (after torchrun has already spun up all ranks), not
  a warning. Confirmed gotcha: the frame-count field is **`max_frames`**, not
  `num_frames` (which doesn't exist and throws `extra_forbidden`).
- Control blocks are keyed `depth` / `edge` / `seg` / `vis` (vis = blur), each
  with `control_path` (mp4) and `control_weight` (0-1, auto-normalized to
  sum ≤1 across multicontrol). `seg` also accepts `control_prompt` (defaults
  to first 128 words of the main prompt) when segmentation is computed
  on-the-fly instead of from a provided video.
- Validate a spec cheaply before a full run:
  ```python
  from pathlib import Path
  from cosmos_transfer2.config import InferenceArguments, InferenceOverrides
  samples, hint_keys = InferenceArguments.from_files([Path("spec.json")], overrides=InferenceOverrides())
  ```
  (Needs the CUDA extra importable, so run inside the synced venv, but this
  step itself doesn't touch the GPU — cheap to run before committing to a
  multi-minute torchrun.)

## 7. Launch command template

```bash
cd cosmos-transfer2.5
LD_LIBRARY_PATH="<nvidia lib dirs, see above>" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=<N_GPUS> --master_port=12341 examples/inference.py \
  -i <spec.json> \
  -o <output_dir> \
  --offload-guardrail-models \
  [--disable-guardrails]
```

Checkpoints auto-download to `HF_HOME` on first use per modality actually
referenced in the spec (e.g. single-control `seg` only pulls the `seg`
checkpoint, not edge/blur/depth) — subsequent runs against the same
`HF_HOME` skip straight to model loading.

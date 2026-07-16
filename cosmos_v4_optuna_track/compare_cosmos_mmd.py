"""Compare Cosmos-Transfer clip sets against the real LOCO reference with DINOv2 + MMD.

This is step 2 ("Compare (DINOv2 MMD)") of cosmos_v4_optuna_track/README.md. It wires
together the two building blocks that already exist in the repo but were only ever used
*inside* the Optuna loop:

  * embedding  -> ``simulation_calibration_loop.data.DINOv2Embedder`` / ``select_real_image_paths``
  * MMD metric -> ``calibration_optuna.metrics.DistributionMetrics.mmd`` (RBF, median-heuristic gamma)

For each named clip set it:
  1. decodes every Nth frame of each Cosmos output video (``<clip>.mp4``, skipping the
     ``_control_*`` conditioning videos) into a PNG cache, OR takes an already-extracted
     image dir as-is (auto-detected: a dir with ``.mp4`` files -> decode; with images -> use)
  2. embeds those frames with DINOv2 (``dinov2_vitb14_reg``, same config as the loop)
It also embeds the real LOCO reference pool, then reports:
  * MMD(set, real) for every set  -- how far each stylized set sits from real footage
  * MMD(set_i, set_j) pairwise    -- e.g. base_v4-Cosmos vs optuna-Cosmos

A single RBF gamma is computed once from the real embeddings (median heuristic) and reused
for every comparison, so all reported MMDs are on one comparable scale -- this mirrors
``SimulationCalibrationController`` / ``compute_distribution_metrics_for_population``.

Usage:
    python cosmos_v4_optuna_track/compare_cosmos_mmd.py \
        --set base_v4=palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260712_cosmos_transfer_all_seeds_2prompts,\
palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260713_cosmos_transfer_new_themes_2seeds \
        --set optuna=palletjack_sdg/palletjack_data/trajectory/cosmos_transfer/20260713_cosmos_transfer_optuna_23runs_1seed \
        --frame-stride 15 \
        --report cosmos_v4_optuna_track/cosmos_mmd_report.json

Real reference defaults to loco_dataset + the warehouse3cls_traj_v2 valid annotations (the
same real pool the trajectory Optuna search scored against). Run with the loop venv, which
has the DINOv2 hub weights cached:  ``.sim_loop_venv/bin/python``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_optuna.metrics import DistributionMetrics  # noqa: E402
from simulation_calibration_loop.data import (  # noqa: E402
    IMAGE_SUFFIXES,
    DINOv2Embedder,
    select_real_image_paths,
)

import cv2  # noqa: E402  (used for video decode; provided by opencv-python-headless)

# DINOv2 defaults, kept in sync with simulation_calibration_loop/config.py DINOv2Config.
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vitb14_reg"
IMAGE_SIZE = 224
RESIZE_SIZE = 256
EMBED_BATCH_SIZE = 32

DEFAULT_REAL_ROOT = "/home/ubuntu/loco_dataset"
DEFAULT_REAL_ANNOTATIONS = "/home/ubuntu/warehouse3cls_traj_v2/valid/_annotations.coco.json"


def parse_set_arg(raw: str) -> tuple[str, list[Path]]:
    """Parse a ``name=dir1,dir2,...`` --set value into (name, [paths])."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--set must be name=dir1,dir2,...  (got: {raw!r})")
    name, dirs = raw.split("=", 1)
    paths = [Path(d).expanduser() for d in dirs.split(",") if d.strip()]
    if not name.strip() or not paths:
        raise argparse.ArgumentTypeError(f"--set must be name=dir1,dir2,...  (got: {raw!r})")
    return name.strip(), paths


def extract_frames_from_videos(video_dir: Path, cache_dir: Path, frame_stride: int) -> list[Path]:
    """Decode every ``frame_stride``-th frame of each non-control mp4 into cache_dir as PNGs.

    Skips ``*_control_*.mp4`` (depth/edge/seg conditioning) and re-uses already-extracted
    frames so repeated runs are cheap.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for mp4 in sorted(video_dir.glob("*.mp4")):
        if "_control_" in mp4.name:
            continue
        stem = mp4.stem
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"    WARN: failed to open {mp4.name}")
            continue
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                out = cache_dir / f"{stem}_{idx:04d}.png"
                if not out.exists():
                    cv2.imwrite(str(out), frame)
                frame_paths.append(out)
            idx += 1
        cap.release()
    return frame_paths


def collect_image_paths(directory: Path) -> list[Path]:
    """Return image files directly under (and below) an already-extracted frames dir."""
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def gather_set_frames(name: str, dirs: list[Path], cache_root: Path, frame_stride: int) -> list[Path]:
    """Resolve a named set's frame paths: decode video dirs, pass through image dirs."""
    frames: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"  [{name}] SKIP (not a directory): {d}")
            continue
        has_video = any(m for m in d.glob("*.mp4") if "_control_" not in m.name)
        if has_video:
            cache_dir = cache_root / name / d.name
            got = extract_frames_from_videos(d, cache_dir, frame_stride)
            print(f"  [{name}] {d.name}: decoded {len(got)} frames (stride {frame_stride})")
            frames.extend(got)
        else:
            got = collect_image_paths(d)
            print(f"  [{name}] {d.name}: found {len(got)} images")
            frames.extend(got)
    return frames


def subsample(paths: list[Path], max_samples: int, rng: np.random.Generator) -> list[Path]:
    if max_samples <= 0 or len(paths) <= max_samples:
        return paths
    idx = rng.choice(len(paths), size=max_samples, replace=False)
    return [paths[i] for i in sorted(idx)]


def embed(embedder: DINOv2Embedder, paths: list[Path], cache_path: Path) -> np.ndarray:
    """Embed a list of image paths, caching by a manifest of (params + sorted paths)."""
    manifest = {
        "model": DINOV2_MODEL,
        "image_size": IMAGE_SIZE,
        "resize_size": RESIZE_SIZE,
        "paths": [str(p) for p in paths],
    }
    return embedder.embed_paths(
        paths,
        batch_size=EMBED_BATCH_SIZE,
        cache_path=cache_path,
        manifest=manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DINOv2 + MMD comparison of Cosmos-Transfer clip sets against the real LOCO reference",
    )
    parser.add_argument("--set", dest="sets", action="append", type=parse_set_arg, required=True,
                        metavar="NAME=DIR1,DIR2,...",
                        help="A named clip set: video dirs (Cosmos outputs) or already-extracted image dirs. "
                             "Repeat --set for each set to compare.")
    parser.add_argument("--real-root", default=DEFAULT_REAL_ROOT,
                        help=f"Real dataset root (default: {DEFAULT_REAL_ROOT})")
    parser.add_argument("--real-annotations", default=DEFAULT_REAL_ANNOTATIONS,
                        help=f"Real LOCO-style annotations json (default: {DEFAULT_REAL_ANNOTATIONS})")
    parser.add_argument("--frame-stride", type=int, default=15,
                        help="Keep every Nth frame when decoding videos (default: 15; ~9 frames/128-frame clip)")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Subsample each distribution to this many frames before MMD (default: 1000; "
                             "MMD is O(n^2), and this matches the loop's mmd_max_samples)")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "od_scripts" / "data" / "cosmos_mmd"),
                        help="Where to cache decoded frames and embeddings (git-ignored under od_scripts/data/)")
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None, help="Optional path to write the MMD report as JSON")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cache_root = Path(args.cache_dir)
    frames_cache = cache_root / "frames"
    embed_cache = cache_root / "embeddings"
    embed_cache.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if _cuda_available() else "cpu")
    print(f"Embedder: DINOv2 {DINOV2_MODEL} on {device}\n")

    # ---- Real reference ----
    print("Real reference:")
    real_paths = select_real_image_paths(args.real_root, args.real_annotations)
    if not real_paths:
        raise SystemExit(f"No real images resolved from {args.real_root} + {args.real_annotations}")
    real_paths = subsample(real_paths, args.max_samples, rng)
    print(f"  loco: {len(real_paths)} images (after subsample to {args.max_samples})\n")

    # ---- Cosmos sets ----
    print("Cosmos sets:")
    set_paths: dict[str, list[Path]] = {}
    for name, dirs in args.sets:
        frames = gather_set_frames(name, dirs, frames_cache, args.frame_stride)
        if not frames:
            print(f"  [{name}] WARNING: no frames resolved, dropping this set")
            continue
        set_paths[name] = subsample(frames, args.max_samples, rng)
        print(f"  [{name}] TOTAL {len(frames)} frames -> {len(set_paths[name])} after subsample\n")
    if not set_paths:
        raise SystemExit("No cosmos sets produced any frames")

    # ---- Embed ----
    embedder = DINOv2Embedder(DINOV2_REPO, DINOV2_MODEL, device, IMAGE_SIZE, RESIZE_SIZE)
    print("Embedding real reference...")
    real_emb = embed(embedder, real_paths, embed_cache / "real.npy")
    embeddings: dict[str, np.ndarray] = {}
    for name, paths in set_paths.items():
        print(f"Embedding set '{name}'...")
        embeddings[name] = embed(embedder, paths, embed_cache / f"set_{name}.npy")

    # ---- MMD: single shared gamma from real (median heuristic), reused everywhere ----
    gamma = DistributionMetrics._compute_gamma_median_heuristic(real_emb, real_emb)
    print(f"\nShared RBF gamma (median heuristic on real): {gamma:.6g}\n")

    vs_real = {name: DistributionMetrics.mmd(emb, real_emb, gamma=gamma) for name, emb in embeddings.items()}

    names = list(embeddings)
    pairwise: dict[str, dict[str, float]] = {a: {} for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = DistributionMetrics.mmd(embeddings[a], embeddings[b], gamma=gamma)
            pairwise[a][b] = m
            pairwise[b][a] = m

    # ---- Report ----
    print("=" * 60)
    print("MMD vs REAL (lower = closer to real footage):")
    for name in sorted(vs_real, key=vs_real.get):
        print(f"  {name:24s} {vs_real[name]:.6f}   (n={len(embeddings[name])})")
    if len(names) > 1:
        print("\nPairwise MMD between cosmos sets:")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"  {a} <-> {b}: {pairwise[a][b]:.6f}")
    print("=" * 60)

    report = {
        "embedder": {"backend": "dinov2", "model": DINOV2_MODEL, "image_size": IMAGE_SIZE, "resize_size": RESIZE_SIZE},
        "real": {"root": args.real_root, "annotations": args.real_annotations, "n": len(real_emb)},
        "frame_stride": args.frame_stride,
        "max_samples": args.max_samples,
        "rbf_gamma": float(gamma),
        "sets": {name: {"n": len(embeddings[name]), "dirs": [str(d) for d in dict(args.sets)[name]]}
                 for name in names},
        "mmd_vs_real": {k: float(v) for k, v in vs_real.items()},
        "mmd_pairwise": {a: {b: float(v) for b, v in row.items()} for a, row in pairwise.items()},
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {args.report}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
